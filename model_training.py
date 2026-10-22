"""
IEEE-CIS Fraud Detection — Spark MLlib Model Training (v3)
Improvements based on xgb-fraud-with-magic-0-9600 notebook:
  - Magic UID: card1_addr1 + floor(day - D1) — behavioural session identity
  - D-column normalization: subtract days since ref to remove time drift
  - Cents feature: fractional part of TransactionAmt
  - Frequency encoding: card/email/uid columns
  - UID aggregations: mean/std amount, count-unique email per session
  - outsider15 anomaly flag: |D1 - D15| > 3
  - Curated V-column selection from notebook's feature importance analysis
  - Stronger GBT: maxIter=150, maxDepth=8, stepSize=0.05
Output : models/fraud_gbt/
         models/fraud_gbt_metadata.json
"""

import os, json, time

os.environ["JAVA_HOME"] = (
    "/Users/shaunakkulkarni/Library/Java/JavaVirtualMachines/ms-17.0.17/Contents/Home"
)
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

DATA_PATH  = "data/cleaned_fraud.parquet"
MODEL_PATH = "models/fraud_gbt"
META_PATH  = "models/fraud_gbt_metadata.json"
REF_EPOCH_SECS = 1512000000  # 2017-11-30 00:00:00 UTC

os.makedirs("models", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("FraudDetection-ModelTraining-v3")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "24")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading cleaned data...")
df = spark.read.parquet(DATA_PATH)
print(f"  {df.count():,} rows × {len(df.columns)} cols")

# ── 2. Drop features the notebook flags as temporally inconsistent ────────────
drop_bad = [c for c in ["C3", "M5"] if c in df.columns]
if drop_bad:
    df = df.drop(*drop_bad)
    print(f"  Dropped temporally inconsistent cols: {drop_bad}")

# ── 3. Curated V-column selection (notebook's feature importance analysis) ────
# Intersection of notebook's chosen indices and cols we actually have
NOTEBOOK_V = {
    1,3,4,6,8,11,13,14,17,20,23,26,27,30,36,37,40,41,44,47,48,
    54,56,59,62,65,67,68,70,76,78,80,82,86,88,89,91,
    107,108,111,115,117,120,121,123,124,127,129,130,136,
    281,283,284,285,286,289,291,294,296,297,301,303,305,
    307,309,310,314,320,
}
v_keep = {f"V{i}" for i in NOTEBOOK_V}
v_all  = {c for c in df.columns if c.startswith("V")}
v_drop = v_all - v_keep
df = df.drop(*v_drop)
print(f"  V cols: kept {len(v_all - v_drop)}, dropped {len(v_drop)} not in notebook selection")

# ── 4. Derive transaction day (replaces TransactionDT which was dropped) ──────
df = df.withColumn(
    "tx_day",
    (F.unix_timestamp("transaction_timestamp") - REF_EPOCH_SECS) / 86400.0,
)

# ── 5. D-column normalization: subtract days-since-ref to remove time drift ───
# Notebook normalises D4, D10, D11, D15 (NOT D1, D2, D3, D5 — those are raw days)
for d_col in ["D4", "D10", "D11", "D15"]:
    if d_col in df.columns:
        df = df.withColumn(d_col, F.col(d_col) - F.col("tx_day"))

# ── 6. Cents feature: fractional part of amount ───────────────────────────────
# Fraudsters tend to use round-number amounts; legitimate txns have varied cents
df = df.withColumn("cents", F.col("TransactionAmt") - F.floor("TransactionAmt"))

# ── 7. Log-transform amount ───────────────────────────────────────────────────
df = df.withColumn("log_amt", F.log1p("TransactionAmt"))

# ── 8. outsider15 anomaly flag ────────────────────────────────────────────────
if "D1" in df.columns and "D15" in df.columns:
    df = df.withColumn(
        "outsider15",
        (F.abs(F.col("D1") - F.col("D15")) > 3).cast("int"),
    )

# ── 9. Combined identity features ────────────────────────────────────────────
df = df.withColumn(
    "card1_addr1",
    F.concat(F.col("card1").cast("string"), F.lit("_"), F.col("addr1").cast("string")),
)
df = df.withColumn(
    "card1_addr1_email",
    F.concat(F.col("card1_addr1"), F.lit("_"), F.col("P_emaildomain").cast("string")),
)

# ── 10. Magic UID: card+address session keyed to account open date (D1) ───────
# floor(tx_day - D1) groups transactions that happen within the same D1-relative day
# This creates a proxy for "same user session" that's far more granular than card alone
if "D1" in df.columns:
    df = df.withColumn(
        "uid",
        F.concat(
            F.col("card1_addr1"),
            F.lit("_"),
            F.floor(F.col("tx_day") - F.col("D1")).cast("string"),
        ),
    )
    has_uid = True
else:
    has_uid = False
    print("  WARNING: D1 not found — skipping UID feature")

# ── 11. Frequency encoding ────────────────────────────────────────────────────
# For each column: replace value with its normalized frequency across the dataset.
# Higher frequency = more common = less anomalous.
print("Computing frequency encodings...")
fe_cols = ["card1", "card2", "card3", "addr1", "P_emaildomain",
           "card1_addr1", "card1_addr1_email"]
if has_uid:
    fe_cols.append("uid")

total_rows = df.count()
for col in fe_cols:
    freq_map = (
        df.groupBy(col)
        .count()
        .withColumn(col + "_FE", F.col("count") / total_rows)
        .select(col, col + "_FE")
    )
    df = df.join(freq_map, on=col, how="left")

# ── 12. UID aggregations ──────────────────────────────────────────────────────
# Per-UID: mean/std of TransactionAmt and D11; count-unique of P_emaildomain
if has_uid:
    print("Computing UID aggregations...")
    uid_win = Window.partitionBy("uid")
    df = (
        df
        .withColumn("uid_amt_mean",    F.mean("TransactionAmt").over(uid_win))
        .withColumn("uid_amt_std",     F.stddev("TransactionAmt").over(uid_win))
        .withColumn("uid_email_nuniq", F.approx_count_distinct("P_emaildomain").over(uid_win))
    )
    if "D11" in df.columns:
        df = (
            df
            .withColumn("uid_D11_mean", F.mean("D11").over(uid_win))
            .withColumn("uid_D11_std",  F.stddev("D11").over(uid_win))
        )
    # Amount z-score within the UID session
    df = df.withColumn(
        "uid_amt_zscore",
        F.when(
            F.col("uid_amt_std") > 0,
            (F.col("TransactionAmt") - F.col("uid_amt_mean")) / F.col("uid_amt_std"),
        ).otherwise(0.0),
    )

# ── 13. Fill any new NaNs introduced by aggregations ─────────────────────────
new_num_cols = [
    "uid_amt_mean", "uid_amt_std", "uid_email_nuniq",
    "uid_D11_mean", "uid_D11_std", "uid_amt_zscore",
    "outsider15", "cents", "log_amt", "tx_day",
]
for c in new_num_cols:
    if c in df.columns:
        df = df.fillna({c: 0.0})

# ── 14. Class weights ─────────────────────────────────────────────────────────
counts    = df.groupBy("isFraud").count().collect()
count_map = {r["isFraud"]: r["count"] for r in counts}
n_total   = count_map[0] + count_map[1]
w0 = n_total / (2.0 * count_map[0])
w1 = n_total / (2.0 * count_map[1])
df = df.withColumn("class_weight", F.when(F.col("isFraud") == 1, w1).otherwise(w0))
print(f"  Class weights → legit={w0:.4f}  fraud={w1:.4f}")

# ── 15. Stratified train / val / test split ───────────────────────────────────
legit = df.filter(F.col("isFraud") == 0)
fraud = df.filter(F.col("isFraud") == 1)
legit_train, legit_val, legit_test = legit.randomSplit([0.7, 0.15, 0.15], seed=42)
fraud_train, fraud_val, fraud_test = fraud.randomSplit([0.7, 0.15, 0.15], seed=42)
train = legit_train.union(fraud_train).cache()
val   = legit_val.union(fraud_val).cache()
test  = legit_test.union(fraud_test).cache()
print(f"  Train: {train.count():,}  Val: {val.count():,}  Test: {test.count():,}")

# ── 16. Feature columns ───────────────────────────────────────────────────────
EXCLUDE = {
    "TransactionID", "isFraud", "transaction_timestamp", "class_weight",
    "card1_addr1", "card1_addr1_email", "uid",  # string cols — use FE versions
}
feature_cols = [c for c in df.columns if c not in EXCLUDE
                and df.schema[c].dataType.typeName() != "string"]
print(f"  {len(feature_cols)} feature columns")

# ── 17. Pipeline ──────────────────────────────────────────────────────────────
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="skip",
)
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=False,
    withStd=True,
)
gbt = GBTClassifier(
    labelCol="isFraud",
    featuresCol="features",
    weightCol="class_weight",
    maxIter=150,
    maxDepth=8,
    stepSize=0.05,         # slower learning rate → less overfit
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
    seed=42,
)
pipeline = Pipeline(stages=[assembler, scaler, gbt])

# ── 18. Train ─────────────────────────────────────────────────────────────────
print(f"\nTraining GBTClassifier (maxIter=150, maxDepth=8, stepSize=0.05)...")
t0    = time.time()
model = pipeline.fit(train)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# ── 19. Threshold sweep on validation set ────────────────────────────────────
print("\nSweeping threshold on validation set...")
prob_udf  = F.udf(lambda v: float(v[1]), DoubleType())
val_preds = model.transform(val).withColumn("prob_fraud", prob_udf("probability"))
val_pd    = val_preds.select("isFraud", "prob_fraud").toPandas()

best_thresh, best_f1 = 0.5, 0.0
results = []
for thresh in [i / 100 for i in range(5, 95, 5)]:
    pred = (val_pd["prob_fraud"] >= thresh).astype(int)
    tp = int(((pred == 1) & (val_pd["isFraud"] == 1)).sum())
    fp = int(((pred == 1) & (val_pd["isFraud"] == 0)).sum())
    fn = int(((pred == 0) & (val_pd["isFraud"] == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results.append((thresh, prec, rec, f1))
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f"  Best threshold: {best_thresh:.2f}  (fraud F1={best_f1:.4f} on val)")
print(f"\n  Threshold sweep (val set):")
print(f"  {'Thresh':>8}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
for t, p, r, f in results:
    marker = " ◄" if t == best_thresh else ""
    print(f"  {t:>8.2f}  {p:>7.4f}  {r:>7.4f}  {f:>7.4f}{marker}")

# ── 20. Evaluate on test set ──────────────────────────────────────────────────
print("\nEvaluating on test set...")
test_preds = (
    model.transform(test)
    .withColumn("prob_fraud", prob_udf("probability"))
    .withColumn("prediction_tuned", (F.col("prob_fraud") >= best_thresh).cast("double"))
)

auc  = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="rawPrediction",
                                      metricName="areaUnderROC").evaluate(test_preds)
aupr = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="rawPrediction",
                                      metricName="areaUnderPR").evaluate(test_preds)

cm = (
    test_preds.groupBy("isFraud", "prediction_tuned").count()
    .orderBy("isFraud", "prediction_tuned").collect()
)
matrix = {(int(r["isFraud"]), int(r["prediction_tuned"])): r["count"] for r in cm}
tn = matrix.get((0,0), 0); fp = matrix.get((0,1), 0)
fn = matrix.get((1,0), 0); tp = matrix.get((1,1), 0)
fraud_prec = tp/(tp+fp) if (tp+fp)>0 else 0
fraud_rec  = tp/(tp+fn) if (tp+fn)>0 else 0
fraud_f1   = 2*fraud_prec*fraud_rec/(fraud_prec+fraud_rec) if (fraud_prec+fraud_rec)>0 else 0

print("\n=== Test Set Results ===")
print(f"  AUC-ROC            : {auc:.4f}")
print(f"  AUC-PR             : {aupr:.4f}")
print(f"  Threshold          : {best_thresh:.2f}")
print(f"\n  Confusion Matrix:")
print(f"                        Pred=0    Pred=1")
print(f"  Actual=0            {tn:7,}   {fp:7,}")
print(f"  Actual=1            {fn:7,}   {tp:7,}")
print(f"\n  Fraud-class only:")
print(f"    Precision : {fraud_prec:.4f}")
print(f"    Recall    : {fraud_rec:.4f}")
print(f"    F1        : {fraud_f1:.4f}")

# ── 21. Feature importances ───────────────────────────────────────────────────
importances = model.stages[-1].featureImportances.toArray()
top20 = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:20]
print("\n  Top 20 feature importances:")
for name, imp in top20:
    bar = "█" * int(imp * 200)
    print(f"    {name:<26s} {imp:.4f}  {bar}")

# ── 22. Save ──────────────────────────────────────────────────────────────────
model.write().overwrite().save(MODEL_PATH)
metadata = {
    "model_type"      : "GBTClassifier_v3",
    "feature_cols"    : feature_cols,
    "label_col"       : "isFraud",
    "best_threshold"  : best_thresh,
    "metrics"         : {
        "auc_roc"        : round(auc,        4),
        "auc_pr"         : round(aupr,       4),
        "fraud_precision": round(fraud_prec, 4),
        "fraud_recall"   : round(fraud_rec,  4),
        "fraud_f1"       : round(fraud_f1,   4),
    },
    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    "top_features"    : [{"feature": n, "importance": round(i,4)} for n,i in top20],
    "training_time_s" : round(elapsed, 1),
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved locally → {MODEL_PATH}")
print(f"Metadata           → {META_PATH}")
spark.stop()

# ── 23. Upload model + metadata + lookups to S3 ───────────────────────────────
from s3_utils import upload_dir, upload_file, PREFIX_MODEL, PREFIX_METADATA, PREFIX_LOOKUPS
print("\nUploading to S3...")
upload_dir(MODEL_PATH,  PREFIX_MODEL)
upload_file(META_PATH,  PREFIX_METADATA + "fraud_gbt_metadata.json")
upload_dir("data/lookups", PREFIX_LOOKUPS)
print("S3 upload complete.")
