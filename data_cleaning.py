"""
IEEE-CIS Fraud Detection — Data Cleaning (Spark Batch Job)
Produces: data/cleaned_fraud.parquet/  (local)
          s3://fraud-detection-pipeline-408697342341/data/cleaned/  (S3)
"""

import os
# Must be set before JVM launches — Hadoop 3.x calls Subject.getSubject()
# which was removed in Java 21+. --add-opens restores reflective access.
os.environ["JAVA_HOME"] = (
    "/Users/shaunakkulkarni/Library/Java/JavaVirtualMachines/ms-17.0.17/Contents/Home"
)
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Imputer, StringIndexer
from pyspark.ml import Pipeline

DATA_DIR = "/Users/shaunakkulkarni/.cache/kagglehub/competitions/ieee-fraud-detection"
OUT_PATH = "data/cleaned_fraud.parquet"

# Reference epoch: TransactionDT is seconds offset from 2017-11-30
REF_EPOCH_SECS = 1512000000  # unix timestamp of 2017-11-30 00:00:00 UTC
MISSING_THRESHOLD = 0.70

spark = (
    SparkSession.builder
    .appName("FraudDetection-DataCleaning")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading CSVs...")
trans = spark.read.csv(f"{DATA_DIR}/train_transaction.csv", header=True, inferSchema=True)
ident = spark.read.csv(f"{DATA_DIR}/train_identity.csv",   header=True, inferSchema=True)
print(f"  transactions : {trans.count()} rows × {len(trans.columns)} cols")
print(f"  identity     : {ident.count()} rows × {len(ident.columns)} cols")

# ── 2. Merge (left join) ──────────────────────────────────────────────────────
df = trans.join(ident, on="TransactionID", how="left")
total_rows = df.count()
print(f"  merged       : {total_rows} rows × {len(df.columns)} cols")

# ── 3. Drop columns with >70 % missing ───────────────────────────────────────
# Compute null counts in one pass with agg
null_counts = df.agg(*[F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0]
drop_cols = [c for c in df.columns if null_counts[c] / total_rows > MISSING_THRESHOLD]
df = df.drop(*drop_cols)
print(f"  dropped {len(drop_cols)} high-missingness cols → {len(df.columns)} remaining")

# ── 4. TransactionDT → real timestamp + time features ────────────────────────
df = (
    df.withColumn(
        "transaction_timestamp",
        F.to_timestamp(F.from_unixtime(F.col("TransactionDT") + REF_EPOCH_SECS))
    )
    .withColumn("hour",        F.hour("transaction_timestamp"))
    .withColumn("day_of_week", F.dayofweek("transaction_timestamp"))
    .withColumn("month",       F.month("transaction_timestamp"))
    .drop("TransactionDT")
)

# ── 5. Identify column types ──────────────────────────────────────────────────
skip_cols  = {"TransactionID", "isFraud", "transaction_timestamp"}
str_cols   = [f.name for f in df.schema.fields
              if str(f.dataType) == "StringType()" and f.name not in skip_cols]
num_cols   = [f.name for f in df.schema.fields
              if str(f.dataType) in ("DoubleType()", "IntegerType()", "LongType()", "FloatType()")
              and f.name not in skip_cols]
print(f"  string cols  : {len(str_cols)}")
print(f"  numeric cols : {len(num_cols)}")

# ── 6. Impute missing values ──────────────────────────────────────────────────
# Numeric → median via Imputer
imputer = Imputer(strategy="median", inputCols=num_cols, outputCols=num_cols)

# String → "unknown"
for c in str_cols:
    df = df.fillna("unknown", subset=[c])

# StringIndexer for each categorical column (encodes to numeric index)
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in str_cols
]

# ── 7. Build and run pipeline ─────────────────────────────────────────────────
stages = [imputer] + indexers
pipeline = Pipeline(stages=stages)

print("Fitting pipeline (imputer medians + string indexers)...")
model = pipeline.fit(df)
df = model.transform(df)

# Drop original string cols, rename _idx cols
df = df.drop(*str_cols)
for c in str_cols:
    df = df.withColumnRenamed(f"{c}_idx", c)

# ── 8. Cast isFraud to int ────────────────────────────────────────────────────
df = df.withColumn("isFraud", F.col("isFraud").cast(IntegerType()))

# ── 9. Sanity checks ──────────────────────────────────────────────────────────
print("\n=== Post-cleaning ===")
print(f"  columns      : {len(df.columns)}")
fraud_rate = df.selectExpr("avg(isFraud) * 100 as fraud_pct").collect()[0]["fraud_pct"]
print(f"  fraud rate   : {fraud_rate:.2f} %")
ts_range = df.selectExpr(
    "min(transaction_timestamp) as min_ts",
    "max(transaction_timestamp) as max_ts"
).collect()[0]
print(f"  date range   : {ts_range['min_ts']} → {ts_range['max_ts']}")

# ── 10. Save locally ─────────────────────────────────────────────────────────
df.write.mode("overwrite").parquet(OUT_PATH)
print(f"\nSaved locally → {OUT_PATH}")

# ── 11. Upload to S3 ─────────────────────────────────────────────────────────
spark.stop()

from s3_utils import upload_dir, upload_file, PREFIX_CLEANED_DATA, PREFIX_RAW_DATA
print("\nUploading to S3...")
upload_dir(OUT_PATH, PREFIX_CLEANED_DATA + "cleaned_fraud.parquet")
upload_file("models/fraud_gbt_metadata.json", PREFIX_RAW_DATA + "fraud_gbt_metadata.json")
print("S3 upload complete.")
