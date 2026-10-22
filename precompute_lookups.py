"""
Precompute lookup tables from training data for use in streaming inference.
Saves to data/lookups/ as parquet files.

Run once after model_training.py before starting the streaming pipeline.
"""

import os

os.environ["JAVA_HOME"] = (
    "/Users/shaunakkulkarni/Library/Java/JavaVirtualMachines/ms-17.0.17/Contents/Home"
)
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_PATH    = "data/cleaned_fraud.parquet"
LOOKUPS_PATH = "data/lookups"
REF_EPOCH_SECS = 1512000000

os.makedirs(LOOKUPS_PATH, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("FraudDetection-PrecomputeLookups")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("Loading cleaned data...")
df = spark.read.parquet(DATA_PATH)
total = df.count()
print(f"  {total:,} rows")

# ── Derive tx_day and combined key columns (mirrors model_training.py) ────────
df = df.withColumn(
    "tx_day",
    (F.unix_timestamp("transaction_timestamp") - REF_EPOCH_SECS) / 86400.0,
)
df = df.withColumn(
    "card1_addr1",
    F.concat(F.col("card1").cast("string"), F.lit("_"), F.col("addr1").cast("string")),
)
df = df.withColumn(
    "card1_addr1_email",
    F.concat(F.col("card1_addr1"), F.lit("_"), F.col("P_emaildomain").cast("string")),
)
df = df.withColumn(
    "uid",
    F.concat(
        F.col("card1_addr1"),
        F.lit("_"),
        F.floor(F.col("tx_day") - F.col("D1")).cast("string"),
    ),
)

# ── 1. Frequency encoding lookups ─────────────────────────────────────────────
print("Computing frequency lookups...")
fe_cols = ["card1", "card2", "card3", "addr1", "P_emaildomain",
           "card1_addr1", "card1_addr1_email", "uid"]

for col in fe_cols:
    lookup = (
        df.groupBy(col)
        .count()
        .withColumn(f"{col}_FE", F.col("count") / total)
        .drop("count")
    )
    out = f"{LOOKUPS_PATH}/fe_{col}"
    lookup.write.mode("overwrite").parquet(out)
    print(f"  Saved {col} freq lookup → {out}  ({lookup.count():,} distinct values)")

# ── 2. UID aggregation lookups ────────────────────────────────────────────────
print("Computing UID aggregation lookups...")
uid_stats = (
    df.groupBy("uid")
    .agg(
        F.mean("TransactionAmt").alias("uid_amt_mean"),
        F.stddev("TransactionAmt").alias("uid_amt_std"),
        F.approx_count_distinct("P_emaildomain").alias("uid_email_nuniq"),
        F.mean("D11").alias("uid_D11_mean"),
        F.stddev("D11").alias("uid_D11_std"),
    )
    .fillna(0.0)
)
uid_stats.write.mode("overwrite").parquet(f"{LOOKUPS_PATH}/uid_stats")
print(f"  Saved UID stats → {LOOKUPS_PATH}/uid_stats  ({uid_stats.count():,} UIDs)")

print("\nAll lookups saved to data/lookups/")
spark.stop()
