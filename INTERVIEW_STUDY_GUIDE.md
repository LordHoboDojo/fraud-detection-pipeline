# Real-Time Fraud Detection Pipeline — Interview Study Guide

---

## The One-Sentence Summary
> Built an end-to-end real-time fraud detection system that ingests financial transactions via Kafka, processes them with Spark Structured Streaming on AWS EMR, scores them with a Gradient Boosted Tree model, and stores results in Redshift — with a live dashboard showing predictions within 5–10 seconds of a transaction occurring.

---

## Project Stats to Remember
| Metric | Value |
|--------|-------|
| Dataset size | 590,540 transactions × 229 features |
| Fraud rate in data | 3.5% (highly imbalanced) |
| Model AUC-ROC | 0.9254 |
| Precision | ~91% |
| Recall | ~67% |
| Streaming latency | 5–10 seconds (Kafka → Redshift) |
| Training time on EMR | 258 seconds |
| Feature columns used | 130 |

---

## Architecture — Know This Cold

```
IEEE-CIS Dataset (CSV)
        ↓
  Spark Batch Cleaning  ←── PySpark on local / EMR
        ↓
     S3 (Parquet)        ←── columnar storage, partitioned
        ↓
  EMR Training Job       ←── Spark 3.5.1, YARN, GBT model
        ↓
  HDFS (model + lookups) ←── fast local access during streaming
        ↓
  Kafka Producer         ←── simulates live payment feed
        ↓
  MSK (managed Kafka)    ←── 2 brokers, 4 partitions, PLAINTEXT
        ↓
  Spark Structured Streaming (EMR)
        ↓
  Redshift               ←── fraud_predictions table
        ↓
  Flask Dashboard        ←── live KPIs, confusion matrix, per-row table
```

**Why each component exists:**
- **Kafka/MSK** — decouples producer from consumer; handles backpressure; if Spark is slow, messages queue instead of being dropped
- **Spark** — distributes work across executors; same code scales from laptop to 1000-node cluster
- **S3** — durable, cheap, natively integrated with EMR via S3A connector
- **HDFS** — faster than S3 for repeated random reads (lookup tables hit on every micro-batch)
- **Redshift** — columnar data warehouse, optimised for analytical queries on the predictions

---

## The Data & Cleaning

**Dataset:** IEEE-CIS Fraud Detection (Kaggle) — two files joined on TransactionID:
- `train_transaction.csv` — 590k rows, payment details
- `train_identity.csv` — 144k rows, device/network info

**Cleaning steps (all in PySpark — distributed):**
1. Left join the two CSVs on `TransactionID`
2. Drop columns with >70% missing values — computed as a distributed `SELECT mean(col IS NULL)` across all partitions
3. Median imputation for remaining nulls — `approxQuantile()` is approximate but works in one distributed pass
4. StringIndexer for categorical columns
5. Write to Parquet on S3 — columnar format, ~10x faster to query than CSV

**Why Spark instead of pandas here:**
- pandas loads everything into RAM on one machine; breaks above ~10GB
- Spark splits the data across partitions, each processed on a separate executor
- The join + null scan on 590k × 229 cols takes ~4 minutes in pandas, ~12 seconds distributed

---

## Feature Engineering — The Most Important Part

Raw transaction data has no useful signal on its own. The features create signal.

### 1. The "Magic UID" (most important feature)
```python
uid = card1 + "_" + addr1 + "_" + floor(tx_day - D1)
```
- `D1` = days since last transaction
- `tx_day - D1` = the day the account was first active
- Combining card number + billing address + account origin day creates a **pseudo-identity** for each cardholder — without any actual user ID in the dataset
- This appeared in the **top 3 feature importances** in the trained model

### 2. Frequency Encoding
```python
freq = df.groupBy(col).count()
     .withColumn(f"{col}_FE", count / total)
df = df.join(freq, on=col, how="left")
```
- Applied to: card1, card2, card3, addr1, P_emaildomain, card1_addr1, card1_addr1_email, uid
- Replaces categorical IDs with **how frequently they appear** in the dataset
- Rare card numbers = unusual = higher fraud risk
- This is a distributed `groupBy` aggregation — runs in parallel across all partitions

### 3. D-Column Normalization
- D4, D10, D11, D15 are "time delta" columns — days since various events
- Subtract `tx_day` to make them relative rather than absolute
- Without this, the model sees these as absolute timestamps which shift as time passes

### 4. Other Features
- `cents` = `TransactionAmt - floor(TransactionAmt)` — round-dollar amounts are suspicious
- `log_amt` = `log1p(TransactionAmt)` — compresses the huge range of transaction amounts
- `outsider15` = `|D1 - D15| > 3` — flags accounts with inconsistent transaction timing
- UID aggregations: mean/std of transaction amounts per UID, email diversity per UID

### 5. V-Column Pruning
- Dataset has 339 anonymized Vesta features (V1–V339)
- Pruned to 68 based on a top Kaggle notebook's analysis
- Removes noise, speeds up training, reduces overfitting

---

## The Model — Gradient Boosted Trees (GBT)

### What it is
- An **ensemble** of 100 decision trees built **sequentially**
- Each tree corrects the errors of the previous one using gradient descent
- Final prediction = weighted vote of all 100 trees

### Why GBT for fraud
- Handles mixed data types naturally (amounts, frequencies, flags, counts)
- Captures non-linear interactions — e.g. a $1 transaction alone is fine; $1 + new email + foreign IP + rare card = fraud
- Tree-based models are robust to outliers (extreme transaction amounts don't break it)
- Works well with class weights for imbalanced data

### Hyperparameters and why
```python
maxIter=100        # 100 trees — more = better but slower
maxDepth=6         # each tree asks 6 levels of yes/no questions
stepSize=0.1       # learning rate — small = stable, less overfitting
subsamplingRate=0.8  # each tree sees 80% of data — prevents overfitting
featureSubsetStrategy="sqrt"  # each split considers sqrt(130) ≈ 11 features — faster + diverse trees
```

### Handling class imbalance
```python
w0 = total / (2 * count_legit)   # weight ~0.52
w1 = total / (2 * count_fraud)   # weight ~14.3
```
- Without this, the model learns "always predict legit" and gets 96.5% accuracy while catching zero fraud
- Class weights force the model to treat 1 missed fraud as equivalent to ~14 false alarms

### Threshold Tuning
- Default threshold of 0.5 gives poor F1 on imbalanced data
- Swept thresholds from 0.05 to 0.95 on validation set
- Optimal threshold: **0.85** — at this point precision is high enough to be operationally useful
- **Why F1 not accuracy?** Accuracy is meaningless when 96.5% of samples are one class

### The ML Pipeline
```python
Pipeline(stages=[
    VectorAssembler,   # combine 130 columns into a single feature vector
    StandardScaler,    # normalize so large amounts don't dominate
    GBTClassifier      # train the model
])
```
- Saved as a single serialized object to HDFS — `model.transform(df)` at inference time runs all three stages automatically

---

## Streaming Architecture

### Kafka
- **Topics** are like queues — producers write, consumers read
- **Partitions** allow parallel consumption — 4 partitions means 4 Spark tasks can read simultaneously
- **Offsets** track where each consumer left off — if Spark crashes, it resumes from the last committed offset
- **`acks="all"`** on the producer — waits for all broker replicas to confirm before acknowledging

### Spark Structured Streaming
```python
raw = spark.readStream.format("kafka")...
# Every 5 seconds, Spark pulls the latest messages as a mini-DataFrame
# Applies the same feature engineering as training
# Runs model.transform() — exact same pipeline
# foreachBatch writes each mini-DataFrame to Redshift
.trigger(processingTime="5 seconds")
```

### The Lookup Table Problem
- Streaming feature engineering needs frequency tables (e.g., how often does this card appear?)
- Can't recompute these at inference time — too slow
- **Solution:** pre-compute lookup tables at training time, save to HDFS, load as **static DataFrames** at startup
- Each micro-batch does a **static join** against these in-memory tables — fast

### Why not window functions in streaming?
- Window functions shuffle all data for a partition — expensive on large streams
- Used `groupBy + join` instead — computes aggregates once, joins efficiently

---

## AWS Infrastructure

| Service | What it does | Why managed |
|---------|-------------|-------------|
| **S3** | Stores raw data, cleaned parquet, model | Unlimited scale, 11 nines durability, natively integrated with EMR |
| **EMR** | Runs Spark (training + streaming) | Pre-configured Hadoop/YARN/Spark; handles cluster lifecycle |
| **MSK** | Managed Kafka | No broker management, automatic replication, VPC-native |
| **Redshift** | Stores predictions | Columnar MPP warehouse; fast analytical queries |

### Networking
- All services in the same VPC — no public internet traffic between components
- MSK only accessible from within VPC — producer runs on EMR master, not laptop
- EMR security group: only port 22 (SSH) public + VPC-internal traffic

### EMR vs local Spark
- Local: PySpark 4.1.1 (latest) — model format uses `treeID` column
- EMR 7.3.0: Spark 3.5.1 — model format uses `_1, _2, _3` tuple columns
- **These are incompatible** — had to retrain on EMR itself to fix
- Lesson: always train and serve on the same Spark version

---

## Challenges & How They Were Solved

### 1. Class Imbalance (3.5% fraud)
- **Problem:** model predicts all-legit and gets 96.5% accuracy
- **Fix:** inverse-frequency class weights, threshold tuning on validation set

### 2. Spark Version Mismatch
- **Problem:** model trained locally (PySpark 4.1.1) couldn't load on EMR (Spark 3.5.1) — GBT internal format changed
- **Fix:** wrote `emr_train.py` to retrain directly on EMR using the correct version

### 3. OOM During Training on EMR
- **Problem:** UID window aggregations (`Window.partitionBy("uid")`) caused heap overflow on 8GB EMR node
- **Fix:** replaced window functions with `groupBy + join` — avoids full shuffle, much lower memory footprint

### 4. Streaming Feature Engineering
- **Problem:** can't run `groupBy` on a live stream to compute frequency tables
- **Fix:** pre-compute all lookup tables at training time, load as static DataFrames at streaming startup, join each micro-batch against them

### 5. MSK Not Accessible from Laptop
- **Problem:** MSK is VPC-internal, can't connect from local machine
- **Fix:** ran Kafka producer on the EMR master node (already in the same VPC)

---

## Results & Metrics

```
AUC-ROC:   0.9254  (1.0 = perfect, 0.5 = random)
Precision: 91%     (of transactions flagged as fraud, 91% actually were)
Recall:    67%     (of all real fraud, 67% was caught)
F1 Score:  0.57
Threshold: 0.85
```

**Confusion matrix on a typical 100-transaction demo batch (15 fraud injected):**
- True Positives (caught fraud): ~10
- False Positives (false alarms): ~1-2
- False Negatives (missed fraud): ~5
- True Negatives (correctly cleared): ~83

**Why 67% recall is acceptable for a demo:**
- The threshold is set high (0.85) to prioritise precision — banks care more about not blocking legitimate customers than catching every single fraud
- Recall vs precision is a business decision, not a model limitation — lowering the threshold to 0.5 raises recall to ~85% but generates more false alarms

---

## Questions You'll Likely Get

**"Why Spark instead of a simple Python script?"**
> Spark distributes computation across a cluster. The same code that processes 590k rows on one machine will process 590 million rows by adding more EMR nodes — zero code changes. pandas crashes above available RAM; Spark spills to disk and parallelises automatically.

**"Why Kafka? Why not just write directly to Spark?"**
> Kafka decouples the producer from the consumer. If Spark is slow or restarts, messages queue in Kafka and nothing is lost. It also allows multiple consumers — you could have Spark streaming AND a separate alerting service both reading the same topic independently.

**"How does the model handle new card numbers it's never seen?"**
> Frequency encoding of unseen keys returns null → filled with 0. The model treats it as "this card has never appeared before" which is itself a signal. The `handleInvalid="skip"` on VectorAssembler handles any remaining nulls gracefully.

**"Why GBT over a neural network?"**
> GBTs typically outperform neural nets on tabular data with engineered features. They're also interpretable — you can extract feature importances and explain to a bank's compliance team why a transaction was flagged. Neural nets are black boxes on tabular data and require far more tuning.

**"What would you do differently?"**
> 1. Add online learning — retrain incrementally as new fraud patterns emerge rather than periodic batch retraining. 2. Use a proper feature store (Feast or Tecton) instead of hand-rolled HDFS lookup tables. 3. Add model monitoring — track prediction drift over time so degradation is caught automatically.

**"What was the hardest part?"**
> The Spark version mismatch between local development (PySpark 4.1.1) and EMR (Spark 3.5.1). The GBT model format changed between versions — a column called `treeID` in 4.x became tuple columns `_1, _2, _3` in 3.x. The error message was cryptic and it took several debugging cycles to identify the root cause. The fix was to retrain directly on EMR rather than transfer a locally-trained model.

**"How does it scale?"**
> Horizontally: add EMR worker nodes → Spark automatically distributes more partitions to new executors. Kafka scales by adding partitions and brokers. Redshift scales by adding nodes to the cluster. The application code doesn't change.

---

## Key Terms to Know

| Term | One-line definition |
|------|-------------------|
| **Micro-batch** | Spark Streaming collects messages for N seconds then processes them as a batch |
| **Partition** | A chunk of data processed independently by one Spark task or Kafka consumer |
| **Shuffle** | Moving data between Spark executors — expensive, should be minimised |
| **Parquet** | Columnar file format — reads only the columns you need, compressed, fast |
| **YARN** | Hadoop's cluster resource manager — allocates CPU/memory to Spark executors |
| **S3A** | Hadoop connector that lets Spark read/write S3 like a filesystem |
| **Offset** | Kafka's pointer to where a consumer last read — enables fault tolerance |
| **Feature Store** | A system for storing and serving pre-computed ML features |
| **AUC-ROC** | Area under the ROC curve — 0.5 = random classifier, 1.0 = perfect |
| **Class weight** | Penalises the model more for misclassifying the minority class |
| **foreachBatch** | Spark Streaming sink that calls a custom function on each micro-batch DataFrame |
