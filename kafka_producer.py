"""
Kafka Producer — streams rows from cleaned_fraud.parquet to Kafka topic.
Simulates a real-time transaction feed at a configurable rate.

Usage:
    python kafka_producer.py [--rate 10] [--topic fraud-transactions] [--limit 5000]
"""

import json
import os
import time
import argparse
import math

from kafka import KafkaProducer
import pandas as pd

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
PARQUET_PATH    = "data/cleaned_fraud.parquet"

def make_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",
        retries=3,
    )

def stream(rate: int, topic: str, limit: int, ensure_fraud: int = 0):
    print(f"Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)

    if ensure_fraud > 0 and limit > 0:
        fraud_rows = df[df["isFraud"] == 1].sample(n=min(ensure_fraud, (df["isFraud"] == 1).sum()), random_state=42)
        legit_rows = df[df["isFraud"] == 0].sample(n=min(limit - len(fraud_rows), (df["isFraud"] == 0).sum()), random_state=42)
        df = pd.concat([legit_rows, fraud_rows]).sample(frac=1, random_state=42).reset_index(drop=True)
    elif limit:
        df = df.head(limit)

    total = len(df)
    print(f"  {total:,} rows to stream → topic '{topic}' at {rate} msg/s")

    producer  = make_producer()
    delay     = 1.0 / rate
    sent = fraud_sent = 0

    for _, row in df.iterrows():
        record = row.to_dict()
        # Convert numpy types to native Python for JSON serialisation
        record = {
            k: (int(v) if hasattr(v, "item") and isinstance(v.item(), int)
                else float(v) if hasattr(v, "item")
                else str(v) if pd.isna(v) is True  # NaT / NaN guard
                else v)
            for k, v in record.items()
        }
        # Convert Timestamp to ISO string
        if "transaction_timestamp" in record:
            ts = record["transaction_timestamp"]
            record["transaction_timestamp"] = str(ts) if ts else None

        producer.send(
            topic,
            key=record.get("TransactionID"),
            value=record,
        )
        sent += 1
        if record.get("isFraud") == 1:
            fraud_sent += 1

        interval = max(25, total // 4)
        if sent % interval == 0:
            print(f"  Sent {sent:,}/{total:,}  (fraud: {fraud_sent})", flush=True)

        time.sleep(delay)

    producer.flush()
    print(f"\nDone. Sent {sent:,} transactions ({fraud_sent} fraud) to '{topic}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate",  type=int, default=10,
                        help="Messages per second (default: 10)")
    parser.add_argument("--topic", type=str, default="fraud-transactions")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max rows to send (0 = all)")
    parser.add_argument("--ensure-fraud", type=int, default=0,
                        help="Guarantee at least N fraud rows in the batch")
    args = parser.parse_args()
    stream(args.rate, args.topic, args.limit, args.ensure_fraud)
