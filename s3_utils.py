"""
S3 utility helpers for the fraud detection pipeline.
Handles upload/download of data, models, and lookup tables.
"""

import os
import boto3
from pathlib import Path

BUCKET      = os.environ.get("S3_BUCKET", "your-s3-bucket-name")
AWS_PROFILE = os.environ.get("AWS_PROFILE_NAME", "default")
REGION      = os.environ.get("AWS_REGION", "us-east-2")

# S3 key prefixes
PREFIX_RAW_DATA     = "data/raw/"
PREFIX_CLEANED_DATA = "data/cleaned/"
PREFIX_LOOKUPS      = "data/lookups/"
PREFIX_MODEL        = "models/fraud_gbt/"
PREFIX_METADATA     = "models/"


def _session():
    """Use named profile locally; fall back to instance role on EC2/EMR."""
    try:
        s = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
        s.client("sts").get_caller_identity()  # validate profile exists
        return s
    except Exception:
        return boto3.Session(region_name=REGION)  # use instance role


def get_client():
    return _session().client("s3")


def get_resource():
    return _session().resource("s3")


def upload_file(local_path: str, s3_key: str):
    client = get_client()
    client.upload_file(local_path, BUCKET, s3_key)
    print(f"  ↑ {local_path} → s3://{BUCKET}/{s3_key}")


def download_file(s3_key: str, local_path: str):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    client = get_client()
    client.download_file(BUCKET, s3_key, local_path)
    print(f"  ↓ s3://{BUCKET}/{s3_key} → {local_path}")


def upload_dir(local_dir: str, s3_prefix: str):
    """Recursively upload a local directory to an S3 prefix."""
    client  = get_client()
    base    = Path(local_dir)
    count   = 0
    for path in base.rglob("*"):
        if path.is_file():
            rel   = path.relative_to(base)
            s3_key = f"{s3_prefix.rstrip('/')}/{rel}"
            client.upload_file(str(path), BUCKET, s3_key)
            count += 1
    print(f"  ↑ {local_dir}/ → s3://{BUCKET}/{s3_prefix}  ({count} files)")


def download_dir(s3_prefix: str, local_dir: str):
    """Recursively download an S3 prefix to a local directory."""
    client = get_client()
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key        = obj["Key"]
            rel        = key[len(s3_prefix):].lstrip("/")
            local_path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            client.download_file(BUCKET, key, local_path)
            count += 1
    print(f"  ↓ s3://{BUCKET}/{s3_prefix} → {local_dir}/  ({count} files)")


def s3_path(prefix: str) -> str:
    """Return a full s3:// path for use in Spark."""
    return f"s3a://{BUCKET}/{prefix.rstrip('/')}"
