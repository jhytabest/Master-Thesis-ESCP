import argparse
import hashlib
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
import sys
from urllib import error as urlerror
from urllib import request as urlrequest

import boto3
import pandas as pd


def env_or_error(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(f"Missing required env var: {name}")
  return value


def now_iso() -> str:
  return datetime.utcnow().isoformat() + "Z"


def build_s3_client() -> boto3.client:
  endpoint = env_or_error("R2_ENDPOINT")
  access_key = env_or_error("R2_ACCESS_KEY_ID")
  secret_key = env_or_error("R2_SECRET_ACCESS_KEY")
  return boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="auto"
  )


def sha256_hex(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def upload_bytes(client: boto3.client, bucket: str, key: str, data: bytes, content_type: str) -> None:
  client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def update_run(run_id: str, payload: Dict[str, Any]) -> None:
  base = os.getenv("WORKER_API_BASE")
  token = os.getenv("WORKER_API_TOKEN")
  if not base or not token:
    return
  url = f"{base}/runs/{run_id}"
  data = json.dumps(payload).encode("utf-8")
  req = urlrequest.Request(url, data=data, method="PATCH")
  req.add_header("Authorization", f"Bearer {token}")
  req.add_header("Content-Type", "application/json")
  cf_client_id = os.getenv("CF_ACCESS_CLIENT_ID")
  cf_client_secret = os.getenv("CF_ACCESS_CLIENT_SECRET")
  if cf_client_id and cf_client_secret:
    req.add_header("CF-Access-Client-Id", cf_client_id)
    req.add_header("CF-Access-Client-Secret", cf_client_secret)
  try:
    with urlrequest.urlopen(req, timeout=30) as resp:
      _ = resp.read()
  except urlerror.HTTPError as exc:
    details = exc.read().decode("utf-8", errors="ignore")
    print(f"[warn] failed to update run: {exc.code} {details}", file=sys.stderr)
  except Exception as exc:
    print(f"[warn] failed to update run: {exc}", file=sys.stderr)


def build_eda_html(df: pd.DataFrame, missing: Dict[str, int]) -> str:
  rows = len(df)
  cols = len(df.columns)
  missing_rows = "".join(
    f"<tr><td>{col}</td><td>{count}</td></tr>" for col, count in missing.items()
  )
  return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>EDA summary</title>
  </head>
  <body>
    <h1>EDA summary</h1>
    <p>Rows: {rows}</p>
    <p>Columns: {cols}</p>
    <h2>Missing values</h2>
    <table border="1" cellpadding="4" cellspacing="0">
      <thead><tr><th>Column</th><th>Missing</th></tr></thead>
      <tbody>{missing_rows}</tbody>
    </table>
  </body>
</html>
"""


def main() -> None:
  parser = argparse.ArgumentParser(description="Deterministic pipeline job")
  parser.add_argument("--version_id", required=True)
  parser.add_argument("--mapping_bundle_id")
  parser.add_argument("--run_id")
  args = parser.parse_args()

  run_id = args.run_id
  started_at = now_iso()
  if run_id:
    update_run(run_id, {"status": "STARTED", "started_at": started_at})

  bucket = env_or_error("R2_BUCKET")
  version_id = args.version_id
  source_key = f"raw/{version_id}/dataset.csv"

  client = build_s3_client()
  try:
    obj = client.get_object(Bucket=bucket, Key=source_key)
    raw_bytes = obj["Body"].read()
    dataset_sha = sha256_hex(raw_bytes)

    df = pd.read_csv(io.BytesIO(raw_bytes))
    missing = df.isna().sum().to_dict()

    quality = {
      "row_count": int(len(df)),
      "column_count": int(len(df.columns)),
      "missing_by_column": {k: int(v) for k, v in missing.items()}
    }

    cohort_buffer = io.BytesIO()
    df.to_parquet(cohort_buffer, index=False)
    cohort_key = f"derived/{version_id}/cohort.parquet"
    upload_bytes(client, bucket, cohort_key, cohort_buffer.getvalue(), "application/octet-stream")

    eda_html = build_eda_html(df, quality["missing_by_column"])
    eda_key = f"reports/{version_id}/eda.html"
    upload_bytes(client, bucket, eda_key, eda_html.encode("utf-8"), "text/html")

    quality_key = f"reports/{version_id}/data_quality.json"
    upload_bytes(client, bucket, quality_key, json.dumps(quality, indent=2).encode("utf-8"), "application/json")

    manifest = {
      "version_id": version_id,
      "dataset_sha256": dataset_sha,
      "run_id": run_id,
      "code_git_sha": os.getenv("CODE_GIT_SHA", "unknown"),
      "container_image_digest": os.getenv("IMAGE_DIGEST", "unknown"),
      "mapping_bundle_id": args.mapping_bundle_id,
      "mapping_manifest_hash": None,
      "parsing_stats": {
        "date_parse_failures": 0,
        "numeric_parse_failures": 0
      },
      "anomaly_counts": {},
      "config_hash": os.getenv("CONFIG_HASH", "unknown")
    }
    manifest_key = f"manifests/{version_id}/cleaning_manifest.json"
    upload_bytes(client, bucket, manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    index = {
      "version_id": version_id,
      "run_id": run_id,
      "created_at": now_iso(),
      "artifacts": {
        "cohort_parquet": cohort_key,
        "eda_html": eda_key,
        "data_quality_json": quality_key,
        "cleaning_manifest": manifest_key
      }
    }
    index_key = f"reports/{version_id}/index.json"
    upload_bytes(client, bucket, index_key, json.dumps(index, indent=2).encode("utf-8"), "application/json")

    if run_id:
      update_run(run_id, {
        "status": "SUCCEEDED",
        "finished_at": now_iso(),
        "artifact_index_path": index_key
      })
  except Exception as exc:
    if run_id:
      update_run(run_id, {
        "status": "FAILED",
        "finished_at": now_iso(),
        "error_step": "pipeline",
        "error_message": str(exc)
      })
    raise


if __name__ == "__main__":
  main()
