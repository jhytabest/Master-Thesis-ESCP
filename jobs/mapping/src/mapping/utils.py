import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import pandas as pd


def env_or_error(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(f"Missing required env var: {name}")
  return value


def now_iso() -> str:
  return datetime.utcnow().isoformat() + "Z"


def sha256_hex(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
  return sha256_hex(text.encode("utf-8"))


def is_missing_value(value: Any) -> bool:
  if value is None:
    return True
  if isinstance(value, float) and pd.isna(value):
    return True
  text = str(value).strip().lower()
  return text in {"", "nan", "#n/a", "n/a", "na", "none", "null"}


def normalize_raw_value(value: Any) -> Optional[str]:
  if is_missing_value(value):
    return None
  return str(value).strip()


def normalize_header(name: str) -> str:
  return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


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


def download_bytes(client: boto3.client, bucket: str, key: str) -> bytes:
  obj = client.get_object(Bucket=bucket, Key=key)
  return obj["Body"].read()


def upload_bytes(client: boto3.client, bucket: str, key: str, data: bytes, content_type: str) -> None:
  client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def list_keys(client: boto3.client, bucket: str, prefix: str) -> List[str]:
  keys: List[str] = []
  token: Optional[str] = None
  while True:
    params: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
    if token:
      params["ContinuationToken"] = token
    resp = client.list_objects_v2(**params)
    for entry in resp.get("Contents", []) or []:
      key = entry.get("Key")
      if key:
        keys.append(key)
    if not resp.get("IsTruncated"):
      break
    token = resp.get("NextContinuationToken")
  return keys


def parse_numeric_value(value: Any) -> Optional[float]:
  if is_missing_value(value):
    return None
  if isinstance(value, (int, float)) and not pd.isna(value):
    return float(value)
  text = str(value).strip().replace(",", ".")
  if not text:
    return None
  numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
  if not numbers:
    return None
  if "-" in text and len(numbers) >= 2:
    vals = [float(num) for num in numbers[:2]]
    return sum(vals) / len(vals)
  return float(numbers[0])


def parse_numeric_series(series: Iterable[Any]) -> Tuple[List[Optional[float]], int]:
  failures = 0
  parsed: List[Optional[float]] = []
  for value in series:
    parsed_value = parse_numeric_value(value)
    if parsed_value is None and not is_missing_value(value):
      failures += 1
    parsed.append(parsed_value)
  return parsed, failures


def detect_list_delimiters(values: Iterable[str], min_fraction: float = 0.15) -> List[str]:
  candidates = [",", ";", "|"]
  counts = {delimiter: 0 for delimiter in candidates}
  total = 0
  for value in values:
    if not value:
      continue
    total += 1
    for delimiter in candidates:
      if delimiter in value:
        parts = [part.strip() for part in value.split(delimiter) if part.strip()]
        if len(parts) >= 2:
          counts[delimiter] += 1
  if total == 0:
    return []
  return [delimiter for delimiter, count in counts.items() if count / total >= min_fraction]


def safe_json_loads(text: str) -> Any:
  cleaned = text.strip()
  if cleaned.startswith("```"):
    cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
    if cleaned.endswith("```"):
      cleaned = cleaned[: -3].strip()
  return json.loads(cleaned)
