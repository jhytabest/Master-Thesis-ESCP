import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
  import boto3
except Exception:  # pragma: no cover - optional for local mode
  boto3 = None


def log(message: str) -> None:
  print(f"[mapping.utils] {message}", flush=True)


def env_or_error(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(f"Missing required env var: {name}")
  return value


def local_storage_root() -> Optional[Path]:
  value = os.getenv("LOCAL_STORAGE_ROOT")
  if not value:
    return None
  root = Path(value).expanduser().resolve()
  root.mkdir(parents=True, exist_ok=True)
  return root


def using_local_storage() -> bool:
  return local_storage_root() is not None


def resolve_bucket_name() -> str:
  if using_local_storage():
    return "__local__"
  return env_or_error("R2_BUCKET")


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


def build_s3_client() -> Any:
  root = local_storage_root()
  if root is not None:
    log(f"build_s3_client local_root={root}")
    return {"mode": "local", "root": str(root)}

  if boto3 is None:
    raise RuntimeError("boto3 is required when LOCAL_STORAGE_ROOT is not set")
  endpoint = env_or_error("R2_ENDPOINT")
  access_key = env_or_error("R2_ACCESS_KEY_ID")
  secret_key = env_or_error("R2_SECRET_ACCESS_KEY")
  log(f"build_s3_client endpoint={endpoint}")
  return boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="auto"
  )


def _local_path(root: Path, key: str) -> Path:
  relative = key.lstrip("/")
  path = (root / relative).resolve()
  if not str(path).startswith(str(root)):
    raise RuntimeError(f"Invalid storage key outside root: {key}")
  return path


def _is_local_client(client: Any) -> bool:
  return isinstance(client, dict) and client.get("mode") == "local"


def download_bytes(client: Any, bucket: str, key: str) -> bytes:
  if _is_local_client(client):
    root = Path(client["root"]).resolve()
    path = _local_path(root, key)
    log(f"download_bytes local key={key}")
    return path.read_bytes()

  log(f"download_bytes bucket={bucket} key={key}")
  obj = client.get_object(Bucket=bucket, Key=key)
  data = obj["Body"].read()
  log(f"download_bytes completed size={len(data)}")
  return data


def upload_bytes(client: Any, bucket: str, key: str, data: bytes, content_type: str) -> None:
  if _is_local_client(client):
    root = Path(client["root"]).resolve()
    path = _local_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log(f"upload_bytes local key={key} bytes={len(data)} content_type={content_type}")
    return

  log(f"upload_bytes bucket={bucket} key={key} bytes={len(data)} content_type={content_type}")
  client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
  log("upload_bytes completed")


def list_keys(client: Any, bucket: str, prefix: str) -> List[str]:
  if _is_local_client(client):
    root = Path(client["root"]).resolve()
    prefix_clean = prefix.lstrip("/")
    base = _local_path(root, prefix_clean)
    if not base.exists():
      return []
    keys: List[str] = []
    for path in base.rglob("*"):
      if path.is_file():
        keys.append(path.relative_to(root).as_posix())
    keys.sort()
    log(f"list_keys local prefix={prefix} count={len(keys)}")
    return keys

  keys: List[str] = []
  token: Optional[str] = None
  log(f"list_keys bucket={bucket} prefix={prefix}")
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
  log(f"list_keys completed count={len(keys)}")
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
