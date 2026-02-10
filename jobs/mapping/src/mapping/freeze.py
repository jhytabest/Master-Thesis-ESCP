import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

try:
  import boto3
except Exception:  # pragma: no cover - optional in local mode
  boto3 = None

from mapping.utils import (
  build_s3_client,
  download_bytes,
  list_keys,
  normalize_raw_value,
  now_iso,
  resolve_bucket_name,
  sha256_hex,
  upload_bytes
)


def log(message: str) -> None:
  print(f"[mapping.freeze] {message}", flush=True)


def parse_approved_flag(value: Any) -> bool:
  if value is None:
    return False
  if isinstance(value, (int, float)) and not pd.isna(value):
    return int(value) == 1
  text = str(value).strip().lower()
  return text in {"1", "true", "yes", "y"}


def load_csv_from_r2(client: boto3.client, bucket: str, key: str) -> pd.DataFrame:
  log(f"load_csv_from_r2 key={key}")
  raw = download_bytes(client, bucket, key)
  return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)


def load_manifest_bytes(client: boto3.client, bucket: str, key: str) -> Tuple[Dict[str, Any], bytes]:
  log(f"load_manifest_bytes key={key}")
  raw = download_bytes(client, bucket, key)
  return json.loads(raw.decode("utf-8")), raw


def resolve_proposal_keys(
  client: boto3.client,
  bucket: str,
  proposal_path: Optional[str],
  proposal_prefix: Optional[str]
) -> Tuple[List[str], Optional[str]]:
  if proposal_prefix:
    prefix = proposal_prefix.rstrip("/") + "/"
    keys = [key for key in list_keys(client, bucket, prefix) if key.endswith(".csv")]
    log(f"resolve_proposal_keys prefix={prefix} count={len(keys)}")
    return keys, prefix.rstrip("/")
  if proposal_path:
    log(f"resolve_proposal_keys path={proposal_path}")
    return [proposal_path], os.path.dirname(proposal_path)
  raise RuntimeError("Provide --proposal_path or --proposal_prefix")


def register_mapping_bundle(payload: Dict[str, Any]) -> None:
  base = os.getenv("WORKER_API_BASE")
  token = os.getenv("WORKER_API_TOKEN")

  local_root = os.getenv("LOCAL_STORAGE_ROOT")
  if local_root and (not base or not token):
    root = Path(local_root).expanduser().resolve()
    meta_path = root / "meta" / "mapping_bundles.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    if meta_path.exists():
      records = json.loads(meta_path.read_text(encoding="utf-8"))
      if not isinstance(records, list):
        records = []
    records = [item for item in records if item.get("mapping_bundle_id") != payload.get("mapping_bundle_id")]
    records.append(payload)
    meta_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log(f"register_mapping_bundle local metadata updated path={meta_path}")
    return

  if not base or not token:
    log("register_mapping_bundle skipped (missing WORKER_API_BASE or WORKER_API_TOKEN)")
    return
  url = f"{base}/mapping_bundles/register"
  data = json.dumps(payload).encode("utf-8")
  req = urlrequest.Request(url, data=data, method="POST")
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
    log("register_mapping_bundle completed")
  except urlerror.HTTPError as exc:
    details = exc.read().decode("utf-8", errors="ignore")
    raise RuntimeError(f"Failed to register mapping bundle: {exc.code} {details}") from exc
  except Exception as exc:
    raise RuntimeError(f"Failed to register mapping bundle: {exc}") from exc


def main() -> None:
  parser = argparse.ArgumentParser(description="Freeze mapping bundle")
  parser.add_argument("--proposal_path")
  parser.add_argument("--proposal_prefix")
  parser.add_argument("--output_bundle_id", required=True)
  parser.add_argument("--auto_approve_all", action="store_true")
  parser.add_argument("--approved_by")
  parser.add_argument("--approved_at")
  parser.add_argument("--output_prefix")
  args = parser.parse_args()

  bucket = resolve_bucket_name()
  client = build_s3_client()
  log(f"start output_bundle_id={args.output_bundle_id} proposal_path={args.proposal_path} proposal_prefix={args.proposal_prefix}")

  proposal_keys, resolved_prefix = resolve_proposal_keys(
    client, bucket, args.proposal_path, args.proposal_prefix
  )
  if not proposal_keys:
    raise RuntimeError("No proposal CSVs found")

  output_prefix = args.output_prefix or f"mappings/{args.output_bundle_id}"
  output_prefix = output_prefix.rstrip("/")
  approved_at = args.approved_at or now_iso()
  log(f"output_prefix={output_prefix} approved_at={approved_at}")

  columns_manifest: List[Dict[str, Any]] = []
  source_version_id: Optional[str] = None

  for csv_key in proposal_keys:
    if csv_key.endswith("consolidations.csv"):
      continue
    base_name = os.path.basename(csv_key)
    fallback_column = base_name[: -4]
    manifest_key = f"{os.path.dirname(csv_key)}/{fallback_column}.manifest.json"

    manifest, manifest_bytes = load_manifest_bytes(client, bucket, manifest_key)
    column_name = manifest.get("column") or fallback_column
    proposal_manifest_hash = sha256_hex(manifest_bytes)
    if not source_version_id:
      source_version_id = manifest.get("version_id")
    log(f"processing column={column_name} csv_key={csv_key}")

    action = manifest.get("action")
    if not action:
      raise RuntimeError(f"Missing action in manifest for {column_name}")
    normalized_column = manifest.get("normalized_column") or column_name

    df = load_csv_from_r2(client, bucket, csv_key)
    required_cols = {"raw_value", "normalized_value", "approved"}
    if not required_cols.issubset(df.columns):
      raise RuntimeError(f"Proposal CSV {csv_key} missing required columns")

    df["approved_flag"] = df["approved"].apply(parse_approved_flag)
    if args.auto_approve_all:
      df["approved_flag"] = True
    approved_rows = df[df["approved_flag"]].copy()
    log(f"approved rows={len(approved_rows)} total rows={len(df)} action={action}")

    approved_rows["raw_value"] = approved_rows["raw_value"].apply(normalize_raw_value)
    approved_rows["normalized_value"] = approved_rows["normalized_value"].apply(normalize_raw_value)

    if approved_rows["raw_value"].isna().any():
      raise RuntimeError(f"Approved rows missing raw_value in {csv_key}")
    if action == "map_values" and approved_rows["normalized_value"].isna().any():
      raise RuntimeError(f"Approved rows missing normalized_value in {csv_key}")

    duplicates = approved_rows["raw_value"].duplicated()
    if duplicates.any():
      raise RuntimeError(f"Duplicate raw_value entries in approved rows for {csv_key}")

    completeness_enforced = False
    if action == "map_values":
      completeness_enforced = True
      if df.empty:
        raise RuntimeError(f"Proposal CSV is empty for mapping column {column_name}")
      if not df["approved_flag"].all():
        raise RuntimeError(f"Incomplete approvals for mapping column {column_name}")

    mapping_path = None
    approved_csv_hash = None
    if action == "map_values":
      for col in ["confidence", "notes"]:
        if col not in approved_rows.columns:
          approved_rows[col] = ""
      mapping_df = approved_rows[["raw_value", "normalized_value", "confidence", "notes"]].copy()
      mapping_df = mapping_df.fillna("")
      mapping_bytes = mapping_df.to_csv(index=False).encode("utf-8")
      filename = (
        "status_mapping.csv"
        if normalized_column == "company_status_clean"
        else f"{normalized_column}_mapping.csv"
      )
      mapping_path = f"{output_prefix}/{filename}"
      upload_bytes(client, bucket, mapping_path, mapping_bytes, "text/csv")
      log(f"wrote mapping_path={mapping_path} rows={len(mapping_df)}")
      approved_csv_hash = sha256_hex(mapping_bytes)

    columns_manifest.append({
      "column": column_name,
      "normalized_column": normalized_column,
      "action": action,
      "mapping_path": mapping_path,
      "proposal_csv_path": csv_key,
      "proposal_manifest_path": manifest_key,
      "proposal_manifest_hash": proposal_manifest_hash,
      "model_id": manifest.get("model_id"),
      "prompt_hash": manifest.get("prompt_hash"),
      "input_hash": manifest.get("input_hash"),
      "list_delimiters": manifest.get("list_delimiters"),
      "unique_count": manifest.get("unique_count"),
      "confidence": manifest.get("confidence"),
      "notes": manifest.get("notes"),
      "approved_rows": int(len(approved_rows)),
      "total_rows": int(len(df)),
      "approved_csv_hash": approved_csv_hash,
      "completeness_enforced": completeness_enforced
    })

  consolidations_path = None
  consolidations_manifest_hash = None
  if resolved_prefix:
    consolidations_key = f"{resolved_prefix}/consolidations.json"
    try:
      consolidations_bytes = download_bytes(client, bucket, consolidations_key)
      consolidations_path = f"{output_prefix}/consolidations.json"
      upload_bytes(client, bucket, consolidations_path, consolidations_bytes, "application/json")
      consolidations_manifest_key = f"{resolved_prefix}/consolidations.manifest.json"
      manifest, manifest_bytes = load_manifest_bytes(client, bucket, consolidations_manifest_key)
      consolidations_manifest_hash = sha256_hex(manifest_bytes)
    except Exception:
      pass

  bundle_manifest = {
    "mapping_bundle_id": args.output_bundle_id,
    "created_at": now_iso(),
    "source_version_id": source_version_id,
    "approved_by": args.approved_by,
    "approved_at": approved_at,
    "columns": columns_manifest,
    "consolidations_path": consolidations_path,
    "consolidations_manifest_hash": consolidations_manifest_hash
  }

  manifest_bytes = json.dumps(bundle_manifest, indent=2).encode("utf-8")
  manifest_key = f"{output_prefix}/bundle_manifest.json"
  upload_bytes(client, bucket, manifest_key, manifest_bytes, "application/json")
  manifest_hash = sha256_hex(manifest_bytes)
  log(f"wrote bundle_manifest={manifest_key}")

  register_mapping_bundle({
    "mapping_bundle_id": args.output_bundle_id,
    "created_at": bundle_manifest["created_at"],
    "source_version_id": source_version_id,
    "columns_json": json.dumps([col["normalized_column"] for col in columns_manifest]),
    "artifact_path": manifest_key,
    "manifest_sha256": manifest_hash
  })

  print(json.dumps({
    "mapping_bundle_id": args.output_bundle_id,
    "output_prefix": output_prefix,
    "bundle_manifest": manifest_key,
    "bundle_manifest_sha256": manifest_hash,
    "columns": len(columns_manifest)
  }, indent=2))
  log("done")


if __name__ == "__main__":
  main()
