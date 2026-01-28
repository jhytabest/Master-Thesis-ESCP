import argparse
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai
from google.genai import types as genai_types

from mapping.utils import (
  build_s3_client,
  detect_list_delimiters,
  download_bytes,
  env_or_error,
  is_missing_value,
  normalize_header,
  normalize_raw_value,
  now_iso,
  parse_numeric_series,
  safe_json_loads,
  sha256_text,
  upload_bytes
)


ALLOWED_ACTIONS = {"map_values", "parse_numeric", "split_list", "consolidate", "skip"}
MAX_MODEL_ATTEMPTS = 3
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "60"))
GROUP_SAMPLE_ROWS = int(os.getenv("GROUP_SAMPLE_ROWS", "20"))
MAX_GROUP_WORKERS = int(os.getenv("MAX_GROUP_WORKERS", "10"))


def build_client() -> genai.Client:
  project = env_or_error("GCP_PROJECT_ID")
  location = os.getenv("GCP_LOCATION", "global")
  return genai.Client(
    vertexai=True,
    project=project,
    location=location,
    http_options=genai_types.HttpOptions(
      api_version="v1",
      timeout=MODEL_TIMEOUT_SECONDS * 1000
    )
  )


def detect_column_type(values: List[str], numeric_failure_rate: float, list_delimiters: List[str]) -> str:
  if list_delimiters:
    return "multi_value_list"
  if numeric_failure_rate <= 0.2 and values:
    return "numeric_ish"
  if values:
    avg_len = sum(len(value) for value in values) / max(len(values), 1)
    space_ratio = sum(1 for value in values if " " in value) / max(len(values), 1)
    if avg_len >= 40 or space_ratio >= 0.6:
      return "free_text"
  return "categorical"


def build_prompt(context: Dict[str, Any]) -> str:
  return (
    "You are a data cleaning assistant. Decide whether to propose a mapping for the column.\n"
    "Return ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "column": "RAW COLUMN",\n'
    '  "normalized_column": "column_clean",\n'
    '  "action": "map_values | parse_numeric | split_list | consolidate | skip",\n'
    '  "value_map": [{"raw": "...", "normalized": "...", "confidence": 0.91}],\n'
    '  "list_delimiters": [",", ";"],\n'
    '  "notes": "...",\n'
    '  "confidence": 0.83\n'
    "}\n"
    "Rules:\n"
    "- If action is map_values, include EVERY distinct raw value in value_map.\n"
    "- If action is parse_numeric, normalized_column should be a numeric column (e.g. *_eur or *_num).\n"
    "- If action is split_list, include list_delimiters you expect.\n"
    "- Use snake_case for normalized_column.\n"
    "- Prefer skip for free-text fields.\n"
    "Context JSON:\n"
    f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
  )


def build_grouping_prompt(columns: List[str], sample_rows: List[Dict[str, Any]]) -> str:
  return (
    "You are a data cleaning assistant. Group related columns into coherent bundles.\n"
    "Return ONLY valid JSON with this schema:\n"
    "{\n"
    '  "groups": [\n'
    '    {"group_name": "location", "columns": ["city", "country"], "notes": "..."}\n'
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- Every column must appear in exactly one group.\n"
    "- Use short, descriptive group_name values.\n"
    "- Groups can be singletons.\n"
    "Columns:\n"
    f"{json.dumps(columns, ensure_ascii=False)}\n"
    "Sample rows (first few rows):\n"
    f"{json.dumps(sample_rows, ensure_ascii=False, indent=2)}\n"
  )


def build_group_prompt(group_name: str, contexts: List[Dict[str, Any]]) -> str:
  return (
    "You are a data cleaning assistant. For this group of related columns, propose mappings.\n"
    "Return ONLY valid JSON with this schema:\n"
    "{\n"
    '  "group_name": "group",\n'
    '  "columns": [\n'
    "    {\n"
    '      "column": "RAW COLUMN",\n'
    '      "normalized_column": "column_clean",\n'
    '      "action": "map_values | parse_numeric | split_list | consolidate | skip",\n'
    '      "value_map": [{"raw": "...", "normalized": "...", "confidence": 0.91}],\n'
    '      "list_delimiters": [",", ";"],\n'
    '      "notes": "...",\n'
    '      "confidence": 0.83\n'
    "    }\n"
    "  ],\n"
    '  "consolidations": [\n'
    "    {\n"
    '      "target_column": "canonical_name",\n'
    '      "source_columns": ["COL A", "COL B"],\n'
    '      "method": "first_non_empty | merge_list | numeric_merge",\n'
    '      "notes": "...",\n'
    '      "confidence": 0.8\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- One entry per input column in columns[].\n"
    "- If action is map_values, include EVERY distinct raw value in value_map.\n"
    "- If action is parse_numeric, normalized_column should be numeric (e.g. *_eur or *_num).\n"
    "- If action is split_list, include list_delimiters you expect.\n"
    "- Use snake_case for normalized_column.\n"
    "- Prefer skip for free-text fields.\n"
    "- If no consolidations, return an empty list [].\n"
    f"Group name: {group_name}\n"
    "Context JSON:\n"
    f"{json.dumps(contexts, ensure_ascii=False, indent=2)}\n"
  )


def build_retry_prompt(base_prompt: str, error: str) -> str:
  return (
    f"{base_prompt}\n"
    "Previous response was invalid.\n"
    f"Reason: {error}\n"
    "Return ONLY valid JSON matching the schema."
  )


def log(message: str) -> None:
  print(message, file=sys.stdout, flush=True)


def extract_usage(response: genai_types.GenerateContentResponse) -> Dict[str, int]:
  usage = getattr(response, "usage_metadata", None)
  if usage is None:
    return {}
  data: Dict[str, int] = {}
  for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
    value = getattr(usage, key, None)
    if value is not None:
      data[key] = value
  for alt_key, key in (
    ("prompt_tokens", "prompt_token_count"),
    ("candidate_tokens", "candidates_token_count"),
    ("total_tokens", "total_token_count"),
  ):
    value = getattr(usage, alt_key, None)
    if value is not None and key not in data:
      data[key] = value
  return data


def generate_with_timeout(client: genai.Client, model_name: str, prompt: str) -> genai_types.GenerateContentResponse:
  def _call() -> genai_types.GenerateContentResponse:
    response = client.models.generate_content(
      model=model_name,
      contents=prompt,
      config=genai_types.GenerateContentConfig(
        temperature=0,
        candidate_count=1,
        response_mime_type="application/json"
      )
    )
    return response

  with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(_call)
    return future.result(timeout=MODEL_TIMEOUT_SECONDS)


def build_consolidation_prompt(columns: List[Dict[str, Any]]) -> str:
  return (
    "You are a data cleaning assistant. Suggest column consolidations when two or more columns overlap.\n"
    "Return ONLY valid JSON as a list with this schema:\n"
    "[\n"
    "  {\n"
    '    "target_column": "canonical_name",\n'
    '    "source_columns": ["COL A", "COL B"],\n'
    '    "method": "first_non_empty | merge_list | numeric_merge",\n'
    '    "notes": "...",\n'
    '    "confidence": 0.8\n'
    "  }\n"
    "]\n"
    "Rules:\n"
    "- Only propose consolidations when overlap is strong.\n"
    "- Use snake_case for target_column.\n"
    "- If no consolidations, return an empty list [].\n"
    "Columns JSON:\n"
    f"{json.dumps(columns, ensure_ascii=False, indent=2)}\n"
  )


def extract_distinct_values(series: pd.Series) -> List[str]:
  seen = set()
  values: List[str] = []
  for value in series.tolist():
    normalized = normalize_raw_value(value)
    if normalized is None:
      continue
    if normalized in seen:
      continue
    seen.add(normalized)
    values.append(normalized)
  return values


def normalize_group_assignments(columns: List[str], groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  seen = set()
  normalized: List[Dict[str, Any]] = []
  for group in groups:
    if not isinstance(group, dict):
      continue
    group_cols = [col for col in group.get("columns", []) if isinstance(col, str)]
    group_cols = [col for col in group_cols if col in columns]
    group_cols = [col for col in group_cols if col not in seen]
    if not group_cols:
      continue
    for col in group_cols:
      seen.add(col)
    normalized.append({
      "group_name": group.get("group_name") or "group",
      "columns": group_cols,
      "notes": group.get("notes") or ""
    })
  missing = [col for col in columns if col not in seen]
  for col in missing:
    normalized.append({"group_name": "singleton", "columns": [col], "notes": ""})
  return normalized


def build_value_map(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
  mapping: Dict[str, Dict[str, Any]] = {}
  for entry in entries:
    raw = normalize_raw_value(entry.get("raw"))
    normalized = entry.get("normalized")
    if raw is None:
      continue
    mapping[raw] = {
      "normalized": normalized,
      "confidence": entry.get("confidence")
    }
  return mapping


def default_normalized_column(column: str, action: str) -> str:
  base = normalize_header(column)
  if action == "map_values":
    if base == "company_status":
      return "company_status_clean"
    return f"{base}_clean"
  if action == "parse_numeric":
    if base == "target_total_funding":
      return "target_total_funding_eur"
    if base == "last_funding":
      return "last_funding_eur"
    if base in {"valuation", "valuation_eur"}:
      return "valuation_eur"
    if base == "target_rounds":
      return "target_rounds_num"
    return f"{base}_num"
  if action == "split_list":
    return f"{base}_list"
  return base


def fallback_skip_output(column: str, reason: str) -> Dict[str, Any]:
  return {
    "column": column,
    "normalized_column": default_normalized_column(column, "skip"),
    "action": "skip",
    "value_map": [],
    "list_delimiters": [],
    "notes": f"auto-skip: {reason}",
    "confidence": 0.0
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Propose mapping bundle")
  parser.add_argument("--version_id", required=True)
  parser.add_argument("--output_prefix")
  parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
  args = parser.parse_args()

  version_id = args.version_id
  output_prefix = args.output_prefix or f"mappings/proposals/{version_id}"
  output_prefix = output_prefix.rstrip("/")
  log(f"[start] version_id={version_id} output_prefix={output_prefix} model={args.model}")

  bucket = env_or_error("R2_BUCKET")
  s3_client = build_s3_client()

  source_key = f"raw/{version_id}/dataset.csv"
  raw_bytes = download_bytes(s3_client, bucket, source_key)
  df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False)
  log(f"[dataset] rows={len(df)} columns={len(df.columns)}")

  ai_client = build_client()

  usage_totals: Dict[str, int] = {}
  usage_lock = threading.Lock()
  consolidations: List[Dict[str, Any]] = []
  consolidation_lock = threading.Lock()

  sample_rows = df.head(GROUP_SAMPLE_ROWS).to_dict(orient="records")
  columns = [str(col) for col in df.columns]
  grouping_prompt = build_grouping_prompt(columns, sample_rows)
  grouping_prompt_hash = sha256_text(grouping_prompt)
  log(f"[grouping] prompt_chars={len(grouping_prompt)} sample_rows={len(sample_rows)}")
  log("[grouping] start")
  try:
    grouping_response = generate_with_timeout(ai_client, args.model, grouping_prompt)
    usage = extract_usage(grouping_response)
    if usage:
      log(f"[grouping] usage={usage}")
      for key, value in usage.items():
        usage_totals[key] = usage_totals.get(key, 0) + int(value)
    grouping_output = safe_json_loads(grouping_response.text or "")
    groups = normalize_group_assignments(columns, grouping_output.get("groups", []))
    group_sizes = [len(group.get("columns", [])) for group in groups]
    log(f"[grouping] groups={len(groups)} max_size={max(group_sizes) if group_sizes else 0} singles={sum(1 for size in group_sizes if size == 1)}")
    log(f"[grouping] sizes={group_sizes}")
  except Exception as exc:
    log(f"[grouping] error: {exc}")
    groups = [{"group_name": "singleton", "columns": [col], "notes": ""} for col in columns]

  def process_group(group: Dict[str, Any]) -> None:
    group_name = group.get("group_name") or "group"
    group_columns = group.get("columns", [])
    log(f"[group {group_name}] start columns={len(group_columns)}")
    group_s3_client = build_s3_client()
    group_start = now_iso()

    contexts: List[Dict[str, Any]] = []
    per_column_stats: Dict[str, Dict[str, Any]] = {}
    for column in group_columns:
      series = df[column]
      total_rows = len(series)
      missing_count = sum(1 for value in series.tolist() if is_missing_value(value))
      distinct_values = extract_distinct_values(series)
      _, numeric_failures = parse_numeric_series(distinct_values)
      numeric_failure_rate = numeric_failures / max(len(distinct_values), 1)
      list_delimiters = detect_list_delimiters(distinct_values)
      detected_type = detect_column_type(distinct_values, numeric_failure_rate, list_delimiters)
      context = {
        "version_id": version_id,
        "column": column,
        "row_count": total_rows,
        "missing_pct": missing_count / max(total_rows, 1),
        "unique_count": len(distinct_values),
        "detected_type": detected_type,
        "list_delimiters": list_delimiters,
        "examples": distinct_values[:8],
        "distinct_values": distinct_values
      }
      contexts.append(context)
      per_column_stats[column] = {
        "distinct_values": distinct_values,
        "missing_count": missing_count,
        "total_rows": total_rows,
        "list_delimiters": list_delimiters,
        "detected_type": detected_type,
        "input_hash": sha256_text(json.dumps(distinct_values, ensure_ascii=False, separators=(",", ":")))
      }

    base_prompt = build_group_prompt(group_name, contexts)
    prompt_hash = ""
    output: Dict[str, Any] = {}
    last_error: Optional[str] = None
    for attempt in range(MAX_MODEL_ATTEMPTS):
      prompt = base_prompt
      if last_error:
        prompt = build_retry_prompt(base_prompt, last_error)
      prompt_hash = sha256_text(prompt)
      try:
        log(f"[group {group_name}] attempt {attempt + 1}/{MAX_MODEL_ATTEMPTS}")
        response = generate_with_timeout(ai_client, args.model, prompt)
        usage = extract_usage(response)
        if usage:
          log(f"[group {group_name}] usage={usage}")
          with usage_lock:
            for key, value in usage.items():
              usage_totals[key] = usage_totals.get(key, 0) + int(value)
        output = safe_json_loads(response.text or "")
        if not isinstance(output, dict):
          raise ValueError("expected JSON object")
        if not isinstance(output.get("columns"), list):
          raise ValueError("missing columns list")
        break
      except TimeoutError:
        last_error = f"timeout after {MODEL_TIMEOUT_SECONDS}s"
        log(f"[group {group_name}] timeout: {last_error}")
        if attempt == MAX_MODEL_ATTEMPTS - 1:
          output = {}
      except Exception as exc:
        last_error = str(exc)
        log(f"[group {group_name}] error: {last_error}")
        if attempt == MAX_MODEL_ATTEMPTS - 1:
          output = {}

    output_columns = output.get("columns", []) if isinstance(output, dict) else []
    columns_by_name = {
      entry.get("column"): entry
      for entry in output_columns
      if isinstance(entry, dict)
    }

    group_consolidations = output.get("consolidations", []) if isinstance(output, dict) else []
    if isinstance(group_consolidations, list):
      with consolidation_lock:
        for item in group_consolidations:
          if isinstance(item, dict):
            consolidations.append(item)

    for column in group_columns:
      stats = per_column_stats[column]
      distinct_values = stats["distinct_values"]
      list_delimiters = stats["list_delimiters"]
      detected_type = stats["detected_type"]
      total_rows = stats["total_rows"]
      missing_count = stats["missing_count"]
      input_hash = stats["input_hash"]

      output_entry = columns_by_name.get(column)
      if not isinstance(output_entry, dict):
        output_entry = fallback_skip_output(column, "missing output for column")

      action = output_entry.get("action")
      if action not in ALLOWED_ACTIONS:
        output_entry = fallback_skip_output(column, "invalid action")
        action = output_entry.get("action")

      normalized_column = output_entry.get("normalized_column") or default_normalized_column(column, action)
      value_map = output_entry.get("value_map", [])
      if not isinstance(value_map, list):
        output_entry = fallback_skip_output(column, "invalid value_map")
        action = output_entry.get("action")
        normalized_column = output_entry.get("normalized_column") or default_normalized_column(column, action)
        value_map = []
      mapping = build_value_map(value_map)
      list_delimiters = output_entry.get("list_delimiters") or list_delimiters
      if not isinstance(list_delimiters, list):
        list_delimiters = []

      rows: List[Dict[str, Any]] = []
      if action == "map_values":
        missing_values = [raw_value for raw_value in distinct_values if raw_value not in mapping]
        if missing_values:
          output_entry = fallback_skip_output(column, "incomplete value_map")
          action = output_entry.get("action")
          normalized_column = output_entry.get("normalized_column") or default_normalized_column(column, action)
          value_map = []
          mapping = {}
        else:
          for raw_value in distinct_values:
            entry = mapping[raw_value]
            rows.append({
              "raw_value": raw_value,
              "normalized_value": entry.get("normalized") or "",
              "confidence": entry.get("confidence") or "",
              "approved": 0,
              "notes": ""
            })
      else:
        for raw_value in distinct_values:
          rows.append({
            "raw_value": raw_value,
            "normalized_value": "",
            "confidence": "",
            "approved": 0,
            "notes": ""
          })

      csv_df = pd.DataFrame(rows, columns=["raw_value", "normalized_value", "confidence", "approved", "notes"])
      csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
      csv_key = f"{output_prefix}/{column}.csv"
      upload_bytes(group_s3_client, bucket, csv_key, csv_bytes, "text/csv")

      manifest = {
        "column": column,
        "normalized_column": normalized_column,
        "action": action,
        "confidence": output_entry.get("confidence"),
        "notes": output_entry.get("notes"),
        "model_id": args.model,
        "prompt_hash": prompt_hash,
        "input_hash": input_hash,
        "unique_count": len(distinct_values),
        "missing_pct": missing_count / max(total_rows, 1),
        "list_delimiters": list_delimiters,
        "value_map_count": len(value_map),
        "proposal_csv_path": csv_key,
        "generated_at": now_iso(),
        "group_name": group_name,
        "detected_type": detected_type
      }
      manifest_key = f"{output_prefix}/{column}.manifest.json"
      upload_bytes(group_s3_client, bucket, manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")
      log(f"[group {group_name}] wrote column={column} action={action} rows={len(rows)}")

    log(f"[group {group_name}] done started_at={group_start} finished_at={now_iso()}")

  max_workers = min(max(len(groups), 1), MAX_GROUP_WORKERS)
  log(f"[grouping] max_workers={max_workers}")
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(executor.map(process_group, groups))

  consolidations_key = f"{output_prefix}/consolidations.json"
  upload_bytes(
    s3_client,
    bucket,
    consolidations_key,
    json.dumps(consolidations, indent=2).encode("utf-8"),
    "application/json"
  )
  consolidations_manifest = {
    "model_id": args.model,
    "prompt_hash": grouping_prompt_hash,
    "input_hash": sha256_text(json.dumps(groups, ensure_ascii=False, separators=(",", ":"))),
    "generated_at": now_iso(),
    "consolidations_path": consolidations_key,
    "count": len(consolidations)
  }
  consolidations_manifest_key = f"{output_prefix}/consolidations.manifest.json"
  upload_bytes(
    s3_client,
    bucket,
    consolidations_manifest_key,
    json.dumps(consolidations_manifest, indent=2).encode("utf-8"),
    "application/json"
  )
  log(f"[consolidations] saved count={len(consolidations)}")

  if usage_totals:
    log(f"[usage totals] {usage_totals}")

  print(json.dumps({
    "version_id": version_id,
    "output_prefix": output_prefix,
    "columns": len(df.columns),
    "generated_at": now_iso(),
    "usage_totals": usage_totals
  }, indent=2))
  log("[done] proposal job completed")


if __name__ == "__main__":
  main()
