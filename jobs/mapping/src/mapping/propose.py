import argparse
import io
import json
import os
import sys
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

  bucket = env_or_error("R2_BUCKET")
  s3_client = build_s3_client()

  source_key = f"raw/{version_id}/dataset.csv"
  raw_bytes = download_bytes(s3_client, bucket, source_key)
  df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False)

  ai_client = build_client()

  column_summaries: List[Dict[str, Any]] = []
  usage_totals: Dict[str, int] = {}
  for idx, column in enumerate(df.columns, start=1):
    log(f"[column {idx}/{len(df.columns)}] start {column}")
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

    base_prompt = build_prompt(context)
    input_hash = sha256_text(json.dumps(distinct_values, ensure_ascii=False, separators=(",", ":")))

    output: Dict[str, Any]
    prompt_hash = ""
    last_error: Optional[str] = None
    for attempt in range(MAX_MODEL_ATTEMPTS):
      prompt = base_prompt
      if last_error:
        prompt = build_retry_prompt(base_prompt, last_error)
      prompt_hash = sha256_text(prompt)
      try:
        log(f"[column {column}] attempt {attempt + 1}/{MAX_MODEL_ATTEMPTS}")
        response = generate_with_timeout(ai_client, args.model, prompt)
        usage = extract_usage(response)
        if usage:
          log(f"[column {column}] usage={usage}")
          for key, value in usage.items():
            usage_totals[key] = usage_totals.get(key, 0) + int(value)
        output = safe_json_loads(response.text or "")
        if not isinstance(output, dict):
          raise ValueError("expected JSON object")
        action = output.get("action")
        if action not in ALLOWED_ACTIONS:
          raise ValueError(f"invalid action '{action}'")
        log(f"[column {column}] action={action}")
        break
      except TimeoutError:
        last_error = f"timeout after {MODEL_TIMEOUT_SECONDS}s"
        output = {}
        log(f"[column {column}] timeout: {last_error}")
        if attempt == MAX_MODEL_ATTEMPTS - 1:
          output = fallback_skip_output(column, last_error)
      except Exception as exc:
        last_error = str(exc)
        output = {}
        log(f"[column {column}] error: {last_error}")
        if attempt == MAX_MODEL_ATTEMPTS - 1:
          output = fallback_skip_output(column, last_error)
    else:
      output = fallback_skip_output(column, "invalid model response")
      prompt_hash = sha256_text(base_prompt)
      action = output.get("action")

    action = output.get("action")
    normalized_column = output.get("normalized_column") or default_normalized_column(column, action)
    value_map = output.get("value_map", [])
    if not isinstance(value_map, list):
      output = fallback_skip_output(column, "invalid value_map")
      action = output.get("action")
      normalized_column = output.get("normalized_column") or default_normalized_column(column, action)
      value_map = []
    mapping = build_value_map(value_map)
    list_delimiters = output.get("list_delimiters") or list_delimiters
    if not isinstance(list_delimiters, list):
      list_delimiters = []

    rows: List[Dict[str, Any]] = []
    if action == "map_values":
      missing_values = [raw_value for raw_value in distinct_values if raw_value not in mapping]
      if missing_values:
        output = fallback_skip_output(column, "incomplete value_map")
        action = output.get("action")
        normalized_column = output.get("normalized_column") or default_normalized_column(column, action)
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
    upload_bytes(s3_client, bucket, csv_key, csv_bytes, "text/csv")

    manifest = {
      "column": column,
      "normalized_column": normalized_column,
      "action": action,
      "confidence": output.get("confidence"),
      "notes": output.get("notes"),
      "model_id": args.model,
      "prompt_hash": prompt_hash,
      "input_hash": input_hash,
      "unique_count": len(distinct_values),
      "missing_pct": missing_count / max(total_rows, 1),
      "list_delimiters": list_delimiters,
      "value_map_count": len(value_map),
      "proposal_csv_path": csv_key,
      "generated_at": now_iso()
    }
    manifest_key = f"{output_prefix}/{column}.manifest.json"
    upload_bytes(s3_client, bucket, manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    column_summaries.append({
      "column": column,
      "detected_type": detected_type,
      "unique_count": len(distinct_values),
      "examples": distinct_values[:5]
    })

  consolidation_prompt = build_consolidation_prompt(column_summaries)
  consolidation_prompt_hash = sha256_text(consolidation_prompt)
  consolidation_input_hash = sha256_text(json.dumps(column_summaries, ensure_ascii=False, separators=(",", ":")))
  log("[consolidations] start")
  try:
    consolidation_response = generate_with_timeout(ai_client, args.model, consolidation_prompt)
    usage = extract_usage(consolidation_response)
    if usage:
      log(f"[consolidations] usage={usage}")
      for key, value in usage.items():
        usage_totals[key] = usage_totals.get(key, 0) + int(value)
    consolidations = safe_json_loads(consolidation_response.text or "")
    if not isinstance(consolidations, list):
      raise ValueError("expected JSON list")
  except Exception as exc:
    log(f"[consolidations] error: {exc}")
    consolidations = []

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
    "prompt_hash": consolidation_prompt_hash,
    "input_hash": consolidation_input_hash,
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

  if usage_totals:
    log(f"[usage totals] {usage_totals}")

  print(json.dumps({
    "version_id": version_id,
    "output_prefix": output_prefix,
    "columns": len(df.columns),
    "generated_at": now_iso(),
    "usage_totals": usage_totals
  }, indent=2))


if __name__ == "__main__":
  main()
