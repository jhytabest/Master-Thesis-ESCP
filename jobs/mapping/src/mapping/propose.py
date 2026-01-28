import argparse
import io
import json
import os
from typing import Any, Dict, List

import pandas as pd
import vertexai
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

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


def build_model(model_name: str) -> GenerativeModel:
  project = env_or_error("GCP_PROJECT_ID")
  location = os.getenv("GCP_LOCATION", "us-central1")
  vertexai.init(project=project, location=location)
  return GenerativeModel(model_name)


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


def main() -> None:
  parser = argparse.ArgumentParser(description="Propose mapping bundle")
  parser.add_argument("--version_id", required=True)
  parser.add_argument("--output_prefix")
  parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-1.5-pro-002"))
  args = parser.parse_args()

  version_id = args.version_id
  output_prefix = args.output_prefix or f"mappings/proposals/{version_id}"
  output_prefix = output_prefix.rstrip("/")

  bucket = env_or_error("R2_BUCKET")
  client = build_s3_client()

  source_key = f"raw/{version_id}/dataset.csv"
  raw_bytes = download_bytes(client, bucket, source_key)
  df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False)

  model = build_model(args.model)

  column_summaries: List[Dict[str, Any]] = []
  for column in df.columns:
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

    prompt = build_prompt(context)
    prompt_hash = sha256_text(prompt)
    input_hash = sha256_text(json.dumps(distinct_values, ensure_ascii=False, separators=(",", ":")))

    response = model.generate_content(
      prompt,
      generation_config=GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json"
      )
    )
    output = safe_json_loads(response.text or "")
    if not isinstance(output, dict):
      raise RuntimeError(f"Invalid AI response for column {column}: expected JSON object")

    action = output.get("action")
    if action not in ALLOWED_ACTIONS:
      raise RuntimeError(f"Invalid action for column {column}: {action}")

    normalized_column = output.get("normalized_column") or default_normalized_column(column, action)
    value_map = output.get("value_map", [])
    if not isinstance(value_map, list):
      raise RuntimeError(f"Invalid value_map for column {column}")
    mapping = build_value_map(value_map)
    list_delimiters = output.get("list_delimiters") or list_delimiters
    if not isinstance(list_delimiters, list):
      list_delimiters = []

    rows: List[Dict[str, Any]] = []
    if action == "map_values":
      for raw_value in distinct_values:
        if raw_value not in mapping:
          raise RuntimeError(f"Missing mapping for raw value '{raw_value}' in column {column}")
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
    upload_bytes(client, bucket, csv_key, csv_bytes, "text/csv")

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
    upload_bytes(client, bucket, manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    column_summaries.append({
      "column": column,
      "detected_type": detected_type,
      "unique_count": len(distinct_values),
      "examples": distinct_values[:5]
    })

  consolidation_prompt = build_consolidation_prompt(column_summaries)
  consolidation_prompt_hash = sha256_text(consolidation_prompt)
  consolidation_input_hash = sha256_text(json.dumps(column_summaries, ensure_ascii=False, separators=(",", ":")))
  consolidation_resp = model.generate_content(
    consolidation_prompt,
    generation_config=GenerationConfig(
      temperature=0.2,
      response_mime_type="application/json"
    )
  )
  consolidations = safe_json_loads(consolidation_resp.text or "")
  if not isinstance(consolidations, list):
    raise RuntimeError("Invalid consolidations response: expected JSON list")

  consolidations_key = f"{output_prefix}/consolidations.json"
  upload_bytes(
    client,
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
    client,
    bucket,
    consolidations_manifest_key,
    json.dumps(consolidations_manifest, indent=2).encode("utf-8"),
    "application/json"
  )

  print(json.dumps({
    "version_id": version_id,
    "output_prefix": output_prefix,
    "columns": len(df.columns),
    "generated_at": now_iso()
  }, indent=2))


if __name__ == "__main__":
  main()
