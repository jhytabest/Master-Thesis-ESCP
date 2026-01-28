import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import boto3
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
  accuracy_score,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  precision_score,
  r2_score,
  recall_score,
  roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_FEATURE_LIST = [
  "company_status_clean",
  "target_total_funding_eur",
  "last_funding_eur",
  "valuation_eur",
  "target_rounds_num",
  "stage_ordinal",
  "has_funding"
]

COLUMN_ALIASES = {
  "company_status_raw": ["COMPANY STATUS", "COMPANY_STATUS", "company_status"],
  "target_total_funding_raw": ["target_total_funding", "TARGET_TOTAL_FUNDING", "TARGET TOTAL FUNDING"],
  "last_funding_raw": ["LAST FUNDING", "LAST_FUNDING", "last_funding"],
  "valuation_raw": ["VALUATION (EUR)", "VALUATION", "valuation", "valuation_eur"],
  "target_rounds_raw": ["target_rounds", "TARGET_ROUNDS", "TARGET ROUNDS"],
  "growth_stage_raw": ["GROWTH STAGE", "GROWTH_STAGE", "growth_stage"],
  "first_funding_date_raw": ["FIRST FUNDING DATE", "FIRST_FUNDING_DATE", "first_funding_date"],
  "company_siren": ["SIREN", "company_siren", "siren"],
  "source_id": ["id", "ID"]
}

CATEGORICAL_FEATURES = {"company_status_clean"}


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
  req.add_header(
    "User-Agent",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
  )
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
    <meta charset=\"utf-8\">
    <title>EDA summary</title>
  </head>
  <body>
    <h1>EDA summary</h1>
    <p>Rows: {rows}</p>
    <p>Columns: {cols}</p>
    <h2>Missing values</h2>
    <table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">
      <thead><tr><th>Column</th><th>Missing</th></tr></thead>
      <tbody>{missing_rows}</tbody>
    </table>
  </body>
</html>
"""


def normalize_header(name: str) -> str:
  return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def build_column_lookup(columns: List[str]) -> Dict[str, str]:
  lookup: Dict[str, str] = {}
  for col in columns:
    normalized = normalize_header(col)
    if normalized and normalized not in lookup:
      lookup[normalized] = col
  return lookup


def find_column(lookup: Dict[str, str], aliases: List[str]) -> Optional[str]:
  for alias in aliases:
    normalized = normalize_header(alias)
    if normalized in lookup:
      return lookup[normalized]
  return None


def load_feature_list() -> List[str]:
  candidates = []
  env_path = os.getenv("FEATURE_CONFIG_PATH")
  if env_path:
    candidates.append(env_path)
  candidates.append(os.path.join(os.getcwd(), "shared", "config", "feature_list.yml"))
  candidates.append(
    os.path.abspath(
      os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "config", "feature_list.yml")
    )
  )
  for path in candidates:
    if not path or not os.path.exists(path):
      continue
    try:
      features: List[str] = []
      with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
          line = line.strip()
          if not line or line.startswith("#"):
            continue
          if line.startswith("-"):
            features.append(line.lstrip("-").strip())
      if features:
        return features
    except Exception:
      continue
  return DEFAULT_FEATURE_LIST.copy()


def is_missing_value(value: Any) -> bool:
  if value is None:
    return True
  if isinstance(value, float) and pd.isna(value):
    return True
  text = str(value).strip().lower()
  return text in {"", "nan", "#n/a", "n/a", "na", "none", "null"}


def parse_numeric_series(series: pd.Series) -> Tuple[pd.Series, int]:
  failures = 0

  def parse_value(value: Any) -> Optional[float]:
    nonlocal failures
    if is_missing_value(value):
      return None
    if isinstance(value, (int, float)) and not pd.isna(value):
      return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
      return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
      failures += 1
      return None
    if "-" in text and len(numbers) >= 2:
      vals = [float(num) for num in numbers[:2]]
      return sum(vals) / len(vals)
    return float(numbers[0])

  parsed = series.apply(parse_value)
  return parsed.astype("float"), failures


def normalize_status(value: Any) -> Optional[str]:
  if is_missing_value(value):
    return None
  text = str(value).strip().lower()
  text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
  return text or None


def derive_stage_ordinal(value: Any) -> Optional[int]:
  if is_missing_value(value):
    return None
  text = str(value).strip().lower()
  if not text:
    return None
  if "late" in text:
    return 4
  if "scale" in text:
    return 4
  if "growth" in text:
    return 3
  if "series d" in text or "series e" in text:
    return 4
  if "series c" in text:
    return 3
  if "series b" in text:
    return 3
  if "series a" in text:
    return 2
  if "early" in text:
    return 2
  if "pre" in text and "seed" in text:
    return 1
  if "seed" in text:
    return 1
  if "mature" in text:
    return 5
  if "ipo" in text or "acquired" in text or "exit" in text:
    return 5
  return None


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
  lookup = build_column_lookup(list(df.columns))

  def pick(alias_key: str) -> pd.Series:
    col_name = find_column(lookup, COLUMN_ALIASES[alias_key])
    if not col_name:
      return pd.Series([None] * len(df))
    return df[col_name]

  numeric_failures = 0
  features = pd.DataFrame({"row_id": df.index.astype(int)})
  meta_cols = ["row_id"]

  source_id_col = find_column(lookup, COLUMN_ALIASES["source_id"])
  if source_id_col:
    features["source_id"] = df[source_id_col]
    meta_cols.append("source_id")

  siren_col = find_column(lookup, COLUMN_ALIASES["company_siren"])
  if siren_col:
    features["company_siren"] = df[siren_col]
    meta_cols.append("company_siren")

  status_raw = pick("company_status_raw")
  features["company_status_clean"] = status_raw.apply(normalize_status)

  target_total_raw = pick("target_total_funding_raw")
  features["target_total_funding_eur"], failures = parse_numeric_series(target_total_raw)
  numeric_failures += failures

  last_funding_raw = pick("last_funding_raw")
  features["last_funding_eur"], failures = parse_numeric_series(last_funding_raw)
  numeric_failures += failures

  valuation_raw = pick("valuation_raw")
  features["valuation_eur"], failures = parse_numeric_series(valuation_raw)
  numeric_failures += failures

  target_rounds_raw = pick("target_rounds_raw")
  features["target_rounds_num"], failures = parse_numeric_series(target_rounds_raw)
  numeric_failures += failures

  growth_stage_raw = pick("growth_stage_raw")
  features["stage_ordinal"] = growth_stage_raw.apply(derive_stage_ordinal)

  funding_date_raw = pick("first_funding_date_raw")
  has_funding = (
    (features["target_total_funding_eur"].fillna(0) > 0)
    | (features["last_funding_eur"].fillna(0) > 0)
    | (features["target_rounds_num"].fillna(0) > 0)
    | (~funding_date_raw.apply(is_missing_value))
  )
  features["has_funding"] = has_funding.astype("boolean")

  parsing_stats = {
    "numeric_parse_failures": int(numeric_failures),
    "date_parse_failures": 0
  }
  return features, parsing_stats, meta_cols


def filter_empty_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
  usable: List[str] = []
  dropped: List[str] = []
  for col in feature_cols:
    if col not in df.columns:
      dropped.append(col)
      continue
    if df[col].dropna().empty:
      dropped.append(col)
    else:
      usable.append(col)
  return usable, dropped


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
  transformers = []
  if numeric_features:
    transformers.append((
      "num",
      Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
      ]),
      numeric_features
    ))
  if categorical_features:
    transformers.append((
      "cat",
      Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
      ]),
      categorical_features
    ))
  return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def split_train_test(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  if len(y) < 5:
    return X, X, y, y
  stratify = None
  if y.nunique() > 1:
    counts = y.value_counts()
    if counts.min() >= 2:
      stratify = y
  return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)


def serialize_model(model: Pipeline) -> bytes:
  with tempfile.NamedTemporaryFile(suffix=".joblib") as handle:
    joblib.dump(model, handle.name)
    handle.seek(0)
    return handle.read()


def train_model(
  *,
  features_df: pd.DataFrame,
  model_id: str,
  target: str,
  feature_cols: List[str],
  task: str,
  dataset_sha: str,
  version_id: str,
  meta_cols: List[str]
) -> Dict[str, Any]:
  if target not in features_df.columns:
    return {
      "model_id": model_id,
      "status": "SKIPPED",
      "reason": f"missing target column: {target}"
    }

  usable_features, dropped_features = filter_empty_features(features_df, feature_cols)
  if not usable_features:
    return {
      "model_id": model_id,
      "status": "SKIPPED",
      "reason": "no usable feature columns",
      "dropped_features": dropped_features
    }

  y = features_df[target]
  mask = y.notna()
  if mask.sum() < 5:
    return {
      "model_id": model_id,
      "status": "SKIPPED",
      "reason": "not enough labeled rows",
      "labeled_rows": int(mask.sum())
    }

  y = y.loc[mask]
  if y.nunique() < 2:
    return {
      "model_id": model_id,
      "status": "SKIPPED",
      "reason": "target has <2 classes",
      "class_distribution": y.value_counts(dropna=False).to_dict()
    }

  X = features_df.loc[mask, usable_features]
  numeric_features = [col for col in usable_features if col not in CATEGORICAL_FEATURES]
  categorical_features = [col for col in usable_features if col in CATEGORICAL_FEATURES]

  preprocessor = build_preprocessor(numeric_features, categorical_features)

  if task == "classification":
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
  elif task == "ordinal":
    estimator = RandomForestClassifier(n_estimators=200, random_state=42)
  else:
    raise ValueError(f"Unsupported task type: {task}")

  model = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
  X_train, X_test, y_train, y_test = split_train_test(X, y)

  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  metrics: Dict[str, Any] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
    "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
  }

  if y_test.nunique() > 1:
    try:
      if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        if probas.shape[1] >= 2:
          positive_index = 1
          if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if True in classes:
              positive_index = classes.index(True)
            elif 1 in classes:
              positive_index = classes.index(1)
          metrics["roc_auc"] = float(roc_auc_score(y_test, probas[:, positive_index]))
    except Exception:
      pass

  if task == "ordinal":
    try:
      metrics["mae"] = float(mean_absolute_error(y_test.astype(float), y_pred.astype(float)))
      metrics["rmse"] = float(mean_squared_error(y_test.astype(float), y_pred.astype(float), squared=False))
      metrics["r2"] = float(r2_score(y_test.astype(float), y_pred.astype(float)))
    except Exception:
      pass

  model_bytes = serialize_model(model)
  X_all = features_df[usable_features]
  predictions = model.predict(X_all)

  predictions_df = features_df[meta_cols].copy()
  if target == "has_funding":
    predictions_df["pred_has_funding"] = predictions
    if hasattr(model, "predict_proba"):
      probas = model.predict_proba(X_all)
      if probas.shape[1] >= 2:
        positive_index = 1
        if hasattr(model, "classes_"):
          classes = list(model.classes_)
          if True in classes:
            positive_index = classes.index(True)
          elif 1 in classes:
            positive_index = classes.index(1)
        predictions_df["pred_has_funding_proba"] = probas[:, positive_index]
  else:
    predictions_df["pred_stage_ordinal"] = predictions

  payload = {
    "model_id": model_id,
    "status": "TRAINED",
    "task": task,
    "target": target,
    "version_id": version_id,
    "dataset_sha256": dataset_sha,
    "trained_at": now_iso(),
    "code_git_sha": os.getenv("CODE_GIT_SHA", "unknown"),
    "container_image_digest": os.getenv("IMAGE_DIGEST", "unknown"),
    "features_used": usable_features,
    "dropped_features": dropped_features,
    "metrics": metrics,
    "class_distribution": y.value_counts(dropna=False).to_dict(),
    "row_count": int(len(features_df)),
    "labeled_rows": int(mask.sum()),
    "train_rows": int(len(y_train)),
    "test_rows": int(len(y_test))
  }

  return {
    "model_id": model_id,
    "status": "TRAINED",
    "metrics": payload,
    "model_bytes": model_bytes,
    "predictions": predictions_df
  }


def train_baseline_models(
  *,
  features_df: pd.DataFrame,
  version_id: str,
  dataset_sha: str,
  meta_cols: List[str]
) -> List[Dict[str, Any]]:
  feature_list = load_feature_list()
  model_specs = [
    {
      "model_id": "baseline_has_funding_v1",
      "target": "has_funding",
      "task": "classification"
    },
    {
      "model_id": "baseline_stage_ordinal_v1",
      "target": "stage_ordinal",
      "task": "ordinal"
    }
  ]
  outputs: List[Dict[str, Any]] = []
  for spec in model_specs:
    target = spec["target"]
    feature_cols = [col for col in feature_list if col != target]
    output = train_model(
      features_df=features_df,
      model_id=spec["model_id"],
      target=target,
      feature_cols=feature_cols,
      task=spec["task"],
      dataset_sha=dataset_sha,
      version_id=version_id,
      meta_cols=meta_cols
    )
    if output.get("status") == "SKIPPED":
      output["metrics"] = {
        "model_id": spec["model_id"],
        "status": "SKIPPED",
        "target": target,
        "version_id": version_id,
        "dataset_sha256": dataset_sha,
        "reason": output.get("reason"),
        "dropped_features": output.get("dropped_features"),
        "class_distribution": output.get("class_distribution"),
        "labeled_rows": output.get("labeled_rows")
      }
    outputs.append(output)
  return outputs


def main() -> None:
  parser = argparse.ArgumentParser(description="Deterministic pipeline job")
  parser.add_argument("--version_id")
  parser.add_argument("--mapping_bundle_id")
  parser.add_argument("--run_id")
  args = parser.parse_args()

  version_id = args.version_id or os.getenv("PIPELINE_VERSION_ID")
  mapping_bundle_id = args.mapping_bundle_id or os.getenv("PIPELINE_MAPPING_BUNDLE_ID")
  run_id = args.run_id or os.getenv("PIPELINE_RUN_ID")
  if not version_id:
    raise RuntimeError("Missing required version_id")
  started_at = now_iso()
  if run_id:
    update_run(run_id, {"status": "STARTED", "started_at": started_at})

  bucket = env_or_error("R2_BUCKET")
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

    features_df, parsing_stats, meta_cols = build_feature_frame(df)
    features_buffer = io.BytesIO()
    features_df.to_parquet(features_buffer, index=False)
    features_key = f"derived/{version_id}/features.parquet"
    upload_bytes(client, bucket, features_key, features_buffer.getvalue(), "application/octet-stream")

    eda_html = build_eda_html(df, quality["missing_by_column"])
    eda_key = f"reports/{version_id}/eda.html"
    upload_bytes(client, bucket, eda_key, eda_html.encode("utf-8"), "text/html")

    quality_key = f"reports/{version_id}/data_quality.json"
    upload_bytes(client, bucket, quality_key, json.dumps(quality, indent=2).encode("utf-8"), "application/json")

    model_outputs = train_baseline_models(
      features_df=features_df,
      version_id=version_id,
      dataset_sha=dataset_sha,
      meta_cols=meta_cols
    )

    models_index: List[Dict[str, Any]] = []
    for output in model_outputs:
      model_id = output["model_id"]
      metrics = output.get("metrics", {})
      metrics_key = f"models/{version_id}/{model_id}/metrics.json"
      upload_bytes(client, bucket, metrics_key, json.dumps(metrics, indent=2).encode("utf-8"), "application/json")

      model_key = None
      predictions_key = None
      if output.get("status") == "TRAINED":
        model_key = f"models/{version_id}/{model_id}/model.bin"
        upload_bytes(client, bucket, model_key, output["model_bytes"], "application/octet-stream")

        predictions_df = output.get("predictions")
        if predictions_df is not None:
          pred_buffer = io.BytesIO()
          predictions_df.to_parquet(pred_buffer, index=False)
          predictions_key = f"predictions/{version_id}/{model_id}/scores.parquet"
          upload_bytes(client, bucket, predictions_key, pred_buffer.getvalue(), "application/octet-stream")

      models_index.append({
        "model_id": model_id,
        "status": output.get("status"),
        "model_path": model_key,
        "metrics_path": metrics_key,
        "predictions_path": predictions_key
      })

    manifest = {
      "version_id": version_id,
      "dataset_sha256": dataset_sha,
      "run_id": run_id,
      "code_git_sha": os.getenv("CODE_GIT_SHA", "unknown"),
      "container_image_digest": os.getenv("IMAGE_DIGEST", "unknown"),
      "mapping_bundle_id": mapping_bundle_id,
      "mapping_manifest_hash": None,
      "parsing_stats": parsing_stats,
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
        "features_parquet": features_key,
        "eda_html": eda_key,
        "data_quality_json": quality_key,
        "cleaning_manifest": manifest_key
      },
      "models": models_index
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
