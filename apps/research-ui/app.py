import io
import json
import os
import secrets
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
import requests
import streamlit as st
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token as google_id_token


st.set_page_config(page_title="Deeptech Research Console", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
}
.main {
  background: radial-gradient(circle at 10% 10%, #f5f0e6 0%, #f8f6f1 35%, #eef2f4 100%);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101820 0%, #1b2a2f 100%);
  color: #f7f4ef;
}
section[data-testid="stSidebar"] * {
  color: #f7f4ef;
}
h1, h2, h3 {
  font-family: 'Source Serif 4', serif;
}
.card {
  background: #ffffff;
  border: 1px solid #e0ddd7;
  border-radius: 12px;
  padding: 1rem;
  box-shadow: 0 10px 25px rgba(16, 24, 32, 0.06);
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


def env(name: str, default: Optional[str] = None) -> Optional[str]:
  value = os.getenv(name, default)
  if value is None:
    return None
  return value.strip() if isinstance(value, str) else value


WORKER_API_BASE = env("WORKER_API_BASE")
WORKER_API_TOKEN = env("WORKER_API_TOKEN")
CF_ACCESS_CLIENT_ID = env("CF_ACCESS_CLIENT_ID")
CF_ACCESS_CLIENT_SECRET = env("CF_ACCESS_CLIENT_SECRET")

R2_ENDPOINT = env("R2_ENDPOINT")
R2_BUCKET = env("R2_BUCKET")
R2_ACCESS_KEY_ID = env("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = env("R2_SECRET_ACCESS_KEY")

GCP_PROJECT_ID = env("GCP_PROJECT_ID")
GCP_LOCATION = env("GCP_LOCATION")
PIPELINE_JOB_NAME = env("PIPELINE_JOB_NAME")
PIPELINE_CONTAINER_NAME = env("PIPELINE_CONTAINER_NAME")

OAUTH_CLIENT_ID = env("GOOGLE_OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = env("GOOGLE_OAUTH_CLIENT_SECRET")
OAUTH_REDIRECT_URI = env("GOOGLE_OAUTH_REDIRECT_URI")
ALLOWED_EMAIL_DOMAIN = env("ALLOWED_EMAIL_DOMAIN", "edu.escp.eu")


def build_oauth_login_url(state: str) -> str:
  if not (OAUTH_CLIENT_ID and OAUTH_REDIRECT_URI):
    return ""
  params = {
    "client_id": OAUTH_CLIENT_ID,
    "redirect_uri": OAUTH_REDIRECT_URI,
    "response_type": "code",
    "scope": "openid email profile",
    "access_type": "offline",
    "prompt": "select_account",
    "state": state,
    "hd": ALLOWED_EMAIL_DOMAIN
  }
  return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(code: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  if not (OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET and OAUTH_REDIRECT_URI):
    raise RuntimeError("Missing Google OAuth configuration")
  payload = {
    "code": code,
    "client_id": OAUTH_CLIENT_ID,
    "client_secret": OAUTH_CLIENT_SECRET,
    "redirect_uri": OAUTH_REDIRECT_URI,
    "grant_type": "authorization_code"
  }
  response = requests.post("https://oauth2.googleapis.com/token", data=payload, timeout=30)
  if response.status_code != 200:
    raise RuntimeError(f"Token exchange failed: {response.text}")
  token_payload = response.json()
  id_token_value = token_payload.get("id_token")
  if not id_token_value:
    raise RuntimeError("No id_token returned from Google")
  claims = google_id_token.verify_oauth2_token(id_token_value, GoogleAuthRequest(), OAUTH_CLIENT_ID)
  return token_payload, claims


def require_google_login() -> None:
  if st.session_state.get("user_email"):
    return

  st.title("Deeptech Research Console")
  st.markdown("<div class='card'>", unsafe_allow_html=True)
  st.subheader("Sign in with Google")
  st.write("Access is restricted to approved educational accounts.")

  if not (OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET and OAUTH_REDIRECT_URI):
    st.error("OAuth is not configured. Set GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, and GOOGLE_OAUTH_REDIRECT_URI.")
    st.stop()

  query_params = st.query_params
  code = query_params.get("code")
  state = query_params.get("state")
  expected_state = st.session_state.get("oauth_state")

  if not expected_state:
    expected_state = secrets.token_urlsafe(16)
    st.session_state["oauth_state"] = expected_state

  if code:
    try:
      if state and expected_state and state != expected_state:
        raise RuntimeError("OAuth state mismatch. Please try again.")
      _, claims = exchange_code_for_tokens(code)
      email = claims.get("email")
      email_verified = claims.get("email_verified")
      if not email or not email_verified:
        raise RuntimeError("Google account email is not verified.")
      if not str(email).lower().endswith(f"@{ALLOWED_EMAIL_DOMAIN}"):
        raise RuntimeError(f"Access restricted to @{ALLOWED_EMAIL_DOMAIN} accounts.")
      st.session_state["user_email"] = email
      st.session_state["user_name"] = claims.get("name") or email
      st.query_params.clear()
      st.success(f"Signed in as {email}")
      st.stop()
    except Exception as exc:
      st.error(str(exc))

  auth_url = build_oauth_login_url(expected_state)
  st.link_button("Sign in with Google", auth_url, type="primary")
  st.markdown("</div>", unsafe_allow_html=True)
  st.stop()


@st.cache_resource
def build_s3_client() -> boto3.client:
  if not (R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
    raise RuntimeError("Missing R2 configuration")
  return boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto"
  )


def worker_headers() -> Dict[str, str]:
  headers = {"Content-Type": "application/json"}
  if WORKER_API_TOKEN:
    headers["Authorization"] = f"Bearer {WORKER_API_TOKEN}"
  if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
    headers["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
    headers["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET
  return headers


@st.cache_data(ttl=60)
def list_runs() -> List[Dict[str, Any]]:
  if not WORKER_API_BASE:
    return []
  resp = requests.get(f"{WORKER_API_BASE}/runs", headers=worker_headers(), timeout=30)
  if resp.status_code != 200:
    return []
  payload = resp.json()
  return payload.get("runs", [])


@st.cache_data(ttl=60)
def list_versions() -> List[Dict[str, Any]]:
  if not WORKER_API_BASE:
    return []
  resp = requests.get(f"{WORKER_API_BASE}/versions", headers=worker_headers(), timeout=30)
  if resp.status_code != 200:
    return []
  payload = resp.json()
  return payload.get("versions", [])


@st.cache_data(ttl=60)
def get_run(run_id: str) -> Optional[Dict[str, Any]]:
  if not WORKER_API_BASE or not run_id:
    return None
  resp = requests.get(f"{WORKER_API_BASE}/runs/{run_id}", headers=worker_headers(), timeout=30)
  if resp.status_code != 200:
    return None
  payload = resp.json()
  return payload.get("run")


def trigger_run(version_id: str, mapping_bundle_id: Optional[str]) -> str:
  if not WORKER_API_BASE:
    raise RuntimeError("WORKER_API_BASE not configured")
  payload: Dict[str, Any] = {"version_id": version_id}
  if mapping_bundle_id:
    payload["mapping_bundle_id"] = mapping_bundle_id
  resp = requests.post(
    f"{WORKER_API_BASE}/runs/trigger",
    headers=worker_headers(),
    json=payload,
    timeout=30
  )
  if resp.status_code != 200:
    raise RuntimeError(f"Failed to create run: {resp.text}")
  run_id = resp.json().get("run_id")
  if not run_id:
    raise RuntimeError("Missing run_id in response")
  return run_id


def gcp_access_token() -> str:
  credentials, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
  if not credentials.valid:
    credentials.refresh(GoogleAuthRequest())
  return credentials.token


def execute_pipeline_job(version_id: str, run_id: str, mapping_bundle_id: Optional[str]) -> Dict[str, Any]:
  if not (GCP_PROJECT_ID and GCP_LOCATION and PIPELINE_JOB_NAME):
    raise RuntimeError("Missing Cloud Run job configuration")
  url = (
    "https://run.googleapis.com/v2/projects/"
    f"{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/jobs/{PIPELINE_JOB_NAME}:run"
  )
  overrides: Dict[str, Any] = {"containerOverrides": []}
  container_override: Dict[str, Any] = {
    "args": ["--version_id", version_id, "--run_id", run_id],
    "env": [
      {"name": "PIPELINE_VERSION_ID", "value": version_id},
      {"name": "PIPELINE_RUN_ID", "value": run_id}
    ]
  }
  if PIPELINE_CONTAINER_NAME:
    container_override["name"] = PIPELINE_CONTAINER_NAME
  if mapping_bundle_id:
    container_override["args"].extend(["--mapping_bundle_id", mapping_bundle_id])
    container_override["env"].append({"name": "PIPELINE_MAPPING_BUNDLE_ID", "value": mapping_bundle_id})
  overrides["containerOverrides"].append(container_override)

  payload = {"overrides": overrides}
  token = gcp_access_token()
  resp = requests.post(
    url,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=payload,
    timeout=30
  )
  if resp.status_code >= 300:
    raise RuntimeError(f"Failed to execute job: {resp.text}")
  return resp.json()


@st.cache_data(ttl=120)
def r2_get_json(key: str) -> Optional[Dict[str, Any]]:
  if not key:
    return None
  if not R2_BUCKET:
    raise RuntimeError("R2_BUCKET not configured")
  client = build_s3_client()
  obj = client.get_object(Bucket=R2_BUCKET, Key=key)
  raw = obj["Body"].read().decode("utf-8")
  return json.loads(raw)


@st.cache_data(ttl=120)
def r2_get_parquet(key: str) -> pd.DataFrame:
  if not R2_BUCKET:
    raise RuntimeError("R2_BUCKET not configured")
  client = build_s3_client()
  obj = client.get_object(Bucket=R2_BUCKET, Key=key)
  raw = obj["Body"].read()
  return pd.read_parquet(io.BytesIO(raw))


def r2_put_text(key: str, text: str, content_type: str) -> None:
  if not R2_BUCKET:
    raise RuntimeError("R2_BUCKET not configured")
  client = build_s3_client()
  client.put_object(Bucket=R2_BUCKET, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


def format_pct(value: float) -> str:
  return f"{value * 100:.1f}%"


def summarize_numeric(df: pd.DataFrame, outcome_col: str, columns: List[str]) -> pd.DataFrame:
  if outcome_col not in df.columns:
    return pd.DataFrame()
  rows = []
  for col in columns:
    if col not in df.columns:
      continue
    subset = df[[col, outcome_col]].dropna()
    if subset.empty:
      continue
    funded = subset[subset[outcome_col] == True][col]
    unfunded = subset[subset[outcome_col] == False][col]
    rows.append({
      "feature": col,
      "funded_mean": float(funded.mean()) if not funded.empty else None,
      "unfunded_mean": float(unfunded.mean()) if not unfunded.empty else None,
      "mean_diff": float(funded.mean() - unfunded.mean()) if not funded.empty and not unfunded.empty else None,
      "missing_rate": float(df[col].isna().mean())
    })
  return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def summarize_categorical(df: pd.DataFrame, outcome_col: str, col: str) -> pd.DataFrame:
  if col not in df.columns or outcome_col not in df.columns:
    return pd.DataFrame()
  counts = df.groupby(col)[outcome_col].agg(["mean", "count"]).reset_index()
  counts = counts.rename(columns={"mean": "funding_rate", "count": "rows"})
  return counts.sort_values("rows", ascending=False)


def build_report(version_id: str, metrics: List[Dict[str, Any]], numeric_summary: pd.DataFrame, cat_summary: pd.DataFrame) -> str:
  lines = [
    f"# Funding predictors report",
    f"",
    f"Version: {version_id}",
    f"Generated at: {datetime.utcnow().isoformat()}Z",
    "",
    "## Baseline model metrics",
  ]
  if metrics:
    for entry in metrics:
      model_id = entry.get("model_id")
      status = entry.get("status")
      lines.append(f"- {model_id} ({status})")
      model_metrics = entry.get("metrics", {})
      if isinstance(model_metrics, dict):
        acc = model_metrics.get("accuracy")
        f1 = model_metrics.get("f1_weighted")
        roc = model_metrics.get("roc_auc")
        if acc is not None:
          lines.append(f"  - accuracy: {acc:.3f}")
        if f1 is not None:
          lines.append(f"  - f1_weighted: {f1:.3f}")
        if roc is not None:
          lines.append(f"  - roc_auc: {roc:.3f}")
  else:
    lines.append("- No metrics available")

  lines.append("")
  lines.append("## Numeric feature gaps")
  if numeric_summary.empty:
    lines.append("- No numeric summary available")
  else:
    for _, row in numeric_summary.head(10).iterrows():
      lines.append(
        f"- {row['feature']}: mean_diff={row['mean_diff']:.2f} missing={row['missing_rate']:.1%}"
      )

  lines.append("")
  lines.append("## Categorical signals (company status)")
  if cat_summary.empty:
    lines.append("- No categorical summary available")
  else:
    category_col = cat_summary.columns[0]
    for _, row in cat_summary.head(10).iterrows():
      lines.append(
        f"- {row[category_col]}: funding_rate={row['funding_rate']:.1%} rows={int(row['rows'])}"
      )

  return "\n".join(lines)

require_google_login()

st.title("Deeptech Research Console")

with st.sidebar:
  st.subheader("Connections")
  st.write("Worker API:", WORKER_API_BASE or "not set")
  st.write("R2 bucket:", R2_BUCKET or "not set")
  st.write("Cloud Run job:", PIPELINE_JOB_NAME or "not set")
  st.caption("Set configuration via environment variables on Cloud Run.")

  user_email = st.session_state.get("user_email")
  if user_email:
    st.subheader("Signed in")
    st.write(user_email)
    if st.button("Sign out"):
      st.session_state.clear()
      st.query_params.clear()
      st.rerun()

  st.subheader("Analysis settings")
  outcome_col = st.selectbox("Outcome", ["has_funding"], index=0)


tab_run, tab_analysis, tab_report = st.tabs(["Run pipeline", "Analysis", "Report"])

with tab_run:
  st.markdown("<div class='card'>", unsafe_allow_html=True)
  versions = list_versions()
  version_ids = [row.get("version_id") for row in versions if row.get("version_id")]
  version_id = st.selectbox("Select version", options=version_ids) if version_ids else ""
  version_id = st.text_input("Version ID", value=version_id)
  mapping_bundle_id = st.text_input("Mapping bundle (optional)")
  if st.button("Trigger pipeline"):
    if not version_id:
      st.error("Please provide a version_id")
    else:
      try:
        with st.spinner("Creating run and executing job..."):
          run_id = trigger_run(version_id, mapping_bundle_id or None)
          response = execute_pipeline_job(version_id, run_id, mapping_bundle_id or None)
        st.success(f"Run started: {run_id}")
        st.json(response)
      except Exception as exc:
        st.error(str(exc))
  st.markdown("</div>", unsafe_allow_html=True)

with tab_analysis:
  st.markdown("<div class='card'>", unsafe_allow_html=True)
  runs = list_runs()
  run_ids = [row.get("run_id") for row in runs if row.get("run_id")]
  selected_run_id = st.selectbox("Select run", options=run_ids) if run_ids else ""
  run_details = get_run(selected_run_id) if selected_run_id else None
  if run_details:
    st.write("Run status:", run_details.get("status"))
    st.write("Artifact index:", run_details.get("artifact_index_path"))

  index_path = st.text_input("Artifact index path", value=(run_details.get("artifact_index_path") if run_details else ""))
  if index_path:
    try:
      index_payload = r2_get_json(index_path)
      if not index_payload:
        st.warning("No index payload found")
      else:
        version_id = index_payload.get("version_id", "")
        artifacts = index_payload.get("artifacts", {})
        models_index = index_payload.get("models", [])
        st.write("Version:", version_id)

        quality_key = artifacts.get("data_quality_json")
        if quality_key:
          quality = r2_get_json(quality_key) or {}
          st.subheader("Data quality")
          st.json(quality)

        features_key = artifacts.get("features_parquet")
        if features_key:
          features_df = r2_get_parquet(features_key)
          if "company_status_clean" in features_df.columns:
            status_options = sorted(features_df["company_status_clean"].dropna().unique().tolist())
            status_filter = st.multiselect("Company status filter", status_options)
            if status_filter:
              features_df = features_df[features_df["company_status_clean"].isin(status_filter)]

          if outcome_col in features_df.columns:
            total = len(features_df)
            funded_rate = float(features_df[outcome_col].mean()) if total > 0 else 0
            st.subheader("Funding prevalence")
            st.metric("Funding rate", format_pct(funded_rate))
            st.caption(f"Rows: {total}")

            numeric_cols = [
              col for col in features_df.select_dtypes(include=["number"]).columns
              if col not in {"row_id"}
            ]
            numeric_summary = summarize_numeric(features_df, outcome_col, numeric_cols)
            st.subheader("Numeric feature gaps")
            st.dataframe(numeric_summary, use_container_width=True)

            cat_summary = summarize_categorical(features_df, outcome_col, "company_status_clean")
            st.subheader("Company status vs funding")
            st.dataframe(cat_summary, use_container_width=True)

        if models_index:
          st.subheader("Model metrics")
          metrics_rows = []
          for entry in models_index:
            metrics_key = entry.get("metrics_path")
            if not metrics_key:
              continue
            metrics_payload = r2_get_json(metrics_key)
            if metrics_payload:
              metrics_rows.append(metrics_payload)
          if metrics_rows:
            st.dataframe(pd.json_normalize(metrics_rows), use_container_width=True)
    except Exception as exc:
      st.error(str(exc))
  st.markdown("</div>", unsafe_allow_html=True)

with tab_report:
  st.markdown("<div class='card'>", unsafe_allow_html=True)
  st.write("Generate a lightweight markdown report from the selected run.")
  report_index_path = st.text_input("Index path for report", value="")
  if st.button("Generate report") and report_index_path:
    try:
      index_payload = r2_get_json(report_index_path)
      if not index_payload:
        st.error("Index not found")
      else:
        version_id = index_payload.get("version_id", "")
        artifacts = index_payload.get("artifacts", {})
        models_index = index_payload.get("models", [])

        features_key = artifacts.get("features_parquet")
        features_df = r2_get_parquet(features_key) if features_key else pd.DataFrame()
        numeric_cols = [
          col for col in features_df.select_dtypes(include=["number"]).columns
          if col not in {"row_id"}
        ]
        numeric_summary = summarize_numeric(features_df, "has_funding", numeric_cols)
        cat_summary = summarize_categorical(features_df, "has_funding", "company_status_clean")

        metrics_rows = []
        for entry in models_index:
          metrics_key = entry.get("metrics_path")
          if metrics_key:
            payload = r2_get_json(metrics_key)
            if payload:
              metrics_rows.append(payload)

        report_text = build_report(version_id, metrics_rows, numeric_summary, cat_summary)
        st.download_button("Download report", report_text, file_name="funding_report.md")

        if st.button("Save report to R2"):
          report_key = f"reports/{version_id}/funding_report.md"
          r2_put_text(report_key, report_text, "text/markdown")
          st.success(f"Saved to {report_key}")
    except Exception as exc:
      st.error(str(exc))
  st.markdown("</div>", unsafe_allow_html=True)
