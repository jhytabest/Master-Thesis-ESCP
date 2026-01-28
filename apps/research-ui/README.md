# Research UI (Streamlit)

Hosted UI to trigger pipeline runs, explore artifacts, and generate reports.

## Required environment variables

Worker API:
- `WORKER_API_BASE`
- `WORKER_API_TOKEN`
- `CF_ACCESS_CLIENT_ID` (if Access is enabled)
- `CF_ACCESS_CLIENT_SECRET` (if Access is enabled)

R2:
- `R2_ENDPOINT`
- `R2_BUCKET`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

Cloud Run Jobs:
- `GCP_PROJECT_ID`
- `GCP_LOCATION`
- `PIPELINE_JOB_NAME`
- `PIPELINE_CONTAINER_NAME` (optional, set if the job has a named container)

Google OAuth (app-level):
- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REDIRECT_URI` (e.g., https://research.deeptech-master-thesis.org)
- `ALLOWED_EMAIL_DOMAIN` (default: edu.escp.eu)

## Local run

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Cloud Run deploy

Build and deploy the image, then set environment variables on the Cloud Run service.
