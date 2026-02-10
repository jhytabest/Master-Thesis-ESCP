# Master-Thesis-ESCP

Research-first, traceable "CSV in -> analysis + predictions out" pipeline for France-only deeptech data.

## Repo layout

- `apps/worker-api`: Cloudflare Worker control plane (D1 + R2 metadata)
- `jobs/pipeline`: Cloud Run job for deterministic pipeline
- `jobs/mapping`: Cloud Run job for mapping bundle workflows
- `jobs/enrichment`: Cloud Run job for enrichment workflows
- `shared/schemas`: JSON schemas for manifests and artifacts
- `shared/config`: Feature lists and column mappings
- `docs`: Architecture notes and variable definitions

## Quick start

- See `docs/architecture.md` for the implementation plan and contracts.
- Start with `apps/worker-api` to create D1 tables and Worker endpoints.

## Admin CLI

A unified repository CLI is available at `tools/repo_admin.py`.

Examples:

- `python tools/repo_admin.py status`
- `python tools/repo_admin.py doctor --component analysis`
- `python tools/repo_admin.py analysis run --step all`
- `python tools/repo_admin.py mapping propose --version-id <id> --dry-run`
- `python tools/repo_admin.py pipeline run --version-id <id> --dry-run`

### Prerequisites by workflow

- Analysis: Python env with `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`.
- Mapping propose: Vertex AI access, GCP auth, and active billing. Required env typically includes `GCP_PROJECT_ID`, `R2_*`, and credentials.
- Pipeline and enrichment: `R2_*` env vars and bucket access.
- Worker checks: `node` + `npm` available.
- API commands: `WORKER_API_BASE` plus `WORKER_API_TOKEN` (and optional Cloudflare Access headers).

### Troubleshooting

- Run `python tools/repo_admin.py doctor --component all` before external workflows.
- If mapping propose fails due billing/auth, use `--dry-run` to validate orchestration and then configure Vertex billing/permissions.
- If API commands return auth errors, verify token + Access headers and call `python tools/repo_admin.py api list-runs`.
