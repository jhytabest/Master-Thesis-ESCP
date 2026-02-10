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
Reusable skill wrappers are available under `skills/` (including `skills/analysis-openalex` for OpenAlex literature mapping).

Examples:

- `python tools/repo_admin.py status`
- `python tools/repo_admin.py doctor --component analysis`
- `python tools/repo_admin.py analysis run --step all`
- `python tools/repo_admin.py local init`
- `python tools/repo_admin.py local register-version --version-id local-main --dataset-path dataset.csv`
- `python tools/repo_admin.py mapping propose --version-id local-main`
- `python tools/repo_admin.py mapping freeze --proposal-prefix mappings/proposals/local-main --output-bundle-id local-main --auto-approve-all`
- `python tools/repo_admin.py pipeline run --version-id local-main`
- `python tools/repo_admin.py analysis literature --dry-run`

### Prerequisites by workflow

- Analysis: Python env with `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`.
- Mapping propose: local heuristic mode is default when running locally; no Vertex billing required.
- Pipeline and enrichment: run against repo-local storage in `local_store/` by default.
- Analysis literature mapping: queries [OpenAlex](https://openalex.org) and writes reports to `local_store/reports/literature/`.
- Worker checks: `node` + `npm` available.
- API commands: `WORKER_API_BASE` plus `WORKER_API_TOKEN` (and optional Cloudflare Access headers).

### Troubleshooting

- Run `python tools/repo_admin.py doctor --component all` before workflows.
- Initialize local storage with `python tools/repo_admin.py local init`.
- Register a local version before mapping/pipeline runs with `python tools/repo_admin.py local register-version ...`.
- Local workflow artifacts are written under `local_store/` and can be committed to GitHub.
- If API commands return auth errors, verify token + Access headers and call `python tools/repo_admin.py api list-runs`.
