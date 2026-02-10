---
name: master-thesis-admin
description: Unified administration for the Master-Thesis-ESCP repository. Use when asked to run, orchestrate, troubleshoot, or audit repository workflows including analysis scripts, mapping proposal/freezing, pipeline runs, enrichment runs, Worker API calls, Streamlit UI operations, cloud build submissions, and environment diagnostics.
---

# Master Thesis Admin Skill

Use the repository CLI as the control plane:

```bash
python <repo_root>/tools/repo_admin.py --help
```

Prefer this wrapper for reusable calls:

```bash
<repo_root>/skills/master-thesis-admin/scripts/run_admin.sh status
```

## Core commands

- `status`: show git status and analysis artifact count.
- `local init`: initialize repo-local storage at `local_store/`.
- `local register-version --version-id <id> --dataset-path <csv>`: copy a dataset into local storage as `raw/<id>/dataset.csv`.
- `doctor --component <name>`: validate env vars, binaries, and Python imports.
- `analysis run --step {eda|correlation|regression|robustness|all}`: run thesis analysis jobs.
- `analysis list-outputs`: list generated reports/figures/CSV files.
- `analysis literature [--finding-source ...]`: extract findings from markdown outputs and query OpenAlex for related papers.
- `research init|ingest-openalex|overview`: maintain a thesis foundation database (papers, claim links, interactions, dependencies).
- `mapping propose ...`: generate mapping proposals (defaults to local heuristic mode).
- `mapping freeze ...`: freeze approved mapping proposals into a bundle.
- `pipeline run ...`: run the deterministic pipeline job.
- `enrichment run ...`: run enrichment CLI (currently scaffolded behavior in repo).
- `worker typecheck|dev|deploy`: execute worker development flows.
- `ui run`: run the Streamlit research UI.
- `api list-*` and `api trigger-run`: call Worker API endpoints.
- `cloudbuild submit --target ...`: submit configured Cloud Build workflows.

## Execution rules

- Run commands from any directory; the CLI resolves repository paths internally.
- Local storage (`local_store/`) is enabled by default for mapping/pipeline/enrichment commands.
- Use `--dry-run` for potentially destructive or cloud-side commands.
- Run `doctor` before external workflows to detect missing env or dependencies.
- Prefer sequence: `analysis run --step all` -> `analysis literature` -> `research ingest-openalex --latest` -> `research overview` -> `local init` -> `local register-version` -> `mapping propose` -> `mapping freeze --auto-approve-all` -> `pipeline run`.
- Cloud-hosted Worker/API deployment is optional and can be deferred.
- Expect `mapping propose` to require Vertex billing and IAM permissions in cloud mode; do not treat billing/auth failures as code defects unless repro occurs with valid credentials.
- Prefer `doctor --component mapping` before invoking mapping, and `doctor --component ui` before launching the Streamlit console.
