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
- `doctor --component <name>`: validate env vars, binaries, and Python imports.
- `analysis run --step {eda|correlation|regression|robustness|all}`: run thesis analysis jobs.
- `analysis list-outputs`: list generated reports/figures/CSV files.
- `mapping propose ...`: generate mapping proposals (requires Vertex/GCP access and billing).
- `mapping freeze ...`: freeze approved mapping proposals into a bundle.
- `pipeline run ...`: run the deterministic pipeline job.
- `enrichment run ...`: run enrichment CLI (currently scaffolded behavior in repo).
- `worker typecheck|dev|deploy`: execute worker development flows.
- `ui run`: run the Streamlit research UI.
- `api list-*` and `api trigger-run`: call Worker API endpoints.
- `cloudbuild submit --target ...`: submit configured Cloud Build workflows.

## Execution rules

- Run commands from any directory; the CLI resolves repository paths internally.
- Use `--dry-run` for mapping/pipeline/cloud commands when credentials or billing are unavailable.
- Run `doctor` before external workflows to detect missing env or dependencies.
