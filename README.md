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
