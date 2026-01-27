# Architecture

This repo implements a research-first, traceable pipeline for France-only deeptech datasets.

## Goals

- Reproducible pipeline: CSV in -> clean cohort, EDA, models, predictions, and versioned artifacts out.
- Human-reviewable mappings for messy categoricals.
- Optional enrichment workflows with compliant, traceable sources.

## Non-goals (MVP)

- Interactive analytics product
- Real-time serving
- Automated LinkedIn scraping

## Workflows

### Workflow A: Mapping Builder

- Inputs: dataset version or distinct raw values
- Outputs: mapping bundle in R2 + metadata in D1
- AI proposes mappings, but only approved mappings are used

### Workflow B: Main Pipeline

- Inputs: version_id and optional mapping bundle
- Outputs: cohort, reports, models, predictions, and manifests in R2

### Workflow C: Enrichment

- Inputs: cohort + compliant external source
- Outputs: enriched features, provenance, coverage report

## Storage contracts

R2 object layout is fixed and versioned:

- raw/{version_id}/dataset.csv
- derived/{version_id}/cohort.parquet
- derived/{version_id}/features.parquet
- reports/{version_id}/eda.html
- reports/{version_id}/data_quality.json
- reports/{version_id}/index.json
- models/{version_id}/{model_id}/model.bin
- models/{version_id}/{model_id}/metrics.json
- predictions/{version_id}/{model_id}/scores.parquet
- predictions/{version_id}/{model_id}/explanations.parquet
- manifests/{version_id}/cleaning_manifest.json
- mappings/{mapping_bundle_id}/status_mapping.csv
- mappings/{mapping_bundle_id}/bundle_manifest.json
- enriched/{version_id}/{source}/features.parquet
- enriched/{version_id}/{source}/provenance.parquet
- enriched/{version_id}/{source}/coverage.json

## Metadata (D1)

- versions
- runs
- mapping_bundles
- models
- enrichment_runs

Schemas live in `apps/worker-api/migrations/0001_init.sql`.

## AI usage policy

Allowed:

- Propose mappings (distinct values only)
- Classify deeptech domains from text as a feature construction step
- Narrative summaries of EDA/quality reports

Not allowed:

- AI rewriting numeric or date values
- Fabricated enrichment data

All AI outputs must store prompt hash, model id, timestamp, and input set hash.

