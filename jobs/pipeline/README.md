# Pipeline Job

Deterministic "CSV in -> artifacts out" pipeline.

## Entrypoint

```
pipeline --version_id <id> [--mapping_bundle_id <id>] [--run_id <id>]
```

## Status

Scaffolded. Steps to implement:

- Download CSV from R2
- Parse + clean
- Apply mapping bundle
- Derive outcomes
- Generate reports
- Train baseline models
- Write artifacts + manifests
