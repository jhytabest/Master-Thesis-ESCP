# Pipeline Job

Deterministic "CSV in -> artifacts out" pipeline.

## Entrypoint

```
pipeline --version_id <id> [--mapping_bundle_id <id>] [--run_id <id>]
```

## Status

Implemented:

- Download CSV from R2
- Generate cohort + basic EDA + quality report
- Build MVP feature matrix
- Train baseline models + write metrics/predictions
- Write artifacts + manifests

Pending:

- Real cleaning + standardization
- Outcome derivations
- Expanded modeling and explainability
