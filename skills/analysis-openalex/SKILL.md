---
name: analysis-openalex
description: Map thesis analysis findings to related academic literature using OpenAlex via the repository admin CLI. Use when asked to find papers connected to current analysis results, hypotheses, priorities, or key findings.
---

# Analysis OpenAlex Skill

Use this command surface:

```bash
python <repo_root>/tools/repo_admin.py analysis literature --help
```

Or use the wrapper:

```bash
<repo_root>/skills/analysis-openalex/scripts/run_literature.sh --max-findings 8 --works-per-finding 5
```

Or run the full curation loop:

```bash
<repo_root>/skills/analysis-openalex/scripts/run_foundation_cycle.sh --max-findings 8 --works-per-finding 5
```

## Workflow

- Confirm analysis markdown outputs exist (`NORTHSTAR.md` and/or `jobs/analysis/output/*.md`).
- Run `analysis literature` to extract findings and query OpenAlex.
- Ingest results into the thesis foundation DB with `research ingest-openalex`.
- Generate a clean curation snapshot with `research overview`.

## Command patterns

- Default source discovery:
  - `analysis literature --max-findings 8 --works-per-finding 5`
- Explicit sources:
  - `analysis literature --finding-source NORTHSTAR.md --finding-source jobs/analysis/output/regression_report.md`
- Query planning only:
  - `analysis literature --dry-run`
- Foundation ingestion:
  - `research ingest-openalex --latest`
  - `research overview`

## Outputs

- JSON report: `local_store/reports/literature/openalex_<run_id>.json`
- Markdown report: `local_store/reports/literature/openalex_<run_id>.md`
- Foundation database: `local_store/research/*.jsonl`
- Derived query DB: `local_store/research/research.db`
- Overview snapshot: `local_store/research/foundation_overview.md`

## Rules

- Prefer `--dry-run` first when changing source selection logic.
- Keep outputs local and commit them if they are part of a reproducible research checkpoint.
- Use `OPENALEX_EMAIL` env var when available for OpenAlex polite pool routing.
