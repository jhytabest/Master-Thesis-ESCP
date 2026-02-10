# Research Foundation Database (Local-First)

This document defines the iterative process and storage model for thesis literature curation.

## Goals

- Maintain a living paper database that evolves with analysis and writing.
- Link thesis claims/findings to supporting or contextual papers.
- Track paper interactions and dependency paths for deeper reading.
- Keep canonical data local, versionable, and Git-friendly.

## Iterative Loop

1. Run/update analysis outputs and `NORTHSTAR.md`.
2. Discover literature:
   - `python tools/repo_admin.py analysis literature --max-findings 8 --works-per-finding 5`
3. Ingest to foundation database:
   - `python tools/repo_admin.py research ingest-openalex --latest`
4. Generate overview snapshot:
   - `python tools/repo_admin.py research overview`
5. Manually enrich relations where needed:
   - `python tools/repo_admin.py research add-edge ...`
   - `python tools/repo_admin.py research add-dependency ...`

## Storage Model

Canonical files (source of truth):

- `local_store/research/papers.jsonl`
- `local_store/research/claims.jsonl`
- `local_store/research/claim_paper_links.jsonl`
- `local_store/research/paper_edges.jsonl`
- `local_store/research/dependencies.jsonl`
- `local_store/research/ingestions.jsonl`
- `local_store/research/runs/openalex_<run_id>.json` (ingested snapshots)

Derived query database:

- `local_store/research/research.db` (rebuilt from JSONL)

Generated overview:

- `local_store/research/foundation_overview.md`

## Semantics

- A **claim** is a thesis statement derived from analysis output or Northstar priorities.
- A **claim-paper link** encodes support/context/method relevance.
- A **paper edge** encodes interactions between papers (e.g., `cites`, `co_supports_claim`, manual `extends` or `contradicts`).
- A **dependency** encodes “read this before going deeper” paths, usually from citation references or manual curation.
- Each paper carries a citation-based quality tier (`foundational`, `high`, `medium`, `emerging`) to prioritize reading.
