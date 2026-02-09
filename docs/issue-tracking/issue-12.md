# Issue #12: No unit tests for analysis scripts

Status: Draft implementation plan

## Acceptance criteria
- [ ] Root cause and implementation approach documented in this issue file
- [ ] Code and/or analysis updates committed in this PR
- [ ] Verification evidence added (tests, logs, or artifact diffs)
- [ ] Follow-up risks and limitations noted

## Implementation plan
- [ ] Identify analysis modules/scripts currently untested and rank by risk (EDA, regression, enrichment)
- [ ] Create deterministic fixture dataset(s) under `tests/fixtures/` for analysis workflows
- [ ] Add unit tests for core helper functions and report-generating functions
- [ ] Add smoke tests that execute each analysis script entrypoint on fixture data
- [ ] Ensure tests validate generated artifacts/columns, not only execution success
\nRefs #12
