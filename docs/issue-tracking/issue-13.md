# Issue #13: CI does not test analysis scripts

Status: Draft implementation plan

## Acceptance criteria
- [ ] Root cause and implementation approach documented in this issue file
- [ ] Code and/or analysis updates committed in this PR
- [ ] Verification evidence added (tests, logs, or artifact diffs)
- [ ] Follow-up risks and limitations noted

## Implementation plan
- [ ] Confirm which analysis scripts are outside current CI test scope
- [ ] Extend CI workflow to run analysis test subset in parallel with existing unit tests
- [ ] Add caching/time-budget controls so pipeline remains stable and fast
- [ ] Fail CI on missing critical analysis artifacts or regressions in expected outputs
- [ ] Document local command parity with CI in README or contributor docs
\nRefs #13
