# Issue #20: Add variable_definitions.yml completeness — most variables undefined

Status: Draft implementation plan

## Acceptance criteria
- [ ] Root cause and implementation approach documented in this issue file
- [ ] Code and/or analysis updates committed in this PR
- [ ] Verification evidence added (tests, logs, or artifact diffs)
- [ ] Follow-up risks and limitations noted

## Implementation plan
- [ ] Inventory all model/report variables and map to existing `variable_definitions.yml` entries
- [ ] Add missing variable definitions with units, source, and transformation notes
- [ ] Standardize naming to match code and output artifacts exactly
- [ ] Add validation check to fail when undefined variables appear in outputs
- [ ] Document maintenance workflow for keeping definitions in sync
\nRefs #20
