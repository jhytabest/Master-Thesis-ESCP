# Issue #18: High missingness (~60%) on targets not formally addressed

Status: Draft implementation plan

## Acceptance criteria
- [ ] Root cause and implementation approach documented in this issue file
- [ ] Code and/or analysis updates committed in this PR
- [ ] Verification evidence added (tests, logs, or artifact diffs)
- [ ] Follow-up risks and limitations noted

## Implementation plan
- [ ] Quantify missingness by target, subgroup, and time slice; persist diagnostic table
- [ ] Evaluate missingness mechanism indicators (MCAR/MAR assumptions where feasible)
- [ ] Prototype mitigation options (imputation strategy vs. models tolerant to missing targets)
- [ ] Select approach and implement in preprocessing/training pipeline
- [ ] Compare downstream metric stability before/after mitigation
\nRefs #18
