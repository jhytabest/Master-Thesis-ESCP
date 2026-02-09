# Issue #17: Kruskal-Wallis significant results lack post-hoc Dunn's tests

Status: Draft implementation plan

## Acceptance criteria
- [ ] Root cause and implementation approach documented in this issue file
- [ ] Code and/or analysis updates committed in this PR
- [ ] Verification evidence added (tests, logs, or artifact diffs)
- [ ] Follow-up risks and limitations noted

## Implementation plan
- [ ] Enumerate all significant Kruskal-Wallis results in current analysis outputs
- [ ] Implement Dunn's post-hoc tests with multiple-comparison correction (e.g., Holm/FDR)
- [ ] Add pairwise effect-size reporting and adjusted p-values to outputs
- [ ] Integrate post-hoc summary table into existing report generation
- [ ] Add tests validating post-hoc table schema and correction behavior
\nRefs #17
