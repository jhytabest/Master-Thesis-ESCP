# Issue #19: Non-normal residuals (JB p=0.0000) across all OLS models

Status: Draft implementation plan

## Acceptance criteria
- [ ] Root cause and implementation approach documented in this issue file
- [ ] Code and/or analysis updates committed in this PR
- [ ] Verification evidence added (tests, logs, or artifact diffs)
- [ ] Follow-up risks and limitations noted

## Implementation plan
- [ ] Reproduce residual diagnostics and isolate variables driving non-normality
- [ ] Test transformations (log/Box-Cox/winsorization) for skewed predictors/targets
- [ ] Evaluate robust or alternative model families (GLM/quantile/robust SEs)
- [ ] Update model/report pipeline with chosen remedy and rationale
- [ ] Add residual diagnostic outputs to artifacts for ongoing monitoring
\nRefs #19
