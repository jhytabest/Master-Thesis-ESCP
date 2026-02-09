# Issue #14: Negative Binomial model may be inappropriate — Poisson dispersion < 1

Status: Draft implementation plan

## Acceptance criteria
- [ ] Root cause and implementation approach documented in this issue file
- [ ] Code and/or analysis updates committed in this PR
- [ ] Verification evidence added (tests, logs, or artifact diffs)
- [ ] Follow-up risks and limitations noted

## Implementation plan
- [ ] Recompute dispersion diagnostics for count outcomes to confirm underdispersion
- [ ] Compare Poisson, quasi-Poisson, NB, and robust alternatives with fit diagnostics
- [ ] Select primary model family using objective criteria (AIC/BIC, residual diagnostics)
- [ ] Update model code paths and output tables to reflect selected specification
- [ ] Add notes explaining when NB should/should not be used for this dataset
\nRefs #14
