# Issue #27: Strategic decision needed — lock analysis track for thesis core

Status: Awaiting human decision

@jhytabest We need your decision to unblock the core analysis path.

## Decision needed
Choose one track for the next sprint:

1. **Option A — Causal-priority (recommended)**
   - Add industry fixed effects to core models
   - Add missingness/selection correction (Heckman or 2-stage selection)
   - Then freeze thesis core results

2. **Option B — Delivery-priority**
   - Freeze current baseline + robustness now
   - Treat missingness correction as appendix/future work

## Recommendation
**Recommend Option A.**
Reason: outcome missingness (~65%) is too large for a defensible final narrative without explicit selection handling.

## Impact
- **Blocked until decision:** final thesis results narrative and interpretation lock
- **Can continue meanwhile:** implementation prep, model scaffolding, and documentation

## Acceptance criteria
- [ ] Decision recorded by @jhytabest
- [ ] NORTHSTAR updated with chosen track
- [ ] Sprint plan adjusted accordingly

Refs #27
