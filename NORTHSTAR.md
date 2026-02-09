# NORTHSTAR — Research Vision & Priorities

**Last updated:** 2026-02-09 | **CTO:** Shober

---

## Research Question

**What founder-level characteristics predict funding outcomes in French deeptech startups?**

Three target variables:
1. `target_total_funding` — total capital raised (M€)
2. `target_rounds` — number of funding rounds
3. `target_delta_1st_round` — months from founding to first round

## Current State of Analysis

### What's Done
- **EDA:** 1,182 French deeptech startups, 68 columns. Heavy missingness (58-65% on targets). Highly skewed funding distributions.
- **Correlation analysis:** 57 bivariate tests, 20 significant (FDR < 0.05). MBA and CEO profile are strongest founder-level predictors. `fundoing_new` is near-perfectly correlated with `target_total_funding` (r=0.991) — correctly flagged as data leakage.
- **Regression:** Three models (OLS log-funding R²=0.255, OLS log-time-to-round R²=0.079, NegBin rounds pseudo-R²=0.009). Only the funding model has meaningful explanatory power.
- **Female founder deep-dive:** Descriptive gap exists (median 2.3M vs 3.0M). OLS coefficient is significant (β=-0.75, p=0.04) but PSM shows stronger effect (ATT=-1.39 log, ~-9.3M€). However: borderline significance, small N (57 female-founded), heavy sector confounding (65% health vs 46%).
- **Robustness:** Bootstrap CIs consistent with HC3. Trimming influential obs doesn't change significance patterns. MBA and scientific CEO profile are robust across all specs.

### Key Findings So Far
| Finding | Strength | Concern |
|---------|----------|---------|
| MBA → more funding & rounds | Strong, robust | Mechanism unclear (signal vs. network?) |
| Scientific/mixte CEO → more funding | Moderate, robust | Reference category = "business" — is this meaningful? |
| Female founder → less funding | Borderline (p=0.04) | Small N, sector confounding, PSM vs OLS diverge |
| Founder count → more funding | Moderate | VIF=5.8 with age — collinearity |
| Time-to-first-round model is weak | R²=0.08 | Founder traits don't explain timing well |

## Top 3 Research Priorities

### 1. Strengthen the Causal Story on MBA Effect
The MBA coefficient (β=1.24, +244% funding) is the strongest and most robust finding. But **why?** Is it:
- A **signaling** effect (investors trust MBA credentials)?
- A **network** effect (MBA programs connect founders to VCs)?
- A **selection** effect (MBA founders choose sectors/strategies that attract more capital)?

**Action:** Add industry fixed effects (use `11_industry_simplified`). Test MBA × industry interactions. If the MBA effect disappears within sectors, it's selection. If it persists, it's signal/network.

### 2. Resolve the Female Founder Puzzle
The current finding is fragile. The OLS says -53% funding, the PSM says -9.3M€, but raw Mann-Whitney is p=0.26 (non-significant). The mediation analysis found little mediation through observables, yet sector distribution differs dramatically.

**Action:**
- Run the funding model **with industry fixed effects** — does `5_has_female` survive?
- Compute the Oster (2019) δ-bounds to assess sensitivity to unobservables
- Consider Heckman selection correction for the 65% missing funding data — are female-founded startups differentially missing?
- Frame carefully: this is an **association finding**, not a discrimination claim

### 3. Address Survivorship & Selection Bias from Missingness
65% of funding data is missing. This is not random — startups without funding data are likely unfunded or early-stage. Current analysis is conditional on having raised capital, which biases everything.

**Action:**
- Profile the missing-data population vs. the observed population
- Consider a two-stage model: (1) Probit for "did the startup raise any funding?" (2) OLS for "how much?"  (Heckman two-step)
- At minimum, report the selection issue prominently as a limitation

## Key Hypotheses to Test

| # | Hypothesis | Test | Priority |
|---|-----------|------|----------|
| H1 | MBA-holding founders raise significantly more total funding, controlling for industry | OLS with industry FE | High |
| H2 | The female founder funding gap is mediated by industry sorting | OLS with industry FE + Oaxaca-Blinder decomposition | High |
| H3 | Scientific CEO profiles are associated with more funding in health/biotech but not other sectors | Interaction model: CEO profile × industry | Medium |
| H4 | Serial entrepreneurship predicts funding rounds but not total amount | Compare NegBin (rounds) vs OLS (funding) coefficients | Medium |
| H5 | Missing funding data is not random — selection into observability correlates with founder characteristics | Probit model for missingness | High |

## Methodology Direction

### What to Do
- **Add industry fixed effects** to all models — this is the single biggest gap right now
- **Heckman correction** or at least a missingness analysis — 65% attrition is too large to ignore
- **Interaction terms** for the key findings (MBA × industry, female × industry, CEO profile × industry)
- **Quantile regression** on total funding — effects at the median vs. top quartile may differ dramatically
- **Clean presentation:** funnel from descriptive → bivariate → multivariate → robustness → deep-dives

### What to Avoid
- ❌ **Don't claim causation.** This is cross-sectional observational data. Say "associated with," not "leads to."
- ❌ **Don't over-interpret the female founder result.** p=0.04 with multiple comparisons and 57 observations is not a strong finding. Present it honestly.
- ❌ **Don't ignore the missingness.** Every model is run on ~230-285 obs out of 1,182. That's 75% attrition. This must be discussed.
- ❌ **Don't add more variables without theory.** The dataset has 68 columns. Feature fishing will produce false positives. Every variable in the model needs a theoretical justification.
- ❌ **Don't confuse `fundoing_new` with an independent variable.** It's essentially `target_total_funding` re-coded. Already flagged but worth repeating.
- ❌ **Don't use GROWTH STAGE as a predictor.** It's endogenous — determined by funding rounds, which is a target variable.

## Thesis Structure Suggestion

1. **Literature Review:** Founder human capital → startup outcomes (Colombo & Grilli 2005, Gimmon & Levie 2010); Gender in VC (Brush et al. 2018, Ewens & Townsend 2020); Deeptech specifics (Laser & Hille 2021)
2. **Data & Methods:** Sample description, variable definitions, missingness analysis, model specifications
3. **Results:** Descriptive → Bivariate → Multivariate (with industry FE) → Female founder deep-dive → Robustness
4. **Discussion:** MBA as strongest predictor (why?), gender gap (cautious interpretation), model limitations, French deeptech ecosystem specifics

---

*This document is the single source of truth for research direction. Challenge it, update it, but always commit changes.*
