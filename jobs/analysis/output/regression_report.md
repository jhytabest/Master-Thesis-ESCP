# Regression Analysis Report

## Overview

Three regression models examining predictors of French deeptech startup funding:

| Model | Target | Method | N | Fit |
|-------|--------|--------|---|-----|
| Log Total Funding | see below | OLS (HC3) | 229 | R²=0.2551, Adj.R²=0.2101 |
| Log Time to First Round | see below | OLS (HC3) | 248 | R²=0.0792, Adj.R²=0.028 |
| Funding Rounds (NegBin) | target_rounds | Neg. Binomial | 285 | Pseudo-R²=0.0092 |

## Predictors

Selected based on bivariate analysis results and theoretical relevance.
**Excluded:** `fundoing_new` and `FOUNDERS FOUNDED COMPANIES TOTAL FUNDING` (endogenous / near-perfect correlation with target = data leakage).

| Type | Features |
|------|----------|
| Numeric | 0_founders_number, 8_age_moyen |
| Binary | 5_has_female, 4_phd_founder, 7_MBA, 3_one_founder_serial, 6_was_1_change_founders |
| Categorical (dummies) | 2_ceo_profile, B2B/B2C |

## Model: Log Total Funding

**N = 229** | R² = 0.2551 | Adj. R² = 0.2101 | F = 5.82 (p = 3.655e-09)

### Coefficients (HC3 robust SEs)

| Feature | Coef | SE | t | p | 95% CI | β (std) |
|---------|------|----|---|---|--------|--------|
| const | -0.8324 | 0.5531 | -1.51 | 0.1323 | [-1.917, 0.252] | — |
| **0_founders_number** | 0.4204 | 0.1501 | 2.80 | 0.0051 | [0.126, 0.715] | 0.289 |
| 8_age_moyen | 0.0152 | 0.0103 | 1.48 | 0.1399 | [-0.005, 0.035] | 0.138 |
| **5_has_female** | -0.7467 | 0.3608 | -2.07 | 0.0385 | [-1.454, -0.040] | -0.237 |
| 4_phd_founder | -0.0355 | 0.2061 | -0.17 | 0.8632 | [-0.440, 0.368] | -0.018 |
| **7_MBA** | 1.2368 | 0.2171 | 5.70 | 0.0000 | [0.811, 1.662] | 0.548 |
| 3_one_founder_serial | -0.2604 | 0.2344 | -1.11 | 0.2666 | [-0.720, 0.199] | -0.115 |
| 6_was_1_change_founders | 0.3431 | 0.4074 | 0.84 | 0.3998 | [-0.456, 1.142] | 0.090 |
| 2_ceo_profile_engineer | -0.3592 | 0.3475 | -1.03 | 0.3012 | [-1.040, 0.322] | -0.150 |
| **2_ceo_profile_mixte** | 0.5474 | 0.2196 | 2.49 | 0.0127 | [0.117, 0.978] | 0.244 |
| **2_ceo_profile_scientific** | 1.0407 | 0.3207 | 3.25 | 0.0012 | [0.412, 1.669] | 0.352 |
| **2_ceo_profile_university** | -0.6514 | 0.3310 | -1.97 | 0.0491 | [-1.300, -0.003] | -0.127 |
| B2B/B2C_business;consumer | -0.8488 | 0.5634 | -1.51 | 0.1319 | [-1.953, 0.255] | -0.190 |
| B2B/B2C_consumer | -0.0726 | 0.3650 | -0.20 | 0.8424 | [-0.788, 0.643] | -0.024 |

### Diagnostics

- **Breusch-Pagan (heteroscedasticity):** p = 0.0004 ⚠️ Significant
- **Jarque-Bera (normality of residuals):** p = 0.0000 ⚠️ Non-normal
- **Influential observations (Cook's d > 4/n):** 17

### VIF (Multicollinearity)

| Feature | VIF |
|---------|-----|
| 0_founders_number | 5.83 ⚠️ |
| 8_age_moyen | 6.41 ⚠️ |
| 5_has_female | 1.19 |
| 4_phd_founder | 2.39 |
| 7_MBA | 1.61 |
| 3_one_founder_serial | 1.47 |
| 6_was_1_change_founders | 1.22 |
| 2_ceo_profile_engineer | 1.64 |
| 2_ceo_profile_mixte | 1.98 |
| 2_ceo_profile_scientific | 1.5 |
| 2_ceo_profile_university | 1.2 |
| B2B/B2C_business;consumer | 1.18 |
| B2B/B2C_consumer | 1.23 |

## Model: Log Time to First Round

**N = 248** | R² = 0.0792 | Adj. R² = 0.028 | F = 1.98 (p = 0.023)

### Coefficients (HC3 robust SEs)

| Feature | Coef | SE | t | p | 95% CI | β (std) |
|---------|------|----|---|---|--------|--------|
| **const** | 2.5814 | 0.3696 | 6.98 | 0.0000 | [1.857, 3.306] | — |
| 0_founders_number | -0.2043 | 0.1054 | -1.94 | 0.0526 | [-0.411, 0.002] | -0.143 |
| **8_age_moyen** | 0.0145 | 0.0071 | 2.05 | 0.0406 | [0.001, 0.028] | 0.139 |
| 5_has_female | 0.2288 | 0.1946 | 1.18 | 0.2397 | [-0.153, 0.610] | 0.077 |
| 4_phd_founder | -0.0474 | 0.1402 | -0.34 | 0.7350 | [-0.322, 0.227] | -0.024 |
| 7_MBA | -0.3328 | 0.1762 | -1.89 | 0.0589 | [-0.678, 0.012] | -0.142 |
| 3_one_founder_serial | 0.0572 | 0.1553 | 0.37 | 0.7126 | [-0.247, 0.362] | 0.025 |
| 6_was_1_change_founders | -0.2538 | 0.2624 | -0.97 | 0.3333 | [-0.768, 0.260] | -0.066 |
| 2_ceo_profile_engineer | -0.1705 | 0.1910 | -0.89 | 0.3720 | [-0.545, 0.204] | -0.071 |
| 2_ceo_profile_mixte | 0.0274 | 0.1821 | 0.15 | 0.8803 | [-0.330, 0.384] | 0.012 |
| 2_ceo_profile_scientific | -0.1061 | 0.2178 | -0.49 | 0.6262 | [-0.533, 0.321] | -0.039 |
| 2_ceo_profile_university | -0.0474 | 0.4191 | -0.11 | 0.9099 | [-0.869, 0.774] | -0.008 |
| B2B/B2C_business;consumer | -0.3593 | 0.2696 | -1.33 | 0.1826 | [-0.888, 0.169] | -0.071 |
| **B2B/B2C_consumer** | 0.3451 | 0.1652 | 2.09 | 0.0367 | [0.021, 0.669] | 0.111 |

### Diagnostics

- **Breusch-Pagan (heteroscedasticity):** p = 0.4894 ✅ Not significant
- **Jarque-Bera (normality of residuals):** p = 0.0000 ⚠️ Non-normal
- **Influential observations (Cook's d > 4/n):** 14

### VIF (Multicollinearity)

| Feature | VIF |
|---------|-----|
| 0_founders_number | 5.64 ⚠️ |
| 8_age_moyen | 6.21 ⚠️ |
| 5_has_female | 1.22 |
| 4_phd_founder | 2.43 |
| 7_MBA | 1.48 |
| 3_one_founder_serial | 1.45 |
| 6_was_1_change_founders | 1.16 |
| 2_ceo_profile_engineer | 1.72 |
| 2_ceo_profile_mixte | 1.93 |
| 2_ceo_profile_scientific | 1.57 |
| 2_ceo_profile_university | 1.18 |
| B2B/B2C_business;consumer | 1.12 |
| B2B/B2C_consumer | 1.19 |

## Model: Funding Rounds (NegBin)

**N = 285** | Pseudo-R² (McFadden) = 0.0092 | AIC = 1142.2

**Poisson dispersion test:** 0.806 (mild overdispersion)

### Coefficients & Incidence Rate Ratios (IRR)

| Feature | Coef | SE | z | p | IRR |
|---------|------|----|---|---|-----|
| const | 0.4943 | 0.4207 | 1.18 | 0.2400 | 1.639 |
| 0_founders_number | 0.0596 | 0.1112 | 0.54 | 0.5921 | 1.061 |
| 8_age_moyen | 0.0005 | 0.0078 | 0.07 | 0.9462 | 1.001 |
| 5_has_female | -0.2505 | 0.2282 | -1.10 | 0.2724 | 0.778 |
| 4_phd_founder | 0.0150 | 0.1559 | 0.10 | 0.9233 | 1.015 |
| **7_MBA** | 0.4265 | 0.1725 | 2.47 | 0.0134 | 1.532 |
| 3_one_founder_serial | -0.0519 | 0.1693 | -0.31 | 0.7593 | 0.949 |
| 6_was_1_change_founders | 0.1568 | 0.2824 | 0.56 | 0.5789 | 1.170 |
| 2_ceo_profile_engineer | 0.0068 | 0.2009 | 0.03 | 0.9730 | 1.007 |
| 2_ceo_profile_mixte | 0.0561 | 0.1991 | 0.28 | 0.7780 | 1.058 |
| 2_ceo_profile_scientific | 0.1266 | 0.2319 | 0.55 | 0.5850 | 1.135 |
| 2_ceo_profile_university | -0.0243 | 0.4377 | -0.06 | 0.9557 | 0.976 |
| B2B/B2C_business;consumer | 0.2147 | 0.3597 | 0.60 | 0.5506 | 1.240 |
| B2B/B2C_consumer | 0.0306 | 0.2292 | 0.13 | 0.8939 | 1.031 |

### VIF

| Feature | VIF |
|---------|-----|
| 0_founders_number | 5.76 ⚠️ |
| 8_age_moyen | 6.3 ⚠️ |
| 5_has_female | 1.22 |
| 4_phd_founder | 2.41 |
| 7_MBA | 1.52 |
| 3_one_founder_serial | 1.46 |
| 6_was_1_change_founders | 1.19 |
| 2_ceo_profile_engineer | 1.75 |
| 2_ceo_profile_mixte | 1.93 |
| 2_ceo_profile_scientific | 1.63 |
| 2_ceo_profile_university | 1.17 |
| B2B/B2C_business;consumer | 1.14 |
| B2B/B2C_consumer | 1.18 |

## Key Findings

*(Auto-generated from significant coefficients at α = 0.05)*

**Log Total Funding:** 0_founders_number, 5_has_female, 7_MBA, 2_ceo_profile_mixte, 2_ceo_profile_scientific, 2_ceo_profile_university

**Log Time to First Round:** 8_age_moyen, B2B/B2C_consumer

**Funding Rounds (NegBin):** 7_MBA


## Limitations

- Sample size is modest after listwise deletion (high missingness on categoricals)
- Cross-sectional design — no causal claims possible
- Omitted variable bias likely (industry effects partially captured via CEO profile)
- Log transforms improve normality but complicate coefficient interpretation
- HC3 SEs are robust to heteroscedasticity but assume correct functional form
