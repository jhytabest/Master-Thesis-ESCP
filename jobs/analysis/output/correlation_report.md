# Correlation & Bivariate Analysis

**Tests run:** 57 | **Significant (FDR < 0.05):** 20

## Methodology

- **Numeric features:** Spearman rank correlation (robust to non-normality)
- **Binary features:** Point-biserial correlation
- **Categorical features:** Kruskal-Wallis H-test with ε² effect size
- **Multiple comparisons:** Benjamini-Hochberg FDR correction (α = 0.05)

## Significant Associations (FDR < 0.05)

- **fundoing_new** → target_total_funding: spearman, r=0.991 (+), p_adj=0.0000, effect=large (0.991), n=420
- **FOUNDERS FOUNDED COMPANIES TOTAL FUNDING** → target_total_funding: spearman, r=0.957 (+), p_adj=0.0000, effect=large (0.957), n=123
- **fundoing_new** → target_rounds: spearman, r=0.622 (+), p_adj=0.0000, effect=large (0.622), n=487
- **FOUNDERS FOUNDED COMPANIES TOTAL FUNDING** → target_rounds: spearman, r=0.513 (+), p_adj=0.0000, effect=large (0.513), n=126
- **GROWTH STAGE** → target_total_funding: kruskal_wallis, r=126.932, p_adj=0.0000, effect=large (0.304), n=418
- **FOUNDERS ESTIMATED AGE** → target_total_funding: spearman, r=0.303 (+), p_adj=0.0009, effect=medium (0.303), n=150
- **7_MBA** → target_rounds: point_biserial, r=0.262 (+), p_adj=0.0000, effect=small (0.262), n=460
- **6_was_1_change_founders** → target_total_funding: point_biserial, r=0.211 (+), p_adj=0.0001, effect=small (0.211), n=409
- **GROWTH STAGE** → target_rounds: kruskal_wallis, r=102.850, p_adj=0.0000, effect=large (0.209), n=493
- **2_ceo_profile** → target_total_funding: kruskal_wallis, r=76.338, p_adj=0.0000, effect=large (0.207), n=370
- **FOUNDERS ESTIMATED AGE** → target_rounds: spearman, r=0.194 (+), p_adj=0.0315, effect=small (0.194), n=182
- **3_one_founder_serial** → target_total_funding: point_biserial, r=0.168 (+), p_adj=0.0080, effect=small (0.168), n=349
- **0_founders_number** → target_rounds: spearman, r=0.161 (+), p_adj=0.0029, effect=small (0.161), n=458
- **8_age_moyen** → target_delta_1st_round: spearman, r=0.151 (+), p_adj=0.0285, effect=small (0.151), n=313
- **7_MBA** → target_total_funding: point_biserial, r=0.146 (+), p_adj=0.0167, effect=small (0.146), n=392
- **CEO ESTIMATED AGE** → target_delta_1st_round: spearman, r=0.139 (+), p_adj=0.0444, effect=small (0.139), n=309
- **0_founders_number** → target_delta_1st_round: spearman, r=-0.134 (−), p_adj=0.0285, effect=small (0.134), n=395
- **3_one_founder_was_serial_at_creation** → target_rounds: point_biserial, r=0.117 (+), p_adj=0.0395, effect=small (0.117), n=462
- **6_was_1_change_founders** → target_rounds: point_biserial, r=0.113 (+), p_adj=0.0411, effect=small (0.113), n=481
- **2_ceo_profile** → target_rounds: kruskal_wallis, r=12.129, p_adj=0.0468, effect=small (0.028), n=433

## Key Takeaways

### target_delta_1st_round
- 8_age_moyen (small effect, r=0.151)
- CEO ESTIMATED AGE (small effect, r=0.139)
- 0_founders_number (small effect, r=-0.134)

### target_total_funding
- fundoing_new (large effect, r=0.991)
- FOUNDERS FOUNDED COMPANIES TOTAL FUNDING (large effect, r=0.957)
- GROWTH STAGE (large effect, r=126.932)

### target_rounds
- fundoing_new (large effect, r=0.622)
- FOUNDERS FOUNDED COMPANIES TOTAL FUNDING (large effect, r=0.513)
- 7_MBA (small effect, r=0.262)

## Figures

- `jobs/analysis/output/fig_correlation_heatmap.png`
- `jobs/analysis/output/fig_top_associations.png`

## Limitations & Assumptions

- Spearman does not assume linearity but captures monotonic relationships only
- High missingness (~60%) on targets limits statistical power
- Binary features assume clean 0/1 encoding; rows with other values are dropped
- Kruskal-Wallis is omnibus — significant results need post-hoc pairwise tests (Dunn's test)
