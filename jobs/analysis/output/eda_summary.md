# Initial EDA summary
- Dataset path: `/home/brewuser/repos/jhytabest/Master-Thesis-ESCP/dataset.csv`
- Shape: **1182 rows x 68 columns**

## Columns & types
Dtype counts:
- str: 52
- float64: 16

## Missingness
Columns with missing values: **65 / 68**
Top missing columns (count, %):
- NUMBER OF ALUMNI EUROPEAN FOUNDERS THAT RAISED > 10M: 1163 (98.4%)
- NUMBER OF ALUMNI FOUNDERS THAT RAISED > 10M: 1160 (98.1%)
- DELIVERY METHOD: 1131 (95.7%)
- TECHNOLOGIES: 869 (73.5%)
- FOUNDERS FOUNDED COMPANIES TOTAL FUNDING: 829 (70.1%)
- VALUATION (EUR): 816 (69.0%)
- VALUATION DATE: 816 (69.0%)
- LAST FUNDING: 777 (65.7%)
- target_total_funding: 762 (64.5%)
- SUB INDUSTRIES: 731 (61.8%)
- FOUNDERS - TOP 25 PAST FOUNDERS: 730 (61.8%)
- FOUNDERS - TOP 25 UNIVERSITIES: 730 (61.8%)
- FOUNDERS LINKEDIN: 725 (61.3%)
- target_rounds: 689 (58.3%)
- LAST FUNDING DATE: 689 (58.3%)

Full table saved to: `/home/brewuser/repos/jhytabest/Master-Thesis-ESCP/jobs/analysis/output/missing_values.csv`

## Targets
- target_delta_1st_round: n=424, missing=758 (64.1%), min=0, median=20, max=650, mean=27.2, skew=10.5, zeros=9, negatives=0
- target_total_funding: n=420, missing=762 (64.5%), min=0.01, median=2.98, max=233, mean=12.5, skew=4.37, zeros=0, negatives=0
- target_rounds: n=493, missing=689 (58.3%), min=1, median=2, max=11, mean=2.42, skew=1.57, zeros=0, negatives=0

Detailed targets table saved to: `/home/brewuser/repos/jhytabest/Master-Thesis-ESCP/jobs/analysis/output/targets_summary.csv`

## Data quality flags (automated heuristics)
No strongly numeric-like object columns detected (>=80% coercible).

## Figures
- `jobs/analysis/output/fig_target_delta_1st_round_hist.png`
- `jobs/analysis/output/fig_target_delta_1st_round_box.png`
- `jobs/analysis/output/fig_target_delta_1st_round_hist_log1p.png`
- `jobs/analysis/output/fig_target_total_funding_hist.png`
- `jobs/analysis/output/fig_target_total_funding_box.png`
- `jobs/analysis/output/fig_target_total_funding_hist_log1p.png`
- `jobs/analysis/output/fig_target_rounds_hist.png`
- `jobs/analysis/output/fig_target_rounds_box.png`
- `jobs/analysis/output/fig_target_rounds_hist_log1p.png`

## Next suggested steps
- Decide a missing-data strategy per feature family (drop, impute, or model-native missing handling).
- Consider log-transforming highly skewed funding targets and/or winsorizing extreme outliers.
- Validate units/currencies for funding and valuation fields (some appear range-like strings).
- Parse and standardize dates (creation, funding dates) to enable time-based features.
