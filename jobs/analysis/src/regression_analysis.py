"""Regression Analysis for French Deeptech Startup Funding

Three models targeting the thesis research questions:

1. OLS on log(target_total_funding)     — What predicts total funding amount?
2. OLS on log(target_delta_1st_round+1) — What predicts time to first funding?
3. Negative Binomial for target_rounds  — What predicts number of funding rounds?

Methodology:
  - OLS with heteroscedasticity-robust standard errors (HC3; Long & Ervin, 2000)
  - Log transforms for skewed continuous targets (Wooldridge, 2016, §6.2)
  - Negative Binomial for count data with overdispersion (Cameron & Trivedi, 2013)
  - VIF checks for multicollinearity (threshold: VIF > 5; Craney & Surles, 2002)
  - Breusch-Pagan test for heteroscedasticity
  - Jarque-Bera test for residual normality
  - Cook's distance for influential observations

Academic references:
  - Cameron, A. C., & Trivedi, P. K. (2013). Regression Analysis of Count Data.
  - Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent SEs.
  - Wooldridge, J. M. (2016). Introductory Econometrics (6th ed.).
  - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.

Run:
  python jobs/analysis/src/regression_analysis.py

Outputs → jobs/analysis/output/regression_*
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
np.random.seed(42)  # Reproducibility
import pandas as pd
from scipy import stats as sp_stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.discrete.discrete_model import NegativeBinomial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "dataset.csv"
OUT_DIR = REPO_ROOT / "jobs" / "analysis" / "output"

# Predictors chosen based on correlation analysis results + theoretical relevance
# Excluding fundoing_new (near-perfect correlation with target_total_funding = data leakage)
# Excluding FOUNDERS FOUNDED COMPANIES TOTAL FUNDING (also likely leakage / endogenous)

NUMERIC_PREDICTORS = [
    "0_founders_number",
    "8_age_moyen",
]

BINARY_PREDICTORS = [
    "5_has_female",
    "4_phd_founder",
    "7_MBA",
    "3_one_founder_serial",
    "6_was_1_change_founders",
]

CATEGORICAL_PREDICTORS = [
    "2_ceo_profile",
    "B2B/B2C",
]

ALL_PREDICTORS = NUMERIC_PREDICTORS + BINARY_PREDICTORS + CATEGORICAL_PREDICTORS


def load_and_prepare() -> pd.DataFrame:
    """Load dataset and prepare analysis-ready DataFrame."""
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Convert targets and numeric predictors
    for col in ["target_delta_1st_round", "target_total_funding", "target_rounds"] + NUMERIC_PREDICTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert binary predictors
    for col in BINARY_PREDICTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log transforms for skewed targets
    df["log_total_funding"] = np.log(df["target_total_funding"].clip(lower=0.001))
    df["log_delta_1st"] = np.log1p(df["target_delta_1st_round"].clip(lower=0))

    return df


def build_design_matrix(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build X matrix with dummies for categoricals, dropping NAs."""
    # Start with numeric + binary
    X = df[NUMERIC_PREDICTORS + BINARY_PREDICTORS].copy()

    # Add dummies for categoricals
    for cat in CATEGORICAL_PREDICTORS:
        dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    y = df[target_col]

    # Drop rows with any NA in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # Add constant
    X = sm.add_constant(X)

    return X, y


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factors (exclude constant)."""
    cols = [c for c in X.columns if c != "const"]
    X_no_const = X[cols].values.astype(float)
    vifs = []
    for i, col in enumerate(cols):
        try:
            vif_val = variance_inflation_factor(X_no_const, i)
        except (np.linalg.LinAlgError, ValueError):
            vif_val = np.nan
        vifs.append({"feature": col, "VIF": round(vif_val, 2)})
    return pd.DataFrame(vifs)


def run_ols(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    """Run OLS with HC3 robust SEs and full diagnostics."""
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Diagnostics
    residuals = model.resid
    fitted = model.fittedvalues

    # Breusch-Pagan
    _, bp_p, _, _ = het_breuschpagan(residuals, X)

    # Jarque-Bera
    jb_stat, jb_p = sp_stats.jarque_bera(residuals)

    # Cook's distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    n_influential = (cooks_d > 4 / len(y)).sum()

    # VIF
    vif_df = compute_vif(X)

    # Effect sizes: fully standardized beta weights (both X and y standardized)
    std_y = (y - y.mean()) / y.std() if y.std() > 0 else y
    std_X = X.drop(columns=["const"]).apply(lambda c: (c - c.mean()) / c.std() if c.std() > 0 else c)
    std_X = sm.add_constant(std_X)
    std_model = sm.OLS(std_y, std_X).fit(cov_type="HC3")
    beta_weights = std_model.params.drop("const")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Residuals vs fitted
    axes[0].scatter(fitted, residuals, alpha=0.4, s=15)
    axes[0].axhline(0, color="red", linewidth=0.8)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    # Q-Q plot
    sp_stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q")

    # Cook's distance
    axes[2].stem(range(len(cooks_d)), cooks_d, markerfmt=",", linefmt="C0-", basefmt="k-")
    axes[2].axhline(4 / len(y), color="red", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("Observation")
    axes[2].set_ylabel("Cook's Distance")
    axes[2].set_title(f"Cook's Distance ({n_influential} influential)")

    fig.suptitle(f"Diagnostics: {model_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path = OUT_DIR / f"fig_regression_diag_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "model_name": model_name,
        "model": model,
        "n": int(model.nobs),
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_stat": round(model.fvalue, 2),
        "f_pvalue": model.f_pvalue,
        "bp_pvalue": bp_p,
        "jb_pvalue": jb_p,
        "n_influential": n_influential,
        "vif": vif_df,
        "beta_weights": beta_weights,
        "fig_path": fig_path,
    }


def run_negbin(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    """Run Negative Binomial regression for count target."""
    # Check for overdispersion first
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    pearson_chi2 = poisson_model.pearson_chi2
    dispersion = pearson_chi2 / poisson_model.df_resid

    # Fit Negative Binomial
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use statsmodels NegativeBinomial with loglink
        nb_model = NegativeBinomial(y, X).fit(disp=0, maxiter=100)

    # Pseudo R² (McFadden, 1974)
    null_model = sm.GLM(y, sm.add_constant(np.ones(len(y))), family=sm.families.NegativeBinomial()).fit()
    pseudo_r2 = 1 - (nb_model.llf / null_model.llf) if null_model.llf != 0 else np.nan

    # VIF
    vif_df = compute_vif(X)

    # Incidence Rate Ratios
    irr = np.exp(nb_model.params)

    return {
        "model_name": model_name,
        "model": nb_model,
        "poisson_dispersion": round(dispersion, 3),
        "n": int(nb_model.nobs),
        "pseudo_r2": round(pseudo_r2, 4) if not np.isnan(pseudo_r2) else None,
        "aic": round(nb_model.aic, 1),
        "bic": round(nb_model.bic, 1) if hasattr(nb_model, "bic") else None,
        "vif": vif_df,
        "irr": irr,
    }


def generate_report(ols_results: list[dict], nb_result: dict) -> str:
    """Generate markdown report."""
    md = []
    md.append("# Regression Analysis Report\n\n")
    md.append("## Overview\n\n")
    md.append("Three regression models examining predictors of French deeptech startup funding:\n\n")
    md.append("| Model | Target | Method | N | Fit |\n")
    md.append("|-------|--------|--------|---|-----|\n")

    for r in ols_results:
        md.append(f"| {r['model_name']} | see below | OLS (HC3) | {r['n']} | "
                  f"R²={r['r_squared']}, Adj.R²={r['adj_r_squared']} |\n")
    md.append(f"| {nb_result['model_name']} | target_rounds | Neg. Binomial | {nb_result['n']} | "
              f"Pseudo-R²={nb_result['pseudo_r2']} |\n")

    md.append("\n## Predictors\n\n")
    md.append("Selected based on bivariate analysis results and theoretical relevance.\n")
    md.append("**Excluded:** `fundoing_new` and `FOUNDERS FOUNDED COMPANIES TOTAL FUNDING` "
              "(endogenous / near-perfect correlation with target = data leakage).\n\n")
    md.append("| Type | Features |\n|------|----------|\n")
    md.append(f"| Numeric | {', '.join(NUMERIC_PREDICTORS)} |\n")
    md.append(f"| Binary | {', '.join(BINARY_PREDICTORS)} |\n")
    md.append(f"| Categorical (dummies) | {', '.join(CATEGORICAL_PREDICTORS)} |\n")

    # OLS results
    for r in ols_results:
        md.append(f"\n## Model: {r['model_name']}\n\n")
        md.append(f"**N = {r['n']}** | R² = {r['r_squared']} | Adj. R² = {r['adj_r_squared']} | "
                  f"F = {r['f_stat']} (p = {r['f_pvalue']:.4g})\n\n")

        md.append("### Coefficients (HC3 robust SEs)\n\n")
        md.append("| Feature | Coef | SE | t | p | 95% CI | β (std) |\n")
        md.append("|---------|------|----|---|---|--------|--------|\n")
        summary = r["model"].summary2().tables[1]
        # HC3 uses z-stats; column name may be P>|z| or P>|t|
        p_col = "P>|z|" if "P>|z|" in summary.columns else "P>|t|"
        t_col = "z" if "z" in summary.columns else "t"
        for feat in summary.index:
            row = summary.loc[feat]
            beta = r["beta_weights"].get(feat, "—")
            beta_str = f"{beta:.3f}" if isinstance(beta, float) else beta
            sig = "**" if row[p_col] < 0.05 else ""
            md.append(f"| {sig}{feat}{sig} | {row['Coef.']:.4f} | {row['Std.Err.']:.4f} | "
                      f"{row[t_col]:.2f} | {row[p_col]:.4f} | [{row['[0.025']:.3f}, {row['0.975]']:.3f}] | "
                      f"{beta_str} |\n")

        md.append("\n### Diagnostics\n\n")
        md.append(f"- **Breusch-Pagan (heteroscedasticity):** p = {r['bp_pvalue']:.4f} "
                  f"{'⚠️ Significant' if r['bp_pvalue'] < 0.05 else '✅ Not significant'}\n")
        md.append(f"- **Jarque-Bera (normality of residuals):** p = {r['jb_pvalue']:.4f} "
                  f"{'⚠️ Non-normal' if r['jb_pvalue'] < 0.05 else '✅ Approx. normal'}\n")
        md.append(f"- **Influential observations (Cook's d > 4/n):** {r['n_influential']}\n")

        md.append("\n### VIF (Multicollinearity)\n\n")
        md.append("| Feature | VIF |\n|---------|-----|\n")
        for _, vrow in r["vif"].iterrows():
            flag = " ⚠️" if vrow["VIF"] > 5 else ""
            md.append(f"| {vrow['feature']} | {vrow['VIF']}{flag} |\n")

    # Negative Binomial
    md.append(f"\n## Model: {nb_result['model_name']}\n\n")
    md.append(f"**N = {nb_result['n']}** | Pseudo-R² (McFadden) = {nb_result['pseudo_r2']} | "
              f"AIC = {nb_result['aic']}\n\n")
    md.append(f"**Poisson dispersion test:** {nb_result['poisson_dispersion']:.3f} "
              f"({'overdispersed → NB appropriate' if nb_result['poisson_dispersion'] > 1.5 else 'mild overdispersion'})\n\n")

    md.append("### Coefficients & Incidence Rate Ratios (IRR)\n\n")
    md.append("| Feature | Coef | SE | z | p | IRR |\n")
    md.append("|---------|------|----|---|---|-----|\n")
    summary_nb = nb_result["model"].summary2().tables[1]
    for feat in summary_nb.index:
        row = summary_nb.loc[feat]
        irr_val = nb_result["irr"].get(feat, np.nan)
        sig = "**" if row["P>|z|"] < 0.05 else ""
        md.append(f"| {sig}{feat}{sig} | {row['Coef.']:.4f} | {row['Std.Err.']:.4f} | "
                  f"{row['z']:.2f} | {row['P>|z|']:.4f} | {irr_val:.3f} |\n")

    md.append("\n### VIF\n\n")
    md.append("| Feature | VIF |\n|---------|-----|\n")
    for _, vrow in nb_result["vif"].iterrows():
        flag = " ⚠️" if vrow["VIF"] > 5 else ""
        md.append(f"| {vrow['feature']} | {vrow['VIF']}{flag} |\n")

    # Interpretation
    md.append("\n## Key Findings\n\n")
    md.append("*(Auto-generated from significant coefficients at α = 0.05)*\n\n")

    for r in ols_results:
        summary = r["model"].summary2().tables[1]
        p_col = "P>|z|" if "P>|z|" in summary.columns else "P>|t|"
        sig_feats = summary[summary[p_col] < 0.05].index.tolist()
        sig_feats = [f for f in sig_feats if f != "const"]
        if sig_feats:
            md.append(f"**{r['model_name']}:** {', '.join(sig_feats)}\n\n")
        else:
            md.append(f"**{r['model_name']}:** No individually significant predictors at α = 0.05\n\n")

    summary_nb = nb_result["model"].summary2().tables[1]
    sig_nb = summary_nb[summary_nb["P>|z|"] < 0.05].index.tolist()
    sig_nb = [f for f in sig_nb if f != "const"]
    if sig_nb:
        md.append(f"**{nb_result['model_name']}:** {', '.join(sig_nb)}\n\n")

    md.append("\n## Limitations\n\n")
    md.append("- Sample size is modest after listwise deletion (high missingness on categoricals)\n")
    md.append("- Cross-sectional design — no causal claims possible\n")
    md.append("- Omitted variable bias likely (industry effects partially captured via CEO profile)\n")
    md.append("- Log transforms improve normality but complicate coefficient interpretation\n")
    md.append("- HC3 SEs are robust to heteroscedasticity but assume correct functional form\n")

    return "".join(md)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare()
    print(f"Loaded {len(df)} rows")

    ols_results = []

    # Model 1: log(total funding)
    print("\n--- Model 1: log(Total Funding) ---")
    X1, y1 = build_design_matrix(df, "log_total_funding")
    print(f"Design matrix: {X1.shape[0]} obs × {X1.shape[1]} features")
    r1 = run_ols(X1, y1, "Log Total Funding")
    print(r1["model"].summary())
    ols_results.append(r1)

    # Model 2: log(delta 1st round + 1)
    print("\n--- Model 2: log(Time to First Round + 1) ---")
    X2, y2 = build_design_matrix(df, "log_delta_1st")
    print(f"Design matrix: {X2.shape[0]} obs × {X2.shape[1]} features")
    r2 = run_ols(X2, y2, "Log Time to First Round")
    print(r2["model"].summary())
    ols_results.append(r2)

    # Model 3: Negative Binomial for rounds
    print("\n--- Model 3: Negative Binomial (Rounds) ---")
    X3, y3 = build_design_matrix(df, "target_rounds")
    print(f"Design matrix: {X3.shape[0]} obs × {X3.shape[1]} features")
    r3 = run_negbin(X3, y3, "Funding Rounds (NegBin)")
    print(r3["model"].summary())

    # Generate report
    report = generate_report(ols_results, r3)
    report_path = OUT_DIR / "regression_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # Save coefficient tables as CSV
    for r in ols_results:
        tbl = r["model"].summary2().tables[1]
        tbl.to_csv(OUT_DIR / f"regression_coef_{r['model_name'].lower().replace(' ', '_')}.csv")

    tbl_nb = r3["model"].summary2().tables[1]
    nb_slug = r3["model_name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
    tbl_nb.to_csv(OUT_DIR / f"regression_coef_{nb_slug}.csv")

    print("Done.")


if __name__ == "__main__":
    main()
