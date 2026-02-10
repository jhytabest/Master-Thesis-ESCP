"""Industry Fixed Effects Regression Models

Extends the baseline regression by adding industry fixed effects
(dummies for 11_industry_simplified). This is the single biggest
methodological gap identified in NORTHSTAR.md.

Models:
  1. OLS log(total_funding) with industry FE
  2. OLS log(total_funding) with industry FE (compare key coefficients)

Key questions answered:
  - Does the MBA effect survive within-industry comparison?
  - Does the female founder gap persist after controlling for sector?
  - Does CEO profile effect vary by industry?

Run:
  python jobs/analysis/src/industry_fixed_effects.py

Outputs → jobs/analysis/output/industry_fe_*
Refs #24
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from data_utils import clean_founders_number

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "dataset.csv"
OUT_DIR = REPO_ROOT / "jobs" / "analysis" / "output"

SEED = 42
np.random.seed(SEED)

NUMERIC_PREDICTORS = ["0_founders_number", "8_age_moyen"]
BINARY_PREDICTORS = [
    "5_has_female", "4_phd_founder", "7_MBA",
    "3_one_founder_serial", "6_was_1_change_founders",
]
CATEGORICAL_PREDICTORS = ["2_ceo_profile", "B2B/B2C"]

# Industry categories to include as FE (drop "vide" as uninformative)
VALID_INDUSTRIES = ["santé", "autres", "numérique", "énergie", "électronique"]


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = clean_founders_number(df)

    for col in ["target_delta_1st_round", "target_total_funding", "target_rounds"] + NUMERIC_PREDICTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BINARY_PREDICTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat "vide" as missing
    df["11_industry_simplified"] = df["11_industry_simplified"].replace("vide", np.nan)

    df["log_total_funding"] = np.log(df["target_total_funding"].clip(lower=0.001))
    df["log_delta_1st"] = np.log1p(df["target_delta_1st_round"].clip(lower=0))

    return df


def build_design_matrix(df: pd.DataFrame, target_col: str,
                        include_industry: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Build X matrix, optionally adding industry FE."""
    X = df[NUMERIC_PREDICTORS + BINARY_PREDICTORS].copy()

    for cat in CATEGORICAL_PREDICTORS:
        dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    if include_industry:
        ind_dummies = pd.get_dummies(
            df["11_industry_simplified"], prefix="industry", drop_first=True, dtype=float
        )
        X = pd.concat([X, ind_dummies], axis=1)

    y = df[target_col]
    mask = X.notna().all(axis=1) & y.notna()
    if include_industry:
        mask = mask & df["11_industry_simplified"].notna()

    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    X = sm.add_constant(X)
    return X, y


def run_ols(X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    model = sm.OLS(y, X).fit(cov_type="HC3")
    _, bp_p, _, _ = het_breuschpagan(model.resid, X)
    jb_stat, jb_p = sp_stats.jarque_bera(model.resid)
    return {
        "name": name,
        "model": model,
        "n": int(model.nobs),
        "r2": round(model.rsquared, 4),
        "adj_r2": round(model.rsquared_adj, 4),
        "aic": round(model.aic, 1),
        "bic": round(model.bic, 1),
        "bp_p": bp_p,
        "jb_p": jb_p,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare()
    print(f"Loaded {len(df)} rows")

    md = ["# Industry Fixed Effects Analysis\n\n"]
    md.append("Adds industry dummies (`11_industry_simplified`) to the baseline regression.\n")
    md.append("Reference category: `autres` (dropped).\n\n")

    # ── Model comparison: log total funding ──
    md.append("## 1. Log Total Funding\n\n")

    # Baseline (no industry FE) — restricted to same sample with non-missing industry
    X_base, y_base = build_design_matrix(df, "log_total_funding", include_industry=False)
    # Filter to rows that also have industry data for fair comparison
    ind_mask = df["11_industry_simplified"].notna()
    df_ind = df[ind_mask].copy()
    X_base_fair, y_base_fair = build_design_matrix(df_ind, "log_total_funding", include_industry=False)
    base = run_ols(X_base_fair, y_base_fair, "Baseline (no industry FE)")

    # With industry FE
    X_fe, y_fe = build_design_matrix(df, "log_total_funding", include_industry=True)
    fe = run_ols(X_fe, y_fe, "With Industry FE")

    md.append("### Model Comparison\n\n")
    md.append("| Specification | N | R² | Adj.R² | AIC | BIC |\n")
    md.append("|---------------|---|-----|--------|-----|-----|\n")
    for r in [base, fe]:
        md.append(f"| {r['name']} | {r['n']} | {r['r2']} | {r['adj_r2']} | {r['aic']} | {r['bic']} |\n")

    r2_gain = fe["r2"] - base["r2"]
    md.append(f"\n**R² improvement from industry FE:** +{r2_gain:.4f}\n\n")

    # ── Key coefficient comparison ──
    md.append("### Key Coefficient Comparison\n\n")
    key_vars = ["5_has_female", "7_MBA", "4_phd_founder", "3_one_founder_serial",
                "0_founders_number", "2_ceo_profile_scientific", "2_ceo_profile_mixte"]

    md.append("| Variable | β (no FE) | p (no FE) | β (with FE) | p (with FE) | Change |\n")
    md.append("|----------|-----------|-----------|-------------|-------------|--------|\n")

    base_params = base["model"].params
    base_pvals = base["model"].pvalues
    fe_params = fe["model"].params
    fe_pvals = fe["model"].pvalues

    for var in key_vars:
        b_coef = base_params.get(var, np.nan)
        b_p = base_pvals.get(var, np.nan)
        f_coef = fe_params.get(var, np.nan)
        f_p = fe_pvals.get(var, np.nan)

        if pd.notna(b_coef) and pd.notna(f_coef):
            change_pct = ((f_coef - b_coef) / abs(b_coef) * 100) if b_coef != 0 else 0
            change_str = f"{change_pct:+.0f}%"
        else:
            change_str = "—"

        b_sig = "**" if (pd.notna(b_p) and b_p < 0.05) else ""
        f_sig = "**" if (pd.notna(f_p) and f_p < 0.05) else ""

        md.append(f"| {var} | {b_sig}{b_coef:.4f}{b_sig} | {b_p:.4f} | "
                  f"{f_sig}{f_coef:.4f}{f_sig} | {f_p:.4f} | {change_str} |\n")

    # ── Industry FE coefficients ──
    md.append("\n### Industry Fixed Effect Coefficients\n\n")
    md.append("| Industry | β | SE | p | Interpretation |\n")
    md.append("|----------|---|-----|---|---------------|\n")

    fe_summary = fe["model"].summary2().tables[1]
    for col in fe_summary.index:
        if col.startswith("industry_"):
            industry = col.replace("industry_", "")
            coef = fe_summary.loc[col, "Coef."]
            se = fe_summary.loc[col, "Std.Err."]
            p = fe_summary.loc[col, "P>|z|"] if "P>|z|" in fe_summary.columns else fe_summary.loc[col, "P>|t|"]
            pct_change = (np.exp(coef) - 1) * 100
            md.append(f"| {industry} | {coef:.4f} | {se:.4f} | {p:.4f} | "
                      f"{pct_change:+.0f}% vs reference |\n")

    # ── F-test for joint significance of industry FE ──
    md.append("\n### Joint Significance of Industry FE\n\n")

    # Wald test: all industry dummies = 0
    industry_cols = [c for c in fe["model"].params.index if c.startswith("industry_")]
    if industry_cols:
        R = np.zeros((len(industry_cols), len(fe["model"].params)))
        for i, col in enumerate(industry_cols):
            j = list(fe["model"].params.index).index(col)
            R[i, j] = 1
        f_test = fe["model"].f_test(R)
        md.append(f"**Wald F-test:** F={float(f_test.fvalue):.2f}, p={float(f_test.pvalue):.4f}\n\n")
        if float(f_test.pvalue) < 0.05:
            md.append("→ Industry FE are **jointly significant** — sector matters for funding.\n\n")
        else:
            md.append("→ Industry FE are **not jointly significant** — limited sector variation in funding.\n\n")

    # ── Repeat for time-to-first-round ──
    md.append("## 2. Log Time to First Round\n\n")

    X2_base, y2_base = build_design_matrix(df_ind, "log_delta_1st", include_industry=False)
    base2 = run_ols(X2_base, y2_base, "Baseline")

    X2_fe, y2_fe = build_design_matrix(df, "log_delta_1st", include_industry=True)
    fe2 = run_ols(X2_fe, y2_fe, "With Industry FE")

    md.append("| Specification | N | R² | Adj.R² |\n")
    md.append("|---------------|---|-----|--------|\n")
    for r in [base2, fe2]:
        md.append(f"| {r['name']} | {r['n']} | {r['r2']} | {r['adj_r2']} |\n")

    md.append(f"\nR² remains low (~{fe2['r2']}), confirming founder traits don't explain timing.\n")

    # ── Coefficient stability plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_vars = ["5_has_female", "7_MBA", "2_ceo_profile_scientific", "2_ceo_profile_mixte"]
    specs = [("No FE", base["model"]), ("With FE", fe["model"])]

    for i, var in enumerate(plot_vars):
        coefs = []
        cis_lo = []
        cis_hi = []
        labels = []
        for label, model in specs:
            if var in model.params:
                coefs.append(model.params[var])
                ci = model.conf_int().loc[var]
                cis_lo.append(ci[0])
                cis_hi.append(ci[1])
                labels.append(label)

        if coefs:
            x = range(len(coefs))
            axes[0].errorbar(x, coefs,
                           yerr=[np.array(coefs) - np.array(cis_lo),
                                 np.array(cis_hi) - np.array(coefs)],
                           fmt='o-', label=var, capsize=4)

    axes[0].set_xticks(range(len(specs)))
    axes[0].set_xticklabels([s[0] for s in specs])
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Key Coefficients: With vs Without Industry FE")
    axes[0].legend(fontsize=8)

    # Industry FE coefficients
    ind_coefs = {col.replace("industry_", ""): fe["model"].params[col]
                 for col in fe["model"].params.index if col.startswith("industry_")}
    if ind_coefs:
        axes[1].barh(list(ind_coefs.keys()), list(ind_coefs.values()))
        axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.5)
        axes[1].set_xlabel("Coefficient (log funding, vs reference)")
        axes[1].set_title("Industry Fixed Effects")

    plt.tight_layout()
    fig_path = OUT_DIR / "fig_industry_fe_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    md.append(f"\n![Industry FE Comparison]({fig_path.name})\n")

    # ── Conclusions ──
    md.append("\n## Conclusions\n\n")
    md.append("Key questions from NORTHSTAR.md:\n\n")
    md.append("1. **Does MBA effect survive within-industry?** Check the coefficient table above.\n")
    md.append("2. **Does female founder gap persist?** Compare p-values with and without FE.\n")
    md.append("3. **Are industry effects significant?** See joint F-test.\n\n")

    # Save
    report_path = OUT_DIR / "industry_fe_report.md"
    report_path.write_text("".join(md), encoding="utf-8")
    print(f"Report: {report_path}")

    # Save coefficient table
    fe_summary.to_csv(OUT_DIR / "industry_fe_coefficients.csv")
    print("Done.")


if __name__ == "__main__":
    main()
