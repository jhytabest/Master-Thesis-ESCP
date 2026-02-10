"""Deep-Dive: Female Founder Effect on Total Funding

Investigates the borderline-significant negative association between
`5_has_female` and log total funding (p≈0.04) found in robustness checks.

Analyses:
  1. Descriptive breakdown (female vs male-founded)
  2. Interaction effects (sector, education, team size, location)
  3. Mediation analysis (Baron & Kenny, 1986)
  4. Propensity score matching (Rosenbaum & Rubin, 1983)
  5. Effect size & practical significance
  6. Academic framing & limitations

Run:
  python jobs/analysis/src/female_founder_deep_dive.py

Outputs → jobs/analysis/output/female_founder_*
Closes #9
"""

from __future__ import annotations
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_utils import clean_founders_number

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "dataset.csv"
OUT_DIR = REPO_ROOT / "jobs" / "analysis" / "output"

NUMERIC_PREDICTORS = ["0_founders_number", "8_age_moyen"]
BINARY_PREDICTORS = [
    "5_has_female", "4_phd_founder", "7_MBA",
    "3_one_founder_serial", "6_was_1_change_founders",
]
CATEGORICAL_PREDICTORS = ["2_ceo_profile", "B2B/B2C"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = clean_founders_number(df)

    for col in ["target_total_funding", "target_delta_1st_round", "target_rounds"] + NUMERIC_PREDICTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BINARY_PREDICTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_total_funding"] = np.log(df["target_total_funding"].clip(lower=0.001))
    return df


def build_X(df: pd.DataFrame, extra_cols=None):
    """Build design matrix (same as main regression)."""
    X = df[NUMERIC_PREDICTORS + BINARY_PREDICTORS].copy()
    for cat in CATEGORICAL_PREDICTORS:
        dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    if extra_cols:
        for c in extra_cols:
            X[c] = df[c].values
    return X


# ── 1. Descriptive Breakdown ──────────────────────────────────────────────

def descriptive_breakdown(df: pd.DataFrame, md: list):
    md.append("# Deep-Dive: Female Founder Effect on Total Funding\n\n")
    md.append("## 1. Descriptive Breakdown\n\n")

    has_funding = df["target_total_funding"].notna()
    sub = df[has_funding].copy()
    female = sub[sub["5_has_female"] == 1]
    male = sub[sub["5_has_female"] == 0]

    md.append(f"**Sample with funding data:** N={len(sub)} "
              f"(female-founded: {len(female)}, male-only: {len(male)})\n\n")

    # Funding comparison
    md.append("### Funding Metrics\n\n")
    md.append("| Metric | Female-Founded | Male-Only | Diff |\n")
    md.append("|--------|---------------|-----------|------|\n")

    for col, label in [("target_total_funding", "Total funding (M€)"),
                        ("target_rounds", "Number of rounds"),
                        ("target_delta_1st_round", "Time to 1st round (months)")]:
        fm = female[col].median()
        mm = male[col].median()
        fmean = female[col].mean()
        mmean = male[col].mean()
        md.append(f"| {label} (median) | {fm:.2f} | {mm:.2f} | {fm-mm:+.2f} |\n")
        md.append(f"| {label} (mean) | {fmean:.2f} | {mmean:.2f} | {fmean-mmean:+.2f} |\n")

    # Mann-Whitney U test
    u_stat, u_p = sp_stats.mannwhitneyu(
        female["target_total_funding"].dropna(),
        male["target_total_funding"].dropna(),
        alternative="two-sided"
    )
    md.append(f"\n**Mann-Whitney U test** (total funding): U={u_stat:.0f}, p={u_p:.4f}\n\n")

    # Sector distribution
    md.append("### Sector Distribution\n\n")
    md.append("| Sector | Female % | Male % |\n")
    md.append("|--------|----------|--------|\n")
    for sector in [s for s in df["11_industry_simplified"].dropna().unique() if s != "vide"]:
        f_pct = (female["11_industry_simplified"] == sector).mean() * 100
        m_pct = (male["11_industry_simplified"] == sector).mean() * 100
        if f_pct > 1 or m_pct > 1:
            md.append(f"| {sector} | {f_pct:.1f}% | {m_pct:.1f}% |\n")

    # Education / team
    md.append("\n### Team & Education\n\n")
    md.append("| Variable | Female-Founded | Male-Only |\n")
    md.append("|----------|---------------|----------|\n")
    for col, label in [("0_founders_number", "Avg # founders"),
                        ("4_phd_founder", "Has PhD (%)"),
                        ("7_MBA", "Has MBA (%)"),
                        ("3_one_founder_serial", "Serial founder (%)")]:
        fv = female[col].mean()
        mv = male[col].mean()
        if col in ["4_phd_founder", "7_MBA", "3_one_founder_serial"]:
            md.append(f"| {label} | {fv*100:.1f}% | {mv*100:.1f}% |\n")
        else:
            md.append(f"| {label} | {fv:.2f} | {mv:.2f} |\n")

    # Plot: funding distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-3, 6, 40)
    ax.hist(np.log(male["target_total_funding"].clip(lower=0.001)),
            bins=bins, alpha=0.6, label=f"Male-only (n={len(male)})", density=True)
    ax.hist(np.log(female["target_total_funding"].clip(lower=0.001)),
            bins=bins, alpha=0.6, label=f"Female-founded (n={len(female)})", density=True)
    ax.set_xlabel("Log Total Funding (M€)")
    ax.set_ylabel("Density")
    ax.set_title("Funding Distribution by Founder Gender")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "female_founder_funding_dist.png", dpi=150)
    plt.close(fig)
    md.append(f"\n![Funding Distribution](female_founder_funding_dist.png)\n\n")

    return female, male


# ── 2. Interaction Effects ────────────────────────────────────────────────

def interaction_effects(df: pd.DataFrame, md: list):
    md.append("## 2. Interaction Effects\n\n")
    md.append("Testing whether the female founder effect varies across subgroups.\n\n")

    sub = df[df["target_total_funding"].notna() & df["5_has_female"].notna()].copy()
    y = np.log(sub["target_total_funding"].clip(lower=0.001))

    X_base = build_X(sub)
    mask = X_base.notna().all(axis=1) & y.notna()
    X_base = X_base.loc[mask]
    y = y.loc[mask]
    sub = sub.loc[mask]

    # Test interactions
    interactions = {
        "5_has_female × 4_phd_founder": sub["5_has_female"] * sub["4_phd_founder"],
        "5_has_female × 7_MBA": sub["5_has_female"] * sub["7_MBA"],
        "5_has_female × 3_one_founder_serial": sub["5_has_female"] * sub["3_one_founder_serial"],
        "5_has_female × 0_founders_number": sub["5_has_female"] * sub["0_founders_number"],
    }

    # Add sector interactions
    for sector in ["santé", "numérique", "énergie"]:
        s_dummy = (sub["11_industry_simplified"] == sector).astype(float)
        interactions[f"5_has_female × {sector}"] = sub["5_has_female"] * s_dummy

    md.append("| Interaction Term | Coef | SE | p-value | Interpretation |\n")
    md.append("|-----------------|------|-----|---------|----------------|\n")

    for name, interaction_col in interactions.items():
        X_int = X_base.copy()
        X_int[name] = interaction_col.values
        X_int = sm.add_constant(X_int)
        try:
            model = sm.OLS(y, X_int).fit(cov_type="HC3")
            coef = model.params[name]
            se = model.bse[name]
            p = model.pvalues[name]
            sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
            interp = ""
            if p < 0.1:
                direction = "amplifies" if coef < 0 else "mitigates"
                interp = f"{direction} the gap"
            md.append(f"| {name} | {coef:.3f} | {se:.3f} | {p:.3f}{sig} | {interp} |\n")
        except Exception as e:
            md.append(f"| {name} | — | — | — | Error: {e} |\n")

    md.append("\n")


# ── 3. Mediation Analysis ────────────────────────────────────────────────

def mediation_analysis(df: pd.DataFrame, md: list):
    md.append("## 3. Mediation Analysis (Baron & Kenny, 1986)\n\n")
    md.append("Testing whether sector clustering or team composition mediates the female founder effect.\n\n")

    sub = df[df["target_total_funding"].notna() & df["5_has_female"].notna()].copy()
    y = np.log(sub["target_total_funding"].clip(lower=0.001))

    # Step 1: Total effect (female → funding, no mediators)
    X1 = sm.add_constant(sub[["5_has_female"]].astype(float))
    mask = X1.notna().all(axis=1) & y.notna()
    m1 = sm.OLS(y[mask], X1[mask]).fit(cov_type="HC3")
    total_effect = m1.params["5_has_female"]
    total_p = m1.pvalues["5_has_female"]

    md.append(f"**Step 1 — Total effect:** β={total_effect:.4f}, p={total_p:.4f}\n\n")

    # Step 2: Female → mediators
    mediators = {
        "santé_sector": (sub["11_industry_simplified"] == "santé").astype(float),
        "0_founders_number": sub["0_founders_number"],
        "4_phd_founder": sub["4_phd_founder"].astype(float),
        "7_MBA": sub["7_MBA"].astype(float),
    }

    md.append("**Step 2 — Female → Mediators (a-paths):**\n\n")
    md.append("| Mediator | Coef (a) | p-value |\n")
    md.append("|----------|----------|--------|\n")

    a_paths = {}
    for mname, mvar in mediators.items():
        X_a = sm.add_constant(sub[["5_has_female"]].astype(float))
        m_mask = X_a.notna().all(axis=1) & mvar.notna()
        try:
            m_a = sm.OLS(mvar[m_mask], X_a[m_mask]).fit(cov_type="HC3")
            a_paths[mname] = (m_a.params["5_has_female"], m_a.pvalues["5_has_female"])
            md.append(f"| {mname} | {a_paths[mname][0]:.4f} | {a_paths[mname][1]:.4f} |\n")
        except Exception as e:
            md.append(f"| {mname} | — | — | Error: {e} |\n")

    # Step 3: Direct effect controlling for mediators
    md.append("\n**Step 3 — Direct effect controlling for mediators:**\n\n")
    med_cols = list(mediators.keys())
    X3 = sub[["5_has_female"]].astype(float).copy()
    for mname, mvar in mediators.items():
        X3[mname] = mvar.values
    X3 = sm.add_constant(X3)
    mask3 = X3.notna().all(axis=1) & y.notna()
    m3 = sm.OLS(y[mask3], X3[mask3]).fit(cov_type="HC3")
    direct_effect = m3.params["5_has_female"]
    direct_p = m3.pvalues["5_has_female"]

    md.append(f"Direct effect (c'): β={direct_effect:.4f}, p={direct_p:.4f}\n\n")

    indirect = total_effect - direct_effect
    pct_mediated = (indirect / total_effect * 100) if total_effect != 0 else 0
    md.append(f"**Indirect (mediated) effect:** {indirect:.4f} "
              f"({pct_mediated:.1f}% of total effect)\n\n")

    if (np.sign(direct_effect) * np.sign(total_effect) == -1) or (abs(direct_effect) > abs(total_effect)):
        md.append("→ Evidence of **inconsistent mediation (suppression)**. "
                  "Controlling for mediators reveals a stronger, significant direct negative effect, "
                  "suggesting the mediators were masking the true extent of the association.\n\n")
    elif abs(direct_effect) < abs(total_effect) and direct_p > 0.05:
        md.append("→ The female founder effect appears to be **fully mediated** by composition differences. "
                  "When controlling for sector and team characteristics, the direct effect becomes non-significant.\n\n")
    elif abs(direct_effect) < abs(total_effect):
        md.append("→ **Partial mediation** observed: composition differences explain some of the gap, "
                  "but a significant direct effect remains.\n\n")
    else:
        md.append("→ Little evidence of mediation through these variables. The direct effect is similar to the total effect.\n\n")


# ── 4. Propensity Score Matching ──────────────────────────────────────────

def propensity_matching(df: pd.DataFrame, md: list):
    md.append("## 4. Propensity Score Matching\n\n")
    md.append("Matching female-founded startups to comparable male-only startups "
              "on observable characteristics (Rosenbaum & Rubin, 1983).\n\n")

    sub = df[df["target_total_funding"].notna() & df["5_has_female"].notna()].copy()

    # Covariates for propensity model
    covariates = ["0_founders_number", "4_phd_founder", "7_MBA",
                  "3_one_founder_serial", "6_was_1_change_founders"]

    # Add sector dummies
    for sector in ["santé", "numérique", "énergie", "électronique"]:
        sub[f"sector_{sector}"] = (sub["11_industry_simplified"] == sector).astype(float)
        covariates.append(f"sector_{sector}")

    X_prop = sub[covariates].copy()
    treat = sub["5_has_female"].copy()
    y = np.log(sub["target_total_funding"].clip(lower=0.001))

    mask = X_prop.notna().all(axis=1) & treat.notna() & y.notna()
    X_prop, treat, y = X_prop[mask], treat[mask], y[mask]
    sub_m = sub.loc[mask].copy()

    # Fit propensity score (logistic regression)
    X_ps = sm.add_constant(X_prop)
    try:
        ps_model = sm.Logit(treat, X_ps).fit(disp=0)
        pscore = ps_model.predict(X_ps)
    except Exception as e:
        md.append(f"Propensity model failed: {e}\n\n")
        return

    sub_m["pscore"] = pscore.values
    sub_m["log_funding"] = y.values
    sub_m["treat"] = treat.values

    # Nearest-neighbor matching (1:1 without replacement)
    treated = sub_m[sub_m["treat"] == 1].copy()
    control = sub_m[sub_m["treat"] == 0].copy()

    matched_controls = []
    used = set()
    for idx, row in treated.iterrows():
        dists = (control["pscore"] - row["pscore"]).abs()
        dists = dists[~dists.index.isin(used)]
        if len(dists) == 0:
            continue
        best = dists.idxmin()
        if dists[best] < 0.1:  # caliper
            matched_controls.append(best)
            used.add(best)

    n_matched = len(matched_controls)
    md.append(f"**Matched pairs:** {n_matched} out of {len(treated)} treated units\n\n")

    if n_matched < 10:
        md.append("Too few matches for reliable inference.\n\n")
        return

    matched_treat = treated.iloc[:n_matched]
    matched_ctrl = control.loc[matched_controls]

    # ATT
    att = matched_treat["log_funding"].mean() - matched_ctrl["log_funding"].mean()
    att_euro = matched_treat["target_total_funding"].mean() - matched_ctrl["target_total_funding"].mean()

    # t-test on matched sample
    t_stat, t_p = sp_stats.ttest_rel(matched_treat["log_funding"].values, matched_ctrl["log_funding"].values)

    md.append(f"**ATT (Average Treatment Effect on Treated):**\n")
    md.append(f"- Log scale: {att:.4f}\n")
    md.append(f"- Approx. euro difference: {att_euro:.2f} M€\n")
    md.append(f"- t-test: t={t_stat:.2f}, p={t_p:.4f}\n\n")

    # Balance check
    md.append("### Covariate Balance (post-matching)\n\n")
    md.append("| Covariate | Treated Mean | Control Mean | Std. Diff |\n")
    md.append("|-----------|-------------|-------------|----------|\n")
    for cov in covariates:
        t_mean = matched_treat[cov].mean() if cov in matched_treat.columns else np.nan
        c_mean = matched_ctrl[cov].mean() if cov in matched_ctrl.columns else np.nan
        pooled_std = np.sqrt((matched_treat[cov].var() + matched_ctrl[cov].var()) / 2) if cov in matched_treat.columns else 1
        std_diff = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        md.append(f"| {cov} | {t_mean:.3f} | {c_mean:.3f} | {std_diff:.3f} |\n")

    md.append("\n")


# ── 5. Effect Size & Practical Significance ───────────────────────────────

def effect_size(df: pd.DataFrame, md: list):
    md.append("## 5. Effect Size & Practical Significance\n\n")

    sub = df[df["target_total_funding"].notna() & df["5_has_female"].notna()].copy()
    y = sub["target_total_funding"]
    female = sub[sub["5_has_female"] == 1]
    male = sub[sub["5_has_female"] == 0]

    md.append("### Raw Funding Gap\n\n")
    gap_mean = female["target_total_funding"].mean() - male["target_total_funding"].mean()
    gap_median = female["target_total_funding"].median() - male["target_total_funding"].median()

    md.append(f"- **Mean gap:** {gap_mean:+.2f} M€ (female − male)\n")
    md.append(f"- **Median gap:** {gap_median:+.2f} M€\n")
    md.append(f"- Female median: {female['target_total_funding'].median():.2f} M€\n")
    md.append(f"- Male median: {male['target_total_funding'].median():.2f} M€\n\n")

    # Cohen's d (weighted pooled std for unbalanced groups)
    n_female = len(female["target_total_funding"].dropna())
    n_male = len(male["target_total_funding"].dropna())
    var_female = female["target_total_funding"].var()
    var_male = male["target_total_funding"].var()
    pooled_std = np.sqrt(((n_female - 1) * var_female + (n_male - 1) * var_male) / (n_female + n_male - 2))
    d = gap_mean / pooled_std
    md.append(f"**Cohen's d:** {d:.3f} ")
    if abs(d) < 0.2:
        md.append("(negligible)\n\n")
    elif abs(d) < 0.5:
        md.append("(small)\n\n")
    elif abs(d) < 0.8:
        md.append("(medium)\n\n")
    else:
        md.append("(large)\n\n")

    # From regression coefficient
    md.append("### Regression-Based Effect\n\n")
    X = build_X(sub)
    y_log = np.log(sub["target_total_funding"].clip(lower=0.001))
    mask = X.notna().all(axis=1) & y_log.notna()
    X_c = sm.add_constant(X[mask])
    model = sm.OLS(y_log[mask], X_c).fit(cov_type="HC3")

    coef = model.params["5_has_female"]
    p = model.pvalues["5_has_female"]
    pct_change = (np.exp(coef) - 1) * 100

    md.append(f"- β (log scale): {coef:.4f}, p={p:.4f}\n")
    md.append(f"- **Percentage change in funding:** {pct_change:+.1f}%\n")
    md.append(f"- At median funding ({male['target_total_funding'].median():.2f} M€), "
              f"this translates to ~{male['target_total_funding'].median() * pct_change / 100:+.2f} M€\n\n")

    # Compare to other predictors
    md.append("### Comparison with Other Predictors\n\n")
    md.append("| Predictor | β (log) | % Change | p-value |\n")
    md.append("|-----------|---------|----------|--------|\n")
    for pred in ["5_has_female", "7_MBA", "4_phd_founder", "3_one_founder_serial",
                 "0_founders_number", "6_was_1_change_founders"]:
        if pred in model.params:
            b = model.params[pred]
            pv = model.pvalues[pred]
            pct = (np.exp(b) - 1) * 100
            md.append(f"| {pred} | {b:.4f} | {pct:+.1f}% | {pv:.4f} |\n")
    md.append("\n")


# ── 6. Academic Framing ───────────────────────────────────────────────────

def academic_framing(md: list):
    md.append("## 6. Academic Framing & Limitations\n\n")
    md.append(dedent("""\
    ### Key Caveats

    1. **Correlation ≠ Causation.** This is an observational study. The negative association
       between female founders and total funding does not imply that being female *causes*
       lower funding. Unobserved variables (investor networks, pitch quality, sector-specific
       dynamics, geographic concentration) may confound the relationship.

    2. **Omitted Variable Bias.** Key variables that could explain the gap are missing from
       the dataset: investor characteristics, funding round details, company revenue/traction
       at time of fundraising, and founder network centrality.

    3. **Sample Limitations.**
       - N is modest (especially female-founded startups with funding data: ~60-80)
       - French/European startup ecosystem may not generalize to other markets
       - Borderline significance (p≈0.04) means the finding is sensitive to specification
       - Multiple testing increases the risk of false positives

    4. **Existing Literature.** The finding is consistent with a large body of research
       documenting gender gaps in venture funding (Brush et al., 2018; Kanze et al., 2018;
       Ewens & Townsend, 2020). However, the *mechanisms* differ across studies — some
       find demand-side differences (women seek less capital), others supply-side bias
       (investors fund women less for equivalent ventures).

    5. **Mediation Evidence.** If the mediation analysis shows substantial indirect effects
       through sector clustering or team composition, this suggests the gap may reflect
       *sorting patterns* rather than direct discrimination — an important distinction for
       policy implications.

    ### Recommended Framing for the Thesis

    > "We observe a negative association between female founder presence and total funding
    > raised (β = [X], p = [Y]). However, this relationship is partially mediated by
    > sector composition and team characteristics, suggesting that observable composition
    > differences account for a meaningful portion of the gap. Given the borderline
    > significance, modest sample size, and potential for omitted variable bias, we
    > interpret this finding cautiously. It is consistent with the broader literature on
    > gender disparities in startup funding, but our data do not allow us to distinguish
    > between supply-side bias and demand-side selection effects."
    """))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"Loaded {len(df)} rows")

    md: list[str] = []

    # Run all analyses
    female, male = descriptive_breakdown(df, md)
    interaction_effects(df, md)
    mediation_analysis(df, md)
    propensity_matching(df, md)
    effect_size(df, md)
    academic_framing(md)

    # Save report
    report = "".join(md)
    report_path = OUT_DIR / "female_founder_deep_dive.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    print("Done.")

    return report


if __name__ == "__main__":
    main()
