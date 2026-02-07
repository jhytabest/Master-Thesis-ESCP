"""Correlation & Bivariate Analysis

Examines relationships between features and the three targets:
  - target_delta_1st_round  (months to first funding round)
  - target_total_funding    (total funding in M€)
  - target_rounds           (number of funding rounds)

Methods:
  - Spearman rank correlation for numeric–numeric pairs (robust to skew; Hauke & Kossowski, 2011)
  - Point-biserial correlation for binary–numeric pairs (equivalent to Pearson on 0/1; Glass & Hopkins, 1996)
  - Kruskal-Wallis H-test for categorical–numeric pairs (non-parametric ANOVA; Kruskal & Wallis, 1952)
  - Effect sizes: r_s, r_pb, epsilon-squared (ε²) for Kruskal-Wallis

Academic notes:
  - Spearman chosen over Pearson because funding variables are heavily right-skewed
  - Multiple comparisons addressed with Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995)
  - Effect size thresholds (Cohen, 1988): small=0.1, medium=0.3, large=0.5

Run:
  python jobs/analysis/src/correlation_analysis.py

Outputs → jobs/analysis/output/correlation_*
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "dataset.csv"
OUT_DIR = REPO_ROOT / "jobs" / "analysis" / "output"

TARGETS = [
    "target_delta_1st_round",
    "target_total_funding",
    "target_rounds",
]

# Features we expect to be numeric or binary
NUMERIC_FEATURES = [
    "0_founders_number",
    "8_age_moyen",
    "FOUNDERS ESTIMATED AGE",
    "CEO ESTIMATED AGE",
    "FOUNDERS YEARS OF EDUCATION",
    "FOUNDERS FOUNDED COMPANIES TOTAL FUNDING",
    "NUMBER OF ALUMNI FOUNDERS THAT RAISED > 10M",
    "NUMBER OF ALUMNI EUROPEAN FOUNDERS THAT RAISED > 10M",
    "fundoing_new",
]

BINARY_FEATURES = [
    "5_has_female",
    "5_bis_mixte?",
    "4_phd_founder",
    "7_MBA",
    "3_one_founder_serial",
    "3_one_founder_was_serial_at_creation",
    "6_was_1_change_founders",
]

CATEGORICAL_FEATURES = [
    "B2B/B2C",
    "REVENUE MODEL",
    "11_industry_simplified",
    "DELIVERY METHOD",
    "GROWTH STAGE",
    "2_ceo_profile",
]

MIN_N = 20  # minimum observations for a test


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    ranked = pvals.rank(method="first")
    adjusted = pvals * n / ranked
    # Enforce monotonicity (step-down)
    adjusted = adjusted.clip(upper=1.0)
    sorted_idx = pvals.sort_values(ascending=False).index
    cummin = adjusted.loc[sorted_idx].cummin()
    return cummin.reindex(pvals.index)


def spearman_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between numeric features and targets."""
    rows = []
    for feat in NUMERIC_FEATURES:
        if feat not in df.columns:
            continue
        x = safe_numeric(df[feat])
        for target in TARGETS:
            y = safe_numeric(df[target])
            mask = x.notna() & y.notna()
            n = mask.sum()
            if n < MIN_N:
                continue
            r_s, p = stats.spearmanr(x[mask], y[mask])
            rows.append({
                "feature": feat,
                "target": target,
                "test": "spearman",
                "n": int(n),
                "statistic": round(r_s, 4),
                "p_value": p,
                "effect_size": abs(round(r_s, 4)),
                "effect_label": _cohen_r_label(abs(r_s)),
            })
    return pd.DataFrame(rows)


def pointbiserial_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial correlations between binary features and targets."""
    rows = []
    for feat in BINARY_FEATURES:
        if feat not in df.columns:
            continue
        x = safe_numeric(df[feat])
        # Ensure truly binary
        unique = x.dropna().unique()
        if not set(unique).issubset({0, 1, 0.0, 1.0}):
            continue
        for target in TARGETS:
            y = safe_numeric(df[target])
            mask = x.notna() & y.notna()
            n = mask.sum()
            if n < MIN_N:
                continue
            r_pb, p = stats.pointbiserialr(x[mask], y[mask])
            rows.append({
                "feature": feat,
                "target": target,
                "test": "point_biserial",
                "n": int(n),
                "statistic": round(r_pb, 4),
                "p_value": p,
                "effect_size": abs(round(r_pb, 4)),
                "effect_label": _cohen_r_label(abs(r_pb)),
            })
    return pd.DataFrame(rows)


def kruskal_wallis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis H-test for categorical features vs targets."""
    rows = []
    for feat in CATEGORICAL_FEATURES:
        if feat not in df.columns:
            continue
        for target in TARGETS:
            y = safe_numeric(df[target])
            # Group by category
            groups = []
            labels = []
            for cat, grp in df.groupby(feat):
                vals = y.loc[grp.index].dropna()
                if len(vals) >= 5:
                    groups.append(vals.values)
                    labels.append(str(cat))
            if len(groups) < 2:
                continue
            total_n = sum(len(g) for g in groups)
            H, p = stats.kruskal(*groups)
            # Epsilon-squared: ε² = H / (n-1)  (Tomczak & Tomczak, 2014)
            eps_sq = H / (total_n - 1) if total_n > 1 else 0
            rows.append({
                "feature": feat,
                "target": target,
                "test": "kruskal_wallis",
                "n": total_n,
                "n_groups": len(groups),
                "statistic": round(H, 4),
                "p_value": p,
                "effect_size": round(eps_sq, 4),
                "effect_label": _eps_sq_label(eps_sq),
                "groups": "; ".join(f"{l}(n={len(g)})" for l, g in zip(labels, groups)),
            })
    return pd.DataFrame(rows)


def _cohen_r_label(r: float) -> str:
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    return "large"


def _eps_sq_label(e: float) -> str:
    """Epsilon-squared thresholds (analogous to eta-squared)."""
    if e < 0.01:
        return "negligible"
    elif e < 0.06:
        return "small"
    elif e < 0.14:
        return "medium"
    return "large"


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Spearman correlation heatmap for numeric features + targets."""
    cols = [c for c in NUMERIC_FEATURES + TARGETS if c in df.columns]
    numeric_df = df[cols].apply(safe_numeric)
    corr = numeric_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Spearman Correlation Matrix\n(Numeric Features × Targets)", fontsize=13)
    plt.tight_layout()
    path = OUT_DIR / "fig_correlation_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_top_associations(results: pd.DataFrame) -> Path:
    """Bar chart of top associations by effect size."""
    top = results.nlargest(20, "effect_size").copy()
    top["label"] = top["feature"] + " → " + top["target"]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2ecc71" if p < 0.05 else "#95a5a6" for p in top["p_value_adj"]]
    ax.barh(range(len(top)), top["effect_size"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=9)
    ax.set_xlabel("Effect Size (|r| or ε²)")
    ax.set_title("Top 20 Feature–Target Associations\n(green = FDR-adjusted p < 0.05)")
    ax.invert_yaxis()
    plt.tight_layout()
    path = OUT_DIR / "fig_top_associations.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run all three test families
    spearman_df = spearman_correlations(df)
    pb_df = pointbiserial_correlations(df)
    kw_df = kruskal_wallis_tests(df)

    # Combine results
    all_results = pd.concat([spearman_df, pb_df, kw_df], ignore_index=True)

    if all_results.empty:
        print("No tests could be run (insufficient data).")
        return

    # FDR correction across all tests
    all_results["p_value_adj"] = benjamini_hochberg(all_results["p_value"])
    all_results["significant_fdr05"] = all_results["p_value_adj"] < 0.05

    # Sort by effect size
    all_results = all_results.sort_values("effect_size", ascending=False)

    # Save full results
    results_path = OUT_DIR / "correlation_results.csv"
    all_results.to_csv(results_path, index=False)
    print(f"Saved {len(all_results)} test results to {results_path}")

    # Plots
    heatmap_path = plot_correlation_heatmap(df)
    assoc_path = plot_top_associations(all_results)

    # Generate markdown report
    sig = all_results[all_results["significant_fdr05"]]
    md = []
    md.append("# Correlation & Bivariate Analysis\n\n")
    md.append(f"**Tests run:** {len(all_results)} | **Significant (FDR < 0.05):** {len(sig)}\n\n")

    md.append("## Methodology\n\n")
    md.append("- **Numeric features:** Spearman rank correlation (robust to non-normality)\n")
    md.append("- **Binary features:** Point-biserial correlation\n")
    md.append("- **Categorical features:** Kruskal-Wallis H-test with ε² effect size\n")
    md.append("- **Multiple comparisons:** Benjamini-Hochberg FDR correction (α = 0.05)\n\n")

    md.append("## Significant Associations (FDR < 0.05)\n\n")
    if sig.empty:
        md.append("*No statistically significant associations after FDR correction.*\n\n")
    else:
        for _, row in sig.iterrows():
            direction = ""
            if row["test"] in ("spearman", "point_biserial"):
                direction = " (+)" if row["statistic"] > 0 else " (−)"
            md.append(
                f"- **{row['feature']}** → {row['target']}: "
                f"{row['test']}, r={row['statistic']:.3f}{direction}, "
                f"p_adj={row['p_value_adj']:.4f}, "
                f"effect={row['effect_label']} ({row['effect_size']:.3f}), "
                f"n={row['n']}\n"
            )

    md.append("\n## Key Takeaways\n\n")

    # Auto-generate takeaways from top results
    for target in TARGETS:
        target_sig = sig[sig["target"] == target]
        if target_sig.empty:
            md.append(f"### {target}\nNo significant predictors found.\n\n")
        else:
            md.append(f"### {target}\n")
            top3 = target_sig.head(3)
            for _, row in top3.iterrows():
                md.append(f"- {row['feature']} ({row['effect_label']} effect, r={row['statistic']:.3f})\n")
            md.append("\n")

    md.append("## Figures\n\n")
    md.append(f"- `{heatmap_path.relative_to(REPO_ROOT)}`\n")
    md.append(f"- `{assoc_path.relative_to(REPO_ROOT)}`\n")

    md.append("\n## Limitations & Assumptions\n\n")
    md.append("- Spearman does not assume linearity but captures monotonic relationships only\n")
    md.append("- High missingness (~60%) on targets limits statistical power\n")
    md.append("- Binary features assume clean 0/1 encoding; rows with other values are dropped\n")
    md.append("- Kruskal-Wallis is omnibus — significant results need post-hoc pairwise tests (Dunn's test)\n")

    report_path = OUT_DIR / "correlation_report.md"
    report_path.write_text("".join(md), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Figures: {heatmap_path}, {assoc_path}")


if __name__ == "__main__":
    main()
