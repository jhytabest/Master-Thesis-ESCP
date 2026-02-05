"""Initial Exploratory Data Analysis (EDA)

Dataset: dataset.csv (French deeptech startups)

Outputs:
- jobs/analysis/output/eda_summary.md
- jobs/analysis/output/missing_values.csv
- jobs/analysis/output/numeric_describe.csv
- jobs/analysis/output/targets_summary.csv
- jobs/analysis/output/fig_*.png

Run:
  python jobs/analysis/src/eda_initial.py

Notes:
- Script is intentionally conservative (read-only); it does not modify the dataset.
- Designed for reproducibility and academic reporting.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")  # for headless runs
import matplotlib.pyplot as plt


# File is at <repo>/jobs/analysis/src/eda_initial.py
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "dataset.csv"
OUT_DIR = REPO_ROOT / "jobs" / "analysis" / "output"

TARGETS = [
    "target_delta_1st_round",
    "target_total_funding",
    "target_rounds",
]


def ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Attempt numeric conversion without raising."""
    return pd.to_numeric(series, errors="coerce")


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df)).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    out = out[out["missing_count"] > 0]
    return out


def numeric_quality_flags(df: pd.DataFrame) -> dict:
    flags: dict[str, list[str] | dict] = {}

    # Attempt to detect numeric-looking object columns with many coercible values
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    numeric_like = []
    for c in object_cols:
        s = df[c]
        coerced = safe_to_numeric(s)
        ok_ratio = coerced.notna().mean()
        if ok_ratio >= 0.80 and s.notna().any():
            numeric_like.append((c, ok_ratio))

    flags["numeric_like_object_columns"] = [f"{c} (numeric_coerce_ok={pct(r)})" for c, r in sorted(numeric_like, key=lambda t: -t[1])]

    # Negative values in targets (if numeric)
    neg_targets = {}
    for t in TARGETS:
        if t in df.columns:
            s = safe_to_numeric(df[t])
            neg = int((s < 0).sum())
            if neg:
                neg_targets[t] = neg
    flags["negative_values_in_targets"] = neg_targets

    return flags


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=["number"])
    if num.shape[1] == 0:
        return pd.DataFrame()

    desc = num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skew"] = num.skew(numeric_only=True)

    # Simple outlier proxy: share outside IQR bounds
    q1 = num.quantile(0.25)
    q3 = num.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_share = ((num.lt(lower)) | (num.gt(upper))).mean(numeric_only=True)
    desc["outlier_share_iqr"] = outlier_share

    return desc.sort_values(by="outlier_share_iqr", ascending=False)


def targets_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        if t not in df.columns:
            rows.append({"target": t, "present": False})
            continue
        s = safe_to_numeric(df[t])
        rows.append(
            {
                "target": t,
                "present": True,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean()),
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                "median": float(s.median(skipna=True)) if s.notna().any() else np.nan,
                "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "skew": float(s.skew(skipna=True)) if s.notna().any() else np.nan,
                "zeros": int((s == 0).sum()) if s.notna().any() else 0,
                "negatives": int((s < 0).sum()) if s.notna().any() else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_target(df: pd.DataFrame, target: str) -> list[Path]:
    if target not in df.columns:
        return []

    s = safe_to_numeric(df[target]).dropna()
    if s.empty:
        return []

    paths: list[Path] = []

    # Histogram (linear)
    plt.figure(figsize=(7, 4))
    sns.histplot(s, bins=40, kde=False)
    plt.title(f"{target} distribution")
    plt.tight_layout()
    p1 = OUT_DIR / f"fig_{target}_hist.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    paths.append(p1)

    # Boxplot
    plt.figure(figsize=(7, 2.2))
    sns.boxplot(x=s)
    plt.title(f"{target} boxplot")
    plt.tight_layout()
    p2 = OUT_DIR / f"fig_{target}_box.png"
    plt.savefig(p2, dpi=160)
    plt.close()
    paths.append(p2)

    # Log1p histogram if strictly non-negative and highly skewed
    if (s >= 0).all():
        s_log = np.log1p(s)
        plt.figure(figsize=(7, 4))
        sns.histplot(s_log, bins=40, kde=False)
        plt.title(f"{target} distribution (log1p)")
        plt.tight_layout()
        p3 = OUT_DIR / f"fig_{target}_hist_log1p.png"
        plt.savefig(p3, dpi=160)
        plt.close()
        paths.append(p3)

    return paths


def main() -> None:
    ensure_outdir()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Basics
    n_rows, n_cols = df.shape

    # Missing values
    missing_df = summarize_missing(df)
    missing_path = OUT_DIR / "missing_values.csv"
    missing_df.to_csv(missing_path, index=True)

    # Numeric describe
    num_desc = describe_numeric(df)
    num_desc_path = OUT_DIR / "numeric_describe.csv"
    if not num_desc.empty:
        num_desc.to_csv(num_desc_path, index=True)

    # Targets
    t_summary = targets_summary(df)
    t_summary_path = OUT_DIR / "targets_summary.csv"
    t_summary.to_csv(t_summary_path, index=False)

    # Plots
    plot_paths = []
    for t in TARGETS:
        plot_paths.extend(plot_target(df, t))

    # Data quality flags
    flags = numeric_quality_flags(df)

    # Assemble markdown report
    md = []
    md.append("# Initial EDA summary\n")
    md.append(f"- Dataset path: `{DATA_PATH}`\n")
    md.append(f"- Shape: **{n_rows} rows x {n_cols} columns**\n")
    md.append("\n## Columns & types\n")
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    md.append("Dtype counts:\n")
    for k, v in sorted(dtype_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        md.append(f"- {k}: {v}\n")

    md.append("\n## Missingness\n")
    if missing_df.empty:
        md.append("No missing values detected.\n")
    else:
        md.append(f"Columns with missing values: **{missing_df.shape[0]} / {n_cols}**\n")
        top = missing_df.head(15)
        md.append("Top missing columns (count, %):\n")
        for col, row in top.iterrows():
            md.append(f"- {col}: {int(row['missing_count'])} ({pct(row['missing_pct'])})\n")
        md.append(f"\nFull table saved to: `{missing_path}`\n")

    md.append("\n## Targets\n")
    for _, r in t_summary.iterrows():
        if not bool(r.get("present", False)):
            md.append(f"- {r['target']}: **NOT FOUND**\n")
            continue
        md.append(
            "- {t}: n={n}, missing={m} ({mp}), min={mn}, median={med}, max={mx}, mean={mean}, skew={skew}, zeros={z}, negatives={neg}\n".format(
                t=r["target"],
                n=int(r["n"]),
                m=int(r["missing"]),
                mp=pct(float(r["missing_pct"])),
                mn=f"{r['min']:.3g}" if pd.notna(r["min"]) else "NA",
                med=f"{r['median']:.3g}" if pd.notna(r["median"]) else "NA",
                mx=f"{r['max']:.3g}" if pd.notna(r["max"]) else "NA",
                mean=f"{r['mean']:.3g}" if pd.notna(r["mean"]) else "NA",
                skew=f"{r['skew']:.3g}" if pd.notna(r["skew"]) else "NA",
                z=int(r["zeros"]) if pd.notna(r.get("zeros")) else 0,
                neg=int(r["negatives"]) if pd.notna(r.get("negatives")) else 0,
            )
        )

    md.append(f"\nDetailed targets table saved to: `{t_summary_path}`\n")

    md.append("\n## Data quality flags (automated heuristics)\n")
    nlike = flags.get("numeric_like_object_columns", [])
    if nlike:
        md.append("Object columns that look numeric (>=80% coercible):\n")
        for item in nlike[:25]:
            md.append(f"- {item}\n")
        if len(nlike) > 25:
            md.append(f"- … ({len(nlike)-25} more)\n")
    else:
        md.append("No strongly numeric-like object columns detected (>=80% coercible).\n")

    neg = flags.get("negative_values_in_targets", {})
    if neg:
        md.append("\nNegative values detected in targets (count):\n")
        for k, v in neg.items():
            md.append(f"- {k}: {v}\n")

    if plot_paths:
        md.append("\n## Figures\n")
        for p in plot_paths:
            md.append(f"- `{p.relative_to(REPO_ROOT)}`\n")

    md.append("\n## Next suggested steps\n")
    md.append("- Decide a missing-data strategy per feature family (drop, impute, or model-native missing handling).\n")
    md.append("- Consider log-transforming highly skewed funding targets and/or winsorizing extreme outliers.\n")
    md.append("- Validate units/currencies for funding and valuation fields (some appear range-like strings).\n")
    md.append("- Parse and standardize dates (creation, funding dates) to enable time-based features.\n")

    report_path = OUT_DIR / "eda_summary.md"
    report_path.write_text("".join(md), encoding="utf-8")

    print(f"Wrote report: {report_path}")
    print(f"Wrote missingness table: {missing_path}")
    if not num_desc.empty:
        print(f"Wrote numeric describe: {num_desc_path}")
    print(f"Wrote targets summary: {t_summary_path}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
