#!/usr/bin/env python3
"""Run paired t-tests on matched item-level between-source generation deltas."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


P_COL = "p_mean_advantage_per_token"
P_PRIME_COL = "p_prime_mean_advantage_per_token"
DIFF_EXPR = r"S' - S"
CONDITION_ORDER = {"just": 0, "only": 1, "not": 2}
TABLE_CONDITION_ORDER = ["just", "not", "only"]
EXPECTED_ITEM_COUNTS = {"just": 40, "not": 30, "only": 30}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired t-tests on matched item-level per-token source differences."
    )
    parser.add_argument(
        "--paired_data_path",
        default="generation/results/figures/generation_context_delta_paired_plot_data.tsv",
        help="Paired item-level plot data TSV from plot_generation_context_deltas.py.",
    )
    parser.add_argument(
        "--output_prefix",
        default="generation/results/figures/generation_context_delta_between_source_ttests",
        help="Output prefix for the TSV and LaTeX table.",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Two-sided confidence level for the reported mean interval.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR threshold for the BH-adjusted reject flag.",
    )
    return parser.parse_args()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return BH-adjusted q-values, preserving NaNs."""
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values.dropna().sort_values()
    m = len(finite)
    if m == 0:
        return q_values

    adjusted = np.empty(m, dtype=float)
    running = 1.0
    values = finite.to_numpy(dtype=float)
    for reverse_rank, p_value in enumerate(values[::-1], start=1):
        rank = m - reverse_rank + 1
        bh_value = p_value * m / rank
        running = min(running, bh_value)
        adjusted[m - reverse_rank] = min(running, 1.0)

    q_values.loc[finite.index] = adjusted
    return q_values


def paired_t_test(p_prime_values: np.ndarray, p_values: np.ndarray, ci_level: float) -> dict[str, float]:
    finite_mask = np.isfinite(p_prime_values) & np.isfinite(p_values)
    finite_p_prime = p_prime_values[finite_mask]
    finite_p = p_values[finite_mask]
    diffs = finite_p_prime - finite_p
    n = len(diffs)
    record: dict[str, float] = {
        "n_items": n,
        "mean_diff_per_token": float(np.nanmean(diffs)) if n else float("nan"),
        "sd_diff_per_token": float(np.nanstd(diffs, ddof=1)) if n > 1 else float("nan"),
        "se_diff_per_token": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "t_stat": float("nan"),
        "df": float(n - 1) if n else float("nan"),
        "p_value_two_sided": float("nan"),
        "cohens_dz": float("nan"),
    }
    if n == 0:
        return record

    mean = record["mean_diff_per_token"]
    if n == 1:
        return record

    sd = record["sd_diff_per_token"]
    se = sd / math.sqrt(n) if np.isfinite(sd) else float("nan")
    record["se_diff_per_token"] = se

    if sd == 0.0:
        if mean > 0:
            t_stat = float("inf")
            p_two_sided = 0.0
            cohens_dz = float("inf")
        elif mean < 0:
            t_stat = float("-inf")
            p_two_sided = 0.0
            cohens_dz = float("-inf")
        else:
            t_stat = 0.0
            p_two_sided = 1.0
            cohens_dz = 0.0
    else:
        result = stats.ttest_rel(finite_p_prime, finite_p)
        t_stat = float(result.statistic)
        p_two_sided = float(result.pvalue)
        cohens_dz = mean / sd

        t_crit = stats.t.ppf((1.0 + ci_level) / 2.0, df=n - 1)
        record["ci_low"] = mean - t_crit * se
        record["ci_high"] = mean + t_crit * se

    record["t_stat"] = t_stat
    record["p_value_two_sided"] = p_two_sided
    record["cohens_dz"] = cohens_dz
    if not np.isfinite(record["ci_low"]) and sd == 0.0:
        record["ci_low"] = mean
        record["ci_high"] = mean
    return record


def load_paired_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    required = [
        "model_dir",
        "model_label",
        "model_order",
        "condition",
        "condition_label",
        P_COL,
        P_PRIME_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Paired data is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df[P_COL] = pd.to_numeric(df[P_COL], errors="coerce")
    df[P_PRIME_COL] = pd.to_numeric(df[P_PRIME_COL], errors="coerce")

    errors: list[str] = []
    grouped = df.groupby(["model_dir", "condition"], sort=False)
    if grouped.ngroups != 24:
        errors.append(f"Expected 24 model/condition groups, found {grouped.ngroups}.")
    for (model_dir, condition), group in grouped:
        expected_n = EXPECTED_ITEM_COUNTS.get(condition)
        if expected_n is not None and len(group) != expected_n:
            errors.append(
                f"{model_dir}/{condition}: expected {expected_n} paired items, found {len(group)}."
            )
    if errors:
        raise ValueError("Coverage validation failed:\n" + "\n".join(errors))
    return df


def compute_tests(paired_data: pd.DataFrame, ci_level: float, alpha: float, paired_data_path: Path) -> pd.DataFrame:
    group_cols = [
        "model_dir",
        "model_label",
        "model_order",
        "condition",
        "condition_label",
    ]
    rows = []
    for keys, group in paired_data.groupby(group_cols, sort=False):
        record = dict(zip(group_cols, keys))
        stats_record = paired_t_test(
            group[P_PRIME_COL].to_numpy(dtype=float),
            group[P_COL].to_numpy(dtype=float),
            ci_level=ci_level,
        )
        record.update(stats_record)
        record["analysis_metric"] = f"{P_PRIME_COL} - {P_COL}"
        record["difference_direction"] = "p_prime_minus_p"
        record["null_mean"] = 0.0
        record["alternative"] = "two-sided"
        record["ci_level"] = ci_level
        record["paired_data_path"] = str(paired_data_path)
        rows.append(record)

    results = pd.DataFrame(rows)
    results["p_value_fdr_bh"] = benjamini_hochberg(results["p_value_two_sided"])
    results["reject_fdr_bh"] = results["p_value_fdr_bh"] <= alpha
    results["alpha"] = alpha
    results["condition_order"] = results["condition"].map(CONDITION_ORDER).fillna(999).astype(int)
    results.sort_values(["condition_order", "model_order"], inplace=True)
    results.reset_index(drop=True, inplace=True)
    return results


def format_p(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if math.isinf(value):
        return r"$\infty$" if value > 0 else r"$-\infty$"
    return f"{value:.{digits}f}"


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def sentence_case(text: str) -> str:
    text = str(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def build_emnlp_table(results: pd.DataFrame) -> str:
    """Build a wide table* export with condition-grouped columns."""
    available_conditions = set(results["condition"].dropna().unique())
    ordered_conditions = [cond for cond in TABLE_CONDITION_ORDER if cond in available_conditions]
    ordered_conditions.extend(
        condition
        for condition in sorted(available_conditions, key=lambda c: CONDITION_ORDER.get(c, 999))
        if condition not in ordered_conditions
    )

    condition_headers: list[str] = []
    cmidrules: list[str] = []
    subheaders: list[str] = [""]
    start_col = 2
    for condition in ordered_conditions:
        subset = results[results["condition"] == condition].copy()
        if subset.empty:
            continue

        condition_label = str(subset["condition_label"].iloc[0])
        unique_n = sorted(subset["n_items"].dropna().astype(int).unique().tolist())
        if len(unique_n) == 1:
            n_text = f"{unique_n[0]} items"
        else:
            n_text = "varying n"

        condition_headers.append(
            rf"\multicolumn{{2}}{{c}}{{\textbf{{{latex_escape(sentence_case(condition_label))}}} ({latex_escape(n_text)})}}"
        )
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{start_col + 1}}}")
        subheaders.extend([r"Mean $\Delta$", r"$q_{\mathrm{BH}}$"])
        start_col += 2

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\caption{Paired t-tests compare matched item-level per-token own-context advantages "
            r"from $S$-sampled and $S'$-sampled response sets within each model and condition. "
            rf"Mean $\Delta$ is the item-level mean paired difference ${DIFF_EXPR}$, "
            r"and $q_{\mathrm{BH}}$ is the Benjamini--Hochberg adjusted two-sided $p$ value across "
            r"all 24 tests; $q_{\mathrm{BH}}<0.05$ indicates a reliable between-source difference "
            r"after false-discovery-rate correction. Positive values mean the $S'$ source has the "
            r"larger effect. Across models, BH-significant between-source effects are concentrated "
            r"in the not (control) condition and usually favor $S'$ over $S$.}"
        ),
        r"\label{tab:generation-context-delta-between-source-ttests}",
        r"\begin{tabular}{l" + "rr" * len(condition_headers) + r"}",
        r"\toprule",
        " & ".join(["Model", *condition_headers]) + r" \\",
        "".join(cmidrules),
        " & ".join(subheaders) + r" \\",
        r"\midrule",
    ]

    row_meta = (
        results[["model_dir", "model_label", "model_order"]]
        .drop_duplicates()
        .sort_values(["model_order"])
    )

    lookup = {}
    for row in results.itertuples(index=False):
        lookup[(row.model_dir, row.condition)] = row

    row_meta_rows = list(row_meta.itertuples(index=False))
    for idx, meta in enumerate(row_meta_rows):
        row_cells = [latex_escape(meta.model_label)]

        for condition in ordered_conditions:
            condition_row = lookup.get((meta.model_dir, condition))
            if condition_row is None:
                row_cells.extend(["--", "--"])
                continue
            row_cells.extend(
                [
                    format_float(condition_row.mean_diff_per_token),
                    format_p(condition_row.p_value_fdr_bh),
                ]
            )

        lines.append(" & ".join(row_cells) + r" \\")
        if idx + 1 < len(row_meta_rows):
            lines.append(r"\addlinespace[2pt]")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paired_data_path = Path(args.paired_data_path)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    paired_data = load_paired_data(paired_data_path)
    results = compute_tests(
        paired_data=paired_data,
        ci_level=args.ci_level,
        alpha=args.alpha,
        paired_data_path=paired_data_path,
    )

    tsv_path = output_prefix.with_suffix(".tsv")
    tex_path = output_prefix.with_suffix(".tex")
    results.to_csv(tsv_path, sep="\t", index=False)
    tex_path.write_text(build_emnlp_table(results))

    print(f"Wrote TSV: {tsv_path}")
    print(f"Wrote LaTeX: {tex_path}")
    print(
        "BH-significant tests at alpha="
        f"{args.alpha:.2f}: {int(results['reject_fdr_bh'].sum())}/{len(results)}"
    )


if __name__ == "__main__":
    main()
