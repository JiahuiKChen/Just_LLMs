#!/usr/bin/env python3
"""Run one-sample t-tests on item-level per-token generation deltas."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DELTA_COL = "mean_own_context_log_prob_advantage_per_token"
CONDITION_ORDER = {"just": 0, "only": 1, "not": 2}
TABLE_CONDITION_ORDER = ["just", "not", "only"]
SOURCE_ORDER = {"P": 0, "P_prime": 1}
SOURCE_LABELS = {
    "P": r"$S$ samples",
    "P_prime": r"$S'$ samples",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-sample t-tests on item-level per-token own-context deltas."
    )
    parser.add_argument(
        "--item_data_path",
        default="generation/results/figures/generation_context_delta_item_plot_data.tsv",
        help="Aggregated item-level plot data TSV from plot_generation_context_deltas.py.",
    )
    parser.add_argument(
        "--output_prefix",
        default="generation/results/figures/generation_context_delta_item_ttests",
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


def t_test_against_zero(values: np.ndarray, ci_level: float) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    n = len(finite)
    record: dict[str, float] = {
        "n_items": n,
        "mean_delta_per_token": float(np.nanmean(finite)) if n else float("nan"),
        "sd_delta_per_token": float(np.nanstd(finite, ddof=1)) if n > 1 else float("nan"),
        "se_delta_per_token": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "t_stat": float("nan"),
        "df": float(n - 1) if n else float("nan"),
        "p_value_one_sided": float("nan"),
        "p_value_two_sided": float("nan"),
        "cohens_d": float("nan"),
    }
    if n == 0:
        return record

    mean = record["mean_delta_per_token"]
    if n == 1:
        return record

    sd = record["sd_delta_per_token"]
    se = sd / math.sqrt(n) if np.isfinite(sd) else float("nan")
    record["se_delta_per_token"] = se

    if sd == 0.0:
        if mean > 0:
            t_stat = float("inf")
            p_one_sided = 0.0
            p_two_sided = 0.0
            cohens_d = float("inf")
        elif mean < 0:
            t_stat = float("-inf")
            p_one_sided = 1.0
            p_two_sided = 0.0
            cohens_d = float("-inf")
        else:
            t_stat = 0.0
            p_one_sided = 0.5
            p_two_sided = 1.0
            cohens_d = 0.0
    else:
        result = stats.ttest_1samp(finite, popmean=0.0, alternative="greater")
        t_stat = float(result.statistic)
        p_one_sided = float(result.pvalue)
        p_two_sided = float(stats.ttest_1samp(finite, popmean=0.0).pvalue)
        cohens_d = mean / sd

        t_crit = stats.t.ppf((1.0 + ci_level) / 2.0, df=n - 1)
        record["ci_low"] = mean - t_crit * se
        record["ci_high"] = mean + t_crit * se

    record["t_stat"] = t_stat
    record["p_value_one_sided"] = p_one_sided
    record["p_value_two_sided"] = p_two_sided
    record["cohens_d"] = cohens_d
    if not np.isfinite(record["ci_low"]) and sd == 0.0:
        record["ci_low"] = mean
        record["ci_high"] = mean
    return record


def load_item_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    required = [
        "model_dir",
        "model_label",
        "model_order",
        "condition",
        "condition_label",
        "generated_from",
        DELTA_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Item data is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df[DELTA_COL] = pd.to_numeric(df[DELTA_COL], errors="coerce")
    if "generated_from_label" not in df.columns:
        df["generated_from_label"] = df["generated_from"].map(
            {"P": "S samples", "P_prime": "S' samples"}
        )
    return df


def compute_tests(item_data: pd.DataFrame, ci_level: float, alpha: float, item_data_path: Path) -> pd.DataFrame:
    group_cols = [
        "model_dir",
        "model_label",
        "model_order",
        "condition",
        "condition_label",
        "generated_from",
        "generated_from_label",
    ]
    rows = []
    for keys, group in item_data.groupby(group_cols, sort=False):
        record = dict(zip(group_cols, keys))
        stats_record = t_test_against_zero(group[DELTA_COL].to_numpy(dtype=float), ci_level=ci_level)
        record.update(stats_record)
        record["analysis_metric"] = DELTA_COL
        record["null_mean"] = 0.0
        record["alternative"] = "greater"
        record["ci_level"] = ci_level
        record["item_data_path"] = str(item_data_path)
        rows.append(record)

    results = pd.DataFrame(rows)
    results["p_value_fdr_bh"] = benjamini_hochberg(results["p_value_one_sided"])
    results["reject_fdr_bh"] = results["p_value_fdr_bh"] <= alpha
    results["alpha"] = alpha
    results["condition_order"] = results["condition"].map(CONDITION_ORDER).fillna(999).astype(int)
    results["source_order"] = results["generated_from"].map(SOURCE_ORDER).fillna(999).astype(int)
    results.sort_values(
        ["condition_order", "model_order", "source_order"],
        inplace=True,
    )
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
    subheaders: list[str] = ["", ""]
    start_col = 3
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
        r"\setlength{\tabcolsep}{4.5pt}",
        (
            r"\caption{One-sample t-tests ask whether the item-level mean per-token own-context "
            r"advantage is greater than zero for each model, source, and condition. Mean $\Delta$ "
            r"is that item-level mean per-token advantage, and $q_{\mathrm{BH}}$ is the "
            r"Benjamini--Hochberg adjusted one-sided $p$ value across all 48 tests. Across models, "
            r"the not (control) condition generally shows larger positive effects than just or only.}"
        ),
        r"\label{tab:generation-context-delta-ttests}",
        r"\begin{tabular}{ll" + "rr" * len(condition_headers) + r"}",
        r"\toprule",
        " & ".join(["Model", "Source", *condition_headers]) + r" \\",
        "".join(cmidrules),
        " & ".join(subheaders) + r" \\",
        r"\midrule",
    ]

    row_meta = (
        results[
            [
                "model_dir",
                "model_label",
                "model_order",
                "generated_from",
                "generated_from_label",
                "source_order",
            ]
        ]
        .drop_duplicates()
        .sort_values(["model_order", "source_order"])
    )

    lookup = {}
    for row in results.itertuples(index=False):
        lookup[(row.model_dir, row.generated_from, row.condition)] = row

    row_meta_rows = list(row_meta.itertuples(index=False))
    for idx, meta in enumerate(row_meta_rows):
        source_label = SOURCE_LABELS.get(meta.generated_from, latex_escape(meta.generated_from_label))
        row_cells = [latex_escape(meta.model_label), source_label]

        for condition in ordered_conditions:
            condition_row = lookup.get((meta.model_dir, meta.generated_from, condition))
            if condition_row is None:
                row_cells.extend(["--", "--"])
                continue
            row_cells.extend(
                [
                    format_float(condition_row.mean_delta_per_token),
                    format_p(condition_row.p_value_fdr_bh),
                ]
            )

        lines.append(" & ".join(row_cells) + r" \\")
        if idx + 1 < len(row_meta_rows) and row_meta_rows[idx + 1].model_dir != meta.model_dir:
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
    item_data_path = Path(args.item_data_path)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    item_data = load_item_data(item_data_path)
    results = compute_tests(
        item_data=item_data,
        ci_level=args.ci_level,
        alpha=args.alpha,
        item_data_path=item_data_path,
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
