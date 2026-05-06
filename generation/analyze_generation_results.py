#!/usr/bin/env python3
"""Analyze P/P' sampled-response cross-scoring outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ITEM_COLS = ["source_dataset", "dataset_role", "source_row_index", "id", "word", "subset"]
ANALYSIS_DELTA_COL = "own_context_log_prob_advantage_per_token"
ANALYSIS_OWN_LOG_PROB_COL = "own_log_prob_per_token"
ANALYSIS_WIN_COL = "own_context_win_per_token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize generated response cross-scoring results.")
    parser.add_argument("--candidates_path", required=True, help="Candidate TSV from generate_topk_responses.py")
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Prefix for summary outputs. Defaults to candidates path without _candidates.tsv.",
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Save diagnostic histograms if matplotlib is available.",
    )
    return parser.parse_args()


def default_output_prefix(candidates_path: Path) -> Path:
    name = candidates_path.name
    if name.endswith("_candidates.tsv"):
        name = name[: -len("_candidates.tsv")]
    elif name.endswith(".tsv"):
        name = name[:-4]
    return candidates_path.with_name(name)


def prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = [ANALYSIS_DELTA_COL, ANALYSIS_OWN_LOG_PROB_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Candidate file is missing token-normalized columns required for analysis: "
            + ", ".join(missing)
        )

    df[ANALYSIS_DELTA_COL] = pd.to_numeric(df[ANALYSIS_DELTA_COL], errors="coerce")
    df[ANALYSIS_OWN_LOG_PROB_COL] = pd.to_numeric(df[ANALYSIS_OWN_LOG_PROB_COL], errors="coerce")
    df[ANALYSIS_WIN_COL] = df[ANALYSIS_DELTA_COL] > 0
    return df


def add_sampling_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["source_dataset", "source_row_index", "id", "generated_from"]
    group_max = df.groupby(group_cols, dropna=False)[ANALYSIS_OWN_LOG_PROB_COL].transform("max")
    df["candidate_weight_unnormalized"] = (df[ANALYSIS_OWN_LOG_PROB_COL] - group_max).apply(math.exp)
    group_sum = df.groupby(group_cols, dropna=False)["candidate_weight_unnormalized"].transform("sum")
    df["candidate_weight"] = df["candidate_weight_unnormalized"] / group_sum
    df["candidate_weight_basis"] = ANALYSIS_OWN_LOG_PROB_COL
    return df


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        weights = group["candidate_weight"]
        deltas = group[ANALYSIS_DELTA_COL]
        wins = group[ANALYSIS_WIN_COL].astype(float)
        top1 = group[group["unique_rank"] == group["unique_rank"].min()]
        weighted_delta = (weights * deltas).sum() / weights.sum()
        weighted_win = (weights * wins).sum() / weights.sum()
        record.update(
            {
                "analysis_metric": ANALYSIS_DELTA_COL,
                "n_candidates": len(group),
                "n_items": group[["source_dataset", "source_row_index", "id"]].drop_duplicates().shape[0],
                "mean_candidates_per_item": len(group)
                / max(group[["source_dataset", "source_row_index", "id"]].drop_duplicates().shape[0], 1),
                "mean_own_context_log_prob_advantage": deltas.mean(),
                "median_own_context_log_prob_advantage": deltas.median(),
                "mean_own_context_log_prob_advantage_per_token": deltas.mean(),
                "median_own_context_log_prob_advantage_per_token": deltas.median(),
                "own_context_win_rate": wins.mean(),
                "own_context_win_rate_per_token": wins.mean(),
                "weighted_own_context_win_rate": weighted_win,
                "weighted_own_context_win_rate_per_token": weighted_win,
                "weighted_own_context_log_prob_advantage": weighted_delta,
                "weighted_own_context_log_prob_advantage_per_token": weighted_delta,
                "top1_own_context_win_rate": top1[ANALYSIS_WIN_COL].astype(float).mean(),
                "top1_own_context_win_rate_per_token": top1[ANALYSIS_WIN_COL].astype(float).mean(),
                "top1_mean_own_context_log_prob_advantage": top1[ANALYSIS_DELTA_COL].mean(),
                "top1_mean_own_context_log_prob_advantage_per_token": top1[ANALYSIS_DELTA_COL].mean(),
                "mean_unique_samples_obtained": group["unique_samples_obtained"].mean(),
                "min_unique_samples_obtained": group["unique_samples_obtained"].min(),
                "mean_sample_draw": group["sample_draw"].mean(),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def item_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_group(df, ITEM_COLS + ["generated_from"])


def paired_item_summary(df: pd.DataFrame) -> pd.DataFrame:
    condition_summary = item_condition_summary(df)
    records = []
    grouped = df.groupby(ITEM_COLS, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(ITEM_COLS, keys))
        group_p = group[group["generated_from"] == "P"]
        group_p_prime = group[group["generated_from"] == "P_prime"]
        set_p = set(group_p["normalized_response"])
        set_p_prime = set(group_p_prime["normalized_response"])
        union = set_p | set_p_prime
        intersection = set_p & set_p_prime
        record.update(
            {
                "p_candidate_count": len(group_p),
                "p_prime_candidate_count": len(group_p_prime),
                "exact_overlap_count": len(intersection),
                "exact_overlap_jaccard": len(intersection) / len(union) if union else float("nan"),
            }
        )
        for label, source in [("p", "P"), ("p_prime", "P_prime")]:
            one = condition_summary[
                (condition_summary["source_dataset"] == record["source_dataset"])
                & (condition_summary["source_row_index"] == record["source_row_index"])
                & (condition_summary["id"] == record["id"])
                & (condition_summary["generated_from"] == source)
            ]
            if one.empty:
                continue
            one_row = one.iloc[0]
            record[f"{label}_win_rate"] = one_row["own_context_win_rate"]
            record[f"{label}_win_rate_per_token"] = one_row["own_context_win_rate_per_token"]
            record[f"{label}_weighted_win_rate"] = one_row["weighted_own_context_win_rate"]
            record[f"{label}_weighted_win_rate_per_token"] = one_row[
                "weighted_own_context_win_rate_per_token"
            ]
            record[f"{label}_mean_advantage"] = one_row["mean_own_context_log_prob_advantage"]
            record[f"{label}_mean_advantage_per_token"] = one_row[
                "mean_own_context_log_prob_advantage_per_token"
            ]
            record[f"{label}_weighted_advantage"] = one_row[
                "weighted_own_context_log_prob_advantage"
            ]
            record[f"{label}_weighted_advantage_per_token"] = one_row[
                "weighted_own_context_log_prob_advantage_per_token"
            ]
            record[f"{label}_top1_win"] = one_row["top1_own_context_win_rate"]
            record[f"{label}_top1_win_per_token"] = one_row["top1_own_context_win_rate_per_token"]
            record[f"{label}_top1_advantage"] = one_row[
                "top1_mean_own_context_log_prob_advantage"
            ]
            record[f"{label}_top1_advantage_per_token"] = one_row[
                "top1_mean_own_context_log_prob_advantage_per_token"
            ]
        records.append(record)
    paired = pd.DataFrame(records)
    if {"p_mean_advantage_per_token", "p_prime_mean_advantage_per_token"}.issubset(paired.columns):
        paired["both_mean_advantages_positive"] = (
            (paired["p_mean_advantage_per_token"] > 0) & (paired["p_prime_mean_advantage_per_token"] > 0)
        )
    if {"p_top1_win_per_token", "p_prime_top1_win_per_token"}.issubset(paired.columns):
        paired["both_top1_own_context_wins"] = (
            (paired["p_top1_win_per_token"] > 0) & (paired["p_prime_top1_win_per_token"] > 0)
        )
    return paired


def build_text_summary(df: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> str:
    lines = [
        "=" * 80,
        "GENERATION CROSS-SCORING SUMMARY",
        "=" * 80,
        f"Total candidates: {len(df)}",
        f"Total stimulus items: {df[['source_dataset', 'source_row_index', 'id']].drop_duplicates().shape[0]}",
        f"Sampling strategy: {', '.join(sorted(df['sampling_strategy'].dropna().astype(str).unique()))}",
        f"Top-p values: {', '.join(str(x) for x in sorted(df['top_p'].dropna().unique()))}",
        f"Candidate weight basis: {ANALYSIS_OWN_LOG_PROB_COL}",
        "",
        f"Main metric: {ANALYSIS_DELTA_COL}",
        "  P samples:        mean token log p(response | P) - mean token log p(response | P')",
        "  P' samples:       mean token log p(response | P') - mean token log p(response | P)",
        "  Positive values:  response is more likely under its source context",
        "",
    ]

    condition_summary = summaries["condition"]
    display_cols = [
        "source_dataset",
        "dataset_role",
        "generated_from",
        "n_candidates",
        "n_items",
        "own_context_win_rate",
        "weighted_own_context_win_rate",
        "mean_own_context_log_prob_advantage_per_token",
        "weighted_own_context_log_prob_advantage_per_token",
        "top1_own_context_win_rate",
        "min_unique_samples_obtained",
    ]
    lines.append("--- BY DATASET AND CONDITION ---")
    for _, row in condition_summary[display_cols].iterrows():
        lines.append(
            f"{row['source_dataset']} ({row['dataset_role']}), {row['generated_from']}: "
            f"win_per_token={row['own_context_win_rate']:.3f}, "
            f"weighted_win_per_token={row['weighted_own_context_win_rate']:.3f}, "
            f"mean_adv_per_token={row['mean_own_context_log_prob_advantage_per_token']:.3f}, "
            f"weighted_adv_per_token={row['weighted_own_context_log_prob_advantage_per_token']:.3f}, "
            f"top1_win_per_token={row['top1_own_context_win_rate']:.3f}, "
            f"candidates={int(row['n_candidates'])}, items={int(row['n_items'])}, "
            f"min_unique={int(row['min_unique_samples_obtained'])}"
        )

    paired = summaries["paired_item"]
    if "both_mean_advantages_positive" in paired.columns:
        lines.extend(
            [
                "",
                "--- ITEM-LEVEL PAIRING ---",
                f"Items where both P and P' sampled sets have positive mean own-context advantage per token: "
                f"{int(paired['both_mean_advantages_positive'].sum())}/{len(paired)}",
                f"Mean exact overlap count: {paired['exact_overlap_count'].mean():.2f}",
                f"Mean exact overlap Jaccard: {paired['exact_overlap_jaccard'].mean():.3f}",
            ]
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def make_plots(df: pd.DataFrame, output_prefix: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Skipping plots: {exc}")
        return

    values = df[ANALYSIS_DELTA_COL].dropna()
    if values.empty:
        print("Skipping plots: no finite context deltas.")
        return

    bins = np.linspace(values.min(), values.max(), 46)
    if len(bins) < 2 or values.min() == values.max():
        bins = 40

    datasets = sorted(df["source_dataset"].dropna().astype(str).unique())
    colors = {
        "just": "#2f7f8f",
        "only": "#5f6f52",
        "not": "#c7662e",
    }
    fallback_colors = ["#2f7f8f", "#c7662e", "#5f6f52", "#8b5c7e"]
    multi_dataset = len(datasets) > 1

    def sample_label(label: str, subset: pd.DataFrame) -> str:
        item_count = subset[["source_dataset", "source_row_index", "id"]].drop_duplicates().shape[0]
        if item_count:
            return f"{label}: {item_count} stimuli"
        return label

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, source in zip(axes, ["P", "P_prime"]):
        subset = df[df["generated_from"] == source]
        if source == "P":
            source_display = "P"
            x_formula = "Per-token Delta = mean token log p(r | P) - mean token log p(r | P')"
        else:
            source_display = "P'"
            x_formula = "Per-token Delta = mean token log p(r' | P') - mean token log p(r' | P)"
        dataset_stat_lines = []
        if multi_dataset:
            for idx, dataset in enumerate(datasets):
                dataset_subset = subset[subset["source_dataset"].astype(str) == dataset]
                dataset_values = dataset_subset[ANALYSIS_DELTA_COL].dropna()
                if dataset_values.empty:
                    continue
                color = colors.get(dataset, fallback_colors[idx % len(fallback_colors)])
                ax.hist(
                    dataset_values,
                    bins=bins,
                    histtype="stepfilled",
                    alpha=0.38,
                    color=color,
                    edgecolor=color,
                    linewidth=1.35,
                    label=sample_label(dataset, dataset_subset),
                )
                dataset_mean = dataset_values.mean()
                dataset_median = dataset_values.median()
                dataset_win = dataset_subset[ANALYSIS_WIN_COL].astype(float).mean()
                ax.axvline(
                    dataset_median,
                    color=color,
                    linewidth=1.6,
                    linestyle=":",
                    label=f"{dataset} median",
                )
                dataset_stat_lines.append(
                    f"{dataset}: win={dataset_win:.3f}, mean={dataset_mean:.3f}, median={dataset_median:.3f}"
                )
        else:
            hist_values = subset[ANALYSIS_DELTA_COL].dropna()
            dataset = datasets[0] if datasets else "all"
            color = colors.get(dataset, "#2f7f8f")
            ax.hist(
                hist_values,
                bins=bins,
                color=color,
                edgecolor="#202020",
                linewidth=0.8,
                alpha=0.82,
                label=sample_label(dataset, subset),
            )
            source_mean = hist_values.mean()
            source_median = hist_values.median()
            source_win = subset[ANALYSIS_WIN_COL].astype(float).mean()
            ax.axvline(
                source_median,
                color="#2e7d32",
                linewidth=1.4,
                linestyle=":",
                label=f"{dataset} median",
            )
            dataset_stat_lines.append(
                f"win={source_win:.3f}\nmean={source_mean:.3f}\nmedian={source_median:.3f}"
            )

        ax.axvline(0, color="black", linewidth=1.4, label="_nolegend_")
        ax.set_title(f"Generated from {source_display}", fontsize=12)
        ax.set_xlabel(x_formula)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.98,
            0.96,
            "\n".join(dataset_stat_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )
        ax.legend(fontsize=8, frameon=True, loc="center right", bbox_to_anchor=(0.98, 0.48))
    axes[0].set_ylabel("Sampled responses")
    fig.suptitle("Per-token Delta Across n=50 Generation Samples", fontsize=13)
    fig.tight_layout()
    plot_path = output_prefix.with_name(output_prefix.name + "_context_delta_hist.png")
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {plot_path}")


def main() -> None:
    args = parse_args()
    candidates_path = Path(args.candidates_path)
    output_prefix = Path(args.output_prefix) if args.output_prefix else default_output_prefix(candidates_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(candidates_path, sep="\t")
    if df.empty:
        raise ValueError(f"No candidates found in {candidates_path}")
    df = prepare_analysis_frame(df)
    df = add_sampling_weights(df)

    summaries = {
        "condition": summarize_group(
            df,
            ["source_dataset", "dataset_role", "word", "generated_from"],
        ),
        "item_condition": item_condition_summary(df),
        "paired_item": paired_item_summary(df),
    }

    weighted_candidates_path = output_prefix.with_name(output_prefix.name + "_candidates_weighted.tsv")
    df.to_csv(weighted_candidates_path, sep="\t", index=False)

    for name, summary_df in summaries.items():
        summary_path = output_prefix.with_name(output_prefix.name + f"_{name}_summary.tsv")
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"Saved {name} summary to: {summary_path}")

    text_summary = build_text_summary(df, summaries)
    text_path = output_prefix.with_name(output_prefix.name + "_summary.txt")
    text_path.write_text(text_summary + "\n")
    print("\n" + text_summary + "\n")
    print(f"Saved weighted candidates to: {weighted_candidates_path}")
    print(f"Saved text summary to: {text_path}")

    if args.make_plots:
        make_plots(df, output_prefix)


if __name__ == "__main__":
    main()
