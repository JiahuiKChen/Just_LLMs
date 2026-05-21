#!/usr/bin/env python3
"""Summaries and statistical analysis for particle-localization ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from localization.common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PARTICLES,
    default_results_dir,
    normalize_particle_name,
)

RAW_SCORE_COLUMNS = [
    "log_prob_with",
    "log_prob_without",
    "context_advantage",
]
TOKEN_COUNT_COLUMNS = [
    "token_count_with",
    "token_count_without",
]
PER_TOKEN_SCORE_COLUMNS = [
    "log_prob_with_per_token",
    "log_prob_without_per_token",
    "context_advantage_per_token",
]
RAW_CHANGE_COLUMNS = [
    "log_prob_with_change_from_none",
    "log_prob_without_change_from_none",
    "context_advantage_change_from_none",
]
PER_TOKEN_CHANGE_COLUMNS = [
    "log_prob_with_per_token_change_from_none",
    "log_prob_without_per_token_change_from_none",
    "context_advantage_per_token_change_from_none",
]
ITEM_KEY_COLUMNS = [
    "particle",
    "mask_percent",
    "train_fold",
    "eval_fold",
    "target_type",
    "source_row_index",
    "id",
]
ITEM_GROUP_COLUMNS = ITEM_KEY_COLUMNS + ["ablation_condition", "random_seed"]
PRIMARY_METRIC = "context_advantage_per_token_change_from_none"
PRIMARY_RANDOM_MEAN_COLUMN = f"{PRIMARY_METRIC}_random_mean"
COMPARISON_METRICS = [
    "log_prob_with_per_token_change_from_none",
    "log_prob_without_per_token_change_from_none",
    PRIMARY_METRIC,
]
MEAN_METRIC_COLUMNS = [
    "log_prob_with",
    "log_prob_without",
    "context_advantage",
    "log_prob_with_per_token",
    "log_prob_without_per_token",
    "context_advantage_per_token",
    "log_prob_with_change_from_none",
    "log_prob_without_change_from_none",
    "context_advantage_change_from_none",
    "log_prob_with_per_token_change_from_none",
    "log_prob_without_per_token_change_from_none",
    "context_advantage_per_token_change_from_none",
]
FLAG_COLUMNS = [
    "baseline_positive",
    "current_positive",
    "positive_to_nonpositive",
    "prompt_with_decreased",
    "context_advantage_decreased",
]
FOLD_MEAN_COLUMNS = MEAN_METRIC_COLUMNS
FOLD_RATE_COLUMNS = [
    "baseline_positive_fraction",
    "current_positive_fraction",
    "positive_to_nonpositive_rate",
    "prompt_with_decrease_fraction",
    "context_advantage_decrease_fraction",
    "localized_stronger_drop_than_random_mean_fraction",
    "localized_random_cdf_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze particle-localization ablation outputs."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument(
        "--particles",
        nargs="+",
        default=list(DEFAULT_PARTICLES),
        help="Particles to analyze.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Optional output root. When omitted, uses localization/results/<model>/<particle>/.",
    )
    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=5000,
        help="Bootstrap replicates for fold and paired-difference confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for bootstrap and sign-flip analyses.",
    )
    parser.add_argument(
        "--signflip_replicates",
        type=int,
        default=20000,
        help="Monte Carlo replicates for paired sign-flip tests.",
    )
    return parser.parse_args()


def resolve_run_dir(model_name: str, particle: str, output_root: str | None) -> Path:
    """Resolve the run directory for one particle."""
    if output_root:
        return Path(output_root) / particle
    return default_results_dir(model_name, dataset_name=particle)


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    denom = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(numer), np.nan, dtype=float)
    valid = np.isfinite(numer) & np.isfinite(denom) & (denom > 0)
    out[valid] = numer[valid] / denom[valid]
    return pd.Series(out, index=numerator.index, dtype=float)


def _finite_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap CI around the mean."""
    finite = _finite_array(values)
    if len(finite) == 0:
        return (float("nan"), float("nan"))
    if replicates <= 0:
        return (float("nan"), float("nan"))
    sample_idx = rng.integers(0, len(finite), size=(replicates, len(finite)))
    sample_means = finite[sample_idx].mean(axis=1)
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return (float(low), float(high))


def sign_flip_pvalues(
    values: Sequence[float],
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    """Return one-sided and two-sided Monte Carlo sign-flip p-values for a paired mean."""
    finite = _finite_array(values)
    if len(finite) == 0:
        return (float("nan"), float("nan"))
    if replicates <= 0:
        return (float("nan"), float("nan"))
    observed = float(np.mean(finite))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(replicates, len(finite)))
    sample_means = (signs * finite[None, :]).mean(axis=1)
    p_less = float((np.sum(sample_means <= observed) + 1) / (replicates + 1))
    p_two_sided = float((np.sum(np.abs(sample_means) >= abs(observed)) + 1) / (replicates + 1))
    return (p_less, p_two_sided)


def add_per_token_metrics(row_df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-token log-probability metrics using the scored token counts."""
    df = _coerce_numeric_columns(row_df, RAW_SCORE_COLUMNS + TOKEN_COUNT_COLUMNS)
    df["log_prob_with_per_token"] = _safe_divide(df["log_prob_with"], df["token_count_with"])
    df["log_prob_without_per_token"] = _safe_divide(df["log_prob_without"], df["token_count_without"])
    df["context_advantage_per_token"] = df["log_prob_with_per_token"] - df["log_prob_without_per_token"]
    return df


def attach_change_from_none(row_df: pd.DataFrame) -> pd.DataFrame:
    """Attach raw and per-token deltas relative to the no-ablation baseline."""
    df = add_per_token_metrics(row_df)
    key_columns = [
        "particle",
        "mask_percent",
        "train_fold",
        "target_type",
        "evaluation_id",
    ]
    baseline_metrics = RAW_SCORE_COLUMNS + PER_TOKEN_SCORE_COLUMNS
    baseline_df = df.loc[df["ablation_condition"] == "none", key_columns + baseline_metrics].copy()
    baseline_df.rename(
        columns={metric: f"{metric}_baseline" for metric in baseline_metrics},
        inplace=True,
    )
    merged = df.merge(baseline_df, on=key_columns, how="left", validate="many_to_one")
    for metric in baseline_metrics:
        merged[f"{metric}_change_from_none"] = merged[metric] - merged[f"{metric}_baseline"]
    return merged


def _add_item_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["baseline_positive"] = result["context_advantage_per_token_baseline"] > 0
    result["current_positive"] = result["context_advantage_per_token"] > 0
    result["positive_to_nonpositive"] = result["baseline_positive"] & (~result["current_positive"])
    result["prompt_with_decreased"] = result["log_prob_with_per_token_change_from_none"] < 0
    result["context_advantage_decreased"] = result[PRIMARY_METRIC] < 0
    return result


def build_item_summary(row_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate generated rows to item level so source items receive equal weight."""
    required_columns = set(
        ITEM_GROUP_COLUMNS
        + ["evaluation_id", "token_count_with", "token_count_without"]
        + MEAN_METRIC_COLUMNS
        + [
            "context_advantage_per_token_baseline",
            "log_prob_with_per_token_baseline",
            "log_prob_without_per_token_baseline",
            "context_advantage_baseline",
            "log_prob_with_baseline",
            "log_prob_without_baseline",
        ]
    )
    missing = sorted(required_columns - set(row_df.columns))
    if missing:
        raise KeyError("Missing columns for item summary: " + ", ".join(missing))

    df = _coerce_numeric_columns(
        row_df,
        ["mask_percent", "train_fold", "eval_fold", "source_row_index"]
        + TOKEN_COUNT_COLUMNS
        + MEAN_METRIC_COLUMNS
        + [
            "context_advantage_per_token_baseline",
            "log_prob_with_per_token_baseline",
            "log_prob_without_per_token_baseline",
            "context_advantage_baseline",
            "log_prob_with_baseline",
            "log_prob_without_baseline",
        ],
    )
    agg_map: dict[str, str] = {
        "evaluation_id": "nunique",
        "token_count_with": "mean",
        "token_count_without": "mean",
        "context_advantage_per_token_baseline": "mean",
        "log_prob_with_per_token_baseline": "mean",
        "log_prob_without_per_token_baseline": "mean",
        "context_advantage_baseline": "mean",
        "log_prob_with_baseline": "mean",
        "log_prob_without_baseline": "mean",
    }
    for column in MEAN_METRIC_COLUMNS:
        agg_map[column] = "mean"

    item_df = (
        df.groupby(ITEM_GROUP_COLUMNS, dropna=False, sort=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={"evaluation_id": "n_member_targets"})
    )
    item_df["n_member_targets"] = pd.to_numeric(item_df["n_member_targets"], errors="coerce").fillna(0).astype(int)
    return _add_item_flags(item_df)


def build_random_mean_item_rows(item_df: pd.DataFrame) -> pd.DataFrame:
    """Average random-seed controls to the item level."""
    random_df = item_df.loc[item_df["ablation_condition"] == "random"].copy()
    if random_df.empty:
        return pd.DataFrame(columns=item_df.columns)

    group_cols = ITEM_KEY_COLUMNS
    value_columns = [
        "n_member_targets",
        "token_count_with",
        "token_count_without",
        "context_advantage_per_token_baseline",
        "log_prob_with_per_token_baseline",
        "log_prob_without_per_token_baseline",
        "context_advantage_baseline",
        "log_prob_with_baseline",
        "log_prob_without_baseline",
    ] + MEAN_METRIC_COLUMNS
    random_mean_df = (
        random_df.groupby(group_cols, dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    random_mean_df["n_member_targets"] = (
        pd.to_numeric(random_mean_df["n_member_targets"], errors="coerce").fillna(0).round().astype(int)
    )
    random_mean_df["ablation_condition"] = "random_mean"
    random_mean_df["random_seed"] = ""
    random_mean_df = _add_item_flags(random_mean_df)
    return random_mean_df.reindex(columns=item_df.columns)


def build_item_comparison(item_df: pd.DataFrame) -> pd.DataFrame:
    """Compare localized item effects against the distribution of random controls."""
    localized_df = item_df.loc[item_df["ablation_condition"] == "localized"].copy()
    random_df = item_df.loc[item_df["ablation_condition"] == "random"].copy()
    random_mean_df = build_random_mean_item_rows(item_df)
    if localized_df.empty:
        return pd.DataFrame()

    if random_df.empty or random_mean_df.empty:
        return localized_df.rename(columns={"ablation_condition": "comparison_source"})

    random_summary_rows = []
    for keys, group in random_df.groupby(ITEM_KEY_COLUMNS, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(ITEM_KEY_COLUMNS, keys))
        record["random_seed_count"] = int(group["random_seed"].astype(str).nunique())
        for metric in COMPARISON_METRICS:
            finite = _finite_array(group[metric].to_numpy(dtype=float))
            if len(finite) == 0:
                record[f"{metric}_random_mean"] = float("nan")
                record[f"{metric}_random_std"] = float("nan")
                record[f"{metric}_random_min"] = float("nan")
                record[f"{metric}_random_q25"] = float("nan")
                record[f"{metric}_random_median"] = float("nan")
                record[f"{metric}_random_q75"] = float("nan")
                record[f"{metric}_random_max"] = float("nan")
            else:
                record[f"{metric}_random_mean"] = float(np.mean(finite))
                record[f"{metric}_random_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
                record[f"{metric}_random_min"] = float(np.min(finite))
                record[f"{metric}_random_q25"] = float(np.percentile(finite, 25))
                record[f"{metric}_random_median"] = float(np.percentile(finite, 50))
                record[f"{metric}_random_q75"] = float(np.percentile(finite, 75))
                record[f"{metric}_random_max"] = float(np.max(finite))
        random_summary_rows.append(record)
    random_summary_df = pd.DataFrame(random_summary_rows)

    keep_columns = ITEM_KEY_COLUMNS + [
        "n_member_targets",
        "token_count_with",
        "token_count_without",
        "context_advantage_per_token_baseline",
        "log_prob_with_per_token_baseline",
        "log_prob_without_per_token_baseline",
        "context_advantage_baseline",
        "log_prob_with_baseline",
        "log_prob_without_baseline",
    ] + MEAN_METRIC_COLUMNS + FLAG_COLUMNS
    comparison_df = localized_df.loc[:, keep_columns].merge(
        random_mean_df.loc[:, ITEM_KEY_COLUMNS + COMPARISON_METRICS],
        on=ITEM_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_random_mean_item"),
    )
    comparison_df = comparison_df.merge(
        random_summary_df,
        on=ITEM_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    comparison_df.rename(
        columns={metric + "_random_mean_item": metric + "_random_mean_item" for metric in COMPARISON_METRICS},
        inplace=True,
    )
    random_groups = {
        tuple(group_key if isinstance(group_key, tuple) else (group_key,)): group
        for group_key, group in random_df.groupby(ITEM_KEY_COLUMNS, dropna=False, sort=False)
    }
    for metric in COMPARISON_METRICS:
        comparison_df[f"{metric}_localized_minus_random_mean"] = (
            comparison_df[metric] - comparison_df[f"{metric}_random_mean"]
        )
        comparison_df[f"{metric}_localized_random_z"] = _safe_divide(
            comparison_df[f"{metric}_localized_minus_random_mean"],
            comparison_df[f"{metric}_random_std"],
        )
        comparison_df[f"{metric}_random_cdf_at_localized"] = np.nan
        for idx, row in comparison_df.iterrows():
            key = tuple(row[column] for column in ITEM_KEY_COLUMNS)
            group = random_groups.get(key)
            if group is None:
                continue
            finite = _finite_array(group[metric].to_numpy(dtype=float))
            if len(finite) == 0 or not np.isfinite(row[metric]):
                continue
            comparison_df.at[idx, f"{metric}_random_cdf_at_localized"] = float(np.mean(finite <= row[metric]))

    comparison_df["localized_stronger_drop_than_random_mean"] = (
        comparison_df[PRIMARY_METRIC] < comparison_df[PRIMARY_RANDOM_MEAN_COLUMN]
    )
    comparison_df["localized_prompt_with_drop_stronger_than_random_mean"] = (
        comparison_df["log_prob_with_per_token_change_from_none"]
        < comparison_df["log_prob_with_per_token_change_from_none_random_mean"]
    )
    comparison_df["localized_random_cdf"] = comparison_df[f"{PRIMARY_METRIC}_random_cdf_at_localized"]
    return comparison_df


def build_fold_summary(item_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level metrics to one row per train fold and condition."""
    combined_df = pd.concat([item_df, build_random_mean_item_rows(item_df)], ignore_index=True, sort=False)
    combined_df = combined_df.loc[combined_df["ablation_condition"] != "random"].copy()
    rows = []
    for keys, group in combined_df.groupby(
        ["particle", "target_type", "mask_percent", "train_fold", "ablation_condition"],
        dropna=False,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(["particle", "target_type", "mask_percent", "train_fold", "ablation_condition"], keys))
        record["n_items"] = int(len(group))
        record["mean_member_targets"] = float(pd.to_numeric(group["n_member_targets"], errors="coerce").mean())
        for metric in FOLD_MEAN_COLUMNS:
            record[f"{metric}_mean"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        baseline_positive = group["baseline_positive"].astype(bool)
        record["baseline_positive_fraction"] = float(baseline_positive.mean())
        record["current_positive_fraction"] = float(group["current_positive"].astype(bool).mean())
        if baseline_positive.any():
            record["positive_to_nonpositive_rate"] = float(
                (~group.loc[baseline_positive, "current_positive"].astype(bool)).mean()
            )
        else:
            record["positive_to_nonpositive_rate"] = float("nan")
        record["prompt_with_decrease_fraction"] = float(group["prompt_with_decreased"].astype(bool).mean())
        record["context_advantage_decrease_fraction"] = float(
            group["context_advantage_decreased"].astype(bool).mean()
        )
        record["localized_stronger_drop_than_random_mean_fraction"] = float("nan")
        record["localized_random_cdf_mean"] = float("nan")
        rows.append(record)
    fold_summary_df = pd.DataFrame(rows)

    if comparison_df.empty:
        return fold_summary_df

    comparison_rows = []
    for keys, group in comparison_df.groupby(
        ["particle", "target_type", "mask_percent", "train_fold"],
        dropna=False,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(["particle", "target_type", "mask_percent", "train_fold"], keys))
        record["localized_stronger_drop_than_random_mean_fraction"] = float(
            group["localized_stronger_drop_than_random_mean"].astype(bool).mean()
        )
        record["localized_random_cdf_mean"] = float(
            pd.to_numeric(group["localized_random_cdf"], errors="coerce").mean()
        )
        comparison_rows.append(record)
    comparison_fold_df = pd.DataFrame(comparison_rows)
    if comparison_fold_df.empty:
        return fold_summary_df

    localized_mask = fold_summary_df["ablation_condition"] == "localized"
    localized_df = fold_summary_df.loc[localized_mask].merge(
        comparison_fold_df,
        on=["particle", "target_type", "mask_percent", "train_fold"],
        how="left",
        suffixes=("", "_comparison"),
        validate="one_to_one",
    )
    for column in ["localized_stronger_drop_than_random_mean_fraction", "localized_random_cdf_mean"]:
        comparison_column = f"{column}_comparison"
        if comparison_column in localized_df.columns:
            localized_df[column] = localized_df[comparison_column]
            localized_df.drop(columns=[comparison_column], inplace=True)
    other_df = fold_summary_df.loc[~localized_mask].copy()
    return pd.concat([other_df, localized_df], ignore_index=True, sort=False).sort_values(
        ["particle", "target_type", "mask_percent", "train_fold", "ablation_condition"]
    )


def build_condition_summary(
    fold_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    """Aggregate fold summaries and attach fold-level uncertainty estimates."""
    rows = []
    columns_to_summarize = [f"{metric}_mean" for metric in FOLD_MEAN_COLUMNS] + FOLD_RATE_COLUMNS
    for keys, group in fold_summary_df.groupby(
        ["particle", "target_type", "mask_percent", "ablation_condition"],
        dropna=False,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(["particle", "target_type", "mask_percent", "ablation_condition"], keys))
        fold_ids = sorted(int(value) for value in pd.unique(group["train_fold"]))
        record["n_folds"] = int(len(group))
        record["train_folds"] = ",".join(str(value) for value in fold_ids)
        record["mean_items_per_fold"] = float(pd.to_numeric(group["n_items"], errors="coerce").mean())
        record["mean_member_targets_per_item"] = float(
            pd.to_numeric(group["mean_member_targets"], errors="coerce").mean()
        )
        for column in columns_to_summarize:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            finite = _finite_array(values)
            record[column] = float(np.mean(finite)) if len(finite) else float("nan")
            record[f"{column}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            ci_low, ci_high = bootstrap_ci(finite, rng, bootstrap_replicates)
            record[f"{column}_ci_low"] = ci_low
            record[f"{column}_ci_high"] = ci_high
        rows.append(record)
    summary_df = pd.DataFrame(rows)
    if summary_df.empty or comparison_df.empty:
        return summary_df

    comparison_rows = []
    for keys, group in comparison_df.groupby(
        ["particle", "target_type", "mask_percent"],
        dropna=False,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(["particle", "target_type", "mask_percent"], keys))
        record["localized_stronger_drop_than_random_mean_fraction_items"] = float(
            group["localized_stronger_drop_than_random_mean"].astype(bool).mean()
        )
        for metric in COMPARISON_METRICS:
            diff_column = f"{metric}_localized_minus_random_mean"
            record[f"{diff_column}_mean"] = float(pd.to_numeric(group[diff_column], errors="coerce").mean())
        comparison_rows.append(record)
    comparison_summary_df = pd.DataFrame(comparison_rows)
    if comparison_summary_df.empty:
        return summary_df

    localized_mask = summary_df["ablation_condition"] == "localized"
    localized_df = summary_df.loc[localized_mask].merge(
        comparison_summary_df,
        on=["particle", "target_type", "mask_percent"],
        how="left",
        validate="one_to_one",
    )
    other_df = summary_df.loc[~localized_mask].copy()
    return pd.concat([other_df, localized_df], ignore_index=True, sort=False).sort_values(
        ["particle", "target_type", "mask_percent", "ablation_condition"]
    )


def build_significance_table(
    comparison_df: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_replicates: int,
    signflip_replicates: int,
) -> pd.DataFrame:
    """Test whether localized item effects differ from the random-control mean."""
    if comparison_df.empty:
        return pd.DataFrame()

    rows = []
    for metric in COMPARISON_METRICS:
        diff_column = f"{metric}_localized_minus_random_mean"
        for keys, group in comparison_df.groupby(
            ["particle", "target_type", "mask_percent"],
            dropna=False,
            sort=False,
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = dict(zip(["particle", "target_type", "mask_percent"], keys))
            values = pd.to_numeric(group[diff_column], errors="coerce").to_numpy(dtype=float)
            finite = _finite_array(values)
            record["metric"] = metric
            record["n_item_units"] = int(len(finite))
            if len(finite) == 0:
                record["mean_difference"] = float("nan")
                record["median_difference"] = float("nan")
                record["difference_sd"] = float("nan")
                record["effect_size_dz"] = float("nan")
                record["ci_low"] = float("nan")
                record["ci_high"] = float("nan")
                record["p_value_less"] = float("nan")
                record["p_value_two_sided"] = float("nan")
                record["negative_fraction"] = float("nan")
            else:
                record["mean_difference"] = float(np.mean(finite))
                record["median_difference"] = float(np.median(finite))
                record["difference_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
                if len(finite) > 1 and record["difference_sd"] > 0:
                    record["effect_size_dz"] = record["mean_difference"] / record["difference_sd"]
                else:
                    record["effect_size_dz"] = float("nan")
                ci_low, ci_high = bootstrap_ci(finite, rng, bootstrap_replicates)
                record["ci_low"] = ci_low
                record["ci_high"] = ci_high
                p_less, p_two_sided = sign_flip_pvalues(finite, rng, signflip_replicates)
                record["p_value_less"] = p_less
                record["p_value_two_sided"] = p_two_sided
                record["negative_fraction"] = float(np.mean(finite < 0))
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["particle", "target_type", "mask_percent", "metric"])


def build_text_summary(condition_summary_df: pd.DataFrame, significance_df: pd.DataFrame) -> str:
    """Create a compact human-readable summary of the key ablation findings."""
    lines = [
        "=" * 80,
        "LOCALIZATION ABLATION SUMMARY",
        "=" * 80,
    ]
    if condition_summary_df.empty:
        lines.append("No ablation summary rows were available.")
        lines.append("=" * 80)
        return "\n".join(lines)

    primary_column = f"{PRIMARY_METRIC}_mean"
    stronger_drop_column = "localized_stronger_drop_than_random_mean_fraction_items"
    display = condition_summary_df.sort_values(
        ["particle", "target_type", "mask_percent", "ablation_condition"]
    )
    for _, row in display.iterrows():
        lines.append(
            f"{row['particle']} | {row['target_type']} | {row['mask_percent']:g}% | {row['ablation_condition']}: "
            f"delta_per_token={row.get(primary_column, float('nan')):.4f}, "
            f"positive_after={row.get('current_positive_fraction', float('nan')):.3f}, "
            f"flip_rate={row.get('positive_to_nonpositive_rate', float('nan')):.3f}"
        )
        if row["ablation_condition"] == "localized":
            localized_minus_random = row.get(f"{PRIMARY_METRIC}_localized_minus_random_mean_mean", float("nan"))
            stronger_drop = row.get(stronger_drop_column, float("nan"))
            if np.isfinite(localized_minus_random):
                lines.append(
                    f"  localized-minus-random mean={localized_minus_random:.4f}, "
                    f"stronger_drop_fraction={stronger_drop:.3f}"
                )

    if not significance_df.empty:
        lines.append("")
        lines.append("--- Localized vs Random Mean Tests ---")
        primary_rows = significance_df.loc[significance_df["metric"] == PRIMARY_METRIC].copy()
        primary_rows.sort_values(["particle", "target_type", "mask_percent"], inplace=True)
        for _, row in primary_rows.iterrows():
            lines.append(
                f"{row['particle']} | {row['target_type']} | {row['mask_percent']:g}%: "
                f"mean_diff={row['mean_difference']:.4f}, "
                f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
                f"p_less={row['p_value_less']:.4f}, "
                f"n={int(row['n_item_units'])}"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


def write_ablation_analysis_outputs(
    run_dir: Path,
    row_df: pd.DataFrame,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 0,
    signflip_replicates: int = 20000,
) -> dict[str, pd.DataFrame | str]:
    """Write item, fold, summary, significance, and text outputs for one run directory."""
    rng = np.random.default_rng(bootstrap_seed)
    enriched_df = row_df.copy()
    if PRIMARY_METRIC not in enriched_df.columns:
        enriched_df = attach_change_from_none(enriched_df)

    item_df = build_item_summary(enriched_df)
    comparison_df = build_item_comparison(item_df)
    fold_summary_df = build_fold_summary(item_df, comparison_df)
    condition_summary_df = build_condition_summary(
        fold_summary_df=fold_summary_df,
        comparison_df=comparison_df,
        rng=rng,
        bootstrap_replicates=bootstrap_replicates,
    )
    significance_df = build_significance_table(
        comparison_df=comparison_df,
        rng=rng,
        bootstrap_replicates=bootstrap_replicates,
        signflip_replicates=signflip_replicates,
    )
    text_summary = build_text_summary(condition_summary_df, significance_df)

    outputs: dict[str, pd.DataFrame | str] = {
        "item": item_df,
        "item_comparison": comparison_df,
        "fold": fold_summary_df,
        "summary": condition_summary_df,
        "significance": significance_df,
        "text": text_summary,
    }
    output_paths = {
        "item": run_dir / "ablation_item_summary.tsv",
        "item_comparison": run_dir / "ablation_item_comparison.tsv",
        "fold": run_dir / "ablation_fold_summary.tsv",
        "summary": run_dir / "ablation_summary.tsv",
        "significance": run_dir / "ablation_significance.tsv",
    }
    for key, path in output_paths.items():
        value = outputs[key]
        assert isinstance(value, pd.DataFrame)
        value.to_csv(path, sep="\t", index=False)
    (run_dir / "ablation_summary.txt").write_text(text_summary + "\n", encoding="utf-8")
    return outputs


def analyze_run_dir(
    run_dir: Path,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 0,
    signflip_replicates: int = 20000,
) -> dict[str, pd.DataFrame | str]:
    """Load row-level ablation outputs for one run directory and regenerate summaries."""
    row_path = run_dir / "ablation_row_metrics.tsv"
    if not row_path.exists():
        raise FileNotFoundError(f"Missing ablation row metrics: {row_path}")
    row_df = pd.read_csv(row_path, sep="\t")
    outputs = write_ablation_analysis_outputs(
        run_dir=run_dir,
        row_df=row_df,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        signflip_replicates=signflip_replicates,
    )
    print(f"Saved ablation analyses under {run_dir}")
    return outputs


def main() -> None:
    args = parse_args()
    for particle in args.particles:
        normalized = normalize_particle_name(particle)
        run_dir = resolve_run_dir(args.model_name, normalized, args.output_root)
        analyze_run_dir(
            run_dir=run_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            signflip_replicates=args.signflip_replicates,
        )


if __name__ == "__main__":
    main()
