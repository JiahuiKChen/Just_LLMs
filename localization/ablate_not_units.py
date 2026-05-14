#!/usr/bin/env python3
"""Ablate localized not-control units and compare against random masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_GENERATED_NEGATIVES,
    DEFAULT_MODEL_NAME,
    build_generated_negative_cache,
    build_prompt_columns,
    compute_row_metrics,
    default_results_dir,
    generated_negative_cache_path,
    get_device,
    json_loads,
    load_model,
    mask_percent_slug,
    merge_generated_negative_cache,
    parse_random_seeds,
    plot_before_after_histogram,
    prepare_not_pilot_dataset,
    sample_random_mask_from_complement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate learned not-control unit masks.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="Path to the not TSV file.")
    parser.add_argument(
        "--mask_dir",
        required=True,
        help="Run directory from localize_not_units.py, or its masks/ subdirectory.",
    )
    parser.add_argument(
        "--random_seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated seeds for same-size random masks.",
    )
    parser.add_argument(
        "--use_generated_negatives",
        action="store_true",
        help="Add generated distractors to the auxiliary held-out evaluation.",
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        default=None,
        help="Optional number of validated rows to keep when rebuilding missing caches.",
    )
    parser.add_argument(
        "--generated_negatives",
        type=int,
        default=DEFAULT_GENERATED_NEGATIVES,
        help="Number of generated distractors to cache per prompt when auxiliary evaluation is enabled.",
    )
    return parser.parse_args()


def resolve_run_dir(mask_dir: str, model_name: str, row_limit: int | None) -> Path:
    """Resolve the root output directory from a user-supplied mask path."""
    candidate = Path(mask_dir)
    if candidate.name == "masks":
        return candidate.parent
    if (candidate / "masks").exists():
        return candidate
    return default_results_dir(model_name, dataset_name="not", row_limit=row_limit)


def load_mask_specs(run_dir: Path) -> list[dict]:
    """Load all saved mask files from the localization run."""
    mask_files = sorted((run_dir / "masks").glob("mask_fold*_percent*.npz"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found under {run_dir / 'masks'}")
    specs = []
    for mask_file in mask_files:
        data = np.load(mask_file, allow_pickle=False)
        specs.append(
            {
                "path": mask_file,
                "fold_id": int(data["fold_id"]),
                "mask_percent": float(data["mask_percent"]),
                "mask": data["mask"].astype(bool),
            }
        )
    return specs


def build_or_load_dataset(run_dir: Path, data_path: str, row_limit: int | None) -> pd.DataFrame:
    """Load the processed pilot dataset, rebuilding it if needed."""
    processed_path = run_dir / "not_pilot_dataset.tsv"
    if processed_path.exists():
        df = pd.read_csv(processed_path, sep="\t")
    else:
        df = prepare_not_pilot_dataset(data_path=data_path, output_path=processed_path, row_limit=row_limit)
    if "prompt_with" not in df.columns or "prompt_without" not in df.columns:
        df = build_prompt_columns(df)
    return df


def build_or_load_generated_cache(run_dir: Path, df: pd.DataFrame, model, tokenizer, generated_negatives: int) -> pd.DataFrame:
    """Load or build the generated distractor cache."""
    cache_path = generated_negative_cache_path(run_dir)
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path, sep="\t")
    else:
        cache_df = build_generated_negative_cache(
            df=df,
            model=model,
            tokenizer=tokenizer,
            output_path=cache_path,
            generated_negatives=generated_negatives,
        )
    return merge_generated_negative_cache(df, cache_df)


def summarize_ablation_results(row_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate means and sign accuracies by mask percent and condition."""
    rows = []
    random_metric_columns = [column for column in ("gold_delta", "effect_delta", "effect_delta_hybrid") if column in row_df.columns]
    for mask_percent in sorted(row_df["mask_percent"].unique()):
        percent_df = row_df.loc[row_df["mask_percent"] == mask_percent].copy()
        random_df = percent_df.loc[percent_df["ablation_condition"] == "random"].copy()
        if not random_df.empty:
            random_mean = random_df.groupby(["pilot_row_index", "fold_id"], dropna=False)[random_metric_columns].mean().reset_index()
            random_mean["ablation_condition"] = "random_mean"
            random_mean["mask_percent"] = mask_percent
            percent_df = pd.concat(
                [
                    percent_df.loc[percent_df["ablation_condition"] != "random"],
                    random_mean,
                ],
                ignore_index=True,
                sort=False,
            )

        for condition, condition_df in percent_df.groupby("ablation_condition", dropna=False):
            record = {
                "mask_percent": mask_percent,
                "ablation_condition": condition,
                "num_rows": len(condition_df),
                "gold_delta_mean": float(pd.to_numeric(condition_df["gold_delta"], errors="coerce").mean()),
                "effect_delta_mean": float(pd.to_numeric(condition_df["effect_delta"], errors="coerce").mean()),
                "gold_delta_sign_accuracy": float((pd.to_numeric(condition_df["gold_delta"], errors="coerce") > 0).mean()),
                "effect_delta_sign_accuracy": float((pd.to_numeric(condition_df["effect_delta"], errors="coerce") > 0).mean()),
            }
            if "effect_delta_hybrid" in condition_df.columns:
                hybrid_values = pd.to_numeric(condition_df["effect_delta_hybrid"], errors="coerce")
                record["effect_delta_hybrid_mean"] = float(hybrid_values.mean())
                record["effect_delta_hybrid_sign_accuracy"] = float((hybrid_values > 0).mean())
            rows.append(record)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    baseline_lookup = summary_df.loc[summary_df["ablation_condition"] == "none"].set_index("mask_percent")
    for metric in ("gold_delta_mean", "effect_delta_mean", "effect_delta_hybrid_mean"):
        if metric not in summary_df.columns:
            continue
        delta_column = metric.replace("_mean", "_change_from_none")
        summary_df[delta_column] = np.nan
        for mask_percent in summary_df["mask_percent"].unique():
            if mask_percent not in baseline_lookup.index:
                continue
            baseline_value = baseline_lookup.loc[mask_percent, metric]
            mask = summary_df["mask_percent"] == mask_percent
            summary_df.loc[mask, delta_column] = summary_df.loc[mask, metric] - baseline_value
    return summary_df


def main() -> None:
    args = parse_args()
    random_seeds = parse_random_seeds(args.random_seeds)
    run_dir = resolve_run_dir(args.mask_dir, model_name=args.model_name, row_limit=args.row_limit)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = build_or_load_dataset(run_dir, data_path=args.data_path, row_limit=args.row_limit)
    mask_specs = load_mask_specs(run_dir)

    baseline_path = run_dir / "baseline_row_metrics.tsv"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline metrics missing at {baseline_path}. Run localization first."
        )
    baseline_df = pd.read_csv(baseline_path, sep="\t")
    baseline_df = baseline_df.loc[:, ~baseline_df.columns.duplicated()]

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    device = get_device(model)

    generated_df = None
    if args.use_generated_negatives:
        generated_df = build_or_load_generated_cache(
            run_dir=run_dir,
            df=dataset_df,
            model=model,
            tokenizer=tokenizer,
            generated_negatives=args.generated_negatives,
        )

    row_records = []
    grouped_masks: dict[tuple[int, float], np.ndarray] = {
        (spec["fold_id"], spec["mask_percent"]): spec["mask"] for spec in mask_specs
    }

    for (fold_id, mask_percent), localized_mask in sorted(grouped_masks.items()):
        held_out = baseline_df.loc[baseline_df["fold_id"] == fold_id].copy()
        percent_slug = mask_percent_slug(mask_percent)
        held_out["mask_percent"] = mask_percent
        held_out["ablation_condition"] = "none"
        held_out["random_seed"] = ""
        if args.use_generated_negatives and generated_df is not None:
            none_rows = []
            source_df = generated_df.loc[generated_df["fold_id"] == fold_id].copy()
            for row in tqdm(source_df.to_dict(orient="records"), total=len(source_df), desc=f"Hybrid baseline fold {fold_id}"):
                series = pd.Series(row)
                metrics = compute_row_metrics(
                    model=model,
                    tokenizer=tokenizer,
                    row=series,
                    device=device,
                    layer_mask=None,
                    generated_negatives_with=list(json_loads(row["generated_negative_followups_with"])),
                    generated_negatives_without=list(json_loads(row["generated_negative_followups_without"])),
                )
                none_rows.append({**row, **metrics, "mask_percent": mask_percent, "ablation_condition": "none", "random_seed": ""})
            held_out = pd.DataFrame(none_rows)
        row_records.append(held_out)

        localized_rows = []
        source_df = generated_df if args.use_generated_negatives and generated_df is not None else dataset_df
        source_df = source_df.loc[source_df["fold_id"] == fold_id].copy()
        for row in tqdm(source_df.to_dict(orient="records"), total=len(source_df), desc=f"Localized ablation fold {fold_id} ({percent_slug})"):
            series = pd.Series(row)
            metrics = compute_row_metrics(
                model=model,
                tokenizer=tokenizer,
                row=series,
                device=device,
                layer_mask=localized_mask,
                generated_negatives_with=list(json_loads(row["generated_negative_followups_with"])) if args.use_generated_negatives and generated_df is not None else None,
                generated_negatives_without=list(json_loads(row["generated_negative_followups_without"])) if args.use_generated_negatives and generated_df is not None else None,
            )
            localized_rows.append(
                {
                    **row,
                    **metrics,
                    "mask_percent": mask_percent,
                    "ablation_condition": "localized",
                    "random_seed": "",
                }
            )
        row_records.append(pd.DataFrame(localized_rows))

        for seed in random_seeds:
            random_mask = sample_random_mask_from_complement(localized_mask, seed=seed)
            random_rows = []
            for row in tqdm(source_df.to_dict(orient="records"), total=len(source_df), desc=f"Random ablation fold {fold_id} ({percent_slug}, seed {seed})"):
                series = pd.Series(row)
                metrics = compute_row_metrics(
                    model=model,
                    tokenizer=tokenizer,
                    row=series,
                    device=device,
                    layer_mask=random_mask,
                    generated_negatives_with=list(json_loads(row["generated_negative_followups_with"])) if args.use_generated_negatives and generated_df is not None else None,
                    generated_negatives_without=list(json_loads(row["generated_negative_followups_without"])) if args.use_generated_negatives and generated_df is not None else None,
                )
                random_rows.append(
                    {
                        **row,
                        **metrics,
                        "mask_percent": mask_percent,
                        "ablation_condition": "random",
                        "random_seed": seed,
                    }
                )
            row_records.append(pd.DataFrame(random_rows))

    row_df = pd.concat(row_records, ignore_index=True, sort=False)
    row_df.to_csv(run_dir / "ablation_row_scores.tsv", sep="\t", index=False)

    summary_df = summarize_ablation_results(row_df)
    summary_df.to_csv(run_dir / "ablation_summary.tsv", sep="\t", index=False)

    primary_percent = 1.0 if 1.0 in summary_df["mask_percent"].tolist() else sorted(summary_df["mask_percent"].unique())[0]
    hist_df = row_df.loc[
        (row_df["mask_percent"] == primary_percent)
        & (row_df["ablation_condition"].isin(["none", "localized"]))
    ].copy()
    plot_before_after_histogram(
        df=hist_df,
        metric_column="gold_delta",
        baseline_condition="none",
        ablated_condition="localized",
        output_path=run_dir / f"gold_delta_hist_percent{mask_percent_slug(primary_percent)}.png",
        title=f"Gold Delta Before vs Localized Ablation ({primary_percent:g}%)",
    )
    plot_before_after_histogram(
        df=hist_df,
        metric_column="effect_delta",
        baseline_condition="none",
        ablated_condition="localized",
        output_path=run_dir / f"effect_delta_hist_percent{mask_percent_slug(primary_percent)}.png",
        title=f"Effect Delta Before vs Localized Ablation ({primary_percent:g}%)",
    )

    print(f"Saved ablation rows to {run_dir / 'ablation_row_scores.tsv'}")
    print(f"Saved ablation summary to {run_dir / 'ablation_summary.tsv'}")


if __name__ == "__main__":
    main()
