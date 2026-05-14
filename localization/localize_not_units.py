#!/usr/bin/env python3
"""Localize residual units predictive of not-control followup effects."""

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
    DEFAULT_HUMAN_NEGATIVES,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_FOLDS,
    assign_folds,
    build_mask_from_scores,
    build_prompt_columns,
    compute_row_metrics,
    compute_signed_correlations,
    default_results_dir,
    extract_prompt_hidden_states,
    get_device,
    get_hidden_size,
    json_dumps,
    load_model,
    mask_percent_slug,
    pairwise_overlap_stats,
    parse_mask_percents,
    plot_layer_counts,
    plot_overlap_heatmap,
    prepare_not_pilot_dataset,
    summarize_metric_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated not-control unit localization.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="Path to the not TSV file.")
    parser.add_argument(
        "--num_folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--mask_percents",
        default="0.5,1,5",
        help="Comma-separated mask percentages to save.",
    )
    parser.add_argument(
        "--human_negatives",
        type=int,
        default=DEFAULT_HUMAN_NEGATIVES,
        help="Number of human distractors per row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed for folds and distractor matching.",
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        default=None,
        help="Optional number of validated rows to keep.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to localization/results/<model>/not[/rowlimit_n].",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Prompt-state extraction batch size.",
    )
    return parser.parse_args()


def build_baseline_metrics(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """Score the baseline no-ablation metrics for every row."""
    device = get_device(model)
    rows = []
    for row in tqdm(df.to_dict(orient="records"), total=len(df), desc="Baseline scoring"):
        series = pd.Series(row)
        metrics = compute_row_metrics(model=model, tokenizer=tokenizer, row=series, device=device)
        rows.append(
            {
                **row,
                **metrics,
                "ablation_condition": "none",
                "mask_percent": 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_mask(mask_path: Path, fold_id: int, percent: float, signed_scores: np.ndarray, mask: np.ndarray) -> None:
    """Persist one fold mask and its signed correlation scores."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_path,
        fold_id=fold_id,
        mask_percent=percent,
        signed_scores=signed_scores.astype(np.float32),
        abs_scores=np.abs(signed_scores).astype(np.float32),
        mask=mask.astype(bool),
    )


def main() -> None:
    args = parse_args()
    mask_percents = parse_mask_percents(args.mask_percents)
    output_dir = Path(args.output_dir) if args.output_dir else default_results_dir(
        args.model_name, dataset_name="not", row_limit=args.row_limit
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = output_dir / "not_pilot_dataset.tsv"
    pilot_df = prepare_not_pilot_dataset(
        data_path=args.data_path,
        output_path=processed_path,
        human_negatives=args.human_negatives,
        seed=args.seed,
        row_limit=args.row_limit,
    )
    pilot_df = build_prompt_columns(pilot_df)
    fold_ids = assign_folds(len(pilot_df), num_folds=args.num_folds, seed=args.seed)
    pilot_df["fold_id"] = fold_ids
    pilot_df.to_csv(processed_path, sep="\t", index=False)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    hidden_size = get_hidden_size(model)

    prompt_state_with = extract_prompt_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompts=pilot_df["prompt_with"].tolist(),
        batch_size=args.batch_size,
    )
    prompt_state_without = extract_prompt_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompts=pilot_df["prompt_without"].tolist(),
        batch_size=args.batch_size,
    )
    delta_states = prompt_state_with - prompt_state_without

    np.savez_compressed(
        output_dir / "prompt_state_cache.npz",
        prompt_state_with=prompt_state_with.astype(np.float32),
        prompt_state_without=prompt_state_without.astype(np.float32),
        delta_states=delta_states.astype(np.float32),
        pilot_row_index=pilot_df["pilot_row_index"].to_numpy(),
        fold_id=fold_ids,
    )

    baseline_df = build_baseline_metrics(pilot_df, model=model, tokenizer=tokenizer)
    baseline_df.to_csv(output_dir / "baseline_row_metrics.tsv", sep="\t", index=False)

    baseline_summary = summarize_metric_frame(
        baseline_df,
        metric_columns=("gold_delta", "effect_delta"),
    )
    baseline_summary_df = pd.DataFrame(
        [
            {
                "model_name": args.model_name,
                "data_path": args.data_path,
                "num_rows": len(baseline_df),
                "num_folds": int(fold_ids.max()) + 1,
                "hidden_size": hidden_size,
                **baseline_summary,
            }
        ]
    )
    baseline_summary_df.to_csv(output_dir / "baseline_summary.tsv", sep="\t", index=False)

    masks_by_percent: dict[float, list[np.ndarray]] = {percent: [] for percent in mask_percents}
    mask_rows = []
    for fold_id in sorted(pilot_df["fold_id"].unique()):
        train_mask = pilot_df["fold_id"].to_numpy() != fold_id
        train_y = baseline_df.loc[train_mask, "effect_delta"].to_numpy()
        train_delta_states = delta_states[train_mask]
        signed_scores = compute_signed_correlations(train_delta_states, train_y)
        for percent in mask_percents:
            mask = build_mask_from_scores(signed_scores, percent=percent)
            masks_by_percent[percent].append(mask)
            percent_slug = mask_percent_slug(percent)
            mask_path = output_dir / "masks" / f"mask_fold{fold_id}_percent{percent_slug}.npz"
            save_mask(mask_path, fold_id=fold_id, percent=percent, signed_scores=signed_scores, mask=mask)
            mask_rows.append(
                {
                    "fold_id": fold_id,
                    "mask_percent": percent,
                    "selected_units": int(mask.sum()),
                    "num_layers": mask.shape[0],
                    "hidden_size": mask.shape[1],
                    "layer_selected_counts": json_dumps(mask.sum(axis=1).astype(int).tolist()),
                }
            )

    pd.DataFrame(mask_rows).to_csv(output_dir / "mask_inventory.tsv", sep="\t", index=False)

    overlap_rows = []
    for percent in mask_percents:
        overlap_df = pairwise_overlap_stats(masks_by_percent[percent], percent=percent)
        overlap_rows.append(overlap_df)
        percent_slug = mask_percent_slug(percent)
        overlap_df.to_csv(
            output_dir / f"mask_overlap_percent{percent_slug}.tsv",
            sep="\t",
            index=False,
        )
        plot_overlap_heatmap(
            overlap_df=overlap_df,
            value_column="jaccard",
            output_path=output_dir / f"mask_overlap_percent{percent_slug}_heatmap.png",
            title=f"Fold Jaccard Overlap ({percent:g}%)",
        )
        plot_layer_counts(
            masks=masks_by_percent[percent],
            percent=percent,
            output_path=output_dir / f"layer_counts_percent{percent_slug}.png",
        )

    pd.concat(overlap_rows, ignore_index=True).to_csv(
        output_dir / "mask_overlap_summary.tsv",
        sep="\t",
        index=False,
    )

    print(f"Saved processed dataset to {processed_path}")
    print(f"Saved prompt-state cache to {output_dir / 'prompt_state_cache.npz'}")
    print(f"Saved baseline metrics to {output_dir / 'baseline_row_metrics.tsv'}")
    print(f"Saved fold masks under {output_dir / 'masks'}")


if __name__ == "__main__":
    main()
