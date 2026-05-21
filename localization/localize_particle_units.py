#!/usr/bin/env python3
"""Localize residual units for pooled particle-sensitive generations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PARTICLES,
    build_mask_from_scores,
    build_mask_from_directional_scores,
    compute_signed_correlations,
    compute_welch_t_scores,
    default_results_dir,
    extract_prompt_hidden_states,
    extract_response_onset_hidden_states,
    extract_full_sequence_hidden_states,
    get_hidden_size,
    json_dumps,
    load_model,
    mask_percent_slug,
    normalize_particle_name,
    pairwise_overlap_stats,
    parse_mask_percents,
)

METHOD_FULL_FINAL_TTEST = "full_final_ttest"
METHOD_RESPONSE_ONSET_TTEST = "response_onset_ttest"
METHOD_PROMPT_BOUNDARY_CORR = "prompt_boundary_corr"
SELECTION_MODE_TOP_POSITIVE_K = "top_positive_k"
SELECTION_MODE_ALL_P_GENERATIONS = "all_p_generations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize particle-responsive residual units from pooled hyper-sensitive generations."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument(
        "--particles",
        nargs="+",
        default=list(DEFAULT_PARTICLES),
        help="Particles to localize.",
    )
    parser.add_argument(
        "--mask_percents",
        default="1",
        help="Comma-separated mask percentages to save.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Full-sequence activation extraction batch size.",
    )
    parser.add_argument(
        "--method",
        choices=[
            METHOD_FULL_FINAL_TTEST,
            METHOD_RESPONSE_ONSET_TTEST,
            METHOD_PROMPT_BOUNDARY_CORR,
        ],
        default=METHOD_FULL_FINAL_TTEST,
        help="Localization method to apply.",
    )
    parser.add_argument(
        "--onset_tokens",
        type=int,
        default=3,
        help="Number of early reply-token positions to average for response_onset_ttest.",
    )
    parser.add_argument(
        "--generation_results_dir",
        default=str(REPO_ROOT / "generation" / "results"),
        help="Retained for compatibility; prompt_boundary_corr now reads item targets from generation_pool.tsv.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Optional output root. When omitted, uses localization/results/<model>/<particle>/.",
    )
    return parser.parse_args()


def resolve_run_dir(model_name: str, particle: str, output_root: str | None) -> Path:
    """Resolve the run directory for one particle."""
    if output_root:
        return Path(output_root) / particle
    return default_results_dir(model_name, dataset_name=particle)


def save_mask(
    mask_path: Path,
    fold_id: int,
    percent: float,
    score_values: np.ndarray,
    score_name: str,
    method: str,
    mask: np.ndarray,
) -> None:
    """Persist one fold-specific localization mask and its scores."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_path,
        fold_id=fold_id,
        mask_percent=percent,
        localization_method=np.array(method),
        score_name=np.array(score_name),
        t_scores=score_values.astype(np.float32),
        abs_t_scores=np.abs(score_values).astype(np.float32),
        mask=mask.astype(bool),
    )


def load_item_effect_targets_from_pool(
    pool_df: pd.DataFrame,
    selected_source_rows: np.ndarray,
) -> pd.Series:
    """Return item-level mean context-advantage targets from the run-local pool."""
    required_columns = {"source_row_index", "own_context_log_prob_advantage_per_token"}
    missing = sorted(required_columns - set(pool_df.columns))
    if missing:
        raise KeyError(f"generation_pool.tsv is missing required columns: {', '.join(missing)}")

    relevant_df = pool_df.copy()
    relevant_df["own_context_log_prob_advantage_per_token"] = pd.to_numeric(
        relevant_df["own_context_log_prob_advantage_per_token"],
        errors="coerce",
    )
    item_targets = (
        relevant_df.groupby("source_row_index", sort=False)["own_context_log_prob_advantage_per_token"]
        .mean()
        .astype(float)
    )
    selected_index = pd.Index(np.asarray(selected_source_rows, dtype=int))
    missing_targets = selected_index.difference(item_targets.index)
    if len(missing_targets) > 0:
        raise ValueError(
            "Missing item-level effect targets for source_row_index values: "
            + ", ".join(str(value) for value in missing_targets.tolist())
        )
    return item_targets.reindex(selected_index)


def load_pool_selection_mode(run_dir: Path) -> str:
    """Return the selection mode recorded in pool_summary.tsv."""
    summary_path = run_dir / "pool_summary.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing pool summary: {summary_path}")
    summary_df = pd.read_csv(summary_path, sep="\t")
    if summary_df.empty:
        raise ValueError(f"{summary_path} was empty.")
    if "selection_mode" not in summary_df.columns:
        return SELECTION_MODE_TOP_POSITIVE_K
    selection_mode = str(summary_df["selection_mode"].iloc[0]).strip()
    if not selection_mode:
        raise ValueError(f"{summary_path} had an empty selection_mode value.")
    return selection_mode


def validate_pool_selection_mode(run_dir: Path, method: str) -> str:
    """Ensure the saved pool matches the localization method's expected input mode."""
    selection_mode = load_pool_selection_mode(run_dir)
    if method == METHOD_PROMPT_BOUNDARY_CORR:
        expected_mode = SELECTION_MODE_ALL_P_GENERATIONS
    else:
        expected_mode = SELECTION_MODE_TOP_POSITIVE_K
    if selection_mode != expected_mode:
        raise ValueError(
            f"{method} requires selection_mode={expected_mode}, "
            f"but {run_dir / 'pool_summary.tsv'} recorded {selection_mode}"
        )
    return selection_mode


def localize_particle(
    model,
    tokenizer,
    particle: str,
    run_dir: Path,
    mask_percents: list[float],
    batch_size: int,
    method: str,
    onset_tokens: int,
    generation_results_dir: Path,
) -> None:
    """Localize fold-specific particle masks for one particle."""
    pool_path = run_dir / "generation_pool.tsv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing generation pool: {pool_path}")
    validate_pool_selection_mode(run_dir, method=method)
    pool_df = pd.read_csv(pool_path, sep="\t")
    required_columns = {
        "pool_row_index",
        "source_row_index",
        "fold_id",
        "response",
        "prompt_with",
        "prompt_without",
        "selection_score_per_token",
    }
    missing = sorted(required_columns - set(pool_df.columns))
    if missing:
        raise KeyError(f"{pool_path} is missing required columns: {', '.join(missing)}")

    masks_by_percent: dict[float, list[np.ndarray]] = {percent: [] for percent in mask_percents}
    mask_rows = []
    fold_ids = sorted(int(value) for value in pd.unique(pool_df["fold_id"]))
    hidden_size = get_hidden_size(model)
    item_effect_targets = None
    if method == METHOD_PROMPT_BOUNDARY_CORR:
        item_effect_targets = load_item_effect_targets_from_pool(
            pool_df=pool_df,
            selected_source_rows=pool_df["source_row_index"].drop_duplicates().to_numpy(dtype=int),
        )

    for fold_id in fold_ids:
        train_df = pool_df.loc[pool_df["fold_id"] == fold_id].copy()
        if train_df.empty:
            raise ValueError(f"Fold {fold_id} contained no rows for {particle}")

        if method == METHOD_PROMPT_BOUNDARY_CORR:
            item_df = (
                train_df.sort_values(["source_row_index", "pool_row_index"])
                .drop_duplicates("source_row_index")
                .reset_index(drop=True)
            )
            states_with = extract_prompt_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=item_df["prompt_with"].tolist(),
                batch_size=batch_size,
            )
            states_without = extract_prompt_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=item_df["prompt_without"].tolist(),
                batch_size=batch_size,
            )
            effect_target = item_effect_targets.reindex(item_df["source_row_index"].to_numpy(dtype=int)).to_numpy(
                dtype=np.float32
            )
            score_values = compute_signed_correlations(states_with - states_without, effect_target)
            score_name = "signed_correlation"
            positive_unit_count = int((score_values > 0).sum())
            np.savez_compressed(
                run_dir / f"activation_cache_fold{fold_id}.npz",
                states_with=states_with.astype(np.float32),
                states_without=states_without.astype(np.float32),
                effect_target=effect_target.astype(np.float32),
                source_row_index=item_df["source_row_index"].to_numpy(dtype=np.int32),
                fold_id=np.full(len(item_df), fold_id, dtype=np.int32),
            )
        elif method == METHOD_RESPONSE_ONSET_TTEST:
            states_with = extract_response_onset_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=train_df["prompt_with"].tolist(),
                targets=train_df["response"].tolist(),
                batch_size=batch_size,
                onset_tokens=onset_tokens,
                desc=f"{particle} fold {fold_id} with onset",
            )
            states_without = extract_response_onset_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=train_df["prompt_without"].tolist(),
                targets=train_df["response"].tolist(),
                batch_size=batch_size,
                onset_tokens=onset_tokens,
                desc=f"{particle} fold {fold_id} without onset",
            )
            score_values = compute_welch_t_scores(states_with, states_without, use_abs=True)
            score_name = f"welch_t_onset_{onset_tokens}tok"
            positive_unit_count = int((score_values > 0).sum())
            np.savez_compressed(
                run_dir / f"activation_cache_fold{fold_id}.npz",
                states_with=states_with.astype(np.float32),
                states_without=states_without.astype(np.float32),
                pool_row_index=train_df["pool_row_index"].to_numpy(),
                source_row_index=train_df["source_row_index"].to_numpy(),
                fold_id=np.full(len(train_df), fold_id, dtype=np.int32),
                selection_score_per_token=train_df["selection_score_per_token"].to_numpy(dtype=np.float32),
            )
        else:
            states_with = extract_full_sequence_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=train_df["prompt_with"].tolist(),
                targets=train_df["response"].tolist(),
                batch_size=batch_size,
                desc=f"{particle} fold {fold_id} with",
            )
            states_without = extract_full_sequence_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=train_df["prompt_without"].tolist(),
                targets=train_df["response"].tolist(),
                batch_size=batch_size,
                desc=f"{particle} fold {fold_id} without",
            )
            score_values = compute_welch_t_scores(states_with, states_without, use_abs=True)
            score_name = "welch_t_full_final"
            positive_unit_count = int((score_values > 0).sum())
            np.savez_compressed(
                run_dir / f"activation_cache_fold{fold_id}.npz",
                states_with=states_with.astype(np.float32),
                states_without=states_without.astype(np.float32),
                pool_row_index=train_df["pool_row_index"].to_numpy(),
                source_row_index=train_df["source_row_index"].to_numpy(),
                fold_id=np.full(len(train_df), fold_id, dtype=np.int32),
                selection_score_per_token=train_df["selection_score_per_token"].to_numpy(dtype=np.float32),
            )

        train_rows = len(train_df)
        train_source_items = int(train_df["source_row_index"].nunique())
        for percent in mask_percents:
            if method == METHOD_PROMPT_BOUNDARY_CORR:
                mask = build_mask_from_scores(score_values, percent=percent)
            else:
                mask = build_mask_from_directional_scores(
                    score_values,
                    percent=percent,
                    require_positive=True,
                )
            masks_by_percent[percent].append(mask)
            percent_slug = mask_percent_slug(percent)
            mask_path = run_dir / "masks" / f"mask_fold{fold_id}_train_percent{percent_slug}.npz"
            save_mask(
                mask_path,
                fold_id=fold_id,
                percent=percent,
                score_values=score_values,
                score_name=score_name,
                method=method,
                mask=mask,
            )
            mask_rows.append(
                {
                    "particle": particle,
                    "localization_method": method,
                    "score_name": score_name,
                    "fold_id": fold_id,
                    "mask_percent": percent,
                    "train_rows": train_rows,
                    "train_source_items": train_source_items,
                    "selected_units": int(mask.sum()),
                    "positive_unit_count": positive_unit_count,
                    "num_layers": mask.shape[0],
                    "hidden_size": mask.shape[1],
                    "t_score_min": float(score_values.min()),
                    "t_score_max": float(score_values.max()),
                    "t_score_mean": float(score_values.mean()),
                    "onset_tokens": onset_tokens if method == METHOD_RESPONSE_ONSET_TTEST else "",
                    "layer_selected_counts": json_dumps(mask.sum(axis=1).astype(int).tolist()),
                }
            )

    mask_inventory_df = pd.DataFrame(mask_rows)
    mask_inventory_df.to_csv(run_dir / "mask_inventory.tsv", sep="\t", index=False)

    overlap_frames = []
    for percent in mask_percents:
        overlap_df = pairwise_overlap_stats(masks_by_percent[percent], percent=percent)
        overlap_df.insert(0, "particle", particle)
        overlap_frames.append(overlap_df)
    pd.concat(overlap_frames, ignore_index=True).to_csv(
        run_dir / "mask_overlap.tsv",
        sep="\t",
        index=False,
    )

    print(f"Saved localization outputs under {run_dir}")


def main() -> None:
    args = parse_args()
    mask_percents = parse_mask_percents(args.mask_percents)
    generation_results_dir = Path(args.generation_results_dir)
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    for particle in args.particles:
        particle = normalize_particle_name(particle)
        run_dir = resolve_run_dir(args.model_name, particle, args.output_root)
        localize_particle(
            model=model,
            tokenizer=tokenizer,
            particle=particle,
            run_dir=run_dir,
            mask_percents=mask_percents,
            batch_size=args.batch_size,
            method=args.method,
            onset_tokens=args.onset_tokens,
            generation_results_dir=generation_results_dir,
        )


if __name__ == "__main__":
    main()
