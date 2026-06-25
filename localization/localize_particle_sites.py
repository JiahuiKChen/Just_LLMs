#!/usr/bin/env python3
"""Localize residual-stream activation-patching sites for discourse particles."""

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
    DEFAULT_MODEL_NAME,
    DEFAULT_PARTICLES,
    build_encoded_sequence_batch,
    build_prediction_position_specs,
    build_residual_patch_plan,
    default_results_dir,
    get_decoder_layers,
    get_device,
    load_model,
    normalize_particle_name,
    resolve_prediction_positions,
    resolve_utterance_final_positions,
    run_sequence_model,
    score_target_log_probs_from_logits,
)

PATCH_SCOPE_UTTERANCE_FINAL = "utterance_final"
PATCH_SCOPE_PROMPT_BOUNDARY = "prompt_boundary"
PATCH_SCOPE_RESPONSE_ONSET = "response_onset"
PATCH_SCOPE_FULL_FINAL = "full_final"
PATCH_SCOPE_HYBRID = "hybrid"
SITE_METRIC_RESTORATION = "restoration"
SITE_METRIC_RAW_DELTA = "raw_delta_per_token"
SITE_METRIC_RESTORATION_CLIPPED = "restoration_clipped"
CORRUPTION_MODE = "prompt_without"
SITE_METRIC_TO_COLUMN = {
    SITE_METRIC_RESTORATION: "mean_restoration",
    SITE_METRIC_RAW_DELTA: "mean_raw_delta_per_token",
    SITE_METRIC_RESTORATION_CLIPPED: "mean_restoration_clipped",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize residual-stream patching sites from pooled particle-sensitive generations."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument(
        "--particles",
        nargs="+",
        default=list(DEFAULT_PARTICLES),
        help="Particles to localize.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for clean/corrupted forward passes.",
    )
    parser.add_argument(
        "--patch_scope",
        choices=[
            PATCH_SCOPE_UTTERANCE_FINAL,
            PATCH_SCOPE_PROMPT_BOUNDARY,
            PATCH_SCOPE_RESPONSE_ONSET,
            PATCH_SCOPE_FULL_FINAL,
            PATCH_SCOPE_HYBRID,
        ],
        default=PATCH_SCOPE_HYBRID,
        help="Relative prediction positions searched by activation patching.",
    )
    parser.add_argument(
        "--onset_tokens",
        type=int,
        default=3,
        help="Number of early target-token prediction positions to search for response_onset and hybrid scopes.",
    )
    parser.add_argument(
        "--site_metric",
        choices=sorted(SITE_METRIC_TO_COLUMN),
        default=SITE_METRIC_RESTORATION,
        help="Metric used to rank patching sites in site_scores.tsv.",
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


def patching_dir(run_dir: Path) -> Path:
    """Return the patching output directory for one localization run."""
    return run_dir / "patching"


def resolve_patch_positions(
    tokenizer,
    batch_df: pd.DataFrame,
    encoded_batch,
    patch_scope: str,
    position_specs: list,
    prompt_column: str,
    utterance_column: str,
) -> np.ndarray:
    """Resolve searched patch positions for one encoded batch."""
    if patch_scope == PATCH_SCOPE_UTTERANCE_FINAL:
        return resolve_utterance_final_positions(
            tokenizer=tokenizer,
            prompts=batch_df[prompt_column].astype(str).tolist(),
            utterances=batch_df[utterance_column].astype(str).tolist(),
        )
    return resolve_prediction_positions(
        prompt_lens=encoded_batch.prompt_lens,
        target_lens=encoded_batch.target_lens,
        position_specs=position_specs,
    )


def _group_site_rows(site_row_df: pd.DataFrame, site_metric: str) -> pd.DataFrame:
    """Aggregate row-level site metrics to source-item means and then fold means."""
    metric_column = SITE_METRIC_TO_COLUMN[site_metric]
    site_key_columns = [
        "particle",
        "train_fold",
        "layer_index",
        "position_label",
        "position_order",
        "corruption_mode",
        "patch_scope",
        "onset_tokens",
    ]
    value_columns = [
        "clean_log_prob_per_token",
        "corrupted_log_prob_per_token",
        "patched_log_prob_per_token",
        "raw_delta_per_token",
        "restoration",
        "restoration_clipped",
    ]
    item_df = (
        site_row_df.groupby(site_key_columns + ["source_row_index"], dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    summary_df = (
        item_df.groupby(site_key_columns, dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
        .rename(
            columns={
                "clean_log_prob_per_token": "mean_clean_log_prob_per_token",
                "corrupted_log_prob_per_token": "mean_corrupted_log_prob_per_token",
                "patched_log_prob_per_token": "mean_patched_log_prob_per_token",
                "raw_delta_per_token": "mean_raw_delta_per_token",
                "restoration": "mean_restoration",
                "restoration_clipped": "mean_restoration_clipped",
            }
        )
    )
    row_counts = (
        site_row_df.groupby(site_key_columns, dropna=False, sort=False)
        .size()
        .reset_index(name="row_count")
    )
    source_counts = (
        item_df.groupby(site_key_columns, dropna=False, sort=False)
        .size()
        .reset_index(name="source_item_count")
    )
    summary_df = summary_df.merge(row_counts, on=site_key_columns, how="left", validate="one_to_one")
    summary_df = summary_df.merge(source_counts, on=site_key_columns, how="left", validate="one_to_one")
    summary_df["site_metric"] = site_metric
    summary_df["ranking_score"] = pd.to_numeric(summary_df[metric_column], errors="coerce")
    return summary_df.sort_values(
        ["train_fold", "ranking_score", "layer_index", "position_order", "position_label"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def localize_particle(
    model,
    tokenizer,
    particle: str,
    run_dir: Path,
    batch_size: int,
    patch_scope: str,
    onset_tokens: int,
    site_metric: str,
) -> None:
    """Localize per-fold residual-stream patching sites for one particle."""
    pool_path = run_dir / "generation_pool.tsv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing generation pool: {pool_path}")
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
    if patch_scope == PATCH_SCOPE_UTTERANCE_FINAL:
        required_columns.update({"w_word", "wo_word"})
    missing = sorted(required_columns - set(pool_df.columns))
    if missing:
        raise KeyError(f"{pool_path} is missing required columns: {', '.join(missing)}")

    patch_dir = patching_dir(run_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    position_specs = build_prediction_position_specs(patch_scope=patch_scope, onset_tokens=onset_tokens)
    decoder_layers = get_decoder_layers(model)
    device = get_device(model)
    site_row_records = []

    for fold_id in sorted(int(value) for value in pd.unique(pool_df["fold_id"])):
        train_df = pool_df.loc[pool_df["fold_id"] == fold_id].copy().reset_index(drop=True)
        if train_df.empty:
            raise ValueError(f"Fold {fold_id} contained no rows for {particle}")

        for start in tqdm(range(0, len(train_df), batch_size), desc=f"{particle} fold {fold_id} patch sites"):
            batch_df = train_df.iloc[start : start + batch_size].copy().reset_index(drop=True)
            clean_batch = build_encoded_sequence_batch(
                tokenizer=tokenizer,
                prompts=batch_df["prompt_with"].astype(str).tolist(),
                targets=batch_df["response"].astype(str).tolist(),
                device=device,
            )
            corrupted_batch = build_encoded_sequence_batch(
                tokenizer=tokenizer,
                prompts=batch_df["prompt_without"].astype(str).tolist(),
                targets=batch_df["response"].astype(str).tolist(),
                device=device,
            )
            clean_positions = resolve_patch_positions(
                tokenizer=tokenizer,
                batch_df=batch_df,
                encoded_batch=clean_batch,
                patch_scope=patch_scope,
                position_specs=position_specs,
                prompt_column="prompt_with",
                utterance_column="w_word",
            )
            corrupted_positions = resolve_patch_positions(
                tokenizer=tokenizer,
                batch_df=batch_df,
                encoded_batch=corrupted_batch,
                patch_scope=patch_scope,
                position_specs=position_specs,
                prompt_column="prompt_without",
                utterance_column="wo_word",
            )

            clean_logits, clean_cache = run_sequence_model(
                model=model,
                batch=clean_batch,
                capture_positions=clean_positions,
            )
            corrupted_logits, _ = run_sequence_model(
                model=model,
                batch=corrupted_batch,
            )
            assert clean_cache is not None
            _, _, clean_per_token = score_target_log_probs_from_logits(
                logits=clean_logits,
                input_ids=clean_batch.input_ids,
                prompt_lens=clean_batch.prompt_lens,
                target_lens=clean_batch.target_lens,
                bos_token_id=tokenizer.bos_token_id,
            )
            _, _, corrupted_per_token = score_target_log_probs_from_logits(
                logits=corrupted_logits,
                input_ids=corrupted_batch.input_ids,
                prompt_lens=corrupted_batch.prompt_lens,
                target_lens=corrupted_batch.target_lens,
                bos_token_id=tokenizer.bos_token_id,
            )

            denominator = np.maximum(clean_per_token - corrupted_per_token, 1e-6)
            for position_index, spec in enumerate(position_specs):
                for layer_index in range(len(decoder_layers)):
                    patch_plan = build_residual_patch_plan(
                        layer_position_pairs=[(layer_index, position_index)],
                        target_positions=corrupted_positions,
                        source_cache=clean_cache,
                    )
                    patched_logits, _ = run_sequence_model(
                        model=model,
                        batch=corrupted_batch,
                        patch_plan=patch_plan,
                    )
                    _, _, patched_per_token = score_target_log_probs_from_logits(
                        logits=patched_logits,
                        input_ids=corrupted_batch.input_ids,
                        prompt_lens=corrupted_batch.prompt_lens,
                        target_lens=corrupted_batch.target_lens,
                        bos_token_id=tokenizer.bos_token_id,
                    )
                    raw_delta_per_token = patched_per_token - corrupted_per_token
                    restoration = raw_delta_per_token / denominator
                    restoration_clipped = np.clip(restoration, 0.0, 1.0)
                    for row_idx, row in batch_df.iterrows():
                        if corrupted_positions[row_idx, position_index] < 0:
                            continue
                        site_row_records.append(
                            {
                                "particle": particle,
                                "train_fold": fold_id,
                                "source_row_index": int(row["source_row_index"]),
                                "layer_index": layer_index,
                                "position_label": spec.label,
                                "position_order": spec.position_order,
                                "corruption_mode": CORRUPTION_MODE,
                                "patch_scope": patch_scope,
                                "onset_tokens": onset_tokens,
                                "clean_patch_position": int(clean_positions[row_idx, position_index]),
                                "corrupted_patch_position": int(corrupted_positions[row_idx, position_index]),
                                "clean_log_prob_per_token": float(clean_per_token[row_idx]),
                                "corrupted_log_prob_per_token": float(corrupted_per_token[row_idx]),
                                "patched_log_prob_per_token": float(patched_per_token[row_idx]),
                                "raw_delta_per_token": float(raw_delta_per_token[row_idx]),
                                "restoration": float(restoration[row_idx]),
                                "restoration_clipped": float(restoration_clipped[row_idx]),
                            }
                        )

    if not site_row_records:
        raise ValueError(f"No activation-patching site rows were generated for {particle}.")

    site_row_df = pd.DataFrame(site_row_records)
    site_scores_df = _group_site_rows(site_row_df=site_row_df, site_metric=site_metric)
    site_scores_df.to_csv(patch_dir / "site_scores.tsv", sep="\t", index=False)
    print(f"Saved patching site scores under {patch_dir}")


def main() -> None:
    args = parse_args()
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    for particle in args.particles:
        normalized = normalize_particle_name(particle)
        run_dir = resolve_run_dir(args.model_name, normalized, args.output_root)
        localize_particle(
            model=model,
            tokenizer=tokenizer,
            particle=normalized,
            run_dir=run_dir,
            batch_size=args.batch_size,
            patch_scope=args.patch_scope,
            onset_tokens=args.onset_tokens,
            site_metric=args.site_metric,
        )


if __name__ == "__main__":
    main()
