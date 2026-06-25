#!/usr/bin/env python3
"""Evaluate held-out activation-patching site sets for discourse particles."""

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
    DEFAULT_RANDOM_MASK_SEEDS,
    build_encoded_sequence_batch,
    build_prediction_position_specs,
    build_residual_patch_plan,
    default_results_dir,
    get_device,
    load_model,
    normalize_particle_name,
    parse_random_seeds,
    resolve_prediction_positions,
    resolve_utterance_final_positions,
    run_sequence_model,
    score_target_log_probs_from_logits,
    stable_int,
)
from localization.patch_analysis import write_patch_analysis_outputs

EVAL_MODE_SUFFICIENCY = "sufficiency"
EVAL_MODE_NECESSITY = "necessity"
SITE_SET_NONE = "none"
SITE_SET_LOCALIZED = "localized"
SITE_SET_RANDOM = "random"
CORRUPTION_MODE = "prompt_without"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out activation-patching site sets for discourse particles."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument(
        "--particles",
        nargs="+",
        default=list(DEFAULT_PARTICLES),
        help="Particles to evaluate.",
    )
    parser.add_argument(
        "--top_ks",
        default="1,3,5,10",
        help="Comma-separated top-k site-set sizes to evaluate.",
    )
    parser.add_argument(
        "--random_seeds",
        default=",".join(str(seed) for seed in DEFAULT_RANDOM_MASK_SEEDS),
        help="Comma-separated seeds for same-size random control site sets.",
    )
    parser.add_argument(
        "--eval_modes",
        default=f"{EVAL_MODE_SUFFICIENCY},{EVAL_MODE_NECESSITY}",
        help="Comma-separated evaluation modes: sufficiency, necessity.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Optional output root. When omitted, uses localization/results/<model>/<particle>/.",
    )
    parser.add_argument(
        "--analysis_bootstrap_replicates",
        type=int,
        default=5000,
        help="Bootstrap replicates for fold and paired-difference confidence intervals.",
    )
    parser.add_argument(
        "--analysis_bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for summary bootstraps and sign-flip tests.",
    )
    parser.add_argument(
        "--analysis_signflip_replicates",
        type=int,
        default=20000,
        help="Monte Carlo replicates for localized-vs-random sign-flip tests.",
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


def uses_utterance_final_position(position_specs: list) -> bool:
    return len(position_specs) == 1 and getattr(position_specs[0], "label", "") == "utterance_final"


def output_position_label(position_specs: list) -> str:
    labels = sorted(str(getattr(spec, "label", "")) for spec in position_specs)
    unique_labels = [label for label in labels if label]
    if len(set(unique_labels)) == 1:
        return unique_labels[0]
    return "mixed"


def resolve_patch_positions(
    tokenizer,
    batch_df: pd.DataFrame,
    encoded_batch,
    position_specs: list,
    prompt_column: str,
    utterance_column: str,
) -> np.ndarray:
    """Resolve searched patch positions for one encoded batch."""
    if uses_utterance_final_position(position_specs):
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


def parse_top_ks(raw: object) -> list[int]:
    """Parse top-k site-set sizes from CLI input."""
    if raw is None:
        return [1, 3, 5, 10]
    if isinstance(raw, (list, tuple)):
        values = [int(value) for value in raw]
    else:
        values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one top-k value is required.")
    if any(value <= 0 for value in values):
        raise ValueError("All top-k values must be positive.")
    return values


def parse_eval_modes(raw: object) -> list[str]:
    """Parse evaluation modes from CLI input."""
    if raw is None:
        return [EVAL_MODE_SUFFICIENCY, EVAL_MODE_NECESSITY]
    if isinstance(raw, (list, tuple)):
        values = [str(value).strip() for value in raw if str(value).strip()]
    else:
        values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one evaluation mode is required.")
    allowed = {EVAL_MODE_SUFFICIENCY, EVAL_MODE_NECESSITY}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Unsupported eval_modes: {', '.join(sorted(set(invalid)))}")
    return values


def build_evaluation_targets(pool_df: pd.DataFrame, train_fold: int) -> pd.DataFrame:
    """Build the held-out generated-target and gold-target evaluation table."""
    held_out_df = pool_df.loc[pool_df["fold_id"] != train_fold].copy()
    if held_out_df.empty:
        raise ValueError(f"No held-out rows remained for train fold {train_fold}")

    generated_df = held_out_df.copy()
    generated_df["target_type"] = "generated"
    generated_df["target_text"] = generated_df["response"].astype(str)
    generated_df["evaluation_id"] = generated_df["pool_row_index"].map(lambda value: f"generated:{value}")

    gold_df = held_out_df.sort_values("source_row_index").drop_duplicates("source_row_index").copy()
    gold_df["target_type"] = "gold"
    gold_df["target_text"] = gold_df["followup"].astype(str)
    gold_df["evaluation_id"] = gold_df["source_row_index"].map(lambda value: f"gold:{value}")

    eval_df = pd.concat([generated_df, gold_df], ignore_index=True, sort=False)
    eval_df["train_fold"] = train_fold
    eval_df["eval_fold"] = eval_df["fold_id"].astype(int)
    return eval_df


def load_site_scores(patch_dir: Path) -> tuple[pd.DataFrame, list, dict[str, int]]:
    """Load patching site scores and rebuild the searched position specs."""
    site_scores_path = patch_dir / "site_scores.tsv"
    if not site_scores_path.exists():
        raise FileNotFoundError(f"Missing site scores: {site_scores_path}")
    site_scores_df = pd.read_csv(site_scores_path, sep="\t")
    required_columns = {
        "train_fold",
        "layer_index",
        "position_label",
        "position_order",
        "patch_scope",
        "onset_tokens",
        "ranking_score",
    }
    missing = sorted(required_columns - set(site_scores_df.columns))
    if missing:
        raise KeyError(f"{site_scores_path} is missing required columns: {', '.join(missing)}")
    if "corruption_mode" not in site_scores_df.columns:
        site_scores_df["corruption_mode"] = CORRUPTION_MODE
    unique_scopes = sorted(str(value) for value in pd.unique(site_scores_df["patch_scope"]))
    if len(unique_scopes) != 1:
        raise ValueError(f"Expected one patch_scope in {site_scores_path}, found: {', '.join(unique_scopes)}")
    unique_onset = sorted(int(value) for value in pd.unique(site_scores_df["onset_tokens"]))
    if len(unique_onset) != 1:
        raise ValueError(f"Expected one onset_tokens value in {site_scores_path}, found: {unique_onset}")
    position_specs = build_prediction_position_specs(patch_scope=unique_scopes[0], onset_tokens=unique_onset[0])
    label_to_index = {spec.label: idx for idx, spec in enumerate(position_specs)}
    unknown_labels = sorted(set(site_scores_df["position_label"]) - set(label_to_index))
    if unknown_labels:
        raise ValueError(f"site_scores.tsv contained unknown position_label values: {', '.join(unknown_labels)}")
    return site_scores_df, position_specs, label_to_index


def select_top_sites(
    fold_site_scores: pd.DataFrame,
    top_k: int,
    label_to_index: dict[str, int],
) -> list[dict[str, int | str]]:
    """Select the top-k ranked sites for one training fold."""
    ranked_df = fold_site_scores.loc[pd.to_numeric(fold_site_scores["ranking_score"], errors="coerce").notna()].copy()
    ranked_df.sort_values(
        ["ranking_score", "layer_index", "position_order", "position_label"],
        ascending=[False, True, True, True],
        inplace=True,
        kind="mergesort",
    )
    selected_df = ranked_df.head(top_k).copy()
    sites = []
    for row in selected_df.itertuples(index=False):
        sites.append(
            {
                "layer_index": int(row.layer_index),
                "position_label": str(row.position_label),
                "position_order": int(row.position_order),
                "position_index": int(label_to_index[str(row.position_label)]),
            }
        )
    return sites


def sample_random_site_set(
    localized_sites: list[dict[str, int | str]],
    site_universe: list[dict[str, int | str]],
    seed: int,
) -> list[dict[str, int | str]]:
    """Sample a same-size random site set from the complement when possible."""
    if not localized_sites:
        return []
    localized_keys = {(int(site["layer_index"]), str(site["position_label"])) for site in localized_sites}
    available = [
        site for site in site_universe
        if (int(site["layer_index"]), str(site["position_label"])) not in localized_keys
    ]
    if len(available) < len(localized_sites):
        available = list(site_universe)
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(available), size=len(localized_sites), replace=False)
    return [available[int(index)] for index in chosen_indices]


def build_site_overlap_rows(
    localized_site_sets: dict[int, dict[int, list[dict[str, int | str]]]],
    site_universes: dict[int, list[dict[str, int | str]]],
    position_label: str,
) -> pd.DataFrame:
    """Summarize fold-to-fold overlap for localized top-k site sets."""
    rows = []
    for top_k, fold_sets in localized_site_sets.items():
        fold_ids = sorted(fold_sets)
        for left_fold in fold_ids:
            left_sites = fold_sets[left_fold]
            left_keys = {(int(site["layer_index"]), str(site["position_label"])) for site in left_sites}
            for right_fold in fold_ids:
                right_sites = fold_sets[right_fold]
                right_keys = {(int(site["layer_index"]), str(site["position_label"])) for site in right_sites}
                intersection = len(left_keys & right_keys)
                union = len(left_keys | right_keys)
                left_universe = len(site_universes[left_fold])
                right_universe = len(site_universes[right_fold])
                if left_universe == right_universe and left_universe > 0:
                    expected_intersection = (len(left_keys) * len(right_keys)) / float(left_universe)
                    expected_union = len(left_keys) + len(right_keys) - expected_intersection
                    expected_jaccard = expected_intersection / expected_union if expected_union > 0 else float("nan")
                else:
                    expected_intersection = float("nan")
                    expected_jaccard = float("nan")
                rows.append(
                    {
                        "position_label": position_label,
                        "corruption_mode": CORRUPTION_MODE,
                        "top_k": top_k,
                        "fold_i": left_fold,
                        "fold_j": right_fold,
                        "intersection_count": intersection,
                        "jaccard": (intersection / union) if union > 0 else float("nan"),
                        "selected_sites_i": len(left_keys),
                        "selected_sites_j": len(right_keys),
                        "site_universe_i": left_universe,
                        "site_universe_j": right_universe,
                        "expected_random_intersection": expected_intersection,
                        "expected_random_jaccard": expected_jaccard,
                    }
                )
    return pd.DataFrame(rows)


def score_patch_site_set(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    particle: str,
    top_k: int,
    eval_mode: str,
    site_set_type: str,
    selected_sites: list[dict[str, int | str]],
    random_seed: str,
    position_specs: list,
    batch_size: int = 4,
) -> pd.DataFrame:
    """Score one held-out site set under sufficiency or necessity patching."""
    device = get_device(model)
    rows = []
    position_label = output_position_label(position_specs)
    desc = (
        f"{particle} {eval_mode} {site_set_type} top_k={top_k} fold {int(eval_df['train_fold'].iloc[0])}"
    )
    for start in tqdm(range(0, len(eval_df), batch_size), desc=desc):
        batch_df = eval_df.iloc[start : start + batch_size].copy().reset_index(drop=True)
        clean_batch = build_encoded_sequence_batch(
            tokenizer=tokenizer,
            prompts=batch_df["prompt_with"].astype(str).tolist(),
            targets=batch_df["target_text"].astype(str).tolist(),
            device=device,
        )
        corrupted_batch = build_encoded_sequence_batch(
            tokenizer=tokenizer,
            prompts=batch_df["prompt_without"].astype(str).tolist(),
            targets=batch_df["target_text"].astype(str).tolist(),
            device=device,
        )
        clean_positions = resolve_patch_positions(
            tokenizer=tokenizer,
            batch_df=batch_df,
            encoded_batch=clean_batch,
            position_specs=position_specs,
            prompt_column="prompt_with",
            utterance_column="w_word",
        )
        corrupted_positions = resolve_patch_positions(
            tokenizer=tokenizer,
            batch_df=batch_df,
            encoded_batch=corrupted_batch,
            position_specs=position_specs,
            prompt_column="prompt_without",
            utterance_column="wo_word",
        )

        capture_sites = [(int(site["layer_index"]), int(site["position_index"])) for site in selected_sites]
        if eval_mode == EVAL_MODE_SUFFICIENCY:
            clean_logits, clean_cache = run_sequence_model(
                model=model,
                batch=clean_batch,
                capture_positions=clean_positions if capture_sites else None,
            )
            corrupted_logits, _ = run_sequence_model(model=model, batch=corrupted_batch)
            baseline_logits = corrupted_logits
            baseline_batch = corrupted_batch
            target_positions = corrupted_positions
            source_cache = clean_cache
            baseline_condition = "corrupted"
            reference_condition = "clean"
        else:
            clean_logits, _ = run_sequence_model(model=model, batch=clean_batch)
            corrupted_logits, corrupted_cache = run_sequence_model(
                model=model,
                batch=corrupted_batch,
                capture_positions=corrupted_positions if capture_sites else None,
            )
            baseline_logits = clean_logits
            baseline_batch = clean_batch
            target_positions = clean_positions
            source_cache = corrupted_cache
            baseline_condition = "clean"
            reference_condition = "corrupted"

        clean_totals, clean_counts, clean_per_token = score_target_log_probs_from_logits(
            logits=clean_logits,
            input_ids=clean_batch.input_ids,
            prompt_lens=clean_batch.prompt_lens,
            target_lens=clean_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        corrupted_totals, corrupted_counts, corrupted_per_token = score_target_log_probs_from_logits(
            logits=corrupted_logits,
            input_ids=corrupted_batch.input_ids,
            prompt_lens=corrupted_batch.prompt_lens,
            target_lens=corrupted_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        baseline_totals, baseline_counts, baseline_per_token = score_target_log_probs_from_logits(
            logits=baseline_logits,
            input_ids=baseline_batch.input_ids,
            prompt_lens=baseline_batch.prompt_lens,
            target_lens=baseline_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )

        if capture_sites:
            assert source_cache is not None
            patch_plan = build_residual_patch_plan(
                layer_position_pairs=capture_sites,
                target_positions=target_positions,
                source_cache=source_cache,
            )
            patched_logits, _ = run_sequence_model(model=model, batch=baseline_batch, patch_plan=patch_plan)
        else:
            patched_logits = baseline_logits

        patched_totals, patched_counts, patched_per_token = score_target_log_probs_from_logits(
            logits=patched_logits,
            input_ids=baseline_batch.input_ids,
            prompt_lens=baseline_batch.prompt_lens,
            target_lens=baseline_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        clean_minus_corrupted = np.maximum(clean_per_token - corrupted_per_token, 1e-6)
        patched_metric_change = patched_per_token - baseline_per_token
        if eval_mode == EVAL_MODE_SUFFICIENCY:
            effect_score = patched_metric_change
        else:
            effect_score = baseline_per_token - patched_per_token
        normalized_effect = effect_score / clean_minus_corrupted

        for row_idx, row in batch_df.iterrows():
            rows.append(
                {
                    "particle": particle,
                    "position_label": position_label,
                    "corruption_mode": CORRUPTION_MODE,
                    "top_k": top_k,
                    "train_fold": int(row["train_fold"]),
                    "eval_fold": int(row["eval_fold"]),
                    "target_type": row["target_type"],
                    "evaluation_id": row["evaluation_id"],
                    "source_row_index": int(row["source_row_index"]),
                    "id": row["id"],
                    "pool_row_index": row.get("pool_row_index", ""),
                    "target_text": row["target_text"],
                    "eval_mode": eval_mode,
                    "site_set_type": site_set_type,
                    "random_seed": random_seed,
                    "selected_site_count": int(len(selected_sites)),
                    "baseline_condition": baseline_condition,
                    "reference_condition": reference_condition,
                    "clean_log_prob": float(clean_totals[row_idx]),
                    "corrupted_log_prob": float(corrupted_totals[row_idx]),
                    "baseline_log_prob": float(baseline_totals[row_idx]),
                    "reference_log_prob": float(clean_totals[row_idx] if eval_mode == EVAL_MODE_SUFFICIENCY else corrupted_totals[row_idx]),
                    "patched_log_prob": float(patched_totals[row_idx]),
                    "clean_token_count": int(clean_counts[row_idx]),
                    "corrupted_token_count": int(corrupted_counts[row_idx]),
                    "baseline_token_count": int(baseline_counts[row_idx]),
                    "reference_token_count": int(clean_counts[row_idx] if eval_mode == EVAL_MODE_SUFFICIENCY else corrupted_counts[row_idx]),
                    "patched_token_count": int(patched_counts[row_idx]),
                    "clean_log_prob_per_token": float(clean_per_token[row_idx]),
                    "corrupted_log_prob_per_token": float(corrupted_per_token[row_idx]),
                    "baseline_log_prob_per_token": float(baseline_per_token[row_idx]),
                    "reference_log_prob_per_token": float(
                        clean_per_token[row_idx] if eval_mode == EVAL_MODE_SUFFICIENCY else corrupted_per_token[row_idx]
                    ),
                    "patched_log_prob_per_token": float(patched_per_token[row_idx]),
                    "patched_metric_change": float(patched_metric_change[row_idx]),
                    "effect_score": float(effect_score[row_idx]),
                    "normalized_effect": float(normalized_effect[row_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_site_set_requests(
    particle: str,
    fold_id: int,
    top_ks: list[int],
    eval_mode: str,
    localized_site_sets: dict[int, dict[int, list[dict[str, int | str]]]],
    site_universe: list[dict[str, int | str]],
    random_seeds: list[int],
) -> list[dict[str, object]]:
    """Build the ordered site-set requests for one train fold and eval mode."""
    requests: list[dict[str, object]] = []
    for top_k in top_ks:
        localized_sites = localized_site_sets[top_k][fold_id]
        requests.append(
            {
                "particle": particle,
                "top_k": top_k,
                "eval_mode": eval_mode,
                "site_set_type": SITE_SET_NONE,
                "selected_sites": [],
                "random_seed": "",
            }
        )
        requests.append(
            {
                "particle": particle,
                "top_k": top_k,
                "eval_mode": eval_mode,
                "site_set_type": SITE_SET_LOCALIZED,
                "selected_sites": localized_sites,
                "random_seed": "",
            }
        )
        for seed in random_seeds:
            random_sites = sample_random_site_set(
                localized_sites=localized_sites,
                site_universe=site_universe,
                seed=stable_int(particle, fold_id, top_k, eval_mode, seed),
            )
            requests.append(
                {
                    "particle": particle,
                    "top_k": top_k,
                    "eval_mode": eval_mode,
                    "site_set_type": SITE_SET_RANDOM,
                    "selected_sites": random_sites,
                    "random_seed": str(seed),
                }
            )
    return requests


def score_patch_site_requests(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    site_requests: list[dict[str, object]],
    position_specs: list,
    batch_size: int = 4,
) -> pd.DataFrame:
    """Score many site sets while reusing shared clean/corrupted baselines."""
    if eval_df.empty:
        return pd.DataFrame()
    if not site_requests:
        return pd.DataFrame()

    device = get_device(model)
    particle = str(site_requests[0]["particle"])
    eval_mode = str(site_requests[0]["eval_mode"])
    fold_id = int(eval_df["train_fold"].iloc[0])
    position_label = output_position_label(position_specs)
    rows = []
    desc = f"{particle} {eval_mode} fold {fold_id}"

    for start in tqdm(range(0, len(eval_df), batch_size), desc=desc):
        batch_df = eval_df.iloc[start : start + batch_size].copy().reset_index(drop=True)
        clean_batch = build_encoded_sequence_batch(
            tokenizer=tokenizer,
            prompts=batch_df["prompt_with"].astype(str).tolist(),
            targets=batch_df["target_text"].astype(str).tolist(),
            device=device,
        )
        corrupted_batch = build_encoded_sequence_batch(
            tokenizer=tokenizer,
            prompts=batch_df["prompt_without"].astype(str).tolist(),
            targets=batch_df["target_text"].astype(str).tolist(),
            device=device,
        )
        clean_positions = resolve_patch_positions(
            tokenizer=tokenizer,
            batch_df=batch_df,
            encoded_batch=clean_batch,
            position_specs=position_specs,
            prompt_column="prompt_with",
            utterance_column="w_word",
        )
        corrupted_positions = resolve_patch_positions(
            tokenizer=tokenizer,
            batch_df=batch_df,
            encoded_batch=corrupted_batch,
            position_specs=position_specs,
            prompt_column="prompt_without",
            utterance_column="wo_word",
        )

        selected_site_union = {
            (int(site["layer_index"]), int(site["position_index"]))
            for request in site_requests
            for site in request["selected_sites"]
        }
        capture_sites = sorted(selected_site_union)
        if eval_mode == EVAL_MODE_SUFFICIENCY:
            clean_logits, clean_cache = run_sequence_model(
                model=model,
                batch=clean_batch,
                capture_positions=clean_positions if capture_sites else None,
            )
            corrupted_logits, _ = run_sequence_model(model=model, batch=corrupted_batch)
            baseline_logits = corrupted_logits
            baseline_batch = corrupted_batch
            target_positions = corrupted_positions
            source_cache = clean_cache
            baseline_condition = "corrupted"
            reference_condition = "clean"
        else:
            clean_logits, _ = run_sequence_model(model=model, batch=clean_batch)
            corrupted_logits, corrupted_cache = run_sequence_model(
                model=model,
                batch=corrupted_batch,
                capture_positions=corrupted_positions if capture_sites else None,
            )
            baseline_logits = clean_logits
            baseline_batch = clean_batch
            target_positions = clean_positions
            source_cache = corrupted_cache
            baseline_condition = "clean"
            reference_condition = "corrupted"

        clean_totals, clean_counts, clean_per_token = score_target_log_probs_from_logits(
            logits=clean_logits,
            input_ids=clean_batch.input_ids,
            prompt_lens=clean_batch.prompt_lens,
            target_lens=clean_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        corrupted_totals, corrupted_counts, corrupted_per_token = score_target_log_probs_from_logits(
            logits=corrupted_logits,
            input_ids=corrupted_batch.input_ids,
            prompt_lens=corrupted_batch.prompt_lens,
            target_lens=corrupted_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        baseline_totals, baseline_counts, baseline_per_token = score_target_log_probs_from_logits(
            logits=baseline_logits,
            input_ids=baseline_batch.input_ids,
            prompt_lens=baseline_batch.prompt_lens,
            target_lens=baseline_batch.target_lens,
            bos_token_id=tokenizer.bos_token_id,
        )
        clean_minus_corrupted = np.maximum(clean_per_token - corrupted_per_token, 1e-6)

        patch_plan_cache: dict[tuple[tuple[int, int], ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for request in site_requests:
            selected_sites = list(request["selected_sites"])
            request_key = tuple(
                sorted((int(site["layer_index"]), int(site["position_index"])) for site in selected_sites)
            )
            if request_key not in patch_plan_cache:
                if request_key:
                    assert source_cache is not None
                    patch_plan = build_residual_patch_plan(
                        layer_position_pairs=list(request_key),
                        target_positions=target_positions,
                        source_cache=source_cache,
                    )
                    patched_logits, _ = run_sequence_model(model=model, batch=baseline_batch, patch_plan=patch_plan)
                    patched_totals, patched_counts, patched_per_token = score_target_log_probs_from_logits(
                        logits=patched_logits,
                        input_ids=baseline_batch.input_ids,
                        prompt_lens=baseline_batch.prompt_lens,
                        target_lens=baseline_batch.target_lens,
                        bos_token_id=tokenizer.bos_token_id,
                    )
                else:
                    patched_totals = baseline_totals
                    patched_counts = baseline_counts
                    patched_per_token = baseline_per_token
                patch_plan_cache[request_key] = (patched_totals, patched_counts, patched_per_token)

            patched_totals, patched_counts, patched_per_token = patch_plan_cache[request_key]
            patched_metric_change = patched_per_token - baseline_per_token
            if eval_mode == EVAL_MODE_SUFFICIENCY:
                effect_score = patched_metric_change
            else:
                effect_score = baseline_per_token - patched_per_token
            normalized_effect = effect_score / clean_minus_corrupted

            for row_idx, row in batch_df.iterrows():
                rows.append(
                    {
                        "particle": str(request["particle"]),
                        "position_label": position_label,
                        "corruption_mode": CORRUPTION_MODE,
                        "top_k": int(request["top_k"]),
                        "train_fold": int(row["train_fold"]),
                        "eval_fold": int(row["eval_fold"]),
                        "target_type": row["target_type"],
                        "evaluation_id": row["evaluation_id"],
                        "source_row_index": int(row["source_row_index"]),
                        "id": row["id"],
                        "pool_row_index": row.get("pool_row_index", ""),
                        "target_text": row["target_text"],
                        "eval_mode": str(request["eval_mode"]),
                        "site_set_type": str(request["site_set_type"]),
                        "random_seed": str(request["random_seed"]),
                        "selected_site_count": int(len(selected_sites)),
                        "baseline_condition": baseline_condition,
                        "reference_condition": reference_condition,
                        "clean_log_prob": float(clean_totals[row_idx]),
                        "corrupted_log_prob": float(corrupted_totals[row_idx]),
                        "baseline_log_prob": float(baseline_totals[row_idx]),
                        "reference_log_prob": float(
                            clean_totals[row_idx]
                            if eval_mode == EVAL_MODE_SUFFICIENCY
                            else corrupted_totals[row_idx]
                        ),
                        "patched_log_prob": float(patched_totals[row_idx]),
                        "clean_token_count": int(clean_counts[row_idx]),
                        "corrupted_token_count": int(corrupted_counts[row_idx]),
                        "baseline_token_count": int(baseline_counts[row_idx]),
                        "reference_token_count": int(
                            clean_counts[row_idx]
                            if eval_mode == EVAL_MODE_SUFFICIENCY
                            else corrupted_counts[row_idx]
                        ),
                        "patched_token_count": int(patched_counts[row_idx]),
                        "clean_log_prob_per_token": float(clean_per_token[row_idx]),
                        "corrupted_log_prob_per_token": float(corrupted_per_token[row_idx]),
                        "baseline_log_prob_per_token": float(baseline_per_token[row_idx]),
                        "reference_log_prob_per_token": float(
                            clean_per_token[row_idx]
                            if eval_mode == EVAL_MODE_SUFFICIENCY
                            else corrupted_per_token[row_idx]
                        ),
                        "patched_log_prob_per_token": float(patched_per_token[row_idx]),
                        "patched_metric_change": float(patched_metric_change[row_idx]),
                        "effect_score": float(effect_score[row_idx]),
                        "normalized_effect": float(normalized_effect[row_idx]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_particle(
    model,
    tokenizer,
    particle: str,
    run_dir: Path,
    top_ks: list[int],
    random_seeds: list[int],
    eval_modes: list[str],
    analysis_bootstrap_replicates: int,
    analysis_bootstrap_seed: int,
    analysis_signflip_replicates: int,
) -> None:
    """Evaluate localized and random activation-patching site sets for one particle."""
    pool_path = run_dir / "generation_pool.tsv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing generation pool: {pool_path}")
    pool_df = pd.read_csv(pool_path, sep="\t")
    required_columns = {
        "pool_row_index",
        "source_row_index",
        "fold_id",
        "id",
        "followup",
        "response",
        "prompt_with",
        "prompt_without",
    }
    patch_dir = patching_dir(run_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    site_scores_df, position_specs, label_to_index = load_site_scores(patch_dir)
    if uses_utterance_final_position(position_specs):
        required_columns.update({"w_word", "wo_word"})
    missing = sorted(required_columns - set(pool_df.columns))
    if missing:
        raise KeyError(f"{pool_path} is missing required columns: {', '.join(missing)}")

    site_universes: dict[int, list[dict[str, int | str]]] = {}
    localized_site_sets: dict[int, dict[int, list[dict[str, int | str]]]] = {top_k: {} for top_k in top_ks}
    for fold_id in sorted(int(value) for value in pd.unique(site_scores_df["train_fold"])):
        fold_site_scores = site_scores_df.loc[site_scores_df["train_fold"] == fold_id].copy()
        universe = select_top_sites(
            fold_site_scores=fold_site_scores,
            top_k=len(fold_site_scores),
            label_to_index=label_to_index,
        )
        site_universes[fold_id] = universe
        for top_k in top_ks:
            localized_site_sets[top_k][fold_id] = select_top_sites(
                fold_site_scores=fold_site_scores,
                top_k=top_k,
                label_to_index=label_to_index,
            )

    row_frames = []
    for fold_id in sorted(int(value) for value in pd.unique(pool_df["fold_id"])):
        eval_df = build_evaluation_targets(pool_df=pool_df, train_fold=fold_id)
        for eval_mode in eval_modes:
            site_requests = build_site_set_requests(
                particle=particle,
                fold_id=fold_id,
                top_ks=top_ks,
                eval_mode=eval_mode,
                localized_site_sets=localized_site_sets,
                site_universe=site_universes[fold_id],
                random_seeds=random_seeds,
            )
            row_frames.append(
                score_patch_site_requests(
                    model=model,
                    tokenizer=tokenizer,
                    eval_df=eval_df,
                    site_requests=site_requests,
                    position_specs=position_specs,
                )
            )

    row_df = pd.concat(row_frames, ignore_index=True, sort=False)
    row_df.to_csv(patch_dir / "eval_rows.tsv", sep="\t", index=False)
    overlap_df = build_site_overlap_rows(
        localized_site_sets=localized_site_sets,
        site_universes=site_universes,
        position_label=output_position_label(position_specs),
    )
    overlap_df.to_csv(patch_dir / "site_overlap.tsv", sep="\t", index=False)
    write_patch_analysis_outputs(
        patch_dir=patch_dir,
        row_df=row_df,
        bootstrap_replicates=analysis_bootstrap_replicates,
        bootstrap_seed=analysis_bootstrap_seed,
        signflip_replicates=analysis_signflip_replicates,
    )
    print(f"Saved patching evaluation outputs under {patch_dir}")


def main() -> None:
    args = parse_args()
    top_ks = parse_top_ks(args.top_ks)
    random_seeds = parse_random_seeds(args.random_seeds)
    eval_modes = parse_eval_modes(args.eval_modes)
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    for particle in args.particles:
        normalized = normalize_particle_name(particle)
        run_dir = resolve_run_dir(args.model_name, normalized, args.output_root)
        evaluate_particle(
            model=model,
            tokenizer=tokenizer,
            particle=normalized,
            run_dir=run_dir,
            top_ks=top_ks,
            random_seeds=random_seeds,
            eval_modes=eval_modes,
            analysis_bootstrap_replicates=args.analysis_bootstrap_replicates,
            analysis_bootstrap_seed=args.analysis_bootstrap_seed,
            analysis_signflip_replicates=args.analysis_signflip_replicates,
        )


if __name__ == "__main__":
    main()
