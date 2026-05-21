#!/usr/bin/env python3
"""Ablate localized particle-responsive units and rescore held-out targets."""

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

from localization.analysis import attach_change_from_none, write_ablation_analysis_outputs
from localization.common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PARTICLES,
    DEFAULT_RANDOM_MASK_SEEDS,
    default_results_dir,
    get_device,
    load_model,
    normalize_particle_name,
    parse_mask_percents,
    parse_random_seeds,
    sample_random_mask_from_complement,
    score_target_pair,
    stable_int,
)

METHOD_FULL_FINAL_TTEST = "full_final_ttest"
METHOD_PROMPT_BOUNDARY_CORR = "prompt_boundary_corr"
SELECTION_MODE_TOP_POSITIVE_K = "top_positive_k"
SELECTION_MODE_ALL_P_GENERATIONS = "all_p_generations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate particle-localized masks and rescore held-out generations and gold followups."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument(
        "--particles",
        nargs="+",
        default=list(DEFAULT_PARTICLES),
        help="Particles to evaluate.",
    )
    parser.add_argument(
        "--mask_percents",
        default="1",
        help="Comma-separated mask percentages to evaluate.",
    )
    parser.add_argument(
        "--random_seeds",
        default=",".join(str(seed) for seed in DEFAULT_RANDOM_MASK_SEEDS),
        help="Comma-separated seeds for same-size random control masks.",
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


def load_mask_specs(run_dir: Path, mask_percents: list[float]) -> list[dict]:
    """Load saved localization masks for the requested percentages."""
    mask_files = sorted((run_dir / "masks").glob("mask_fold*_train_percent*.npz"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found under {run_dir / 'masks'}")
    requested = {float(value) for value in mask_percents}
    specs = []
    for mask_file in mask_files:
        data = np.load(mask_file, allow_pickle=False)
        mask_percent = float(data["mask_percent"])
        if requested and mask_percent not in requested:
            continue
        specs.append(
            {
                "path": mask_file,
                "fold_id": int(data["fold_id"]),
                "mask_percent": mask_percent,
                "localization_method": str(data["localization_method"].item())
                if "localization_method" in data.files
                else METHOD_FULL_FINAL_TTEST,
                "mask": data["mask"].astype(bool),
            }
        )
    if not specs:
        raise FileNotFoundError(
            f"No mask files matching mask_percents={sorted(requested)} found under {run_dir / 'masks'}"
        )
    return specs


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


def validate_pool_selection_mode(run_dir: Path, localization_methods: set[str]) -> str:
    """Ensure the evaluation pool matches the localization method used to create the masks."""
    if not localization_methods:
        raise ValueError("At least one localization method is required for selection-mode validation.")
    if len(localization_methods) != 1:
        raise ValueError(
            f"Expected one localization method per run, found: {', '.join(sorted(localization_methods))}"
        )
    localization_method = next(iter(localization_methods))
    if localization_method == METHOD_PROMPT_BOUNDARY_CORR:
        expected_mode = SELECTION_MODE_ALL_P_GENERATIONS
    else:
        expected_mode = SELECTION_MODE_TOP_POSITIVE_K
    selection_mode = load_pool_selection_mode(run_dir)
    if selection_mode != expected_mode:
        raise ValueError(
            f"{localization_method} requires selection_mode={expected_mode}, "
            f"but {run_dir / 'pool_summary.tsv'} recorded {selection_mode}"
        )
    return selection_mode


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


def score_evaluation_targets(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    particle: str,
    mask_percent: float,
    ablation_condition: str,
    layer_mask: np.ndarray | None,
    random_seed: str,
) -> pd.DataFrame:
    """Score all held-out targets for one ablation condition."""
    device = get_device(model)
    rows = []
    desc = f"{particle} {ablation_condition} {mask_percent:g}% fold {int(eval_df['train_fold'].iloc[0])}"
    for row in tqdm(eval_df.to_dict(orient="records"), total=len(eval_df), desc=desc):
        metrics = score_target_pair(
            model=model,
            tokenizer=tokenizer,
            prompt_with=row["prompt_with"],
            prompt_without=row["prompt_without"],
            target_text=row["target_text"],
            device=device,
            layer_mask=layer_mask,
        )
        rows.append(
            {
                "particle": particle,
                "mask_percent": mask_percent,
                "train_fold": int(row["train_fold"]),
                "eval_fold": int(row["eval_fold"]),
                "target_type": row["target_type"],
                "evaluation_id": row["evaluation_id"],
                "source_row_index": int(row["source_row_index"]),
                "id": row["id"],
                "pool_row_index": row.get("pool_row_index", ""),
                "target_text": row["target_text"],
                "ablation_condition": ablation_condition,
                "random_seed": random_seed,
                **metrics,
            }
        )
    return pd.DataFrame(rows)

def evaluate_particle(
    model,
    tokenizer,
    particle: str,
    run_dir: Path,
    mask_percents: list[float],
    random_seeds: list[int],
    analysis_bootstrap_replicates: int,
    analysis_bootstrap_seed: int,
    analysis_signflip_replicates: int,
) -> None:
    """Evaluate localized and random masks for one particle."""
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
    missing = sorted(required_columns - set(pool_df.columns))
    if missing:
        raise KeyError(f"{pool_path} is missing required columns: {', '.join(missing)}")

    mask_specs = load_mask_specs(run_dir, mask_percents=mask_percents)
    validate_pool_selection_mode(
        run_dir,
        localization_methods={str(spec["localization_method"]) for spec in mask_specs},
    )
    row_frames = []
    for spec in mask_specs:
        train_fold = spec["fold_id"]
        mask_percent = spec["mask_percent"]
        localized_mask = spec["mask"]
        eval_df = build_evaluation_targets(pool_df, train_fold=train_fold)

        row_frames.append(
            score_evaluation_targets(
                model=model,
                tokenizer=tokenizer,
                eval_df=eval_df,
                particle=particle,
                mask_percent=mask_percent,
                ablation_condition="none",
                layer_mask=None,
                random_seed="",
            )
        )
        row_frames.append(
            score_evaluation_targets(
                model=model,
                tokenizer=tokenizer,
                eval_df=eval_df,
                particle=particle,
                mask_percent=mask_percent,
                ablation_condition="localized",
                layer_mask=localized_mask,
                random_seed="",
            )
        )
        for seed in random_seeds:
            random_mask = sample_random_mask_from_complement(
                localized_mask,
                seed=stable_int(particle, train_fold, mask_percent, seed),
            )
            row_frames.append(
                score_evaluation_targets(
                    model=model,
                    tokenizer=tokenizer,
                    eval_df=eval_df,
                    particle=particle,
                    mask_percent=mask_percent,
                    ablation_condition="random",
                    layer_mask=random_mask,
                    random_seed=str(seed),
                )
            )

    row_df = pd.concat(row_frames, ignore_index=True, sort=False)
    row_df = attach_change_from_none(row_df)
    row_df.to_csv(run_dir / "ablation_row_metrics.tsv", sep="\t", index=False)
    write_ablation_analysis_outputs(
        run_dir=run_dir,
        row_df=row_df,
        bootstrap_replicates=analysis_bootstrap_replicates,
        bootstrap_seed=analysis_bootstrap_seed,
        signflip_replicates=analysis_signflip_replicates,
    )
    print(f"Saved ablation outputs under {run_dir}")


def main() -> None:
    args = parse_args()
    mask_percents = parse_mask_percents(args.mask_percents)
    random_seeds = parse_random_seeds(args.random_seeds)
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    for particle in args.particles:
        particle = normalize_particle_name(particle)
        run_dir = resolve_run_dir(args.model_name, particle, args.output_root)
        evaluate_particle(
            model=model,
            tokenizer=tokenizer,
            particle=particle,
            run_dir=run_dir,
            mask_percents=mask_percents,
            random_seeds=random_seeds,
            analysis_bootstrap_replicates=args.analysis_bootstrap_replicates,
            analysis_bootstrap_seed=args.analysis_bootstrap_seed,
            analysis_signflip_replicates=args.analysis_signflip_replicates,
        )


if __name__ == "__main__":
    main()
