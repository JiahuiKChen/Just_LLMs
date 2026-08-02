#!/usr/bin/env python3
"""Compare localized component/layer rankings between two source groups.

The analysis consumes row-level component-localization outputs, so it does not
load a language model or generate/evaluate any new continuations.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.plot_style import (  # noqa: E402
    CONTROL_COLOR,
    PARTICLE_COLORS,
    configure_paper_style,
)


DEFAULT_TOP_KS = (1, 3, 5, 10)
REQUIRED_COLUMNS = {
    "train_fold",
    "source_row_index",
    "component",
    "layer_index",
    "position_label",
    "corruption_mode",
    "restoration",
}
COMPONENT_ORDER = ("attn", "mlp", "resid")
COMPONENT_LABELS = {"attn": "Attention", "mlp": "MLP", "resid": "Residual"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare top-k component/layer localization overlap between two groups."
    )
    parser.add_argument("--left-site-rows", required=True, help="Left group site_rows.tsv.")
    parser.add_argument("--right-site-rows", required=True, help="Right group site_rows.tsv.")
    parser.add_argument("--left-label", default="exclusive_just")
    parser.add_argument("--right-label", default="only")
    parser.add_argument(
        "--top-ks",
        default=",".join(str(value) for value in DEFAULT_TOP_KS),
        help="Comma-separated top-k site-set sizes.",
    )
    parser.add_argument(
        "--hide-rank-correlation",
        action="store_true",
        help="Do not annotate pooled Spearman rank-correlation statistics on the plot.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_top_ks(raw_value: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw_value.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--top-ks must contain positive integers.")
    return values


def load_site_rows(path: Path, group_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Site-row table does not exist: {path}")
    frame = pd.read_csv(path, sep="\t")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["group"] = group_label
    frame["restoration"] = pd.to_numeric(frame["restoration"], errors="coerce")
    if frame["restoration"].isna().any():
        raise ValueError(f"{path} contains nonnumeric restoration values.")
    return frame


def aggregate_site_scores(site_rows: pd.DataFrame) -> pd.DataFrame:
    item_keys = [
        "group",
        "train_fold",
        "component",
        "layer_index",
        "position_label",
        "corruption_mode",
        "source_row_index",
    ]
    item_rows = site_rows.groupby(item_keys, dropna=False, sort=False)["restoration"].mean().reset_index()

    score_frames = []
    fold_keys = [
        "group",
        "train_fold",
        "component",
        "layer_index",
        "position_label",
        "corruption_mode",
    ]
    fold_scores = (
        item_rows.groupby(fold_keys, dropna=False, sort=False)["restoration"]
        .agg(ranking_score="mean", source_item_count="size")
        .reset_index()
    )
    fold_scores.insert(1, "scope", "fold")
    score_frames.append(fold_scores)

    pooled_keys = ["group", "component", "layer_index", "position_label", "corruption_mode"]
    pooled_scores = (
        item_rows.groupby(pooled_keys, dropna=False, sort=False)["restoration"]
        .agg(ranking_score="mean", source_item_count="size")
        .reset_index()
    )
    pooled_scores.insert(1, "scope", "pooled")
    pooled_scores.insert(2, "train_fold", "all")
    score_frames.append(pooled_scores)

    scores = pd.concat(score_frames, ignore_index=True, sort=False)
    return scores.sort_values(
        ["group", "scope", "train_fold", "component", "ranking_score", "layer_index"],
        ascending=[True, True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def selected_layers(scores: pd.DataFrame, top_k: int) -> list[int]:
    ordered = scores.sort_values(
        ["ranking_score", "layer_index"], ascending=[False, True], kind="mergesort"
    )
    return ordered.head(top_k)["layer_index"].astype(int).tolist()


def overlap_record(
    left_scores: pd.DataFrame,
    right_scores: pd.DataFrame,
    left_label: str,
    right_label: str,
    component: str,
    top_k: int,
    scope: str,
    left_fold: int | str,
    right_fold: int | str,
) -> dict[str, object]:
    left_universe = set(left_scores["layer_index"].astype(int))
    right_universe = set(right_scores["layer_index"].astype(int))
    if left_universe != right_universe:
        raise ValueError(
            f"Layer universes differ for {component}, {scope}, folds {left_fold}/{right_fold}."
        )
    universe_size = len(left_universe)
    if top_k > universe_size:
        raise ValueError(f"top_k={top_k} exceeds the {component} universe of {universe_size} sites.")

    left_sites = selected_layers(left_scores, top_k=top_k)
    right_sites = selected_layers(right_scores, top_k=top_k)
    left_set, right_set = set(left_sites), set(right_sites)
    shared = sorted(left_set & right_set)
    intersection = len(shared)
    union = len(left_set | right_set)
    expected_intersection = (len(left_set) * len(right_set)) / float(universe_size)
    expected_union = len(left_set) + len(right_set) - expected_intersection
    expected_jaccard = expected_intersection / expected_union if expected_union else np.nan
    expected_fraction = expected_intersection / len(left_set) if left_set else np.nan
    p_value = float(
        hypergeom.sf(intersection - 1, universe_size, len(right_set), len(left_set))
    )
    return {
        "left_group": left_label,
        "right_group": right_label,
        "scope": scope,
        "component": component,
        "top_k": top_k,
        "left_fold": left_fold,
        "right_fold": right_fold,
        "intersection_count": intersection,
        "overlap_fraction": intersection / len(left_set) if left_set else np.nan,
        "jaccard": intersection / union if union else np.nan,
        "selected_sites_left": len(left_set),
        "selected_sites_right": len(right_set),
        "site_universe": universe_size,
        "expected_random_intersection": expected_intersection,
        "expected_random_overlap_fraction": expected_fraction,
        "expected_random_jaccard": expected_jaccard,
        "intersection_enrichment": intersection / expected_intersection if expected_intersection else np.nan,
        "hypergeom_p_value": p_value,
        "left_layers": ",".join(str(value) for value in left_sites),
        "right_layers": ",".join(str(value) for value in right_sites),
        "shared_layers": ",".join(str(value) for value in shared),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.full(len(values), np.nan, dtype=float)
    running = 1.0
    for reverse_rank, idx in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[idx] * len(values) / rank)
        adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def build_overlap_tables(
    scores: pd.DataFrame,
    left_label: str,
    right_label: str,
    top_ks: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = [value for value in COMPONENT_ORDER if value in set(scores["component"])]
    pooled_records = []
    fold_records = []
    for component in components:
        left_pooled = scores.loc[
            (scores["group"] == left_label)
            & (scores["scope"] == "pooled")
            & (scores["component"] == component)
        ]
        right_pooled = scores.loc[
            (scores["group"] == right_label)
            & (scores["scope"] == "pooled")
            & (scores["component"] == component)
        ]
        for top_k in top_ks:
            pooled_records.append(
                overlap_record(
                    left_pooled,
                    right_pooled,
                    left_label,
                    right_label,
                    component,
                    top_k,
                    "pooled",
                    "all",
                    "all",
                )
            )

        left_folds = sorted(
            scores.loc[
                (scores["group"] == left_label)
                & (scores["scope"] == "fold")
                & (scores["component"] == component),
                "train_fold",
            ].unique()
        )
        right_folds = sorted(
            scores.loc[
                (scores["group"] == right_label)
                & (scores["scope"] == "fold")
                & (scores["component"] == component),
                "train_fold",
            ].unique()
        )
        for left_fold, right_fold, top_k in product(left_folds, right_folds, top_ks):
            left_fold_scores = scores.loc[
                (scores["group"] == left_label)
                & (scores["scope"] == "fold")
                & (scores["component"] == component)
                & (scores["train_fold"] == left_fold)
            ]
            right_fold_scores = scores.loc[
                (scores["group"] == right_label)
                & (scores["scope"] == "fold")
                & (scores["component"] == component)
                & (scores["train_fold"] == right_fold)
            ]
            fold_records.append(
                overlap_record(
                    left_fold_scores,
                    right_fold_scores,
                    left_label,
                    right_label,
                    component,
                    top_k,
                    "fold_pair",
                    int(left_fold),
                    int(right_fold),
                )
            )

    pooled = pd.DataFrame(pooled_records)
    pooled["hypergeom_q_value_bh"] = benjamini_hochberg(pooled["hypergeom_p_value"])
    return pooled, pd.DataFrame(fold_records)


def correlation_record(
    left_scores: pd.DataFrame,
    right_scores: pd.DataFrame,
    left_label: str,
    right_label: str,
    component: str,
    scope: str,
    left_fold: int | str,
    right_fold: int | str,
) -> dict[str, object]:
    merged = left_scores[["layer_index", "ranking_score"]].merge(
        right_scores[["layer_index", "ranking_score"]],
        on="layer_index",
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    rho, p_value = spearmanr(merged["ranking_score_left"], merged["ranking_score_right"])
    return {
        "left_group": left_label,
        "right_group": right_label,
        "scope": scope,
        "component": component,
        "left_fold": left_fold,
        "right_fold": right_fold,
        "layer_count": len(merged),
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
    }


def build_rank_correlations(
    scores: pd.DataFrame, left_label: str, right_label: str
) -> pd.DataFrame:
    records = []
    components = [value for value in COMPONENT_ORDER if value in set(scores["component"])]
    for component in components:
        left_pooled = scores.loc[
            (scores.group == left_label) & (scores.scope == "pooled") & (scores.component == component)
        ]
        right_pooled = scores.loc[
            (scores.group == right_label) & (scores.scope == "pooled") & (scores.component == component)
        ]
        records.append(
            correlation_record(
                left_pooled,
                right_pooled,
                left_label,
                right_label,
                component,
                "pooled",
                "all",
                "all",
            )
        )
        left_folds = sorted(
            scores.loc[
                (scores.group == left_label)
                & (scores.scope == "fold")
                & (scores.component == component),
                "train_fold",
            ].unique()
        )
        right_folds = sorted(
            scores.loc[
                (scores.group == right_label)
                & (scores.scope == "fold")
                & (scores.component == component),
                "train_fold",
            ].unique()
        )
        for left_fold, right_fold in product(left_folds, right_folds):
            left_fold_scores = scores.loc[
                (scores.group == left_label)
                & (scores.scope == "fold")
                & (scores.component == component)
                & (scores.train_fold == left_fold)
            ]
            right_fold_scores = scores.loc[
                (scores.group == right_label)
                & (scores.scope == "fold")
                & (scores.component == component)
                & (scores.train_fold == right_fold)
            ]
            records.append(
                correlation_record(
                    left_fold_scores,
                    right_fold_scores,
                    left_label,
                    right_label,
                    component,
                    "fold_pair",
                    int(left_fold),
                    int(right_fold),
                )
            )
    return pd.DataFrame(records)


def plot_top_k_overlap(
    pooled: pd.DataFrame,
    correlations: pd.DataFrame,
    left_label: str,
    right_label: str,
    output_dir: Path,
    show_rank_correlation: bool = True,
) -> None:
    def display_label(value: str) -> str:
        return value.replace("nonexclusive", "non-exclusive").replace("_", " ")

    components = [value for value in COMPONENT_ORDER if value in set(pooled.component)]
    figure, axes = plt.subplots(1, len(components), figsize=(10.4, 3.35), sharey=True)
    if len(components) == 1:
        axes = [axes]
    for axis, component in zip(axes, components):
        component_pooled = pooled.loc[pooled.component == component].sort_values("top_k")
        axis.plot(
            component_pooled.top_k,
            component_pooled.overlap_fraction,
            color=PARTICLE_COLORS["just"],
            marker="o",
            linewidth=2.4,
            markersize=6.5,
            label=f"{display_label(left_label).capitalize()}–{display_label(right_label)}",
        )
        axis.plot(
            component_pooled.top_k,
            component_pooled.expected_random_overlap_fraction,
            color=CONTROL_COLOR,
            linestyle="--",
            linewidth=2.0,
            label="Random expectation",
        )
        axis.set_title(COMPONENT_LABELS.get(component, component), pad=5)
        axis.set_xlabel("Top-k layer sites")
        axis.set_xticks(component_pooled.top_k)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.set_ylim(-0.02, 1.05)
        if show_rank_correlation:
            correlation = correlations.loc[
                (correlations.scope == "pooled") & (correlations.component == component)
            ]
            if len(correlation) != 1:
                raise ValueError(f"Expected one pooled rank correlation for {component}.")
            rho = float(correlation.iloc[0].spearman_rho)
            p_value = float(correlation.iloc[0].spearman_p_value)
            axis.text(
                0.04,
                0.96,
                rf"Spearman $\rho$ = {rho:.2f}" + "\n" + rf"$p$ = {p_value:.2g}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            )
    axes[0].set_ylabel("Shared top-k fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.92))
    figure.suptitle(
        "Component–Layer Site Overlap",
        y=0.995,
    )
    figure.tight_layout(rect=(0.005, 0.005, 0.995, 0.88), pad=0.4, w_pad=0.65)
    stem = "component_overlap"
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def plot_rank_scatter(
    scores: pd.DataFrame,
    correlations: pd.DataFrame,
    left_label: str,
    right_label: str,
    output_dir: Path,
) -> None:
    components = [value for value in COMPONENT_ORDER if value in set(scores.component)]
    figure, axes = plt.subplots(1, len(components), figsize=(10.4, 3.35), sharex=True, sharey=True)
    if len(components) == 1:
        axes = [axes]
    for axis, component in zip(axes, components):
        left = scores.loc[
            (scores.group == left_label) & (scores.scope == "pooled") & (scores.component == component),
            ["layer_index", "ranking_score"],
        ].copy()
        right = scores.loc[
            (scores.group == right_label) & (scores.scope == "pooled") & (scores.component == component),
            ["layer_index", "ranking_score"],
        ].copy()
        left["rank_left"] = left.ranking_score.rank(method="first", ascending=False)
        right["rank_right"] = right.ranking_score.rank(method="first", ascending=False)
        merged = left.merge(right, on="layer_index", suffixes=("_left", "_right"), validate="one_to_one")
        shared_top_ten = (merged.rank_left <= 10) & (merged.rank_right <= 10)
        axis.scatter(
            merged.loc[~shared_top_ten, "rank_left"],
            merged.loc[~shared_top_ten, "rank_right"],
            color="#adb5bd",
            alpha=0.8,
            s=28,
        )
        axis.scatter(
            merged.loc[shared_top_ten, "rank_left"],
            merged.loc[shared_top_ten, "rank_right"],
            color=PARTICLE_COLORS["just"],
            s=42,
            label="Shared top-10",
        )
        axis.plot([1, len(merged)], [1, len(merged)], color="#ced4da", linestyle="--", linewidth=1)
        axis.set_title(COMPONENT_LABELS.get(component, component))
        axis.set_xlabel(f"{left_label.replace('_', ' ').title()} rank")
        axis.grid(alpha=0.18)
        axis.invert_xaxis()
        axis.invert_yaxis()
    axes[0].set_ylabel(f"{right_label.replace('_', ' ').title()} rank")
    figure.suptitle("Component–Layer Rank Agreement", y=0.995)
    figure.tight_layout(rect=(0.005, 0.005, 0.995, 0.91), pad=0.4, w_pad=0.65)
    stem = "exclusive_just_only_component_rank_agreement"
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_paper_style()
    args = parse_args()
    top_ks = parse_top_ks(args.top_ks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_rows = load_site_rows(Path(args.left_site_rows), group_label=args.left_label)
    right_rows = load_site_rows(Path(args.right_site_rows), group_label=args.right_label)
    if set(left_rows.component) != set(right_rows.component):
        raise ValueError("The two site-row tables do not contain the same components.")
    if set(left_rows.position_label) != set(right_rows.position_label):
        raise ValueError("The two site-row tables use different localization positions.")
    if set(left_rows.corruption_mode) != set(right_rows.corruption_mode):
        raise ValueError("The two site-row tables use different corruption modes.")

    scores = aggregate_site_scores(pd.concat([left_rows, right_rows], ignore_index=True, sort=False))
    pooled, fold_pairs = build_overlap_tables(
        scores=scores,
        left_label=args.left_label,
        right_label=args.right_label,
        top_ks=top_ks,
    )
    correlations = build_rank_correlations(
        scores=scores,
        left_label=args.left_label,
        right_label=args.right_label,
    )

    scores.to_csv(output_dir / "component_site_scores.tsv", sep="\t", index=False)
    pooled.to_csv(output_dir / "component_overlap_pooled.tsv", sep="\t", index=False)
    fold_pairs.to_csv(output_dir / "component_overlap_fold_pairs.tsv", sep="\t", index=False)
    correlations.to_csv(output_dir / "component_rank_correlations.tsv", sep="\t", index=False)
    plot_top_k_overlap(
        pooled=pooled,
        correlations=correlations,
        left_label=args.left_label,
        right_label=args.right_label,
        output_dir=output_dir,
        show_rank_correlation=not args.hide_rank_correlation,
    )
    print(f"Saved component-overlap analysis under {output_dir}")


if __name__ == "__main__":
    main()
