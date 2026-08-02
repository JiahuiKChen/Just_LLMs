#!/usr/bin/env python3
"""Compare four directional just/only cross-particle transfer effects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.plot_style import (  # noqa: E402
    PARTICLE_COLORS,
    configure_paper_style,
    use_scientific_y_axis,
)


COMPONENT_ORDER = ("attn", "mlp", "resid")
COMPONENT_LABELS = {"attn": "Attention", "mlp": "MLP", "resid": "Residual"}
TOP_KS = (1, 3, 5, 10)
METRIC_MEAN = "effect_score_mean"
METRIC_LOW = "effect_score_mean_ci_low"
METRIC_HIGH = "effect_score_mean_ci_high"
TRANSFER_ORDER = (
    ("just_to_only", "nonexclusive"),
    ("only_to_just", "nonexclusive"),
    ("just_to_only", "exclusive"),
    ("only_to_just", "exclusive"),
)
TRANSFER_COLORS = {
    ("just_to_only", "nonexclusive"): "#4C78A8",
    ("only_to_just", "nonexclusive"): "#B279A2",
    ("just_to_only", "exclusive"): PARTICLE_COLORS["just"],
    ("only_to_just", "exclusive"): PARTICLE_COLORS["only"],
}
TRANSFER_LABELS = {
    ("just_to_only", "nonexclusive"): "Non-exclusive just → only",
    ("only_to_just", "nonexclusive"): "Only → non-exclusive just",
    ("just_to_only", "exclusive"): "Exclusive just → only",
    ("only_to_just", "exclusive"): "Only → exclusive just",
}
CONDITION_STYLES = {
    "cross": {"linestyle": "-", "marker": "o", "linewidth": 2.4, "label": "Cross-particle"},
    "random": {
        "linestyle": "--",
        "marker": "^",
        "linewidth": 1.9,
        "label": "Matched random baseline",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot four directional just/only cross-particle transfers and random controls."
    )
    parser.add_argument(
        "--exclusive-transfer-root",
        default=(
            "localization/results/component_transfer_runs/"
            "20260728_llama3_instruct_exclusive_just_only_cross_patch"
        ),
    )
    parser.add_argument(
        "--nonexclusive-transfer-root",
        default=(
            "localization/results/component_transfer_runs/"
            "20260729_llama3_instruct_nonexclusive_just_only_cross_patch"
        ),
    )
    parser.add_argument("--eval-mode", choices=("sufficiency", "necessity"), default="necessity")
    parser.add_argument(
        "--target-type",
        choices=("generated", "gold", "both"),
        default="both",
        help="Follow-up target set to plot (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        default="localization/results/plots/just_type_component_transfer/instruct",
    )
    return parser.parse_args()


def summary_path(run_dir: Path) -> Path:
    return run_dir / "component_patching" / "eval_summary.tsv"


def load_summary(path: Path, eval_mode: str, target_type: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing component evaluation summary: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {
        "component",
        "eval_mode",
        "target_type",
        "top_k",
        "site_set_type",
        METRIC_MEAN,
        METRIC_LOW,
        METRIC_HIGH,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {', '.join(missing)}")
    frame = frame.loc[
        (frame.eval_mode == eval_mode) & (frame.target_type == target_type)
    ].copy()
    if frame.empty:
        raise ValueError(f"No {eval_mode}/{target_type} rows in {path}")
    frame["source_path"] = str(path)
    frame["top_k"] = pd.to_numeric(frame.top_k, errors="raise").astype(int)
    for column in (METRIC_MEAN, METRIC_LOW, METRIC_HIGH):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def apply_ablation_sign(frame: pd.DataFrame, eval_mode: str) -> pd.DataFrame:
    signed = frame.copy()
    signed["effect_score_original_mean"] = signed[METRIC_MEAN]
    signed["effect_score_original_ci_low"] = signed[METRIC_LOW]
    signed["effect_score_original_ci_high"] = signed[METRIC_HIGH]
    if eval_mode == "necessity":
        signed[METRIC_MEAN] = -signed["effect_score_original_mean"]
        signed[METRIC_LOW] = -signed["effect_score_original_ci_high"]
        signed[METRIC_HIGH] = -signed["effect_score_original_ci_low"]
        signed["metric_sign_convention"] = "patched_minus_baseline"
    else:
        signed["metric_sign_convention"] = "baseline_minus_patched"
    return signed


def select_curve(
    frame: pd.DataFrame,
    site_set_type: str,
    direction: str,
    group: str,
    condition: str,
) -> pd.DataFrame:
    curve = frame.loc[frame.site_set_type == site_set_type].copy()
    if curve.empty:
        raise ValueError(f"No {site_set_type} rows for {direction}/{group}/{condition}")
    curve["direction"] = direction
    curve["group"] = group
    curve["condition"] = condition
    return curve


def build_plot_data(args: argparse.Namespace, target_type: str) -> pd.DataFrame:
    exclusive_root = Path(args.exclusive_transfer_root)
    nonexclusive_root = Path(args.nonexclusive_transfer_root)
    transfer_specs = (
        (
            exclusive_root / "exclusive_just_to_only",
            "just_to_only",
            "exclusive",
        ),
        (
            nonexclusive_root / "nonexclusive_just_to_only",
            "just_to_only",
            "nonexclusive",
        ),
        (
            exclusive_root / "only_to_exclusive_just",
            "only_to_just",
            "exclusive",
        ),
        (
            nonexclusive_root / "only_to_nonexclusive_just",
            "only_to_just",
            "nonexclusive",
        ),
    )
    curves = []
    for run_dir, direction, group in transfer_specs:
        frame = load_summary(
            summary_path(run_dir),
            eval_mode=args.eval_mode,
            target_type=target_type,
        )
        curves.append(select_curve(frame, "localized", direction, group, "cross"))
        curves.append(select_curve(frame, "random_mean", direction, group, "random"))

    plot_data = apply_ablation_sign(pd.concat(curves, ignore_index=True), args.eval_mode)
    expected_components = set(COMPONENT_ORDER)
    if set(plot_data.component) != expected_components:
        raise ValueError(
            f"Expected components {sorted(expected_components)}, found {sorted(set(plot_data.component))}"
        )
    if set(plot_data.top_k) != set(TOP_KS):
        raise ValueError(f"Expected top-k values {TOP_KS}, found {sorted(set(plot_data.top_k))}")
    duplicate_keys = ["direction", "group", "condition", "component", "top_k"]
    if plot_data.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate comparison curves were loaded.")
    return plot_data.sort_values(duplicate_keys, kind="mergesort").reset_index(drop=True)


def row_limits(frame: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat(
        [frame[METRIC_MEAN], frame[METRIC_LOW], frame[METRIC_HIGH]], ignore_index=True
    ).dropna()
    lower = min(float(values.min()), 0.0)
    upper = max(float(values.max()), 0.0)
    span = upper - lower
    padding = max(span * 0.12, 5e-4)
    return lower - padding, upper + padding


def plot_curve(axis, curve: pd.DataFrame) -> None:
    group = str(curve.group.iloc[0])
    direction = str(curve.direction.iloc[0])
    condition = str(curve.condition.iloc[0])
    style = CONDITION_STYLES[condition]
    color = TRANSFER_COLORS[(direction, group)]
    zorder = 4 if condition == "cross" else 2
    ordered = curve.sort_values("top_k")
    x_values = ordered.top_k.to_numpy(dtype=float)
    y_values = ordered[METRIC_MEAN].to_numpy(dtype=float)
    axis.plot(
        x_values,
        y_values,
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=style["linewidth"],
        markersize=6.0,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=1.0,
        zorder=zorder,
    )
    axis.fill_between(
        x_values,
        ordered[METRIC_LOW].to_numpy(dtype=float),
        ordered[METRIC_HIGH].to_numpy(dtype=float),
        color=color,
        alpha=0.09 if condition == "random" else 0.11,
        linewidth=0,
        zorder=1,
    )


def plot_comparison(
    plot_data: pd.DataFrame,
    output_dir: Path,
    target_type: str,
) -> None:
    figure, axes = plt.subplots(
        nrows=len(COMPONENT_ORDER),
        ncols=1,
        figsize=(7.4, 8.0),
        sharex=True,
        squeeze=False,
    )
    for row_index, component in enumerate(COMPONENT_ORDER):
        component_rows = plot_data.loc[plot_data.component == component]
        limits = row_limits(component_rows)
        axis = axes[row_index][0]
        axis.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.5)
        for direction, group in TRANSFER_ORDER:
            pair_rows = component_rows.loc[
                (component_rows.direction == direction) & (component_rows.group == group)
            ]
            for condition in ("cross", "random"):
                curve = pair_rows.loc[pair_rows.condition == condition]
                if curve.empty:
                    raise ValueError(f"Missing {direction}/{group}/{condition}/{component} curve")
                plot_curve(axis, curve)
        axis.set_ylim(*limits)
        axis.set_xticks(TOP_KS)
        axis.grid(True, axis="y", alpha=0.18)
        use_scientific_y_axis(axis)
        axis.set_title(COMPONENT_LABELS[component])
        axis.set_ylabel("Δ log p / token")
        if row_index == len(COMPONENT_ORDER) - 1:
            axis.set_xlabel("Top-k component–layer sites")

    transfer_handles = [
        Line2D(
            [],
            [],
            color=TRANSFER_COLORS[key],
            linewidth=3.0,
            label=TRANSFER_LABELS[key],
        )
        for key in TRANSFER_ORDER
    ]
    condition_handles = [
        Line2D(
            [],
            [],
            color="#252525",
            linestyle=CONDITION_STYLES[condition]["linestyle"],
            marker=CONDITION_STYLES[condition]["marker"],
            linewidth=2.1,
            label=CONDITION_STYLES[condition]["label"],
        )
        for condition in ("cross", "random")
    ]
    figure.legend(
        transfer_handles,
        [handle.get_label() for handle in transfer_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.35,
        handletextpad=0.45,
    )
    figure.legend(
        condition_handles,
        [handle.get_label() for handle in condition_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.875),
        columnspacing=1.35,
        handletextpad=0.45,
    )
    target_label = "Generated Follow-ups" if target_type == "generated" else "Gold Follow-ups"
    figure.suptitle(f"Cross-Particle Effects: {target_label}", y=0.998)
    figure.tight_layout(rect=(0.005, 0.005, 0.995, 0.825), pad=0.55, h_pad=0.65)
    stem = f"component_transfer_{target_type}"
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    configure_paper_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_types = ("generated", "gold") if args.target_type == "both" else (args.target_type,)
    for target_type in target_types:
        plot_data = build_plot_data(args, target_type=target_type)
        plot_data.to_csv(
            output_dir / f"component_transfer_{target_type}_plot_data.tsv",
            sep="\t",
            index=False,
        )
        plot_comparison(
            plot_data,
            output_dir=output_dir,
            target_type=target_type,
        )
    print(f"Saved just-type component-transfer comparison under {output_dir}")


if __name__ == "__main__":
    main()
