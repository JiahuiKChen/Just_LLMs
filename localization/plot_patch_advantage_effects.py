#!/usr/bin/env python3
"""Plot localized-vs-random patch advantage effects from evaluation summaries."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CONDITIONS = ("localized", "random_mean")
CONDITION_STYLES = {
    "localized": {
        "color": "#0B6E4F",
        "linestyle": "-",
        "marker": "o",
        "label": "Localized",
    },
    "random_mean": {
        "color": "#6B7280",
        "linestyle": "--",
        "marker": "^",
        "label": "Random",
    },
}
COMPONENT_ORDER = ("attn", "mlp", "resid")
EVAL_MODE_ORDER = ("sufficiency", "necessity")
METRIC_MEAN_COLUMN = "effect_score_mean"
METRIC_CI_LOW_COLUMN = "effect_score_mean_ci_low"
METRIC_CI_HIGH_COLUMN = "effect_score_mean_ci_high"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-token patch advantage changes for localized and random site sets."
    )
    parser.add_argument("--particles", nargs="+", required=True, help="Particles to include in column order.")
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        metavar="LABEL=ROOT",
        help="Run root containing per-particle subdirectories.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        metavar="LABEL:PARTICLE=DIR",
        help="Per-particle run directory override.",
    )
    parser.add_argument(
        "--method",
        choices=("component", "residual", "both"),
        default="both",
        help="Which localization result type to plot.",
    )
    parser.add_argument(
        "--layout",
        choices=("full", "ablation_style"),
        default="full",
        help="full keeps sufficiency/necessity panels; ablation_style plots the selected eval mode only.",
    )
    parser.add_argument(
        "--ablation-style-eval-mode",
        choices=EVAL_MODE_ORDER,
        default="necessity",
        help="Evaluation mode to use for ablation_style plots.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for figures and plotting data.")
    return parser.parse_args()


def pretty_label(label: str) -> str:
    words = label.replace("_", " ").replace("-", " ").split()
    return " ".join(word.capitalize() for word in words)


def parse_run_roots(specs: list[str]) -> OrderedDict[str, Path]:
    run_roots: OrderedDict[str, Path] = OrderedDict()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=ROOT, got: {spec}")
        label, root = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Missing label in --run-root spec: {spec}")
        run_roots[label] = Path(root.strip())
    return run_roots


def parse_run_dir_overrides(specs: list[str]) -> tuple[OrderedDict[str, None], dict[tuple[str, str], Path]]:
    run_labels: OrderedDict[str, None] = OrderedDict()
    overrides: dict[tuple[str, str], Path] = {}
    for spec in specs:
        if "=" not in spec or ":" not in spec.split("=", 1)[0]:
            raise ValueError(f"Expected LABEL:PARTICLE=DIR, got: {spec}")
        lhs, run_dir = spec.split("=", 1)
        label, particle = lhs.split(":", 1)
        label = label.strip()
        particle = particle.strip().lower()
        if not label or not particle:
            raise ValueError(f"Malformed --run-dir spec: {spec}")
        run_labels[label] = None
        overrides[(label, particle)] = Path(run_dir.strip())
    return run_labels, overrides


def resolve_run_dir(
    label: str,
    particle: str,
    run_roots: OrderedDict[str, Path],
    overrides: dict[tuple[str, str], Path],
) -> Path:
    override = overrides.get((label, particle))
    if override is not None:
        return override
    root = run_roots.get(label)
    if root is None:
        raise KeyError(f"No run directory found for {label}:{particle}")
    return root / particle


def summary_subdir(method: str) -> str:
    if method == "component":
        return "component_patching"
    if method == "residual":
        return "patching"
    raise ValueError(f"Unsupported method: {method}")


def load_method_rows(
    method: str,
    run_labels: list[str],
    particles: list[str],
    run_roots: OrderedDict[str, Path],
    overrides: dict[tuple[str, str], Path],
) -> pd.DataFrame:
    rows = []
    subdir = summary_subdir(method)
    for label in run_labels:
        for particle in particles:
            run_dir = resolve_run_dir(label, particle, run_roots, overrides)
            summary_path = run_dir / subdir / "eval_summary.tsv"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing {method} patch summary: {summary_path}")
            summary_df = pd.read_csv(summary_path, sep="\t")
            if "component" not in summary_df.columns:
                summary_df["component"] = "residual_stream"
            if "position_label" not in summary_df.columns:
                summary_df["position_label"] = ""
            if "corruption_mode" not in summary_df.columns:
                summary_df["corruption_mode"] = ""
            subset = summary_df.loc[summary_df["site_set_type"].isin(CONDITIONS)].copy()
            needed = [
                "particle",
                "component",
                "position_label",
                "corruption_mode",
                "eval_mode",
                "target_type",
                "top_k",
                "site_set_type",
                METRIC_MEAN_COLUMN,
            ]
            missing = sorted(set(needed) - set(subset.columns))
            if missing:
                raise KeyError(f"{summary_path} is missing columns: {', '.join(missing)}")
            subset = subset.loc[:, needed]
            subset[METRIC_CI_LOW_COLUMN] = (
                pd.to_numeric(summary_df.loc[subset.index, METRIC_CI_LOW_COLUMN], errors="coerce")
                if METRIC_CI_LOW_COLUMN in summary_df.columns
                else float("nan")
            )
            subset[METRIC_CI_HIGH_COLUMN] = (
                pd.to_numeric(summary_df.loc[subset.index, METRIC_CI_HIGH_COLUMN], errors="coerce")
                if METRIC_CI_HIGH_COLUMN in summary_df.columns
                else float("nan")
            )
            subset["method"] = method
            subset["run_label"] = label
            subset["run_label_pretty"] = pretty_label(label)
            subset["run_dir"] = str(run_dir)
            rows.append(subset)
    if not rows:
        raise ValueError(f"No {method} plotting rows were loaded.")
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined["top_k"] = pd.to_numeric(combined["top_k"], errors="raise").astype(int)
    for column in (METRIC_MEAN_COLUMN, METRIC_CI_LOW_COLUMN, METRIC_CI_HIGH_COLUMN):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def ordered_components(values: pd.Series) -> list[str]:
    present = {str(value) for value in values.dropna().unique()}
    ordered = [component for component in COMPONENT_ORDER if component in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def ordered_eval_modes(values: pd.Series) -> list[str]:
    present = {str(value) for value in values.dropna().unique()}
    ordered = [mode for mode in EVAL_MODE_ORDER if mode in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def row_limits(panel_df: pd.DataFrame) -> tuple[float, float] | None:
    finite_values = pd.concat(
        [
            pd.to_numeric(panel_df[METRIC_MEAN_COLUMN], errors="coerce"),
            pd.to_numeric(panel_df[METRIC_CI_LOW_COLUMN], errors="coerce"),
            pd.to_numeric(panel_df[METRIC_CI_HIGH_COLUMN], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if finite_values.empty:
        return None
    y_min = min(float(finite_values.min()), 0.0)
    y_max = max(float(finite_values.max()), 0.0)
    y_span = y_max - y_min
    padding = max(y_span * 0.12, 5e-4)
    return y_min - padding, y_max + padding


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def add_condition_curves(ax, panel_df: pd.DataFrame) -> None:
    ax.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.5)
    for condition in CONDITIONS:
        condition_df = panel_df.loc[panel_df["site_set_type"] == condition].sort_values("top_k")
        if condition_df.empty:
            continue
        style = CONDITION_STYLES[condition]
        x_values = condition_df["top_k"].to_numpy(dtype=float)
        y_values = condition_df[METRIC_MEAN_COLUMN].to_numpy(dtype=float)
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.5,
            label=style["label"],
        )
        ci_low = condition_df[METRIC_CI_LOW_COLUMN].to_numpy(dtype=float)
        ci_high = condition_df[METRIC_CI_HIGH_COLUMN].to_numpy(dtype=float)
        if pd.notna(ci_low).all() and pd.notna(ci_high).all():
            ax.fill_between(x_values, ci_low, ci_high, color=style["color"], alpha=0.12, linewidth=0)


def ablation_style_signed_rows(plot_df: pd.DataFrame, eval_mode: str) -> pd.DataFrame:
    """Use the ablation sign convention for necessity: patched minus baseline."""
    signed_df = plot_df.loc[plot_df["eval_mode"] == eval_mode].copy()
    if signed_df.empty:
        raise ValueError(f"No rows found for eval_mode={eval_mode}")
    signed_df["effect_score_original_mean"] = signed_df[METRIC_MEAN_COLUMN]
    signed_df["effect_score_original_ci_low"] = signed_df[METRIC_CI_LOW_COLUMN]
    signed_df["effect_score_original_ci_high"] = signed_df[METRIC_CI_HIGH_COLUMN]
    signed_df[METRIC_MEAN_COLUMN] = -signed_df["effect_score_original_mean"]
    signed_df[METRIC_CI_LOW_COLUMN] = -signed_df["effect_score_original_ci_high"]
    signed_df[METRIC_CI_HIGH_COLUMN] = -signed_df["effect_score_original_ci_low"]
    signed_df["metric_sign_convention"] = "patched_minus_baseline"
    return signed_df


def legend_handles(ax):
    handles = []
    labels = []
    for condition in CONDITIONS:
        style = CONDITION_STYLES[condition]
        handle = ax.plot(
            [],
            [],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.5,
            label=style["label"],
        )[0]
        handles.append(handle)
        labels.append(style["label"])
    return handles, labels


def plot_component_target(plot_df: pd.DataFrame, particles: list[str], target_type: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    target_df = plot_df.loc[plot_df["target_type"] == target_type].copy()
    if target_df.empty:
        raise ValueError(f"No component rows for target_type={target_type}")
    components = ordered_components(target_df["component"])
    eval_modes = ordered_eval_modes(target_df["eval_mode"])
    row_specs = [(component, eval_mode) for component in components for eval_mode in eval_modes]

    fig, axes = plt.subplots(
        nrows=len(row_specs),
        ncols=len(particles),
        figsize=(4.2 * len(particles), 2.55 * len(row_specs)),
        sharex=True,
        squeeze=False,
    )
    top_ks = sorted(target_df["top_k"].dropna().unique())

    for row_idx, (component, eval_mode) in enumerate(row_specs):
        row_df = target_df.loc[
            (target_df["component"] == component) & (target_df["eval_mode"] == eval_mode)
        ].copy()
        limits = row_limits(row_df)
        for col_idx, particle in enumerate(particles):
            ax = axes[row_idx][col_idx]
            panel_df = row_df.loc[row_df["particle"] == particle].copy()
            if panel_df.empty:
                continue
            add_condition_curves(ax, panel_df)
            if limits is not None:
                ax.set_ylim(*limits)
            ax.grid(True, axis="y", alpha=0.18)
            ax.set_xticks(top_ks)
            if row_idx == 0:
                ax.set_title(particle, fontsize=12)
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Top-k component/layer sites")
            if col_idx == 0:
                ax.set_ylabel(f"{pretty_label(component)}\n{pretty_label(eval_mode)}\nDelta advantage / token")

    title_target = "Generated holdout" if target_type == "generated" else "Gold followups"
    fig.suptitle(
        f"{title_target}: component localization patch advantage change",
        fontsize=14,
        y=0.995,
    )
    handles, labels = legend_handles(axes[0][0])
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_figure(fig, output_dir, f"component_patch_advantage_effects_{target_type}")
    plt.close(fig)


def plot_residual_target(plot_df: pd.DataFrame, particles: list[str], target_type: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    target_df = plot_df.loc[plot_df["target_type"] == target_type].copy()
    if target_df.empty:
        raise ValueError(f"No residual rows for target_type={target_type}")
    eval_modes = ordered_eval_modes(target_df["eval_mode"])

    fig, axes = plt.subplots(
        nrows=len(eval_modes),
        ncols=len(particles),
        figsize=(4.2 * len(particles), 2.8 * len(eval_modes)),
        sharex=True,
        squeeze=False,
    )
    top_ks = sorted(target_df["top_k"].dropna().unique())

    for row_idx, eval_mode in enumerate(eval_modes):
        row_df = target_df.loc[target_df["eval_mode"] == eval_mode].copy()
        limits = row_limits(row_df)
        for col_idx, particle in enumerate(particles):
            ax = axes[row_idx][col_idx]
            panel_df = row_df.loc[row_df["particle"] == particle].copy()
            if panel_df.empty:
                continue
            add_condition_curves(ax, panel_df)
            if limits is not None:
                ax.set_ylim(*limits)
            ax.grid(True, axis="y", alpha=0.18)
            ax.set_xticks(top_ks)
            if row_idx == 0:
                ax.set_title(particle, fontsize=12)
            if row_idx == len(eval_modes) - 1:
                ax.set_xlabel("Top-k residual-stream sites")
            if col_idx == 0:
                ax.set_ylabel(f"{pretty_label(eval_mode)}\nDelta advantage / token")

    title_target = "Generated holdout" if target_type == "generated" else "Gold followups"
    fig.suptitle(
        f"{title_target}: residual-stream localization patch advantage change",
        fontsize=14,
        y=0.995,
    )
    handles, labels = legend_handles(axes[0][0])
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    save_figure(fig, output_dir, f"residual_patch_advantage_effects_{target_type}")
    plt.close(fig)


def plot_component_ablation_style_target(
    plot_df: pd.DataFrame,
    particles: list[str],
    target_type: str,
    eval_mode: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    target_df = plot_df.loc[
        (plot_df["target_type"] == target_type) & (plot_df["eval_mode"] == eval_mode)
    ].copy()
    if target_df.empty:
        raise ValueError(f"No component rows for target_type={target_type}, eval_mode={eval_mode}")
    components = ordered_components(target_df["component"])

    fig, axes = plt.subplots(
        nrows=len(components),
        ncols=len(particles),
        figsize=(4.2 * len(particles), 2.8 * len(components)),
        sharex=True,
        squeeze=False,
    )
    top_ks = sorted(target_df["top_k"].dropna().unique())

    for row_idx, component in enumerate(components):
        row_df = target_df.loc[target_df["component"] == component].copy()
        limits = row_limits(row_df)
        for col_idx, particle in enumerate(particles):
            ax = axes[row_idx][col_idx]
            panel_df = row_df.loc[row_df["particle"] == particle].copy()
            if panel_df.empty:
                continue
            add_condition_curves(ax, panel_df)
            if limits is not None:
                ax.set_ylim(*limits)
            ax.grid(True, axis="y", alpha=0.18)
            ax.set_xticks(top_ks)
            if row_idx == 0:
                ax.set_title(particle, fontsize=12)
            if row_idx == len(components) - 1:
                ax.set_xlabel("Top-k component/layer sites")
            if col_idx == 0:
                ax.set_ylabel(f"{pretty_label(component)}\nPatch change / token")

    title_target = "Generated holdout" if target_type == "generated" else "Gold followups"
    fig.suptitle(
        f"{title_target}: component {pretty_label(eval_mode)} patch change (patched - baseline)",
        fontsize=14,
        y=0.995,
    )
    handles, labels = legend_handles(axes[0][0])
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save_figure(fig, output_dir, f"component_patch_advantage_effects_ablation_style_{target_type}")
    plt.close(fig)


def plot_residual_ablation_style_target(
    plot_df: pd.DataFrame,
    particles: list[str],
    target_type: str,
    eval_mode: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    target_df = plot_df.loc[
        (plot_df["target_type"] == target_type) & (plot_df["eval_mode"] == eval_mode)
    ].copy()
    if target_df.empty:
        raise ValueError(f"No residual rows for target_type={target_type}, eval_mode={eval_mode}")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(particles),
        figsize=(4.2 * len(particles), 2.95),
        sharex=True,
        squeeze=False,
    )
    top_ks = sorted(target_df["top_k"].dropna().unique())
    limits = row_limits(target_df)

    for col_idx, particle in enumerate(particles):
        ax = axes[0][col_idx]
        panel_df = target_df.loc[target_df["particle"] == particle].copy()
        if panel_df.empty:
            continue
        add_condition_curves(ax, panel_df)
        if limits is not None:
            ax.set_ylim(*limits)
        ax.grid(True, axis="y", alpha=0.18)
        ax.set_xticks(top_ks)
        ax.set_title(particle, fontsize=12)
        ax.set_xlabel("Top-k residual-stream sites")
        if col_idx == 0:
            ax.set_ylabel("Patch change / token")

    title_target = "Generated holdout" if target_type == "generated" else "Gold followups"
    fig.suptitle(
        f"{title_target}: residual-stream {pretty_label(eval_mode)} patch change (patched - baseline)",
        fontsize=14,
        y=0.995,
    )
    handles, labels = legend_handles(axes[0][0])
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, output_dir, f"residual_patch_advantage_effects_ablation_style_{target_type}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    particles = [particle.strip().lower() for particle in args.particles]
    run_roots = parse_run_roots(args.run_root)
    override_labels, overrides = parse_run_dir_overrides(args.run_dir)

    run_labels: list[str] = []
    seen = set()
    for label in list(override_labels.keys()) + list(run_roots.keys()):
        if label not in seen:
            seen.add(label)
            run_labels.append(label)
    if not run_labels:
        raise ValueError("Provide at least one --run-root or --run-dir.")
    if len(run_labels) != 1:
        raise ValueError("This plot layout expects one run label; provide one --run-root or run-dir label.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ("component", "residual") if args.method == "both" else (args.method,)
    for method in methods:
        plot_df = load_method_rows(
            method=method,
            run_labels=run_labels,
            particles=particles,
            run_roots=run_roots,
            overrides=overrides,
        )
        output_df = (
            ablation_style_signed_rows(plot_df=plot_df, eval_mode=args.ablation_style_eval_mode)
            if args.layout == "ablation_style"
            else plot_df
        )
        output_df.to_csv(output_dir / f"{method}_patch_advantage_plot_data.tsv", sep="\t", index=False)
        for target_type in ("generated", "gold"):
            if method == "component" and args.layout == "full":
                plot_component_target(
                    plot_df=plot_df,
                    particles=particles,
                    target_type=target_type,
                    output_dir=output_dir,
                )
            elif method == "component":
                plot_component_ablation_style_target(
                    plot_df=output_df,
                    particles=particles,
                    target_type=target_type,
                    eval_mode=args.ablation_style_eval_mode,
                    output_dir=output_dir,
                )
            elif args.layout == "full":
                plot_residual_target(
                    plot_df=plot_df,
                    particles=particles,
                    target_type=target_type,
                    output_dir=output_dir,
                )
            else:
                plot_residual_ablation_style_target(
                    plot_df=output_df,
                    particles=particles,
                    target_type=target_type,
                    eval_mode=args.ablation_style_eval_mode,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    main()
