#!/usr/bin/env python3
"""Plot component prompt-boundary patching effects from evaluation summaries."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PATCH_DIR_NAME = "component_patching"
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
METRIC_LABELS = {
    "effect_score_mean": "Effect / token",
    "normalized_effect_mean": "Normalized effect",
    "patched_metric_change_mean": "Patched change / token",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot localized-vs-random effects for component prompt-boundary patching."
    )
    parser.add_argument(
        "--particles",
        nargs="+",
        required=True,
        help="Particles to include.",
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        metavar="LABEL=ROOT",
        help="Run root containing per-particle directories.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        metavar="LABEL:PARTICLE=DIR",
        help="Per-particle run directory override.",
    )
    parser.add_argument(
        "--metric",
        default="effect_score_mean",
        choices=sorted(METRIC_LABELS),
        help="Summary metric to plot for localized and random curves.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where figures and plotting data will be written.",
    )
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
        root_path = Path(root.strip())
        if not label:
            raise ValueError(f"Missing label in --run-root spec: {spec}")
        run_roots[label] = root_path
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


def metric_ci_columns(metric: str) -> tuple[str, str]:
    suffix = "_mean"
    if not metric.endswith(suffix):
        raise ValueError(f"Metric must end in {suffix!r}: {metric}")
    stem = metric[: -len(suffix)]
    return f"{stem}_ci_low", f"{stem}_ci_high"


def load_plot_rows(
    run_labels: list[str],
    particles: list[str],
    run_roots: OrderedDict[str, Path],
    overrides: dict[tuple[str, str], Path],
    metric: str,
) -> pd.DataFrame:
    rows = []
    ci_low_column, ci_high_column = metric_ci_columns(metric)
    for label in run_labels:
        for particle in particles:
            run_dir = resolve_run_dir(label, particle, run_roots, overrides)
            summary_path = run_dir / PATCH_DIR_NAME / "eval_summary.tsv"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing component patch summary: {summary_path}")
            summary_df = pd.read_csv(summary_path, sep="\t")
            if "component" not in summary_df.columns:
                summary_df["component"] = "resid"
            if "position_label" not in summary_df.columns:
                summary_df["position_label"] = "prompt_boundary"
            if "corruption_mode" not in summary_df.columns:
                summary_df["corruption_mode"] = "prompt_without"

            needed = [
                "particle",
                "component",
                "position_label",
                "corruption_mode",
                "target_type",
                "eval_mode",
                "top_k",
                "site_set_type",
                metric,
            ]
            missing = sorted(set(needed) - set(summary_df.columns))
            if missing:
                raise KeyError(f"{summary_path} is missing columns: {', '.join(missing)}")

            subset = summary_df.loc[summary_df["site_set_type"].isin(CONDITIONS)].copy()
            subset = subset.loc[:, needed]
            subset[ci_low_column] = (
                pd.to_numeric(summary_df.loc[subset.index, ci_low_column], errors="coerce")
                if ci_low_column in summary_df.columns
                else float("nan")
            )
            subset[ci_high_column] = (
                pd.to_numeric(summary_df.loc[subset.index, ci_high_column], errors="coerce")
                if ci_high_column in summary_df.columns
                else float("nan")
            )
            subset["run_label"] = label
            subset["run_label_pretty"] = pretty_label(label)
            subset["run_dir"] = str(run_dir)
            rows.append(subset)

    if not rows:
        raise ValueError("No plotting rows were loaded.")
    combined = pd.concat(rows, ignore_index=True)
    combined["top_k"] = pd.to_numeric(combined["top_k"], errors="raise").astype(int)
    combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
    return combined


def component_order(values: pd.Series) -> list[str]:
    present = {str(value) for value in values.dropna().unique()}
    ordered = [component for component in COMPONENT_ORDER if component in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def y_limits(panel_df: pd.DataFrame, metric: str) -> tuple[float, float] | None:
    ci_low_column, ci_high_column = metric_ci_columns(metric)
    finite_values = pd.concat(
        [
            pd.to_numeric(panel_df[metric], errors="coerce"),
            pd.to_numeric(panel_df[ci_low_column], errors="coerce"),
            pd.to_numeric(panel_df[ci_high_column], errors="coerce"),
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


def plot_effect_curves(
    plot_df: pd.DataFrame,
    run_labels: list[str],
    particle: str,
    target_type: str,
    metric: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    target_df = plot_df.loc[
        (plot_df["particle"] == particle) & (plot_df["target_type"] == target_type)
    ].copy()
    if target_df.empty:
        return

    components = component_order(target_df["component"])
    eval_modes = sorted(str(value) for value in target_df["eval_mode"].dropna().unique())
    row_specs = [(label, eval_mode) for label in run_labels for eval_mode in eval_modes]
    fig, axes = plt.subplots(
        nrows=len(row_specs),
        ncols=len(components),
        figsize=(4.1 * len(components), 2.7 * len(row_specs)),
        sharex=True,
        squeeze=False,
    )
    ci_low_column, ci_high_column = metric_ci_columns(metric)

    for row_idx, (label, eval_mode) in enumerate(row_specs):
        row_df = target_df.loc[
            (target_df["run_label"] == label) & (target_df["eval_mode"] == eval_mode)
        ].copy()
        limits = y_limits(row_df, metric) if not row_df.empty else None
        for col_idx, component in enumerate(components):
            ax = axes[row_idx][col_idx]
            panel_df = row_df.loc[row_df["component"] == component].sort_values("top_k")
            ax.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.5)
            for condition in CONDITIONS:
                condition_df = panel_df.loc[panel_df["site_set_type"] == condition].copy()
                if condition_df.empty:
                    continue
                style = CONDITION_STYLES[condition]
                x_values = condition_df["top_k"].to_numpy(dtype=float)
                y_values = condition_df[metric].to_numpy(dtype=float)
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
                ci_low = condition_df[ci_low_column].to_numpy(dtype=float)
                ci_high = condition_df[ci_high_column].to_numpy(dtype=float)
                if pd.notna(ci_low).all() and pd.notna(ci_high).all():
                    ax.fill_between(x_values, ci_low, ci_high, color=style["color"], alpha=0.12, linewidth=0)

            if limits is not None:
                ax.set_ylim(*limits)
            ax.grid(True, axis="y", alpha=0.18)
            ax.set_xticks(sorted(target_df["top_k"].unique()))
            if row_idx == 0:
                ax.set_title(component, fontsize=12)
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Top-k component/layer sites")
            if col_idx == 0:
                ax.set_ylabel(f"{pretty_label(label)}\n{pretty_label(eval_mode)}\n{METRIC_LABELS[metric]}")

    handles = []
    labels = []
    first_ax = axes[0][0]
    for condition in CONDITIONS:
        style = CONDITION_STYLES[condition]
        handle = first_ax.plot(
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

    target_label = "Generated targets" if target_type == "generated" else "Gold followups"
    fig.suptitle(f"{particle}: {target_label} prompt-boundary patch effects", fontsize=14, y=0.995)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, f"component_patch_effects_{particle}_{target_type}")
    plt.close(fig)


def build_difference_rows(plot_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    key_columns = [
        "run_label",
        "run_label_pretty",
        "run_dir",
        "particle",
        "component",
        "position_label",
        "corruption_mode",
        "target_type",
        "eval_mode",
        "top_k",
    ]
    pivot_df = plot_df.pivot_table(
        index=key_columns,
        columns="site_set_type",
        values=metric,
        aggfunc="first",
    ).reset_index()
    if not {"localized", "random_mean"}.issubset(pivot_df.columns):
        return pd.DataFrame()
    pivot_df["localized_minus_random_mean"] = pivot_df["localized"] - pivot_df["random_mean"]
    return pivot_df


def plot_difference_curves(
    diff_df: pd.DataFrame,
    run_labels: list[str],
    particle: str,
    target_type: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    target_df = diff_df.loc[
        (diff_df["particle"] == particle) & (diff_df["target_type"] == target_type)
    ].copy()
    if target_df.empty:
        return

    components = component_order(target_df["component"])
    eval_modes = sorted(str(value) for value in target_df["eval_mode"].dropna().unique())
    row_specs = [(label, eval_mode) for label in run_labels for eval_mode in eval_modes]
    fig, axes = plt.subplots(
        nrows=len(row_specs),
        ncols=len(components),
        figsize=(4.1 * len(components), 2.6 * len(row_specs)),
        sharex=True,
        squeeze=False,
    )
    finite_values = pd.to_numeric(target_df["localized_minus_random_mean"], errors="coerce").dropna()
    if finite_values.empty:
        limits = None
    else:
        y_min = min(float(finite_values.min()), 0.0)
        y_max = max(float(finite_values.max()), 0.0)
        y_span = y_max - y_min
        padding = max(y_span * 0.12, 5e-4)
        limits = (y_min - padding, y_max + padding)

    for row_idx, (label, eval_mode) in enumerate(row_specs):
        row_df = target_df.loc[
            (target_df["run_label"] == label) & (target_df["eval_mode"] == eval_mode)
        ].copy()
        for col_idx, component in enumerate(components):
            ax = axes[row_idx][col_idx]
            panel_df = row_df.loc[row_df["component"] == component].sort_values("top_k")
            ax.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.5)
            ax.plot(
                panel_df["top_k"].to_numpy(dtype=float),
                panel_df["localized_minus_random_mean"].to_numpy(dtype=float),
                color="#2563EB",
                linestyle="-",
                marker="o",
                linewidth=2.0,
                markersize=5.5,
            )
            if limits is not None:
                ax.set_ylim(*limits)
            ax.grid(True, axis="y", alpha=0.18)
            ax.set_xticks(sorted(target_df["top_k"].unique()))
            if row_idx == 0:
                ax.set_title(component, fontsize=12)
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Top-k component/layer sites")
            if col_idx == 0:
                ax.set_ylabel(f"{pretty_label(label)}\n{pretty_label(eval_mode)}\nLocalized - random")

    target_label = "Generated targets" if target_type == "generated" else "Gold followups"
    fig.suptitle(f"{particle}: {target_label} localized advantage over random", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir, f"component_patch_localized_minus_random_{particle}_{target_type}")
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

    plot_df = load_plot_rows(
        run_labels=run_labels,
        particles=particles,
        run_roots=run_roots,
        overrides=overrides,
        metric=args.metric,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(output_dir / "component_patch_plot_data.tsv", sep="\t", index=False)
    diff_df = build_difference_rows(plot_df, metric=args.metric)
    if not diff_df.empty:
        diff_df.to_csv(output_dir / "component_patch_localized_minus_random_plot_data.tsv", sep="\t", index=False)

    for particle in particles:
        for target_type in ("generated", "gold"):
            plot_effect_curves(
                plot_df=plot_df,
                run_labels=run_labels,
                particle=particle,
                target_type=target_type,
                metric=args.metric,
                output_dir=output_dir,
            )
            if not diff_df.empty:
                plot_difference_curves(
                    diff_df=diff_df,
                    run_labels=run_labels,
                    particle=particle,
                    target_type=target_type,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    main()
