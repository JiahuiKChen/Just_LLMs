#!/usr/bin/env python3
"""Plot localization ablation effects from saved summary tables."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METRIC_MEAN_COLUMN = "context_advantage_per_token_change_from_none_mean"
METRIC_CI_LOW_COLUMN = "context_advantage_per_token_change_from_none_ci_low"
METRIC_CI_HIGH_COLUMN = "context_advantage_per_token_change_from_none_ci_high"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-token advantage changes for localized and random ablations."
    )
    parser.add_argument(
        "--particles",
        nargs="+",
        required=True,
        help="Particles to include, in subplot column order.",
    )
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


def load_plot_rows(
    run_labels: list[str],
    particles: list[str],
    run_roots: OrderedDict[str, Path],
    overrides: dict[tuple[str, str], Path],
) -> pd.DataFrame:
    rows = []
    for label in run_labels:
        for particle in particles:
            run_dir = resolve_run_dir(label, particle, run_roots, overrides)
            summary_path = run_dir / "ablation_summary.tsv"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing ablation summary: {summary_path}")
            summary_df = pd.read_csv(summary_path, sep="\t")
            subset = summary_df.loc[summary_df["ablation_condition"].isin(CONDITIONS)].copy()
            needed = [
                "particle",
                "target_type",
                "mask_percent",
                "ablation_condition",
                METRIC_MEAN_COLUMN,
            ]
            missing = sorted(set(needed) - set(subset.columns))
            if missing:
                raise KeyError(f"{summary_path} is missing columns: {', '.join(missing)}")
            subset = subset.loc[:, needed]
            if METRIC_CI_LOW_COLUMN not in summary_df.columns:
                subset[METRIC_CI_LOW_COLUMN] = float("nan")
            else:
                subset[METRIC_CI_LOW_COLUMN] = pd.to_numeric(summary_df.loc[subset.index, METRIC_CI_LOW_COLUMN], errors="coerce")
            if METRIC_CI_HIGH_COLUMN not in summary_df.columns:
                subset[METRIC_CI_HIGH_COLUMN] = float("nan")
            else:
                subset[METRIC_CI_HIGH_COLUMN] = pd.to_numeric(summary_df.loc[subset.index, METRIC_CI_HIGH_COLUMN], errors="coerce")
            subset["run_label"] = label
            subset["run_label_pretty"] = pretty_label(label)
            subset["run_dir"] = str(run_dir)
            rows.append(subset)
    if not rows:
        raise ValueError("No plotting rows were loaded.")
    combined = pd.concat(rows, ignore_index=True)
    combined["mask_percent"] = pd.to_numeric(combined["mask_percent"], errors="raise")
    return combined


def plot_target_type(
    plot_df: pd.DataFrame,
    run_labels: list[str],
    particles: list[str],
    target_type: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    target_df = plot_df.loc[plot_df["target_type"] == target_type].copy()
    if target_df.empty:
        raise ValueError(f"No rows found for target_type={target_type}")

    row_y_limits: dict[str, tuple[float, float]] = {}
    for label in run_labels:
        row_df = target_df.loc[target_df["run_label"] == label].copy()
        if row_df.empty:
            continue
        limit_columns = [
            row_df[METRIC_MEAN_COLUMN],
            row_df[METRIC_CI_LOW_COLUMN],
            row_df[METRIC_CI_HIGH_COLUMN],
        ]
        finite_values = pd.concat(limit_columns, ignore_index=True).dropna()
        if finite_values.empty:
            continue

        y_min = min(finite_values.min(), 0.0)
        y_max = max(finite_values.max(), 0.0)
        y_span = y_max - y_min
        padding = max(y_span * 0.12, 5e-4)
        row_y_limits[label] = (y_min - padding, y_max + padding)

    fig, axes = plt.subplots(
        nrows=len(run_labels),
        ncols=len(particles),
        figsize=(4.2 * len(particles), 2.8 * len(run_labels)),
        sharex=True,
        squeeze=False,
    )

    for row_idx, label in enumerate(run_labels):
        label_pretty = pretty_label(label)
        for col_idx, particle in enumerate(particles):
            ax = axes[row_idx][col_idx]
            panel_df = target_df.loc[
                (target_df["run_label"] == label) & (target_df["particle"] == particle)
            ].sort_values("mask_percent")
            if panel_df.empty:
                raise ValueError(f"No rows found for run={label}, particle={particle}, target={target_type}")

            ax.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.5)
            for condition in CONDITIONS:
                condition_df = panel_df.loc[panel_df["ablation_condition"] == condition].copy()
                if condition_df.empty:
                    continue
                style = CONDITION_STYLES[condition]
                x_values = condition_df["mask_percent"].to_numpy(dtype=float)
                y_values = condition_df[METRIC_MEAN_COLUMN].to_numpy(dtype=float)
                ci_low = condition_df[METRIC_CI_LOW_COLUMN].to_numpy(dtype=float)
                ci_high = condition_df[METRIC_CI_HIGH_COLUMN].to_numpy(dtype=float)

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
                if pd.notna(ci_low).all() and pd.notna(ci_high).all():
                    ax.fill_between(
                        x_values,
                        ci_low,
                        ci_high,
                        color=style["color"],
                        alpha=0.12,
                        linewidth=0,
                    )

            ax.grid(True, axis="y", alpha=0.18)
            ax.set_xticks([0.5, 1.0, 5.0], labels=["0.5%", "1%", "5%"])
            if label in row_y_limits:
                ax.set_ylim(*row_y_limits[label])
            if row_idx == 0:
                ax.set_title(particle, fontsize=12)
            if row_idx == len(run_labels) - 1:
                ax.set_xlabel("Ablated units")
            if col_idx == 0:
                ax.set_ylabel(f"{label_pretty}\nDelta advantage / token")

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

    title_target = "Generated targets" if target_type == "generated" else "Gold followups"
    fig.suptitle(
        f"{title_target}: per-token advantage change from original model",
        fontsize=14,
        y=0.995,
    )
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"ablation_effects_{target_type}.png"
    pdf_path = output_dir / f"ablation_effects_{target_type}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
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
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(output_dir / "ablation_effect_plot_data.tsv", sep="\t", index=False)
    for target_type in ("generated", "gold"):
        plot_target_type(
            plot_df=plot_df,
            run_labels=run_labels,
            particles=particles,
            target_type=target_type,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
