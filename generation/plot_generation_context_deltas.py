#!/usr/bin/env python3
"""Create EMNLP-style context-delta figures for generation results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelRun:
    result_dir: str
    label: str


MODEL_RUNS: tuple[ModelRun, ...] = (
    ModelRun("llama3", "Llama 3 8B"),
    ModelRun("llama3_instruct", "Llama 3 8B Inst."),
    ModelRun("gemma2", "Gemma 2 9B"),
    ModelRun("gemma2_it", "Gemma 2 9B IT"),
    ModelRun("olmo2", "OLMo 2 7B"),
    ModelRun("olmo2_instruct", "OLMo 2 7B Inst."),
    ModelRun("qwen3_5", "Qwen3.5 9B"),
    ModelRun("qwen35_instruct", "Qwen3.5 9B Chat"),
)

CONDITIONS: tuple[str, ...] = ("just", "only", "not")
CONDITION_LABELS = {
    "just": "just",
    "only": "only",
    "not": "not (control)",
}
EXPECTED_ITEM_COUNTS = {
    "just": 40,
    "only": 30,
    "not": 30,
}
GENERATED_FROM: tuple[str, ...] = ("P", "P_prime")
SOURCE_LABELS = {
    "P": "S samples",
    "P_prime": "S' samples",
}

# Okabe-Ito colors.
SOURCE_COLORS = {
    "P": "#0072B2",
    "P_prime": "#D55E00",
}
SOURCE_MARKERS = {
    "P": "o",
    "P_prime": "^",
}
MODEL_COLORS = [
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#D55E00",
    "#999999",
    "#000000",
]
PANEL_A_Y_POSITIONS = np.array([0.0, 0.46, 1.34, 1.80, 2.68, 3.14, 4.02, 4.48])
PANEL_A_XLABEL = r"$\Delta=\log p(y\mid S_{\mathrm{src}})-\log p(y\mid S_{\mathrm{alt}})$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generation own-context log-probability advantages."
    )
    parser.add_argument(
        "--results_dir",
        default="generation/results",
        help="Directory containing per-model generation results.",
    )
    parser.add_argument(
        "--figures_dir",
        default=None,
        help="Output directory. Defaults to RESULTS_DIR/figures.",
    )
    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=5000,
        help="Bootstrap replicates for item-level 95%% CIs.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=20260422,
        help="Random seed for bootstrap CIs and jitter.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        choices=["pdf", "svg", "png"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def one_file(files: Iterable[Path], description: str) -> Path:
    paths = sorted(files)
    if len(paths) != 1:
        joined = "\n  ".join(str(path) for path in paths) or "<none>"
        raise FileNotFoundError(f"Expected exactly one {description}, found {len(paths)}:\n  {joined}")
    return paths[0]


def model_index() -> dict[str, int]:
    return {run.result_dir: idx for idx, run in enumerate(MODEL_RUNS)}


def model_label_map() -> dict[str, str]:
    return {run.result_dir: run.label for run in MODEL_RUNS}


def add_common_metadata(df: pd.DataFrame, model_dir: str, condition: str) -> pd.DataFrame:
    out = df.copy()
    out["model_dir"] = model_dir
    out["model_label"] = model_label_map()[model_dir]
    out["model_order"] = model_index()[model_dir]
    out["condition"] = condition
    out["condition_label"] = CONDITION_LABELS[condition]
    if "generated_from" in out.columns:
        out["generated_from_label"] = out["generated_from"].map(SOURCE_LABELS)
    return out


def read_item_data(results_dir: Path) -> pd.DataFrame:
    frames = []
    for run in MODEL_RUNS:
        model_dir = results_dir / run.result_dir
        for condition in CONDITIONS:
            path = one_file(
                model_dir.glob(f"*_context_{condition}_generation_item_condition_summary.tsv"),
                f"item summary for {run.result_dir}/{condition}",
            )
            frame = pd.read_csv(path, sep="\t")
            frame["source_file"] = str(path)
            frames.append(add_common_metadata(frame, run.result_dir, condition))
    item_data = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "source_row_index",
        "n_candidates",
        "mean_own_context_log_prob_advantage",
        "median_own_context_log_prob_advantage",
        "mean_own_context_log_prob_advantage_per_token",
        "own_context_win_rate",
        "weighted_own_context_win_rate",
        "weighted_own_context_log_prob_advantage",
        "top1_own_context_win_rate",
        "top1_mean_own_context_log_prob_advantage",
    ]
    for col in numeric_cols:
        if col in item_data.columns:
            item_data[col] = pd.to_numeric(item_data[col], errors="coerce")
    item_data["item_key"] = (
        item_data["source_dataset"].astype(str)
        + ":"
        + item_data["source_row_index"].astype(str)
        + ":"
        + item_data["id"].astype(str)
    )
    return item_data


def read_condition_data(results_dir: Path) -> pd.DataFrame:
    frames = []
    for run in MODEL_RUNS:
        model_dir = results_dir / run.result_dir
        for condition in CONDITIONS:
            path = one_file(
                model_dir.glob(f"*_context_{condition}_generation_condition_summary.tsv"),
                f"condition summary for {run.result_dir}/{condition}",
            )
            frame = pd.read_csv(path, sep="\t")
            frame["source_file"] = str(path)
            frames.append(add_common_metadata(frame, run.result_dir, condition))
    condition_data = pd.concat(frames, ignore_index=True)
    for col in [
        "mean_own_context_log_prob_advantage",
        "own_context_win_rate",
        "weighted_own_context_win_rate",
        "weighted_own_context_log_prob_advantage",
    ]:
        condition_data[col] = pd.to_numeric(condition_data[col], errors="coerce")
    return condition_data


def read_paired_data(results_dir: Path) -> pd.DataFrame:
    frames = []
    for run in MODEL_RUNS:
        model_dir = results_dir / run.result_dir
        for condition in CONDITIONS:
            path = one_file(
                model_dir.glob(f"*_context_{condition}_generation_paired_item_summary.tsv"),
                f"paired item summary for {run.result_dir}/{condition}",
            )
            frame = pd.read_csv(path, sep="\t")
            frame["source_file"] = str(path)
            frames.append(add_common_metadata(frame, run.result_dir, condition))
    paired = pd.concat(frames, ignore_index=True)
    for col in [
        "p_mean_advantage",
        "p_prime_mean_advantage",
        "p_win_rate",
        "p_prime_win_rate",
        "exact_overlap_jaccard",
    ]:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")
    return paired


def read_candidate_data(results_dir: Path) -> pd.DataFrame:
    wanted = {
        "source_dataset",
        "dataset_role",
        "source_row_index",
        "id",
        "word",
        "subset",
        "generated_from",
        "unique_rank",
        "own_context_log_prob_advantage",
        "own_context_log_prob_advantage_per_token",
        "own_context_win",
        "candidate_weight",
    }
    frames = []
    for run in MODEL_RUNS:
        model_dir = results_dir / run.result_dir
        for condition in CONDITIONS:
            weighted = sorted(model_dir.glob(f"*_context_{condition}_generation_candidates_weighted.tsv"))
            raw = sorted(model_dir.glob(f"*_context_{condition}_generation_candidates.tsv"))
            path = one_file(weighted or raw, f"candidate file for {run.result_dir}/{condition}")
            frame = pd.read_csv(path, sep="\t", usecols=lambda col: col in wanted)
            frame["source_file"] = str(path)
            frames.append(add_common_metadata(frame, run.result_dir, condition))
    candidate = pd.concat(frames, ignore_index=True)
    for col in [
        "source_row_index",
        "unique_rank",
        "own_context_log_prob_advantage",
        "own_context_log_prob_advantage_per_token",
        "candidate_weight",
    ]:
        if col in candidate.columns:
            candidate[col] = pd.to_numeric(candidate[col], errors="coerce")
    return candidate


def validate_coverage(item_data: pd.DataFrame) -> None:
    errors = []
    expected_groups = len(MODEL_RUNS) * len(CONDITIONS) * len(GENERATED_FROM)
    actual_groups = item_data.groupby(["model_dir", "condition", "generated_from"]).ngroups
    if actual_groups != expected_groups:
        errors.append(f"Expected {expected_groups} model/condition/source groups, found {actual_groups}.")

    for run in MODEL_RUNS:
        for condition in CONDITIONS:
            for source in GENERATED_FROM:
                group = item_data[
                    (item_data["model_dir"] == run.result_dir)
                    & (item_data["condition"] == condition)
                    & (item_data["generated_from"] == source)
                ]
                expected_n = EXPECTED_ITEM_COUNTS[condition]
                if len(group) != expected_n:
                    errors.append(
                        f"{run.result_dir}/{condition}/{source}: expected {expected_n} items, found {len(group)}."
                    )
                datasets = sorted(group["source_dataset"].dropna().unique())
                if datasets != [condition]:
                    errors.append(
                        f"{run.result_dir}/{condition}/{source}: expected source_dataset={condition}, found {datasets}."
                    )

    if any(item_data["model_dir"].str.contains("thinking_bad", regex=False)):
        errors.append("Excluded qwen35_instruct_thinking_bad run appeared in plotting data.")

    if errors:
        raise ValueError("Coverage validation failed:\n" + "\n".join(errors))


def compare_to_condition_summaries(item_data: pd.DataFrame, condition_data: pd.DataFrame) -> pd.DataFrame:
    item_means = (
        item_data.groupby(["model_dir", "condition", "generated_from"], as_index=False)[
            "mean_own_context_log_prob_advantage"
        ]
        .mean()
        .rename(columns={"mean_own_context_log_prob_advantage": "mean_delta_from_items"})
    )
    condition_means = condition_data[
        ["model_dir", "condition", "generated_from", "mean_own_context_log_prob_advantage"]
    ].rename(columns={"mean_own_context_log_prob_advantage": "mean_delta_from_condition_summary"})
    checks = item_means.merge(condition_means, on=["model_dir", "condition", "generated_from"], how="outer")
    checks["abs_diff"] = (
        checks["mean_delta_from_items"] - checks["mean_delta_from_condition_summary"]
    ).abs()
    max_diff = checks["abs_diff"].max()
    if pd.isna(max_diff) or max_diff > 1e-8:
        raise ValueError(
            "Item means do not match condition summaries; max abs diff = "
            f"{max_diff}. See generated condition check TSV."
        )
    return checks


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (np.nan, np.nan)
    if replicates <= 0:
        return (np.nan, np.nan)
    sample_idx = rng.integers(0, len(finite), size=(replicates, len(finite)))
    sample_means = finite[sample_idx].mean(axis=1)
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return (float(low), float(high))


def build_model_summary(
    item_data: pd.DataFrame,
    condition_checks: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    rows = []
    group_cols = ["model_dir", "model_label", "model_order", "condition", "condition_label", "generated_from"]
    for keys, group in item_data.groupby(group_cols, sort=False):
        record = dict(zip(group_cols, keys))
        values = group["mean_own_context_log_prob_advantage"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(values, rng, bootstrap_replicates)
        record.update(
            {
                "n_items": len(values),
                "mean_delta": float(np.nanmean(values)),
                "median_item_delta": float(np.nanmedian(values)),
                "item_delta_q1": float(np.nanpercentile(values, 25)),
                "item_delta_q3": float(np.nanpercentile(values, 75)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        rows.append(record)

    model_summary = pd.DataFrame(rows)
    model_summary = model_summary.merge(
        condition_checks,
        on=["model_dir", "condition", "generated_from"],
        how="left",
    )
    return model_summary.sort_values(["condition", "model_order", "generated_from"]).reset_index(drop=True)


def padded_limits(values: Sequence[float], include_zero: bool = True, pad_frac: float = 0.08) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if include_zero:
        arr = np.concatenate([arr, np.array([0.0])])
    if len(arr) == 0:
        return (-1.0, 1.0)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if low == high:
        pad = max(abs(low) * 0.1, 0.1)
    else:
        pad = (high - low) * pad_frac
    return (low - pad, high + pad)


def quantile_limits(
    values: Sequence[float],
    lower: float = 0.01,
    upper: float = 0.99,
    include_zero: bool = True,
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (-1.0, 1.0)
    low, high = np.nanquantile(arr, [lower, upper])
    if include_zero:
        low = min(float(low), 0.0)
        high = max(float(high), 0.0)
    if low == high:
        pad = max(abs(low) * 0.1, 0.1)
    else:
        pad = (high - low) * pad_frac
    return (float(low - pad), float(high + pad))


def save_figure(fig: plt.Figure, prefix: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    written = []
    for fmt in formats:
        path = prefix.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def condition_xlim_from_summary(
    model_summary: pd.DataFrame,
    conditions: Sequence[str],
    condition: str | None = None,
) -> tuple[float, float]:
    subset = model_summary[model_summary["condition"].isin(conditions)]
    if condition is not None:
        subset = subset[subset["condition"] == condition]
    values = pd.concat([subset["ci_low"], subset["ci_high"], subset["mean_delta"]]).to_numpy(dtype=float)
    return padded_limits(values)


def draw_zero_axis(ax: plt.Axes, dense_x_grid: bool = False) -> None:
    ax.axvline(0, color="#222222", linewidth=0.9, zorder=0)
    if dense_x_grid:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="x", which="major", color="#d3d3d3", linewidth=0.55, alpha=0.68)
        ax.grid(axis="x", which="minor", color="#e5e5e5", linewidth=0.45, alpha=0.52)
    else:
        ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.8)


def draw_model_pair_guides(ax: plt.Axes, y_positions: np.ndarray) -> None:
    for pair_idx in range(0, len(y_positions), 2):
        low = y_positions[pair_idx] - 0.23
        high = y_positions[pair_idx + 1] + 0.23
        if (pair_idx // 2) % 2 == 0:
            ax.axhspan(low, high, color="#f7f7f7", alpha=0.55, zorder=-3)
        if pair_idx:
            separator = (y_positions[pair_idx - 1] + y_positions[pair_idx]) / 2
            ax.axhline(separator, color="#e9e9e9", linewidth=0.7, zorder=-2)


def draw_panel_a(
    fig: plt.Figure,
    header_ax: plt.Axes,
    axes: Sequence[plt.Axes],
    model_summary: pd.DataFrame,
    include_panel_label: bool = True,
) -> None:
    header_ax.axis("off")
    particle_forest_xlim = condition_xlim_from_summary(model_summary, ["just", "only"])
    not_forest_xlim = condition_xlim_from_summary(model_summary, ["not"])
    y_positions = PANEL_A_Y_POSITIONS
    offsets = {"P": -0.09, "P_prime": 0.09}
    legend_handles = []
    legend_labels = []

    for col_idx, condition in enumerate(CONDITIONS):
        ax = axes[col_idx]
        draw_model_pair_guides(ax, y_positions)
        draw_zero_axis(ax, dense_x_grid=True)
        ax.set_title(CONDITION_LABELS[condition])
        for source in GENERATED_FROM:
            subset = model_summary[
                (model_summary["condition"] == condition)
                & (model_summary["generated_from"] == source)
            ].sort_values("model_order")
            x = subset["mean_delta"].to_numpy(dtype=float)
            low = subset["ci_low"].to_numpy(dtype=float)
            high = subset["ci_high"].to_numpy(dtype=float)
            model_orders = subset["model_order"].to_numpy(dtype=int)
            y = y_positions[model_orders] + offsets[source]
            container = ax.errorbar(
                x,
                y,
                xerr=np.vstack([x - low, high - x]),
                fmt=SOURCE_MARKERS[source],
                color=SOURCE_COLORS[source],
                ecolor=SOURCE_COLORS[source],
                markersize=4.2,
                elinewidth=1.0,
                capsize=2.0,
                label=SOURCE_LABELS[source],
                zorder=3,
            )
            if col_idx == 0:
                legend_handles.append(container)
                legend_labels.append(SOURCE_LABELS[source])
        ax.set_ylim(y_positions[0] - 0.35, y_positions[-1] + 0.35)
        ax.invert_yaxis()
        ax.set_yticks(y_positions)
        if col_idx == 0:
            ax.set_yticklabels([run.label for run in MODEL_RUNS])
        else:
            ax.set_yticklabels([])
        ax.set_xlim(not_forest_xlim if condition == "not" else particle_forest_xlim)
        ax.set_xlabel(PANEL_A_XLABEL, labelpad=6)

    title = "Model-level mean Delta"
    if include_panel_label:
        title = f"A. {title}"
    header_ax.text(
        0.0,
        0.84,
        title,
        transform=header_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    header_ax.text(
        0.0,
        0.28,
        "Points are model means; bars are 95% bootstrap CIs over items. Positive values favor the source context.",
        transform=header_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
    )
    header_ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.88),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handlelength=2.1,
        borderaxespad=0.0,
    )


def draw_panel_b(
    header_ax: plt.Axes,
    axes: Sequence[plt.Axes],
    item_data: pd.DataFrame,
    rng: np.random.Generator,
    include_panel_label: bool = True,
) -> None:
    header_ax.axis("off")
    particle_item_values = item_data[item_data["condition"].isin(["just", "only"])][
        "mean_own_context_log_prob_advantage"
    ].to_numpy(dtype=float)
    not_item_values = item_data[item_data["condition"] == "not"][
        "mean_own_context_log_prob_advantage"
    ].to_numpy(dtype=float)
    particle_item_xlim = quantile_limits(particle_item_values)
    not_item_xlim = quantile_limits(not_item_values)

    for col_idx, condition in enumerate(CONDITIONS):
        ax = axes[col_idx]
        draw_zero_axis(ax)
        ax.set_title(CONDITION_LABELS[condition])
        xlim = not_item_xlim if condition == "not" else particle_item_xlim
        arrays = []
        positions = [0, 1]
        for source in GENERATED_FROM:
            vals = item_data[
                (item_data["condition"] == condition)
                & (item_data["generated_from"] == source)
            ]["mean_own_context_log_prob_advantage"].dropna().to_numpy(dtype=float)
            arrays.append(vals)
        parts = ax.violinplot(
            arrays,
            positions=positions,
            vert=False,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, source in zip(parts["bodies"], GENERATED_FROM):
            body.set_facecolor(SOURCE_COLORS[source])
            body.set_edgecolor(SOURCE_COLORS[source])
            body.set_alpha(0.22)

        for pos, source, vals in zip(positions, GENERATED_FROM, arrays):
            capped = np.clip(vals, xlim[0], xlim[1])
            jitter = rng.normal(0.0, 0.055, size=len(vals))
            ax.scatter(
                capped,
                np.full(len(vals), pos) + jitter,
                s=6,
                alpha=0.18,
                color=SOURCE_COLORS[source],
                linewidths=0,
                zorder=2,
            )
            q1, median, q3 = np.nanpercentile(vals, [25, 50, 75])
            ax.plot(
                [np.clip(q1, *xlim), np.clip(q3, *xlim)],
                [pos, pos],
                color=SOURCE_COLORS[source],
                linewidth=2.1,
                solid_capstyle="round",
                zorder=4,
            )
            ax.scatter(
                [np.clip(median, *xlim)],
                [pos],
                s=18,
                color="white",
                edgecolor=SOURCE_COLORS[source],
                linewidth=1.0,
                zorder=5,
            )

        ax.set_xlim(xlim)
        ax.set_yticks(positions)
        if col_idx == 0:
            ax.set_yticklabels([SOURCE_LABELS[source] for source in GENERATED_FROM])
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Item mean Delta")
        ax.set_ylim(-0.65, 1.65)

    title = "Item-level mean Delta distributions"
    if include_panel_label:
        title = f"B. {title}"
    header_ax.text(
        0.0,
        0.88,
        title,
        transform=header_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    header_ax.text(
        0.0,
        0.24,
        "Axes show the 1st-99th percentile range; extreme dots are capped at the nearest axis edge.",
        transform=header_ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#333333",
    )


def plot_main_figure(
    model_summary: pd.DataFrame,
    item_data: pd.DataFrame,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
    rng: np.random.Generator,
) -> list[Path]:
    fig = plt.figure(figsize=(7.4, 8.7))
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=[0.16, 1.25, 0.18, 1.0],
        hspace=0.18,
        wspace=0.22,
        left=0.15,
        right=0.98,
        top=0.98,
        bottom=0.08,
    )

    top_header = fig.add_subplot(grid[0, :])
    bottom_header = fig.add_subplot(grid[2, :])
    for header_ax in (top_header, bottom_header):
        header_ax.axis("off")

    top_axes = [fig.add_subplot(grid[1, idx]) for idx in range(3)]
    bottom_axes = [fig.add_subplot(grid[3, idx]) for idx in range(3)]

    draw_panel_a(fig, top_header, top_axes, model_summary)
    draw_panel_b(bottom_header, bottom_axes, item_data, rng)

    prefix = figures_dir / "generation_context_delta_main"
    return save_figure(fig, prefix, formats, dpi)


def plot_split_main_panels(
    model_summary: pd.DataFrame,
    item_data: pd.DataFrame,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
    rng: np.random.Generator,
) -> list[Path]:
    written = []

    fig_a = plt.figure(figsize=(7.4, 4.25))
    grid_a = fig_a.add_gridspec(
        2,
        3,
        height_ratios=[0.22, 1.0],
        hspace=0.15,
        wspace=0.22,
        left=0.15,
        right=0.98,
        top=0.98,
        bottom=0.14,
    )
    header_a = fig_a.add_subplot(grid_a[0, :])
    axes_a = [fig_a.add_subplot(grid_a[1, idx]) for idx in range(3)]
    draw_panel_a(fig_a, header_a, axes_a, model_summary)
    written.extend(save_figure(fig_a, figures_dir / "generation_context_delta_panel_a", formats, dpi))

    fig_b = plt.figure(figsize=(7.4, 3.65))
    grid_b = fig_b.add_gridspec(
        2,
        3,
        height_ratios=[0.24, 1.0],
        hspace=0.14,
        wspace=0.22,
        left=0.15,
        right=0.98,
        top=0.98,
        bottom=0.17,
    )
    header_b = fig_b.add_subplot(grid_b[0, :])
    axes_b = [fig_b.add_subplot(grid_b[1, idx]) for idx in range(3)]
    draw_panel_b(header_b, axes_b, item_data, rng)
    written.extend(save_figure(fig_b, figures_dir / "generation_context_delta_panel_b", formats, dpi))

    return written


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if len(finite) == 0:
        return finite, finite
    y = np.arange(1, len(finite) + 1) / len(finite)
    return finite, y


def plot_candidate_ecdf(
    candidate_data: pd.DataFrame,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(
        len(MODEL_RUNS),
        len(CONDITIONS),
        figsize=(7.4, 10.8),
        sharey=True,
        squeeze=False,
    )
    particle_values = candidate_data[candidate_data["condition"].isin(["just", "only"])][
        "own_context_log_prob_advantage"
    ].to_numpy(dtype=float)
    not_values = candidate_data[candidate_data["condition"] == "not"][
        "own_context_log_prob_advantage"
    ].to_numpy(dtype=float)
    particle_xlim = quantile_limits(particle_values)
    not_xlim = quantile_limits(not_values)

    for row_idx, run in enumerate(MODEL_RUNS):
        for col_idx, condition in enumerate(CONDITIONS):
            ax = axes[row_idx, col_idx]
            draw_zero_axis(ax)
            for source in GENERATED_FROM:
                vals = candidate_data[
                    (candidate_data["model_dir"] == run.result_dir)
                    & (candidate_data["condition"] == condition)
                    & (candidate_data["generated_from"] == source)
                ]["own_context_log_prob_advantage"].to_numpy(dtype=float)
                x, y = ecdf(vals)
                ax.plot(
                    x,
                    y,
                    color=SOURCE_COLORS[source],
                    linewidth=1.0,
                    label=SOURCE_LABELS[source],
                )
            ax.set_xlim(not_xlim if condition == "not" else particle_xlim)
            ax.set_ylim(0, 1)
            if row_idx == 0:
                ax.set_title(CONDITION_LABELS[condition])
            if col_idx == 0:
                ax.set_ylabel(run.label)
            else:
                ax.set_yticklabels([])
            if row_idx == len(MODEL_RUNS) - 1:
                ax.set_xlabel("Candidate Delta")
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc="lower right", frameon=False)

    fig.suptitle("Candidate-level ECDFs of own-context Delta", x=0.52, y=0.995)
    fig.text(
        0.99,
        0.005,
        "Axes show 1st-99th percentile range; ECDFs are computed from all candidate values.",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    prefix = figures_dir / "generation_context_delta_candidate_ecdf"
    return save_figure(fig, prefix, formats, dpi)


def plot_paired_scatter(
    paired_data: pd.DataFrame,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), squeeze=False)
    axes = axes[0]
    particle_values = pd.concat(
        [
            paired_data[paired_data["condition"].isin(["just", "only"])]["p_mean_advantage"],
            paired_data[paired_data["condition"].isin(["just", "only"])]["p_prime_mean_advantage"],
        ]
    ).to_numpy(dtype=float)
    not_values = pd.concat(
        [
            paired_data[paired_data["condition"] == "not"]["p_mean_advantage"],
            paired_data[paired_data["condition"] == "not"]["p_prime_mean_advantage"],
        ]
    ).to_numpy(dtype=float)
    particle_xlim = quantile_limits(particle_values)
    not_xlim = quantile_limits(not_values)

    for col_idx, condition in enumerate(CONDITIONS):
        ax = axes[col_idx]
        lim = not_xlim if condition == "not" else particle_xlim
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.plot(lim, lim, color="#bbbbbb", linewidth=0.7, linestyle="--", zorder=0)
        for idx, run in enumerate(MODEL_RUNS):
            subset = paired_data[
                (paired_data["model_dir"] == run.result_dir)
                & (paired_data["condition"] == condition)
            ]
            x = np.clip(subset["p_mean_advantage"].to_numpy(dtype=float), lim[0], lim[1])
            y = np.clip(subset["p_prime_mean_advantage"].to_numpy(dtype=float), lim[0], lim[1])
            ax.scatter(
                x,
                y,
                s=12,
                alpha=0.72,
                color=MODEL_COLORS[idx],
                linewidths=0,
                label=run.label if col_idx == 2 else None,
            )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(CONDITION_LABELS[condition])
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("S item mean Delta")
        if col_idx == 0:
            ax.set_ylabel("S' item mean Delta")
        else:
            ax.set_yticklabels([])
        ax.text(
            0.98,
            0.96,
            "both > 0",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#333333",
        )
    axes[2].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        markerscale=1.2,
        handletextpad=0.3,
    )
    fig.suptitle("Paired item mean Deltas by generation source", y=1.04)
    fig.tight_layout()
    prefix = figures_dir / "generation_context_delta_paired_scatter"
    return save_figure(fig, prefix, formats, dpi)


def plot_winrate_heatmap(
    condition_data: pd.DataFrame,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    vmin, vmax = 0.5, 1.0
    images = []
    for ax, source in zip(axes, GENERATED_FROM):
        matrix = np.full((len(MODEL_RUNS), len(CONDITIONS)), np.nan)
        for row_idx, run in enumerate(MODEL_RUNS):
            for col_idx, condition in enumerate(CONDITIONS):
                row = condition_data[
                    (condition_data["model_dir"] == run.result_dir)
                    & (condition_data["condition"] == condition)
                    & (condition_data["generated_from"] == source)
                ]
                if not row.empty:
                    matrix[row_idx, col_idx] = float(row.iloc[0]["own_context_win_rate"])
        image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        images.append(image)
        ax.set_title(SOURCE_LABELS[source])
        ax.set_xticks(np.arange(len(CONDITIONS)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
        ax.set_yticks(np.arange(len(MODEL_RUNS)))
        ax.set_yticklabels([run.label for run in MODEL_RUNS])
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                color = "white" if value < 0.72 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    fig.subplots_adjust(left=0.19, right=0.88, top=0.83, bottom=0.14, wspace=0.06)
    cbar = fig.colorbar(images[0], ax=axes, shrink=0.86, pad=0.02)
    cbar.set_label("Own-context win rate")
    fig.suptitle("Win-rate heatmap for positive own-context Delta", y=1.02)
    prefix = figures_dir / "generation_context_delta_winrate_heatmap"
    return save_figure(fig, prefix, formats, dpi)


def write_plot_data(
    figures_dir: Path,
    item_data: pd.DataFrame,
    model_summary: pd.DataFrame,
    condition_checks: pd.DataFrame,
    condition_data: pd.DataFrame,
    paired_data: pd.DataFrame,
    candidate_data: pd.DataFrame,
) -> list[Path]:
    outputs = []
    tables = {
        "generation_context_delta_item_plot_data.tsv": item_data,
        "generation_context_delta_model_summary.tsv": model_summary,
        "generation_context_delta_condition_check.tsv": condition_checks,
        "generation_context_delta_winrate_plot_data.tsv": condition_data,
        "generation_context_delta_paired_plot_data.tsv": paired_data,
        "generation_context_delta_candidate_plot_data.tsv": candidate_data,
    }
    for name, frame in tables.items():
        path = figures_dir / name
        frame.to_csv(path, sep="\t", index=False)
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    item_data = read_item_data(results_dir)
    condition_data = read_condition_data(results_dir)
    paired_data = read_paired_data(results_dir)
    candidate_data = read_candidate_data(results_dir)

    validate_coverage(item_data)
    condition_checks = compare_to_condition_summaries(item_data, condition_data)
    model_summary = build_model_summary(
        item_data=item_data,
        condition_checks=condition_checks,
        rng=rng,
        bootstrap_replicates=args.bootstrap_replicates,
    )

    data_paths = write_plot_data(
        figures_dir=figures_dir,
        item_data=item_data,
        model_summary=model_summary,
        condition_checks=condition_checks,
        condition_data=condition_data,
        paired_data=paired_data,
        candidate_data=candidate_data,
    )

    figure_paths = []
    figure_paths.extend(
        plot_main_figure(
            model_summary=model_summary,
            item_data=item_data,
            figures_dir=figures_dir,
            formats=args.formats,
            dpi=args.dpi,
            rng=rng,
        )
    )
    figure_paths.extend(
        plot_split_main_panels(
            model_summary=model_summary,
            item_data=item_data,
            figures_dir=figures_dir,
            formats=args.formats,
            dpi=args.dpi,
            rng=rng,
        )
    )
    figure_paths.extend(
        plot_candidate_ecdf(
            candidate_data=candidate_data,
            figures_dir=figures_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )
    figure_paths.extend(
        plot_paired_scatter(
            paired_data=paired_data,
            figures_dir=figures_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )
    figure_paths.extend(
        plot_winrate_heatmap(
            condition_data=condition_data,
            figures_dir=figures_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )

    print("Wrote plot data:")
    for path in data_paths:
        print(f"  {path}")
    print("Wrote figures:")
    for path in figure_paths:
        print(f"  {path}")
    print(
        "Validated coverage: "
        f"{len(MODEL_RUNS)} models x {len(CONDITIONS)} conditions x {len(GENERATED_FROM)} generation sources."
    )


if __name__ == "__main__":
    main()
