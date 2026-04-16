#!/usr/bin/env python3
"""
Generate a paper-style summary figure and LaTeX table from existing removal runs.

Panel A shows violin plots of per-token removal delta distributions for each model/particle pair.
Panel B shows the matching box plots for the same distributions.
Panel C overlays box plots on violins to preserve both quartile markers and density shape.
An optional comparison figure stacks the base-model Panel C above the instruction-tuned Panel C.

The summary table reports:
- % of examples increased (same as % positive for the chosen delta metric)
- median per-token removal delta
- 95% bootstrap confidence interval for the median
- N retained after IQR outlier filtering
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Patch


BASE_MODEL_SPECS = [
    {
        "label": "Llama 3 8B",
        "results_dir": "llama3",
        "prefix": "Meta-Llama-3-8B",
    },
    {
        "label": "OLMo-2 7B",
        "results_dir": "olmo2",
        "prefix": "OLMo-2-1124-7B",
    },
    {
        "label": "Qwen3.5 9B",
        "results_dir": "qwen3_5",
        "prefix": "Qwen3.5-9B",
    },
    {
        "label": "Gemma-2 9B",
        "results_dir": "gemma2",
        "prefix": "gemma-2-9b",
    },
]

INSTRUCT_MODEL_SPECS = [
    {
        "label": "Llama 3 8B Instruct",
        "results_dir": "llama3_instruct",
        "prefix": "Meta-Llama-3-8B-Instruct",
    },
    {
        "label": "OLMo-2 7B Instruct",
        "results_dir": "olmo2_instruct",
        "prefix": "OLMo-2-1124-7B-Instruct",
    },
    {
        "label": "Qwen3.5 9B CT",
        "results_dir": "qwen35_instruct",
        "prefix": "Qwen3.5-9B",
    },
    {
        "label": "Gemma-2 9B IT",
        "results_dir": "gemma2_it",
        "prefix": "gemma-2-9b-it",
    },
]

PARTICLE_SPECS = [
    {"key": "just", "stem": "filtered_just_dd", "label": "just"},
    {"key": "only", "stem": "filtered_only_dd", "label": "only"},
    {"key": "not", "stem": "filtered_not_dd", "label": "not"},
]


def get_model_specs(preset: str) -> list[dict]:
    if preset == "instruct":
        return INSTRUCT_MODEL_SPECS
    return BASE_MODEL_SPECS


def _results_path(results_root: Path, model_spec: dict, particle_spec: dict) -> Path:
    return (
        results_root
        / model_spec["results_dir"]
        / f'{model_spec["prefix"]}_context_{particle_spec["stem"]}_removal.csv'
    )


def _load_results_frame(results_path: Path, exclude_outliers: bool) -> tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(results_path, sep="\t")
    total_n = len(df)
    if exclude_outliers:
        if "removal_upper_iqr_outlier" not in df.columns:
            raise KeyError(
                f"Missing removal_upper_iqr_outlier column in {results_path}"
            )
        mask = df["removal_upper_iqr_outlier"].fillna(False).astype(bool)
        df = df.loc[~mask].copy()
    used_n = len(df)
    return df, total_n, used_n


def _bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int,
    ci: float,
    seed: int,
    batch_size: int = 1000,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(values)
    medians = []
    remaining = n_boot

    while remaining > 0:
        current = min(batch_size, remaining)
        sample_idx = rng.integers(0, n, size=(current, n))
        medians.append(np.median(values[sample_idx], axis=1))
        remaining -= current

    medians = np.concatenate(medians)
    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(medians, alpha))
    upper = float(np.percentile(medians, 100.0 - alpha))
    return lower, upper


def collect_summary(
    results_root: Path,
    model_specs: list[dict],
    exclude_outliers: bool,
    n_boot: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    records = []

    for model_idx, model_spec in enumerate(model_specs):
        for particle_idx, particle_spec in enumerate(PARTICLE_SPECS):
            results_path = _results_path(results_root, model_spec, particle_spec)
            df, total_n, used_n = _load_results_frame(results_path, exclude_outliers)
            values = df["removal_log_prob_delta_per_token"].astype(float).to_numpy()
            median_delta = float(np.median(values))
            ci_low, ci_high = _bootstrap_median_ci(
                values=values,
                n_boot=n_boot,
                ci=ci,
                seed=seed + (model_idx * 100) + particle_idx,
            )
            pct_positive = float((values > 0).mean() * 100.0)

            records.append(
                {
                    "model": model_spec["label"],
                    "particle": particle_spec["key"],
                    "median_delta": median_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "pct_positive": pct_positive,
                    "n_used": int(used_n),
                    "n_total": int(total_n),
                    "n_outliers": int(total_n - used_n),
                    "source_path": str(results_path),
                }
            )

    summary_df = pd.DataFrame(records)
    summary_df["particle"] = pd.Categorical(
        summary_df["particle"],
        categories=[spec["key"] for spec in PARTICLE_SPECS],
        ordered=True,
    )
    summary_df["model"] = pd.Categorical(
        summary_df["model"],
        categories=[spec["label"] for spec in model_specs],
        ordered=True,
    )
    return summary_df.sort_values(["model", "particle"]).reset_index(drop=True)


def collect_delta_distributions(
    results_root: Path,
    model_specs: list[dict],
    exclude_outliers: bool,
) -> pd.DataFrame:
    records = []

    for model_spec in model_specs:
        for particle_spec in PARTICLE_SPECS:
            results_path = _results_path(results_root, model_spec, particle_spec)
            df, _, _ = _load_results_frame(results_path, exclude_outliers)
            if "removal_log_prob_delta_per_token" not in df.columns:
                raise KeyError(
                    f"Missing removal_log_prob_delta_per_token column in {results_path}"
                )

            delta_values = df["removal_log_prob_delta_per_token"].astype(float).to_numpy()
            delta_values = delta_values[np.isfinite(delta_values)]

            for value in delta_values:
                records.append(
                    {
                        "model": model_spec["label"],
                        "particle": particle_spec["key"],
                        "delta_value": float(value),
                    }
                )

    plot_df = pd.DataFrame(records)
    plot_df["particle"] = pd.Categorical(
        plot_df["particle"],
        categories=[spec["key"] for spec in PARTICLE_SPECS],
        ordered=True,
    )
    plot_df["model"] = pd.Categorical(
        plot_df["model"],
        categories=[spec["label"] for spec in model_specs],
        ordered=True,
    )
    return plot_df.sort_values(["model", "particle"]).reset_index(drop=True)


PARTICLE_COLORS = {
    "just": "#1b9e77",
    "only": "#d95f02",
    "not": "#7570b3",
}


def _build_distribution_series(
    plot_df: pd.DataFrame,
    model_specs: list[dict],
) -> tuple[list[np.ndarray], list[float], np.ndarray, list[str], list[str]]:
    model_labels = [spec["label"] for spec in model_specs]
    group_centers = np.arange(len(model_labels), dtype=float) * 1.35
    particle_offsets = np.array([-0.28, 0.0, 0.28])

    distributions = []
    positions = []
    particle_keys = []

    for group_center, model_label in zip(group_centers, model_labels):
        model_rows = plot_df.loc[plot_df["model"] == model_label]
        for particle_offset, particle_spec in zip(particle_offsets, PARTICLE_SPECS):
            particle_key = particle_spec["key"]
            particle_rows = model_rows.loc[model_rows["particle"] == particle_key]
            values = particle_rows["delta_value"].astype(float).to_numpy()
            if len(values) == 0:
                continue
            distributions.append(values)
            positions.append(float(group_center + particle_offset))
            particle_keys.append(particle_key)

    return distributions, positions, group_centers, model_labels, particle_keys


def _style_distribution_axis(
    ax,
    group_centers: np.ndarray,
    model_labels: list[str],
    title: Optional[str],
) -> None:
    ax.set_xticks(group_centers)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Per-token removal delta", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def _draw_violin_panel(
    ax,
    plot_df: pd.DataFrame,
    model_specs: list[dict],
    title: Optional[str],
) -> None:
    distributions, positions, group_centers, model_labels, particle_keys = _build_distribution_series(
        plot_df,
        model_specs,
    )
    violin = ax.violinplot(
        distributions,
        positions=positions,
        widths=0.23,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, particle_key in zip(violin["bodies"], particle_keys):
        color = PARTICLE_COLORS[particle_key]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
        body.set_linewidth(1.0)

    _style_distribution_axis(ax, group_centers, model_labels, title)


def _draw_boxplot(
    ax,
    distributions: list[np.ndarray],
    positions: list[float],
    particle_keys: list[str],
    fill_alpha: float,
    widths: float,
) -> None:
    boxplot = ax.boxplot(
        distributions,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 2.2},
        whiskerprops={"linewidth": 1.8},
        capprops={"linewidth": 1.8},
        boxprops={"linewidth": 1.8},
    )

    for idx, (patch, particle_key) in enumerate(zip(boxplot["boxes"], particle_keys)):
        color = PARTICLE_COLORS[particle_key]
        patch.set_facecolor(mcolors.to_rgba(color, alpha=fill_alpha))
        patch.set_edgecolor(color)
        patch.set_linewidth(1.8)

        for whisker in boxplot["whiskers"][2 * idx : 2 * idx + 2]:
            whisker.set_color(color)
            whisker.set_linewidth(1.8)

        for cap in boxplot["caps"][2 * idx : 2 * idx + 2]:
            cap.set_color(color)
            cap.set_linewidth(1.8)

        median = boxplot["medians"][idx]
        median.set_color(color)
        median.set_linewidth(2.4)


def _draw_box_panel(
    ax,
    plot_df: pd.DataFrame,
    model_specs: list[dict],
    title: Optional[str],
) -> None:
    distributions, positions, group_centers, model_labels, particle_keys = _build_distribution_series(
        plot_df,
        model_specs,
    )
    _draw_boxplot(
        ax,
        distributions=distributions,
        positions=positions,
        particle_keys=particle_keys,
        fill_alpha=0.42,
        widths=0.18,
    )
    _style_distribution_axis(ax, group_centers, model_labels, title)


def _draw_combined_panel(
    ax,
    plot_df: pd.DataFrame,
    model_specs: list[dict],
    title: Optional[str],
) -> None:
    distributions, positions, group_centers, model_labels, particle_keys = _build_distribution_series(
        plot_df,
        model_specs,
    )
    violin = ax.violinplot(
        distributions,
        positions=positions,
        widths=0.23,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, particle_key in zip(violin["bodies"], particle_keys):
        color = PARTICLE_COLORS[particle_key]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
        body.set_linewidth(1.0)

    _draw_boxplot(
        ax,
        distributions=distributions,
        positions=positions,
        particle_keys=particle_keys,
        fill_alpha=0.40,
        widths=0.10,
    )
    _style_distribution_axis(ax, group_centers, model_labels, title)


def make_stacked_panel_c_figure(
    base_plot_df: pd.DataFrame,
    instruct_plot_df: pd.DataFrame,
    output_prefix: Path,
    exclude_outliers: bool,
) -> None:
    fig = plt.figure(figsize=(13.8, 8.3))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 0.20, 1.0],
        hspace=0.07,
    )
    top_ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[1, 0])
    bottom_ax = fig.add_subplot(grid[2, 0], sharey=top_ax)

    _draw_combined_panel(
        top_ax,
        plot_df=base_plot_df,
        model_specs=BASE_MODEL_SPECS,
        title="Base Models",
    )
    _draw_combined_panel(
        bottom_ax,
        plot_df=instruct_plot_df,
        model_specs=INSTRUCT_MODEL_SPECS,
        title="Instruction-Tuned Models",
    )

    for ax in (top_ax, bottom_ax):
        ax.set_title(ax.get_title(), fontsize=16, fontweight="bold", pad=4)
        ax.yaxis.label.set_fontsize(15)
        ax.tick_params(axis="x", labelsize=13, pad=1)
        ax.tick_params(axis="y", labelsize=13, pad=1)
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(0)
            tick_label.set_ha("center")

    legend_handles = [
        Patch(
            facecolor=PARTICLE_COLORS[spec["key"]],
            edgecolor=PARTICLE_COLORS[spec["key"]],
            alpha=0.5,
            label=spec["label"],
        )
        for spec in PARTICLE_SPECS
    ]
    legend_ax.axis("off")
    legend_ax.text(
        0.26,
        0.50,
        "Particle",
        ha="right",
        va="center",
        fontsize=15,
        fontweight="bold",
        transform=legend_ax.transAxes,
    )
    legend_ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.30, 0.50),
        ncol=3,
        frameon=False,
        fontsize=15,
        handlelength=2.1,
        handleheight=1.4,
        columnspacing=2.8,
        labelspacing=1.0,
    )

    fig.subplots_adjust(top=0.985, bottom=0.07, left=0.10, right=0.995)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_main_figure(
    plot_df: pd.DataFrame,
    output_prefix: Path,
    exclude_outliers: bool,
    model_specs: list[dict],
    figure_title_prefix: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19.0, 5.8),
        gridspec_kw={"wspace": 0.16},
    )

    _draw_violin_panel(
        axes[0],
        plot_df=plot_df,
        model_specs=model_specs,
        title="Panel A. Violin Plots",
    )
    _draw_box_panel(
        axes[1],
        plot_df=plot_df,
        model_specs=model_specs,
        title="Panel B. Box Plots",
    )
    _draw_combined_panel(
        axes[2],
        plot_df=plot_df,
        model_specs=model_specs,
        title="Panel C. Violin + Box",
    )

    legend_handles = [
        Patch(facecolor=PARTICLE_COLORS[spec["key"]], edgecolor=PARTICLE_COLORS[spec["key"]], alpha=0.5, label=spec["label"])
        for spec in PARTICLE_SPECS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        title="Particle",
    )

    filtering_note = (
        "after excluding IQR outliers"
        if exclude_outliers
        else "including all examples"
    )
    fig.suptitle(
        f"{figure_title_prefix} per-token removal delta distributions ({filtering_note})",
        fontsize=12.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.02,
        "Each group corresponds to one LLM; within each group, just/only/not show the same per-example delta that the previous heatmap summarized by median.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.74, bottom=0.20, wspace=0.16)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def _save_single_panel(
    plot_df: pd.DataFrame,
    model_specs: list[dict],
    plot_kind: str,
    output_path: Path,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    if plot_kind == "violin":
        _draw_violin_panel(ax, plot_df=plot_df, model_specs=model_specs, title=None)
    elif plot_kind == "box":
        _draw_box_panel(ax, plot_df=plot_df, model_specs=model_specs, title=None)
    else:
        _draw_combined_panel(ax, plot_df=plot_df, model_specs=model_specs, title=None)

    legend_handles = [
        Patch(facecolor=PARTICLE_COLORS[spec["key"]], edgecolor=PARTICLE_COLORS[spec["key"]], alpha=0.5, label=spec["label"])
        for spec in PARTICLE_SPECS
    ]
    fig.legend(
        handles=legend_handles,
        title="Particle",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.26)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_separate_panels(plot_df: pd.DataFrame, output_dir: Path, model_specs: list[dict]) -> None:
    _save_single_panel(
        plot_df=plot_df,
        model_specs=model_specs,
        plot_kind="violin",
        output_path=output_dir / "particle_removal_panel_a.png",
        figsize=(6.1, 4.8),
    )
    _save_single_panel(
        plot_df=plot_df,
        model_specs=model_specs,
        plot_kind="box",
        output_path=output_dir / "particle_removal_panel_b.png",
        figsize=(5.4, 4.8),
    )
    _save_single_panel(
        plot_df=plot_df,
        model_specs=model_specs,
        plot_kind="combined",
        output_path=output_dir / "particle_removal_panel_c.png",
        figsize=(6.1, 4.8),
    )


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    escaped = value
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def write_summary_tsv(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, sep="\t", index=False)


def write_latex_table(
    summary_df: pd.DataFrame,
    output_path: Path,
    exclude_outliers: bool,
    caption_prefix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrlr}",
        r"\toprule",
        r"Model & Particle & \% increased & Median $\Delta_{\mathrm{rem}}$ & 95\% bootstrap CI & N \\",
        r"\midrule",
    ]

    previous_model = None
    for row in summary_df.itertuples(index=False):
        model_label = "" if previous_model == row.model else _latex_escape(str(row.model))
        ci_text = f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        lines.append(
            f"{model_label} & {_latex_escape(str(row.particle))} & "
            f"{row.pct_positive:.1f} & {row.median_delta:.3f} & {ci_text} & {row.n_used} \\\\"
        )
        previous_model = row.model

    note = (
        "N is the number of non-outlier examples retained after IQR filtering. "
        if exclude_outliers
        else "N is the number of examples included in the summary. "
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                rf"\caption{{{caption_prefix} sensitivity to particle removal measured with the per-token "
                r"removal delta $\Delta_{\mathrm{rem}} = \log P(S \mid C) - \log P(S' \mid C)$. "
                r"\% increased is the share of examples with positive $\Delta_{\mathrm{rem}}$ "
                r"(i.e., the same quantity you might also call \% positive). "
                + note
                + r"}"
            ),
            r"\label{tab:particle-removal-sensitivity}",
            r"\end{table}",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate a paper-style figure and LaTeX table from existing removal results."
    )
    parser.add_argument(
        "--preset",
        choices=["base", "instruct"],
        default="base",
        help="Which model set to summarize.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=None,
        help="Directory containing the per-model result folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for the generated figure, summary TSV, and LaTeX table.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples used for the median confidence intervals.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Bootstrap confidence level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--include_outliers",
        action="store_true",
        help="Use all rows instead of excluding IQR outliers.",
    )
    parser.add_argument(
        "--write_panel_c_comparison",
        action="store_true",
        help="Also write a stacked comparison figure with the base Panel C on top and the instruction-tuned Panel C below.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude_outliers = not args.include_outliers
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.preset == "instruct":
        default_results_root = repo_root / "results"
        default_output_dir = default_results_root / "paper_summary_instruct"
        model_specs = INSTRUCT_MODEL_SPECS
        figure_title_prefix = "Instruction-tuned LLM"
        caption_prefix = "Instruction-tuned LLM"
    else:
        default_results_root = script_dir / "results"
        default_output_dir = default_results_root / "paper_summary"
        model_specs = BASE_MODEL_SPECS
        figure_title_prefix = "LLM"
        caption_prefix = "LLM"

    results_root = args.results_root or default_results_root
    output_dir = args.output_dir or default_output_dir

    summary_df = collect_summary(
        results_root=results_root,
        model_specs=model_specs,
        exclude_outliers=exclude_outliers,
        n_boot=args.bootstrap_samples,
        ci=args.ci,
        seed=args.seed,
    )
    plot_df = collect_delta_distributions(
        results_root=results_root,
        model_specs=model_specs,
        exclude_outliers=exclude_outliers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "particle_removal_summary.tsv"
    figure_prefix = output_dir / "particle_removal_main_figure"
    latex_path = output_dir / "particle_removal_summary_table.tex"
    comparison_prefix = (
        results_root / "paper_summary_comparison" / "particle_removal_panel_c_stacked"
    )

    write_summary_tsv(summary_df, summary_path)
    make_main_figure(
        plot_df,
        figure_prefix,
        exclude_outliers=exclude_outliers,
        model_specs=model_specs,
        figure_title_prefix=figure_title_prefix,
    )
    write_separate_panels(plot_df, output_dir, model_specs=model_specs)
    write_latex_table(
        summary_df,
        latex_path,
        exclude_outliers=exclude_outliers,
        caption_prefix=caption_prefix,
    )
    if args.write_panel_c_comparison:
        if args.preset == "base":
            base_plot_df = plot_df
            instruct_plot_df = collect_delta_distributions(
                results_root=results_root,
                model_specs=INSTRUCT_MODEL_SPECS,
                exclude_outliers=exclude_outliers,
            )
        else:
            base_plot_df = collect_delta_distributions(
                results_root=results_root,
                model_specs=BASE_MODEL_SPECS,
                exclude_outliers=exclude_outliers,
            )
            instruct_plot_df = plot_df

        make_stacked_panel_c_figure(
            base_plot_df=base_plot_df,
            instruct_plot_df=instruct_plot_df,
            output_prefix=comparison_prefix,
            exclude_outliers=exclude_outliers,
        )

    print(f"Summary TSV saved to: {summary_path}")
    print(f"Main figure saved to: {figure_prefix}.png and {figure_prefix}.pdf")
    print(f"Panel A saved to: {output_dir / 'particle_removal_panel_a.png'}")
    print(f"Panel B saved to: {output_dir / 'particle_removal_panel_b.png'}")
    print(f"Panel C saved to: {output_dir / 'particle_removal_panel_c.png'}")
    if args.write_panel_c_comparison:
        print(f"Stacked Panel C figure saved to: {comparison_prefix}.png and {comparison_prefix}.pdf")
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
