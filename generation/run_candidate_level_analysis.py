#!/usr/bin/env python3
"""Reproduce the candidate-level analysis workflow from generation/just.R."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from scipy import stats


WORD_ORDER = ["not", "just", "only"]
WORD_COLORS = {
    "just": "#1b9e77",
    "only": "#d95f02",
    "not": "#7570b3",
}
PARTICLE_LEGEND_ORDER = ["just", "only", "not"]
SOURCE_COLORS = {
    "P": "#0072B2",
    "P_prime": "#D55E00",
}
SOURCE_MARKERS = {
    "P": "o",
    "P_prime": "^",
}
SOURCE_Y_OFFSETS = {
    "P": -0.04,
    "P_prime": 0.04,
}
INSTRUCTION_MARKERS = {
    False: "o",
    True: "^",
}
FAMILY_ORDER = ["Llama3", "OLMo2", "Gemma2", "Qwen3.5"]
MODEL_WORD_OFFSETS = {
    "not": 0.0,
    "just": -0.13,
    "only": 0.13,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the candidate-level generation analysis implemented in generation/just.R."
    )
    parser.add_argument(
        "--candidates_path",
        default="generation/results/all_models_all_particles_generation_candidates.tsv",
        help="Candidate TSV to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        default="generation/results/figures",
        help="Directory for summary tables and plots.",
    )
    parser.add_argument(
        "--output_prefix",
        default="generation_candidate_level",
        help="Filename prefix for outputs written under output_dir.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf"],
        help="Plot formats to write.",
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
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def model_meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_model": "Meta-Llama-3-8B",
                "short": "Llama3-8B",
                "family": "Llama3",
                "instruction": False,
                "model_order": 0,
            },
            {
                "source_model": "Meta-Llama-3-8B-Instruct",
                "short": "Llama3-8B-I",
                "family": "Llama3",
                "instruction": True,
                "model_order": 1,
            },
            {
                "source_model": "OLMo-2-1124-7B",
                "short": "OLMo2-7B",
                "family": "OLMo2",
                "instruction": False,
                "model_order": 2,
            },
            {
                "source_model": "OLMo-2-1124-7B-Instruct",
                "short": "OLMo2-7B-I",
                "family": "OLMo2",
                "instruction": True,
                "model_order": 3,
            },
            {
                "source_model": "gemma-2-9b",
                "short": "Gemma2-9B",
                "family": "Gemma2",
                "instruction": False,
                "model_order": 4,
            },
            {
                "source_model": "gemma-2-9b-it",
                "short": "Gemma2-9B-I",
                "family": "Gemma2",
                "instruction": True,
                "model_order": 5,
            },
            {
                "source_model": "Qwen3.5-9B",
                "short": "Qwen3.5-9B",
                "family": "Qwen3.5",
                "instruction": False,
                "model_order": 6,
            },
            {
                "source_model": "Qwen3.5-9B-Instruct",
                "short": "Qwen3.5-9B-I",
                "family": "Qwen3.5",
                "instruction": True,
                "model_order": 7,
            },
        ]
    )


def load_relevant_candidates(candidates_path: Path) -> pd.DataFrame:
    df = pd.read_csv(candidates_path, sep="\t")
    required = [
        "source_model",
        "source_results_dir",
        "word",
        "generated_from",
        "response",
        "own_log_prob_per_token",
        "other_log_prob_per_token",
        "own_context_log_prob_advantage_per_token",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Candidate file is missing required columns: " + ", ".join(missing)
        )

    relevant = df.copy()
    relevant["source_model"] = np.where(
        (relevant["source_model"] == "Qwen3.5-9B")
        & (relevant["source_results_dir"] == "qwen35_instruct"),
        "Qwen3.5-9B-Instruct",
        relevant["source_model"],
    )
    relevant["word"] = pd.Categorical(
        relevant["word"],
        categories=WORD_ORDER,
        ordered=True,
    )
    relevant["own_context_log_prob_advantage_per_token"] = pd.to_numeric(
        relevant["own_context_log_prob_advantage_per_token"], errors="coerce"
    )
    relevant["own_log_prob_per_token"] = pd.to_numeric(
        relevant["own_log_prob_per_token"], errors="coerce"
    )
    relevant["other_log_prob_per_token"] = pd.to_numeric(
        relevant["other_log_prob_per_token"], errors="coerce"
    )
    return relevant[required[:-1] + ["own_context_log_prob_advantage_per_token"]]


def aggregate_candidates(relevant: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        relevant.groupby(["source_model", "word", "generated_from"], observed=True, sort=False)
        .agg(
            n=("own_context_log_prob_advantage_per_token", lambda s: int(np.isfinite(s).sum())),
            std=("own_context_log_prob_advantage_per_token", "std"),
            advantage=("own_context_log_prob_advantage_per_token", "mean"),
            advantage_pct=(
                "own_context_log_prob_advantage_per_token",
                lambda s: float(np.mean(s[np.isfinite(s)] > 0)) if np.isfinite(s).any() else float("nan"),
            ),
        )
        .reset_index()
    )

    def ci_half_width(row: pd.Series) -> float:
        if row["n"] <= 1 or not np.isfinite(row["std"]):
            return float("nan")
        t_crit = stats.t.ppf(0.975, df=row["n"] - 1)
        return float(t_crit * row["std"] / math.sqrt(row["n"]))

    grouped["cb"] = grouped.apply(ci_half_width, axis=1)
    merged = grouped.merge(meta, on="source_model", how="inner")
    merged.sort_values(["model_order", "word", "generated_from"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def t_test_against_zero(values: pd.Series) -> dict[str, float | str]:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    n = len(finite)
    record: dict[str, float | str] = {
        "n": n,
        "estimate": float(np.mean(finite)) if n else float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "parameter": float(n - 1) if n else float("nan"),
        "conf_low": float("nan"),
        "conf_high": float("nan"),
        "method": "One Sample t-test",
        "alternative": "two-sided",
        "effect_size": float("nan"),
    }
    if n < 2:
        return record

    sd = float(np.std(finite, ddof=1))
    mean = float(record["estimate"])
    if sd == 0.0:
        if mean > 0:
            record["statistic"] = float("inf")
            record["p_value"] = 0.0
            record["effect_size"] = float("inf")
        elif mean < 0:
            record["statistic"] = float("-inf")
            record["p_value"] = 0.0
            record["effect_size"] = float("-inf")
        else:
            record["statistic"] = 0.0
            record["p_value"] = 1.0
            record["effect_size"] = 0.0
        record["conf_low"] = mean
        record["conf_high"] = mean
        return record

    test = stats.ttest_1samp(finite, popmean=0.0)
    se = sd / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    record["statistic"] = float(test.statistic)
    record["p_value"] = float(test.pvalue)
    record["conf_low"] = mean - t_crit * se
    record["conf_high"] = mean + t_crit * se
    record["effect_size"] = mean / sd
    return record


def candidate_ttests(relevant: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = relevant.groupby(
        ["source_model", "word", "generated_from"],
        observed=True,
        sort=False,
    )
    for keys, group in grouped:
        row = dict(zip(["source_model", "word", "generated_from"], keys))
        row.update(t_test_against_zero(group["own_context_log_prob_advantage_per_token"]))
        rows.append(row)

    results = pd.DataFrame(rows).merge(meta, on="source_model", how="inner")
    results.sort_values(["model_order", "word", "generated_from"], inplace=True)
    results.reset_index(drop=True, inplace=True)
    return results


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def save_figure(fig: plt.Figure, base_path: Path, formats: list[str], dpi: int) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(base_path.with_suffix(f".{fmt}"), **save_kwargs)
    plt.close(fig)


def add_shape_and_color_legends(
    ax: plt.Axes,
    *,
    color_items: list[tuple[str, str]],
    marker_items: list[tuple[str, str, str]],
) -> None:
    color_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=label)
        for label, color in color_items
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="black",
            label=label,
        )
        for label, marker, _ in marker_items
    ]
    first = ax.legend(handles=color_handles, title="Color", loc="upper right")
    ax.add_artist(first)
    ax.legend(handles=marker_handles, title="Shape", loc="lower right")


def plot_advantage_by_model(
    agg: pd.DataFrame,
    out_base: Path,
    formats: list[str],
    dpi: int,
) -> None:
    model_rows = (
        agg[["short", "family", "model_order"]]
        .drop_duplicates()
        .sort_values("model_order")
    )
    y_positions: dict[str, float] = {}
    short_order: list[str] = []
    family_ranges: list[tuple[str, float, float]] = []
    family_boundaries: list[float] = []
    current_y = 0.0
    current_family = None
    family_start = 0.0
    row_step = 1.38
    family_gap = 0.52

    for row in model_rows.itertuples(index=False):
        if current_family is None:
            current_family = row.family
            family_start = current_y
        elif row.family != current_family:
            family_ranges.append((current_family, family_start, current_y - row_step))
            family_boundaries.append(current_y - row_step / 2.0 + family_gap / 2.0)
            current_y += family_gap
            current_family = row.family
            family_start = current_y

        y_positions[row.short] = current_y
        short_order.append(row.short)
        current_y += row_step

    if current_family is not None:
        family_ranges.append((current_family, family_start, current_y - row_step))

    fig, ax = plt.subplots(figsize=(10.1, 7.5))

    for _, y_start, y_end in family_ranges:
        ax.axhspan(
            y_start - 0.42,
            y_end + 0.42,
            facecolor="#F7F7F7",
            edgecolor="none",
            zorder=0,
        )
    for boundary in family_boundaries:
        ax.axhline(boundary, color="#D0D0D0", linewidth=0.9, zorder=1)

    for _, row in agg.iterrows():
        base_y = y_positions[row["short"]]
        y = (
            base_y
            + MODEL_WORD_OFFSETS[str(row["word"])]
            + SOURCE_Y_OFFSETS[str(row["generated_from"])]
        )
        marker = SOURCE_MARKERS[row["generated_from"]]
        color = WORD_COLORS[str(row["word"])]
        xerr = row["cb"] if np.isfinite(row["cb"]) else None
        ax.errorbar(
            row["advantage"],
            y,
            xerr=xerr,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=4.4,
            markersize=10.4,
            markeredgecolor="white",
            markeredgewidth=1.0,
            linestyle="none",
            zorder=3,
        )

    grouped = agg.groupby(["short", "word"], observed=True, sort=False)
    for (short, word), group in grouped:
        if len(group) != 2:
            continue
        ordered = group.sort_values("generated_from")
        y_center = y_positions[short] + MODEL_WORD_OFFSETS[str(ordered["word"].iloc[0])]
        x_values = ordered["advantage"].to_numpy(dtype=float)
        x_low = float(np.nanmin(x_values))
        x_high = float(np.nanmax(x_values))
        x_pad = 0.005
        half_height = 0.11
        ax.add_patch(
            Rectangle(
                (x_low - x_pad, y_center - half_height),
                max((x_high - x_low) + 2 * x_pad, 0.008),
                2 * half_height,
                facecolor=WORD_COLORS[str(word)],
                edgecolor="none",
                alpha=0.10,
                zorder=1.5,
            )
        )

    ci_low = agg["advantage"] - agg["cb"].fillna(0.0)
    ci_high = agg["advantage"] + agg["cb"].fillna(0.0)
    family_label_x = float(np.nanmin(ci_low)) - 0.02

    ax.axvline(0.0, color="#8A8A8A", linewidth=1.2, linestyle="--", zorder=1)
    ax.text(
        0.002,
        1.002,
        "no advantage",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#666666",
        clip_on=False,
    )
    ax.set_yticks([y_positions[short] for short in short_order])
    ax.set_yticklabels(short_order)
    ax.set_ylim(max(y_positions.values()) + 0.62, -0.62)
    ax.set_xlim(family_label_x - 0.002, float(np.nanmax(ci_high)) + 0.018)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.grid(True, which="major", color="#D2D2D2", linewidth=0.8, alpha=0.9)
    ax.xaxis.grid(True, which="minor", color="#E8E8E8", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "Mean candidate-level advantage per token\n"
        r"$\Delta_{\mathrm{tok}} = \log p(y \mid S_{\mathrm{src}}) - \log p(y \mid S_{\mathrm{alt}})$",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylabel("Model", fontsize=15, fontweight="bold")
    ax.set_title(
        "Own-context advantage by model (mean per-token $\\Delta$; 95% CIs)",
        pad=12,
        fontsize=17,
        fontweight="bold",
    )

    word_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=WORD_COLORS[word],
            markeredgecolor=WORD_COLORS[word],
            color=WORD_COLORS[word],
            markersize=8,
            label=word,
        )
        for word in PARTICLE_LEGEND_ORDER
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker=SOURCE_MARKERS[source],
            linestyle="none",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        for source, label in [("P", r"$S$ samples"), ("P_prime", r"$S'$ samples")]
    ]
    first_legend = ax.legend(
        handles=word_handles,
        title="Particle",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C7C7C7",
        fontsize=12,
        title_fontsize=14,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=source_handles,
        title="Response Source",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.78),
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C7C7C7",
        fontsize=12,
        title_fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    save_figure(fig, out_base, formats, dpi)


def family_axes() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), sharex=True)
    return fig, axes.reshape(-1)


def combo_offsets() -> dict[tuple[str, bool], float]:
    return {
        ("P", False): -0.18,
        ("P_prime", False): -0.06,
        ("P", True): 0.06,
        ("P_prime", True): 0.18,
    }


def plot_family_metric(
    df: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    title: str,
    out_base: Path,
    formats: list[str],
    dpi: int,
    yerr_col: str | None = None,
) -> None:
    fig, axes = family_axes()
    word_positions = {word: idx for idx, word in enumerate(WORD_ORDER)}
    offsets = combo_offsets()

    for ax, family in zip(axes, FAMILY_ORDER):
        subset = df[df["family"] == family].copy()
        for _, row in subset.iterrows():
            x = word_positions[str(row["word"])] + offsets[(row["generated_from"], bool(row["instruction"]))]
            y = row[y_col]
            yerr = row[yerr_col] if yerr_col and np.isfinite(row[yerr_col]) else None
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=INSTRUCTION_MARKERS[bool(row["instruction"])],
                color=SOURCE_COLORS[row["generated_from"]],
                ecolor=SOURCE_COLORS[row["generated_from"]],
                elinewidth=1.0,
                capsize=2.5,
                markersize=5.5,
                linestyle="none",
            )

        ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
        ax.set_xticks(range(len(WORD_ORDER)))
        ax.set_xticklabels(WORD_ORDER)
        ax.set_title(family)
        ax.set_ylabel(y_label)

    add_shape_and_color_legends(
        axes[0],
        color_items=[(label, SOURCE_COLORS[label]) for label in ["P", "P_prime"]],
        marker_items=[
            ("Base", INSTRUCTION_MARKERS[False], "instruction"),
            ("Instruction tuned", INSTRUCTION_MARKERS[True], "instruction"),
        ],
    )
    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    save_figure(fig, out_base, formats, dpi)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    candidates_path = Path(args.candidates_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.output_prefix

    meta = model_meta()
    relevant = load_relevant_candidates(candidates_path)
    agg = aggregate_candidates(relevant, meta)
    tests = candidate_ttests(relevant, meta)

    save_dataframe(agg, prefix.with_name(f"{prefix.name}_summary.tsv"))
    save_dataframe(tests, prefix.with_name(f"{prefix.name}_ttests.tsv"))

    plot_advantage_by_model(
        agg,
        prefix.with_name(f"{prefix.name}_advantage_by_model"),
        args.formats,
        args.dpi,
    )
    plot_family_metric(
        agg,
        y_col="advantage",
        y_label="Mean candidate-level own-context advantage per token",
        title="Candidate-level mean advantages by family",
        out_base=prefix.with_name(f"{prefix.name}_advantage_by_family"),
        formats=args.formats,
        dpi=args.dpi,
        yerr_col="cb",
    )
    plot_family_metric(
        agg,
        y_col="advantage_pct",
        y_label="Share of candidate deltas > 0",
        title="Candidate-level positive-delta share by family",
        out_base=prefix.with_name(f"{prefix.name}_positive_rate_by_family"),
        formats=args.formats,
        dpi=args.dpi,
    )
    plot_family_metric(
        tests,
        y_col="effect_size",
        y_label="One-sample Cohen's d",
        title="Candidate-level effect sizes by family",
        out_base=prefix.with_name(f"{prefix.name}_effect_size_by_family"),
        formats=args.formats,
        dpi=args.dpi,
    )

    print(f"Wrote candidate-level summary to {prefix.with_name(f'{prefix.name}_summary.tsv')}")
    print(f"Wrote candidate-level t-tests to {prefix.with_name(f'{prefix.name}_ttests.tsv')}")


if __name__ == "__main__":
    main()
