"""Shared publication styling for localization figures."""

from __future__ import annotations

import matplotlib
from matplotlib.ticker import ScalarFormatter


# Match the particle colors used by particle_removal/generate_paper_summary.py.
PARTICLE_COLORS = {
    "just": "#1b9e77",
    "only": "#d95f02",
    "not": "#7570b3",
}
CONTROL_COLOR = "#6B7280"
TIMES_SERIF_STACK = [
    "Times New Roman",
    "Liberation Serif",
    "Times",
    "Nimbus Roman",
]


def configure_times_font() -> None:
    """Prefer Times New Roman with a metrically compatible local fallback."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": TIMES_SERIF_STACK,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def configure_paper_style() -> None:
    """Apply a compact, Times-style configuration suitable for paper figures."""
    configure_times_font()
    matplotlib.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": (-2, 3),
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
        }
    )


def use_scientific_y_axis(axis) -> None:
    """Use scientific notation for sufficiently small or large y values."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    formatter.set_useOffset(False)
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.get_offset_text().set_fontsize(11)
