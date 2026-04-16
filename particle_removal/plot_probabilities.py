"""
Plot histogram of log probability differences with and without discourse particles.

Reads a results CSV file and creates a histogram showing:
- Full-sequence log probability difference (WITH particle - WITHOUT particle) per-token by default
- Conditional (followup) log probability difference per-token when --conditional is used
- Only non-outliers after applying the IQR-based removal filter

Summary statistics are computed from the retained non-outlier rows.

USAGE:
    # Default: plots FULL-SEQUENCE probability differences per-token
    python plot_probabilities.py

    # Plot CONDITIONAL probability differences per-token
    python plot_probabilities.py --conditional

    # Use different results file
    python plot_probabilities.py --results_path results/my_results.csv

    # Specify custom output path
    python plot_probabilities.py --output_path custom_plot.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def _abbreviate_model_name(model_name: str) -> str:
    tokens = model_name.split('-')
    if len(tokens) <= 2:
        return model_name
    size_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i]
        if tok.upper().endswith(('B', 'M', 'K')) and any(ch.isdigit() for ch in tok):
            size_idx = i
            break
    if size_idx is None:
        head = ''.join(t[0] for t in tokens[:-1] if t)
        return f"{head}{tokens[-1]}" if head else model_name
    start_tail = size_idx
    if size_idx - 1 >= 0 and all(ch.isdigit() or ch == '.' for ch in tokens[size_idx - 1]):
        start_tail = size_idx - 1
    head_tokens = tokens[:start_tail]
    tail_tokens = tokens[start_tail:]
    head_abbrev = ''.join(t[0] for t in head_tokens if t)
    if head_abbrev:
        return f"{head_abbrev}-{'-'.join(tail_tokens)}"
    return '-'.join(tail_tokens)


def plot_probabilities(results_path, output_path=None, use_conditional=False):
    """
    Create histogram of log probability differences (with particle - without particle).

    Args:
        results_path: Path to the results CSV file
        output_path: Path to save the plot (optional, will show if not provided)
        use_conditional: If True, plot conditional differences; otherwise full-sequence (per-token)
    """
    # Load results
    df = pd.read_csv(results_path, sep='\t')
    print(f"Loaded {len(df)} examples from {results_path}")
    if 'removal_upper_iqr_outlier' not in df.columns:
        raise KeyError(
            "Missing IQR outlier column; regenerate results with calculate_followup_probabilities.py"
        )

    outlier_mask = df['removal_upper_iqr_outlier'].fillna(False).astype(bool)
    excluded_count = int(outlier_mask.sum())
    df = df.loc[~outlier_mask].copy()
    print(f"Excluded {excluded_count} IQR outliers; plotting {len(df)} non-outliers")

    # Select which difference to plot based on flags
    if use_conditional:
        if 'log_prob_followup_with_per_token' not in df.columns or 'log_prob_followup_without_per_token' not in df.columns:
            raise KeyError("Missing per-token conditional columns; regenerate results with calculate_followup_probabilities.py")
        diff = df['log_prob_followup_with_per_token'] - df['log_prob_followup_without_per_token']
        plot_label = 'Conditional Log Probability Difference (Per-Token)'
    else:
        if 'log_prob_full_with_per_token' not in df.columns or 'log_prob_full_without_per_token' not in df.columns:
            raise KeyError("Missing per-token full-sequence columns; regenerate results with calculate_followup_probabilities.py")
        diff = df['log_prob_full_with_per_token'] - df['log_prob_full_without_per_token']
        plot_label = 'Full-Sequence Log Probability Difference (Per-Token)'

    mean_diff = diff.mean()
    median_diff = diff.median()
    std_diff = diff.std()
    increased_count = int((diff > 0).sum())
    decreased_count = int((diff < 0).sum())
    total_count = len(diff)
    increased_pct = (increased_count / total_count * 100) if total_count else 0.0
    decreased_pct = (decreased_count / total_count * 100) if total_count else 0.0

    # Create figure with 2 subplots in one row (1 histogram + 1 stats panel)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5),
        gridspec_kw={'width_ratios': [1, 0.35], 'wspace': 0.01}
    )

    # Histogram for difference
    axes[0].hist(diff, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(f'{plot_label} (WITH - WITHOUT)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Zero')
    axes[0].axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_diff:.4f}")
    axes[0].axvline(median_diff, color='green', linestyle=':', linewidth=2, label=f"Median: {median_diff:.4f}")
    axes[0].legend(fontsize=9)

    # Summary statistics panel for the filtered distribution
    axes[1].axis('off')
    summary_text = (
        "Summary Statistics\n"
        "------------------\n"
        f"Plotted samples: {total_count}\n"
        f"IQR outliers removed: {excluded_count}\n"
        "\n"
        f"Mean difference:   {mean_diff:.4f}\n"
        f"Median difference: {median_diff:.4f}\n"
        f"Std dev:           {std_diff:.4f}\n"
        "\n"
        f"INCREASED: {increased_count} ({increased_pct:.1f}%)\n"
        f"DECREASED: {decreased_count} ({decreased_pct:.1f}%)"
    )
    axes[1].text(0.0, 0.5, summary_text, transform=axes[1].transAxes,
                 fontsize=10, family='monospace', verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Extract info for title from filename
    basename = os.path.basename(results_path).replace('_removal.csv', '')
    norm_indicator = "(Conditional, Per-Token, IQR-Filtered)" if use_conditional else "(Full-Sequence, Per-Token, IQR-Filtered)"
    fig.suptitle(
        f'Follow-up Probability Distribution {norm_indicator}: {basename}',
        fontsize=11,
        fontweight='bold',
        y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    # Save or show
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of per-token log probabilities with/without discourse particles"
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default='results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv',
        help='Path to the results CSV file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save the plot (default: same as results file but .png)'
    )
    parser.add_argument(
        '--conditional',
        action='store_true',
        default=False,
        help='Plot conditional (followup) probability differences per-token'
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified (replace .csv with .png)
    if args.output_path is None:
        base_no_ext = args.results_path.rsplit('.', 1)[0]
        base_dir = os.path.dirname(base_no_ext)
        base_name = os.path.basename(base_no_ext)
        if "_" in base_name:
            model, rest = base_name.split("_", 1)
            model = _abbreviate_model_name(model)
            if args.conditional:
                base_name = f"{model}_Cond_{rest}"
            else:
                base_name = f"{model}_Uncond_{rest}"
        else:
            model = _abbreviate_model_name(base_name)
            base_name = f"{model}_Cond" if args.conditional else f"{model}_Uncond"
        args.output_path = os.path.join(base_dir, f"{base_name}.png")

    plot_probabilities(
        args.results_path,
        args.output_path,
        use_conditional=args.conditional
    )


if __name__ == "__main__":
    main()
