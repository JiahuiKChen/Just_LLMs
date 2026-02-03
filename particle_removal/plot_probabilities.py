"""
Plot histogram of log probability differences with and without discourse particles.

Reads a results CSV file and creates a histogram showing:
- Normalized log probability difference (WITH particle - WITHOUT particle) by default
- Raw log probability difference when --unnormalized flag is used

Also displays summary statistics from the corresponding .txt file.

USAGE:
    # Default: plots NORMALIZED probability differences
    python plot_probabilities.py

    # Plot UNNORMALIZED (raw) probability differences
    python plot_probabilities.py --unnormalized

    # Use different results file
    python plot_probabilities.py --results_path results/my_results.csv

    # Specify custom output path
    python plot_probabilities.py --output_path custom_plot.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_summary_stats(txt_path):
    """Load summary statistics from the .txt file (both raw and normalized)."""
    stats = {'raw': {}, 'normalized': {}}
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, 'r') as f:
        content = f.read()

    # Track which section we're in
    in_normalized_section = False

    # Parse the relevant lines
    for line in content.split('\n'):
        # Check for section headers
        if '--- NORMALIZED LOG PROBABILITIES' in line:
            in_normalized_section = True
            continue
        elif '--- RAW FOLLOWUP LOG PROBABILITIES' in line:
            in_normalized_section = False
            continue

        # Parse statistics based on current section
        if in_normalized_section:
            if 'Average normalized log prob WITH particle:' in line:
                stats['normalized']['avg_with'] = float(line.split(':')[1].strip())
            elif 'Average normalized log prob WITHOUT particle:' in line:
                stats['normalized']['avg_without'] = float(line.split(':')[1].strip())
            elif 'Average normalized log prob difference:' in line:
                stats['normalized']['avg_diff'] = float(line.split(':')[1].strip())
            elif 'Median normalized log prob difference:' in line:
                stats['normalized']['median_diff'] = float(line.split(':')[1].strip())
            elif 'Std dev of normalized log prob difference:' in line:
                stats['normalized']['std_diff'] = float(line.split(':')[1].strip())
            elif 'Particle INCREASED normalized probability:' in line:
                parts = line.split(':')[1].strip()
                count = int(parts.split('(')[0].strip())
                pct = float(parts.split('(')[1].replace('%)', '').strip())
                stats['normalized']['increased_count'] = count
                stats['normalized']['increased_pct'] = pct
            elif 'Particle DECREASED normalized probability:' in line:
                parts = line.split(':')[1].strip()
                count = int(parts.split('(')[0].strip())
                pct = float(parts.split('(')[1].replace('%)', '').strip())
                stats['normalized']['decreased_count'] = count
                stats['normalized']['decreased_pct'] = pct
        else:
            # Raw section
            if 'Average log probability WITH particle:' in line:
                stats['raw']['avg_with'] = float(line.split(':')[1].strip())
            elif 'Average log probability WITHOUT particle:' in line:
                stats['raw']['avg_without'] = float(line.split(':')[1].strip())
            elif 'Average log probability difference:' in line:
                stats['raw']['avg_diff'] = float(line.split(':')[1].strip())
            elif 'Median log probability difference:' in line:
                stats['raw']['median_diff'] = float(line.split(':')[1].strip())
            elif 'Std dev of log probability difference:' in line:
                stats['raw']['std_diff'] = float(line.split(':')[1].strip())
            elif 'Particle INCREASED followup probability:' in line:
                parts = line.split(':')[1].strip()
                count = int(parts.split('(')[0].strip())
                pct = float(parts.split('(')[1].replace('%)', '').strip())
                stats['raw']['increased_count'] = count
                stats['raw']['increased_pct'] = pct
            elif 'Particle DECREASED followup probability:' in line:
                parts = line.split(':')[1].strip()
                count = int(parts.split('(')[0].strip())
                pct = float(parts.split('(')[1].replace('%)', '').strip())
                stats['raw']['decreased_count'] = count
                stats['raw']['decreased_pct'] = pct

    return stats


def plot_probabilities(results_path, output_path=None, use_unnormalized=False):
    """
    Create histogram of log probability differences (with particle - without particle).

    Args:
        results_path: Path to the results CSV file
        output_path: Path to save the plot (optional, will show if not provided)
        use_unnormalized: If True, plot raw (unnormalized) differences; if False, plot normalized
    """
    # Load results
    df = pd.read_csv(results_path, sep='\t')
    print(f"Loaded {len(df)} examples from {results_path}")

    # Load summary stats from corresponding .txt file
    txt_path = results_path.rsplit('.', 1)[0] + '.txt'
    stats = load_summary_stats(txt_path)

    # Select which difference to plot based on flag
    if use_unnormalized:
        # Use raw log probability difference
        if 'log_prob_diff' in df.columns:
            diff = df['log_prob_diff']
        else:
            diff = df['log_prob_with_particle'] - df['log_prob_without_particle']
        plot_label = 'Raw Log Probability Difference'
        stats_section = 'raw' if stats else None
    else:
        # Use normalized log probability difference (default)
        if 'normalized_log_prob_diff' in df.columns:
            diff = df['normalized_log_prob_diff']
        else:
            # Fallback to raw if normalized not available
            print("Warning: normalized_log_prob_diff not found, falling back to raw difference")
            diff = df['log_prob_diff'] if 'log_prob_diff' in df.columns else df['log_prob_with_particle'] - df['log_prob_without_particle']
        plot_label = 'Normalized Log Probability Difference'
        stats_section = 'normalized' if stats else None

    mean_diff = diff.mean()
    median_diff = diff.median()

    # Create figure with 2 subplots in one row (1 histogram + 1 stats panel)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 0.5], 'wspace': 0.02})

    # Histogram for difference
    axes[0].hist(diff, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(f'{plot_label} (WITH - WITHOUT)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'{plot_label} Distribution', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Zero')
    axes[0].axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_diff:.4f}")
    axes[0].axvline(median_diff, color='green', linestyle=':', linewidth=2, label=f"Median: {median_diff:.4f}")
    axes[0].legend(fontsize=9)

    # Summary statistics panel - show BOTH raw and normalized
    axes[1].axis('off')
    if stats and stats.get('raw') and stats.get('normalized'):
        raw = stats['raw']
        norm = stats['normalized']
        summary_text = (
            f"{'=' * 40}\n"
            f"RAW LOG PROBABILITIES\n"
            f"{'─' * 40}\n"
            f"Avg log prob WITH:    {raw.get('avg_with', 'N/A'):.4f}\n"
            f"Avg log prob WITHOUT: {raw.get('avg_without', 'N/A'):.4f}\n"
            f"Avg difference:       {raw.get('avg_diff', 'N/A'):.4f}\n"
            f"Median difference:    {raw.get('median_diff', 'N/A'):.4f}\n"
            f"Std dev:              {raw.get('std_diff', 'N/A'):.4f}\n"
            f"INCREASED: {raw.get('increased_count', 'N/A')} ({raw.get('increased_pct', 'N/A'):.1f}%)\n"
            f"DECREASED: {raw.get('decreased_count', 'N/A')} ({raw.get('decreased_pct', 'N/A'):.1f}%)\n"
            f"\n"
            f"{'=' * 40}\n"
            f"NORMALIZED LOG PROBABILITIES\n"
            f"{'─' * 40}\n"
            f"Avg norm prob WITH:    {norm.get('avg_with', 'N/A'):.4f}\n"
            f"Avg norm prob WITHOUT: {norm.get('avg_without', 'N/A'):.4f}\n"
            f"Avg difference:        {norm.get('avg_diff', 'N/A'):.4f}\n"
            f"Median difference:     {norm.get('median_diff', 'N/A'):.4f}\n"
            f"Std dev:               {norm.get('std_diff', 'N/A'):.4f}\n"
            f"INCREASED: {norm.get('increased_count', 'N/A')} ({norm.get('increased_pct', 'N/A'):.1f}%)\n"
            f"DECREASED: {norm.get('decreased_count', 'N/A')} ({norm.get('decreased_pct', 'N/A'):.1f}%)"
        )
        axes[1].text(0.0, 0.5, summary_text, transform=axes[1].transAxes,
                     fontsize=9, family='monospace', verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    elif stats and stats_section and stats.get(stats_section):
        # Fallback to single section if only one available
        s = stats[stats_section]
        section_name = "RAW" if stats_section == 'raw' else "NORMALIZED"
        summary_text = (
            f"Summary Statistics ({section_name})\n"
            f"{'─' * 32}\n"
            f"Avg log prob WITH:    {s.get('avg_with', 'N/A'):.4f}\n"
            f"Avg log prob WITHOUT: {s.get('avg_without', 'N/A'):.4f}\n"
            f"Avg difference:       {s.get('avg_diff', 'N/A'):.4f}\n"
            f"  (positive = particle\n"
            f"   increases probability)\n"
            f"\n"
            f"Median difference: {s.get('median_diff', 'N/A'):.4f}\n"
            f"Std dev:           {s.get('std_diff', 'N/A'):.4f}\n"
            f"\n"
            f"INCREASED: {s.get('increased_count', 'N/A')} ({s.get('increased_pct', 'N/A'):.1f}%)\n"
            f"DECREASED: {s.get('decreased_count', 'N/A')} ({s.get('decreased_pct', 'N/A'):.1f}%)"
        )
        axes[1].text(0.0, 0.5, summary_text, transform=axes[1].transAxes,
                     fontsize=10, family='monospace', verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Extract info for title from filename
    basename = os.path.basename(results_path).replace('_removal.csv', '')
    norm_indicator = "(Raw)" if use_unnormalized else "(Normalized)"
    fig.suptitle(f'Follow-up Probability Distribution {norm_indicator}: {basename}', fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of log probabilities with/without discourse particles"
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
        '--unnormalized',
        action='store_true',
        default=False,
        help='Plot raw (unnormalized) probability differences instead of normalized (default: False)'
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified (replace .csv with .png)
    if args.output_path is None:
        args.output_path = args.results_path.rsplit('.', 1)[0] + '.png'

    plot_probabilities(args.results_path, args.output_path, use_unnormalized=args.unnormalized)


if __name__ == "__main__":
    main()
