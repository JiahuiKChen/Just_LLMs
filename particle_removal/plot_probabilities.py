"""
Plot histograms of log probabilities with and without discourse particles.

Reads a results CSV file and creates side-by-side histograms comparing:
- Log probability of followup WITH particle
- Log probability of followup WITHOUT particle

Also displays summary statistics from the corresponding .txt file.

USAGE:
    # Default: saves to results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.png
    python plot_probabilities.py

    # Use different results file (saves to results/my_results.png)
    python plot_probabilities.py --results_path results/my_results.csv

    # Specify custom output path
    python plot_probabilities.py --output_path custom_plot.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_summary_stats(txt_path):
    """Load summary statistics from the .txt file."""
    stats = {}
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, 'r') as f:
        content = f.read()

    # Parse the relevant lines
    for line in content.split('\n'):
        if 'Average log probability WITH particle:' in line:
            stats['avg_with'] = float(line.split(':')[1].strip())
        elif 'Average log probability WITHOUT particle:' in line:
            stats['avg_without'] = float(line.split(':')[1].strip())
        elif 'Average log probability difference:' in line:
            stats['avg_diff'] = float(line.split(':')[1].strip())
        elif 'Median log probability difference:' in line:
            stats['median_diff'] = float(line.split(':')[1].strip())
        elif 'Std dev of log probability difference:' in line:
            stats['std_diff'] = float(line.split(':')[1].strip())
        elif 'Particle INCREASED' in line:
            # Parse "Particle INCREASED followup probability: 770 (55.2%)"
            parts = line.split(':')[1].strip()
            count = int(parts.split('(')[0].strip())
            pct = float(parts.split('(')[1].replace('%)', '').strip())
            stats['increased_count'] = count
            stats['increased_pct'] = pct
        elif 'Particle DECREASED' in line:
            parts = line.split(':')[1].strip()
            count = int(parts.split('(')[0].strip())
            pct = float(parts.split('(')[1].replace('%)', '').strip())
            stats['decreased_count'] = count
            stats['decreased_pct'] = pct

    return stats


def plot_probabilities(results_path, output_path=None):
    """
    Create side-by-side histograms of log probabilities with and without particles.

    Args:
        results_path: Path to the results CSV file
        output_path: Path to save the plot (optional, will show if not provided)
    """
    # Load results
    df = pd.read_csv(results_path, sep='\t')
    print(f"Loaded {len(df)} examples from {results_path}")

    # Load summary stats from corresponding .txt file
    txt_path = results_path.rsplit('.', 1)[0] + '.txt'
    stats = load_summary_stats(txt_path)

    # Calculate medians from data
    median_with = df['log_prob_with_particle'].median()
    median_without = df['log_prob_without_particle'].median()

    # Create figure with 3 subplots in one row (2 histograms + 1 stats panel)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1, 0.6], 'wspace': 0.15})

    # Histogram for WITH particle
    axes[0].hist(df['log_prob_with_particle'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Log Probability', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('WITH Particle', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    if stats:
        axes[0].axvline(stats['avg_with'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['avg_with']:.2f}")
    axes[0].axvline(median_with, color='green', linestyle=':', linewidth=2, label=f"Median: {median_with:.2f}")
    axes[0].legend(fontsize=9)

    # Histogram for WITHOUT particle
    axes[1].hist(df['log_prob_without_particle'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Log Probability', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('WITHOUT Particle', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    if stats:
        axes[1].axvline(stats['avg_without'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['avg_without']:.2f}")
    axes[1].axvline(median_without, color='green', linestyle=':', linewidth=2, label=f"Median: {median_without:.2f}")
    axes[1].legend(fontsize=9)

    # Summary statistics panel
    axes[2].axis('off')
    if stats:
        summary_text = (
            f"Summary Statistics\n"
            f"{'─' * 32}\n"
            f"Avg log prob WITH:    {stats['avg_with']:.4f}\n"
            f"Avg log prob WITHOUT: {stats['avg_without']:.4f}\n"
            f"Avg difference:       {stats['avg_diff']:.4f}\n"
            f"  (positive = particle\n"
            f"   increases probability)\n"
            f"\n"
            f"Median difference: {stats['median_diff']:.4f}\n"
            f"Std dev:           {stats['std_diff']:.4f}\n"
            f"\n"
            f"INCREASED: {stats['increased_count']} ({stats['increased_pct']:.1f}%)\n"
            f"DECREASED: {stats['decreased_count']} ({stats['decreased_pct']:.1f}%)"
        )
        axes[2].text(0.0, 0.5, summary_text, transform=axes[2].transAxes,
                     fontsize=10, family='monospace', verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Extract info for title from filename
    basename = os.path.basename(results_path).replace('_removal.csv', '')
    fig.suptitle(f'Follow-up Probability Distribution: {basename}', fontsize=13, fontweight='bold')

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

    args = parser.parse_args()

    # Auto-generate output path if not specified (replace .csv with .png)
    if args.output_path is None:
        args.output_path = args.results_path.rsplit('.', 1)[0] + '.png'

    plot_probabilities(args.results_path, args.output_path)


if __name__ == "__main__":
    main()
