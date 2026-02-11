"""
Plot histogram of log probability differences with and without discourse particles.

Reads a results CSV file and creates a histogram showing:
- Full-sequence log probability difference (WITH particle - WITHOUT particle) by default
- Conditional (followup) log probability difference when --conditional is used

Also displays summary statistics from the corresponding .txt file.

USAGE:
    # Default: plots FULL-SEQUENCE probability differences
    python plot_probabilities.py

    # Plot CONDITIONAL probability differences
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


def load_summary_stats(txt_path):
    """Load summary statistics from the .txt file (raw and full-sequence)."""
    stats = {'raw': {}, 'full': {}}
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, 'r') as f:
        content = f.read()

    # Track which section we're in
    in_full_section = False

    # Parse the relevant lines
    for line in content.split('\n'):
        # Check for section headers
        if '--- RAW FOLLOWUP LOG PROBABILITIES' in line:
            in_full_section = False
            continue
        elif '--- FULL-SEQUENCE LOG PROBABILITIES' in line:
            in_full_section = True
            continue

        # Parse statistics based on current section
        if in_full_section:
            if 'Average log probability WITH particle:' in line:
                stats['full']['avg_with'] = float(line.split(':')[1].strip())
            elif 'Average log probability WITHOUT particle:' in line:
                stats['full']['avg_without'] = float(line.split(':')[1].strip())
            elif 'Average log probability difference:' in line:
                stats['full']['avg_diff'] = float(line.split(':')[1].strip())
            elif 'Median log probability difference:' in line:
                stats['full']['median_diff'] = float(line.split(':')[1].strip())
            elif 'Std dev of log probability difference:' in line:
                stats['full']['std_diff'] = float(line.split(':')[1].strip())
            elif 'Particle INCREASED full-sequence probability:' in line:
                parts = line.split(':')[1].strip()
                count = int(parts.split('(')[0].strip())
                pct = float(parts.split('(')[1].replace('%)', '').strip())
                stats['full']['increased_count'] = count
                stats['full']['increased_pct'] = pct
            elif 'Particle DECREASED full-sequence probability:' in line:
                parts = line.split(':')[1].strip()
                count = int(parts.split('(')[0].strip())
                pct = float(parts.split('(')[1].replace('%)', '').strip())
                stats['full']['decreased_count'] = count
                stats['full']['decreased_pct'] = pct
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


def plot_probabilities(results_path, output_path=None, use_conditional=False):
    """
    Create histogram of log probability differences (with particle - without particle).

    Args:
        results_path: Path to the results CSV file
        output_path: Path to save the plot (optional, will show if not provided)
        use_conditional: If True, plot conditional differences; otherwise full-sequence
    """
    # Load results
    df = pd.read_csv(results_path, sep='\t')
    print(f"Loaded {len(df)} examples from {results_path}")

    # Load summary stats from corresponding .txt file
    txt_path = results_path.rsplit('.', 1)[0] + '.txt'
    stats = load_summary_stats(txt_path)

    # Select which difference to plot based on flags
    if use_conditional:
        diff = df['log_prob_followup_with'] - df['log_prob_followup_without']
        plot_label = 'Conditional Log Probability Difference'
        stats_section = 'raw' if stats and stats.get('raw') else None
    else:
        if 'log_prob_full_with' not in df.columns or 'log_prob_full_without' not in df.columns:
            raise KeyError("Missing full-sequence columns; regenerate results with calculate_followup_probabilities.py")
        diff = df['log_prob_full_with'] - df['log_prob_full_without']
        plot_label = 'Full-Sequence Log Probability Difference'
        stats_section = 'full' if stats and stats.get('full') else None

    mean_diff = diff.mean()
    median_diff = diff.median()

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

    # Summary statistics panel - show the available section
    axes[1].axis('off')
    if stats and stats_section and stats.get(stats_section):
        # Fallback to single section if only one available
        s = stats[stats_section]
        if stats_section == 'raw':
            summary_text = (
                f"Summary Statistics (RAW)\n"
                f"{'─' * 26}\n"
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
        elif stats_section == 'full':
            summary_text = (
                f"Summary Statistics (FULL-SEQUENCE)\n"
                f"{'─' * 34}\n"
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
    norm_indicator = "(Conditional)" if use_conditional else "(Full-Sequence)"
    fig.suptitle(
        f'Follow-up Probability Distribution {norm_indicator}: {basename}',
        fontsize=11,
        fontweight='bold',
        y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

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
        '--conditional',
        action='store_true',
        default=False,
        help='Plot conditional (followup) probability differences'
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
                base_name = f"{model}_{rest}"
            else:
                base_name = f"{model}_Uncond_{rest}"
        else:
            model = _abbreviate_model_name(base_name)
            base_name = model if args.conditional else f"{model}_Uncond"
        args.output_path = os.path.join(base_dir, f"{base_name}.png")

    plot_probabilities(
        args.results_path,
        args.output_path,
        use_conditional=args.conditional
    )


if __name__ == "__main__":
    main()
