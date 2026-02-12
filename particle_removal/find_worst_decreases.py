"""
Find examples where the particle most decreases followup probability.

Displays context, wo_word, w_word, followup, and the probability difference,
sorted by most negative difference.

USAGE:
    # Default: full-sequence (unconditional) differences per-token, top 10
    python find_worst_decreases.py

    # Use conditional (followup) differences per-token
    python find_worst_decreases.py --conditional

    # Show more examples
    python find_worst_decreases.py --top_k 25

    # Use a different results file
    python find_worst_decreases.py --results_path results/my_results.csv
"""

import argparse
import os
import textwrap
import pandas as pd


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


def _get_diff_series(df, use_conditional=False):
    if not use_conditional:
        if 'log_prob_full_with_per_token' not in df.columns or 'log_prob_full_without_per_token' not in df.columns:
            raise KeyError("Missing per-token full-sequence columns; regenerate results with calculate_followup_probabilities.py")
        return (
            df['log_prob_full_with_per_token'] - df['log_prob_full_without_per_token'],
            'Full-Sequence Log Probability Difference (Per-Token)'
        )
    if 'log_prob_followup_with_per_token' not in df.columns or 'log_prob_followup_without_per_token' not in df.columns:
        raise KeyError("Missing per-token conditional columns; regenerate results with calculate_followup_probabilities.py")
    return (
        df['log_prob_followup_with_per_token'] - df['log_prob_followup_without_per_token'],
        'Conditional Log Probability Difference (Per-Token)'
    )


def _format_block(label, text, width=120, indent='    '):
    if text is None:
        text = ""
    text = str(text)
    wrapped = textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)
    return f"{label}:\n{wrapped}"


def main():
    parser = argparse.ArgumentParser(
        description="Find examples where the particle most decreases followup probability"
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default='results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv',
        help='Path to the results CSV file'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of examples to display'
    )
    parser.add_argument(
        '--conditional',
        action='store_true',
        default=False,
        help='Use conditional (followup) differences per-token'
    )
    parser.add_argument(
        '--wrap_width',
        type=int,
        default=120,
        help='Line wrap width for context and text fields'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save CSV output (default: results/<results_file>_worst_decreases.csv)'
    )

    args = parser.parse_args()

    df = pd.read_csv(args.results_path, sep='\t')
    print(f"Loaded {len(df)} examples from {args.results_path}")

    diff, label = _get_diff_series(df, use_conditional=args.conditional)
    df = df.copy()
    df['_diff'] = diff

    required = ['context', 'wo_word', 'w_word', 'followup']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    worst = df.sort_values('_diff', ascending=True).head(args.top_k)

    # Write CSV output
    if args.output_path is None:
        base = os.path.splitext(os.path.basename(args.results_path))[0]
        if "_" in base:
            model, rest = base.split("_", 1)
            model = _abbreviate_model_name(model)
            base = f"{model}_{rest}"
        else:
            base = _abbreviate_model_name(base)
        if args.conditional:
            if "_" in base:
                model, rest = base.split("_", 1)
                base = f"{model}_Cond_{rest}"
            else:
                base = f"{base}_Cond"
        else:
            if "_" in base:
                model, rest = base.split("_", 1)
                base = f"{model}_Uncond_{rest}"
            else:
                base = f"{base}_Uncond"
        os.makedirs("results", exist_ok=True)
        args.output_path = os.path.join("results", f"{base}_worst_decreases.csv")

    worst.to_csv(args.output_path, index=False)
    print(f"Saved CSV: {args.output_path}")

    print(f"\nShowing {len(worst)} examples with MOST NEGATIVE {label}\n")

    for i, row in enumerate(worst.iterrows(), start=1):
        _, row = row
        print("=" * 80)
        print(f"Rank: {i}")
        if 'id' in row:
            print(f"ID: {row['id']}")
        if 'word' in row:
            print(f"Particle: {row['word']}")
        print(f"{label}: {row['_diff']:.6f}")
        print(_format_block("Context", row.get('context'), width=args.wrap_width))
        print(_format_block("wo_word (without particle)", row.get('wo_word'), width=args.wrap_width))
        print(_format_block("w_word (with particle)", row.get('w_word'), width=args.wrap_width))
        print(_format_block("Followup", row.get('followup'), width=args.wrap_width))

    print("=" * 80)


if __name__ == "__main__":
    main()
