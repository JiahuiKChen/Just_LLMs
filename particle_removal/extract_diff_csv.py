"""
Extract a simplified CSV with the per-token full-sequence log probability difference.

Reads a results CSV (tab-separated) produced by calculate_followup_probabilities.py,
computes  log_prob_full_with_per_token - log_prob_full_without_per_token  as a new
column called "with-without_diff", and writes a new CSV containing only:

    id  word  context  w_word  wo_word  followup  with-without_diff

USAGE:
    # Default output: results/diff_csv/my_results.csv
    python extract_diff_csv.py --input results/my_results.csv

    # Custom output path
    python extract_diff_csv.py --input results/my_results.csv --output results/my_diff.csv
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-token full-sequence log prob difference into a slim CSV"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input results CSV (tab-separated)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for the output CSV (default: <input_dir>/diff_csv/<input_filename>)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    df["with-without_diff"] = (
        df["log_prob_full_with_per_token"] - df["log_prob_full_without_per_token"]
    )

    keep_cols = ["id", "word", "context", "w_word", "wo_word", "followup", "with-without_diff"]
    df_out = df[keep_cols]

    if args.output is None:
        input_dir = os.path.dirname(args.input)
        diff_dir = os.path.join(input_dir, "diff_csv")
        os.makedirs(diff_dir, exist_ok=True)
        output_path = os.path.join(diff_dir, os.path.basename(args.input))
    else:
        output_path = args.output

    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {len(df_out)} rows to {output_path}")


if __name__ == "__main__":
    main()
