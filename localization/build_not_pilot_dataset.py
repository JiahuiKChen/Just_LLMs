#!/usr/bin/env python3
"""Build the processed not-only localization pilot dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_HUMAN_NEGATIVES,
    prepare_not_pilot_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the processed not-control pilot dataset.")
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help="Path to the raw TSV source data.",
    )
    parser.add_argument(
        "--output_path",
        default="localization/results/not/not_pilot_dataset.tsv",
        help="Where to save the processed pilot TSV.",
    )
    parser.add_argument(
        "--human_negatives",
        type=int,
        default=DEFAULT_HUMAN_NEGATIVES,
        help="Number of human distractor followups to attach to each row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed for deterministic distractor matching.",
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        default=None,
        help="Optional number of validated rows to keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    pilot_df = prepare_not_pilot_dataset(
        data_path=args.data_path,
        output_path=output_path,
        human_negatives=args.human_negatives,
        seed=args.seed,
        row_limit=args.row_limit,
    )
    print(f"Saved {len(pilot_df)} rows to {output_path}")
    print(f"Mean human negatives per row: {pilot_df['human_negative_count'].mean():.2f}")
    print(
        "Preview columns: "
        + ", ".join(
            column
            for column in (
                "pilot_row_index",
                "source_row_index",
                "id",
                "word",
                "followup",
                "human_negative_followups",
            )
            if column in pilot_df.columns
        )
    )


if __name__ == "__main__":
    main()
