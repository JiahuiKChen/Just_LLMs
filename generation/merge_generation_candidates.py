#!/usr/bin/env python3
"""Merge per-run generation candidate TSVs into one table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "generation" / "results"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "all_models_all_particles_generation_candidates.tsv"
DEFAULT_DROP_COLUMNS = ("source_row_index",)
PROVENANCE_COLUMNS = ("source_model", "source_results_dir", "source_file")
DEFAULT_INCLUDE_DATASETS = ("just", "only", "not")
DEFAULT_EXCLUDE_RESULT_DIRS = ("figures", "qwen35_instruct_thinking_bad_20260416_221335")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-model generation candidate TSVs into one TSV."
    )
    parser.add_argument(
        "--results_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing per-model generation result folders.",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for the merged TSV.",
    )
    parser.add_argument(
        "--drop_columns",
        nargs="*",
        default=list(DEFAULT_DROP_COLUMNS),
        help="Optional columns to drop before merging.",
    )
    parser.add_argument(
        "--include_datasets",
        nargs="+",
        default=list(DEFAULT_INCLUDE_DATASETS),
        help="Keep only candidate files whose source dataset stem is listed here.",
    )
    parser.add_argument(
        "--exclude_result_dirs",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_RESULT_DIRS),
        help="Skip these result subdirectories entirely.",
    )
    return parser.parse_args()


def candidate_paths(
    results_dir: Path,
    include_datasets: list[str],
    exclude_result_dirs: list[str],
) -> list[Path]:
    paths: list[Path] = []
    include_set = set(include_datasets)
    exclude_set = set(exclude_result_dirs)
    for subdir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        if subdir.name in exclude_set:
            continue
        paths.extend(
            path
            for path in sorted(subdir.glob("*_generation_candidates.tsv"))
            if not path.name.endswith("_generation_candidates_weighted.tsv")
            and any(
                path.name.endswith(f"_context_{dataset}_generation_candidates.tsv")
                for dataset in include_set
            )
        )
    if not paths:
        raise ValueError(f"No candidate TSVs found under {results_dir}")
    return paths


def source_model_name(path: Path) -> str:
    if "_context_" in path.name:
        return path.name.split("_context_", 1)[0]
    return path.stem.removesuffix("_generation_candidates")


def configure_csv_limits() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def kept_columns(path: Path, drop_columns: list[str]) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    return [
        column
        for column in header
        if column not in drop_columns and not column.startswith("Unnamed:")
    ]


def merged_columns(paths: list[Path], drop_columns: list[str]) -> list[str]:
    ordered = list(PROVENANCE_COLUMNS)
    seen = set(ordered)
    for path in paths:
        for column in kept_columns(path, drop_columns):
            if column in seen:
                continue
            seen.add(column)
            ordered.append(column)
    return ordered


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_csv_limits()
    paths = candidate_paths(
        results_dir,
        include_datasets=args.include_datasets,
        exclude_result_dirs=args.exclude_result_dirs,
    )
    columns = merged_columns(paths, args.drop_columns)

    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()

        for path in paths:
            with path.open(newline="", encoding="utf-8") as in_handle:
                reader = csv.DictReader(in_handle, delimiter="\t")
                for row in reader:
                    row_count += 1
                    row["source_model"] = source_model_name(path)
                    row["source_results_dir"] = path.parent.name
                    row["source_file"] = str(path.relative_to(REPO_ROOT))
                    for column in args.drop_columns:
                        row.pop(column, None)
                    for column in list(row):
                        if column.startswith("Unnamed:"):
                            row.pop(column, None)
                    writer.writerow(row)

    print(f"Merged files: {len(paths)}")
    print(f"Merged rows: {row_count}")
    print(f"Wrote TSV: {output_path}")


if __name__ == "__main__":
    main()
