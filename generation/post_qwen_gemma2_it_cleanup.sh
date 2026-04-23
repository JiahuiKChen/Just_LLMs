#!/usr/bin/env bash
set -euo pipefail

MAIN_PID="${1:?main batch PID is required}"
MAIN_LOG="${2:?main batch log path is required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"

cd "${REPO_ROOT}"

echo "Waiting for main detached batch PID ${MAIN_PID}"
while kill -0 "${MAIN_PID}" 2>/dev/null; do
  sleep 60
done
echo "Main batch finished at $(date)"

if ! grep -q "All-model generation batch complete." "${MAIN_LOG}"; then
  echo "Main batch log does not show successful completion; skipping Gemma IT rerun."
  echo "Check ${MAIN_LOG}"
  exit 1
fi

"${PYTHON_BIN}" generation/run_parallel_generation_experiment.py \
  --model_name google/gemma-2-9b-it \
  --output_dir generation/results/gemma2_it \
  --prompt_mode chat \
  --data_paths stimulus_set/just.tsv stimulus_set/not.tsv \
  --samples_per_condition 50 \
  --top_p 0.9 \
  --temperature 1.0 \
  --sample_batch_size 32 \
  --max_sample_draws 500 \
  --max_new_tokens 64 \
  --min_free_mem_mb 18000 \
  --make_plots

rm -rf generation/results/*/shards generation/__pycache__

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import pandas as pd

rows = []
for path in sorted(Path("generation/results").glob("*/*_condition_summary.tsv")):
    df = pd.read_csv(path, sep="\t")
    df.insert(0, "summary_path", str(path))
    rows.append(df)

if not rows:
    print("No condition summary files found.")
    raise SystemExit

out = pd.concat(rows, ignore_index=True)
out.to_csv("generation/results/generation_condition_summary_all_models.tsv", sep="\t", index=False)
print("Wrote generation/results/generation_condition_summary_all_models.tsv")

short = out[out["min_unique"] < 50]
if short.empty:
    print("All condition summaries have min_unique >= 50.")
else:
    print("Conditions still below min_unique=50:")
    print(
        short[
            [
                "summary_path",
                "source_dataset",
                "generated_from",
                "candidates",
                "items",
                "min_unique",
            ]
        ].to_string(index=False)
    )
PY

echo "Post-run cleanup complete at $(date)"
