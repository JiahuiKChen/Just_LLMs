#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" generation/run_parallel_generation_experiment.py \
  --model_name "meta-llama/Meta-Llama-3-8B" \
  --output_dir "generation/results/llama3" \
  --data_paths "stimulus_set/just.tsv" "stimulus_set/not.tsv" \
  --samples_per_condition "${SAMPLES_PER_CONDITION:-50}" \
  --top_p "${TOP_P:-0.9}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --sample_batch_size "${SAMPLE_BATCH_SIZE:-64}" \
  --max_sample_draws "${MAX_SAMPLE_DRAWS:-200}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
  --min_free_mem_mb "${MIN_FREE_MEM_MB:-18000}" \
  --make_plots
