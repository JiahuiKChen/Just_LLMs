#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
RUNNER="${REPO_ROOT}/generation/run_parallel_generation_experiment.py"
ANALYZER="${REPO_ROOT}/generation/analyze_generation_results.py"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-18000}"
START_AT_MODEL="${START_AT_MODEL:-}"

DATA_PATH="stimulus_set/only.tsv"

# model | output_dir | prompt_mode
MODELS=(
  "meta-llama/Meta-Llama-3-8B|generation/results/llama3|base"
  "meta-llama/Meta-Llama-3-8B-Instruct|generation/results/llama3_instruct|chat"
  "google/gemma-2-9b|generation/results/gemma2|base"
  "google/gemma-2-9b-it|generation/results/gemma2_it|chat"
  "allenai/OLMo-2-1124-7B|generation/results/olmo2|base"
  "allenai/OLMo-2-1124-7B-Instruct|generation/results/olmo2_instruct|chat"
  "Qwen/Qwen3.5-9B|generation/results/qwen3_5|base"
  "Qwen/Qwen3.5-9B|generation/results/qwen35_instruct|chat"
)

build_only_not_comparison() {
  local model_name="$1"
  local output_dir="$2"
  local model_short="${model_name##*/}"
  local only_candidates="${output_dir}/${model_short}_context_only_generation_candidates.tsv"
  local not_candidates="${output_dir}/${model_short}_context_not_generation_candidates.tsv"
  local combined_candidates="${output_dir}/${model_short}_context_only_not_generation_candidates.tsv"

  if [[ ! -f "${only_candidates}" ]]; then
    echo "Missing only candidates: ${only_candidates}" >&2
    return 1
  fi
  if [[ ! -f "${not_candidates}" ]]; then
    echo "Missing not control candidates: ${not_candidates}" >&2
    return 1
  fi

  "${PYTHON_BIN}" - "${only_candidates}" "${not_candidates}" "${combined_candidates}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

only_path = Path(sys.argv[1])
not_path = Path(sys.argv[2])
combined_path = Path(sys.argv[3])

df = pd.concat(
    [pd.read_csv(only_path, sep="\t"), pd.read_csv(not_path, sep="\t")],
    ignore_index=True,
)
df.sort_values(
    ["source_dataset", "source_row_index", "generated_from", "unique_rank"],
    inplace=True,
)
combined_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(combined_path, sep="\t", index=False)
print(f"Combined only/not candidates saved to: {combined_path}", flush=True)
PY

  "${PYTHON_BIN}" "${ANALYZER}" \
    --candidates_path "${combined_candidates}" \
    --make_plots
}

write_aggregate_summary() {
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import pandas as pd

paths = sorted(Path("generation/results").glob("*/*_context_only_not_generation_condition_summary.tsv"))
rows = []
for path in paths:
    df = pd.read_csv(path, sep="\t")
    df.insert(0, "summary_path", str(path))
    rows.append(df)

if not rows:
    print("No only/not condition summaries found; aggregate not written.", flush=True)
else:
    out = pd.concat(rows, ignore_index=True)
    out_path = Path("generation/results/generation_condition_summary_only_not_all_models.tsv")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}", flush=True)
PY
}

cd "${REPO_ROOT}"

echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Data path: ${DATA_PATH}"
echo "Minimum free VRAM per replica: ${MIN_FREE_MEM_MB} MiB"
echo "Start at model: ${START_AT_MODEL:-<first>}"
echo

started=0
for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_name output_dir prompt_mode <<< "${model_spec}"

  if [[ -n "${START_AT_MODEL}" && "${started}" -eq 0 && "${model_name}" != "${START_AT_MODEL}" ]]; then
    echo "Skipping model before restart point: ${model_name}"
    continue
  fi
  started=1

  echo "======================================================================"
  echo "Starting model: ${model_name}"
  echo "Output dir: ${output_dir}"
  echo "Prompt mode: ${prompt_mode}"
  date
  mkdir -p "${output_dir}"

  default_sample_batch_size=64
  if [[ "${prompt_mode}" == "chat" ]]; then
    default_sample_batch_size=32
  fi
  sample_batch_size="${SAMPLE_BATCH_SIZE:-${default_sample_batch_size}}"

  default_max_sample_draws=200
  if [[ "${prompt_mode}" == "chat" ]]; then
    default_max_sample_draws=500
  fi
  max_sample_draws="${MAX_SAMPLE_DRAWS:-${default_max_sample_draws}}"

  echo "Sample batch size: ${sample_batch_size}"
  echo "Max sample draws: ${max_sample_draws}"

  "${PYTHON_BIN}" "${RUNNER}" \
    --model_name "${model_name}" \
    --output_dir "${output_dir}" \
    --prompt_mode "${prompt_mode}" \
    --data_paths "${DATA_PATH}" \
    --samples_per_condition "${SAMPLES_PER_CONDITION:-50}" \
    --top_p "${TOP_P:-0.9}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --sample_batch_size "${sample_batch_size}" \
    --max_sample_draws "${max_sample_draws}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
    --min_free_mem_mb "${MIN_FREE_MEM_MB}" \
    --skip_existing \
    --skip_analysis

  model_short="${model_name##*/}"
  only_candidates="${output_dir}/${model_short}_context_only_generation_candidates.tsv"
  "${PYTHON_BIN}" "${ANALYZER}" \
    --candidates_path "${only_candidates}" \
    --make_plots

  build_only_not_comparison "${model_name}" "${output_dir}"

  echo "Finished model: ${model_name}"
  date
  echo
done

write_aggregate_summary

echo "Only-generation batch complete."
