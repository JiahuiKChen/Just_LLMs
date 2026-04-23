#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
RUNNER="${REPO_ROOT}/generation/run_parallel_generation_experiment.py"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-18000}"
START_AT_MODEL="${START_AT_MODEL:-}"

DATA_PATHS=(
  "stimulus_set/just.tsv"
  "stimulus_set/not.tsv"
)

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

cd "${REPO_ROOT}"

echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
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
    --data_paths "${DATA_PATHS[@]}" \
    --samples_per_condition "${SAMPLES_PER_CONDITION:-50}" \
    --top_p "${TOP_P:-0.9}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --sample_batch_size "${sample_batch_size}" \
    --max_sample_draws "${max_sample_draws}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
    --min_free_mem_mb "${MIN_FREE_MEM_MB}" \
    --skip_existing \
    --make_plots

  echo "Finished model: ${model_name}"
  date
  echo
done

echo "All-model generation batch complete."
