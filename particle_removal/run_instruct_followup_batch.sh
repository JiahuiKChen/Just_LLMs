#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
RUNNER="${REPO_ROOT}/particle_removal/run_parallel_probability_experiment.py"
EXPERIMENT_SCRIPT="particle_removal/calculate_followup_probabilities_instruct.py"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-18000}"
MODEL_PARALLEL_MIN_FREE_MEM_MB="${MODEL_PARALLEL_MIN_FREE_MEM_MB:-4000}"
START_AT_MODEL="${START_AT_MODEL:-}"

DATA_PATHS=(
  "dd_data/filtered_just_dd.tsv"
  "dd_data/filtered_not_dd.tsv"
  "dd_data/filtered_only_dd.tsv"
)

MODELS=(
  "allenai/OLMo-2-1124-7B-Instruct results/olmo2_instruct"
  "Qwen/Qwen3.5-9B results/qwen35_instruct"
  "google/gemma-2-9b-it results/gemma2_it"
  "meta-llama/Meta-Llama-3-8B-Instruct results/llama3_instruct"
)

cd "${REPO_ROOT}"

echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Experiment script: ${EXPERIMENT_SCRIPT}"
echo "Minimum free VRAM per replica: ${MIN_FREE_MEM_MB} MiB"
echo "Minimum free VRAM per model-parallel GPU: ${MODEL_PARALLEL_MIN_FREE_MEM_MB} MiB"
echo "Start at model: ${START_AT_MODEL:-<first>}"
echo

detect_gpu_ids() {
  local min_free_mem_mb="$1"
  "${PYTHON_BIN}" - <<'PY' "${min_free_mem_mb}"
import subprocess
import sys

min_free = int(sys.argv[1])
cmd = [
    "nvidia-smi",
    "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
]
output = subprocess.check_output(cmd, text=True)
gpu_ids = []
for line in output.strip().splitlines():
    gpu_idx_str, free_mem_str = [part.strip() for part in line.split(",", 1)]
    if int(free_mem_str) >= min_free:
        gpu_ids.append(gpu_idx_str)
print(",".join(gpu_ids))
PY
}

run_model_parallel_dataset() {
  local model_name="$1"
  local output_dir="$2"
  local data_path="$3"
  local visible_gpus="$4"
  local model_short="${model_name##*/}"
  local data_stem
  data_stem="$(basename "${data_path%.tsv}")"
  local output_path="${output_dir}/${model_short}_context_${data_stem}_removal.csv"

  echo "Running model-parallel dataset on GPUs ${visible_gpus}: ${data_path}"
  CUDA_VISIBLE_DEVICES="${visible_gpus}" "${PYTHON_BIN}" "${EXPERIMENT_SCRIPT}" \
    --model_name "${model_name}" \
    --data_path "${data_path}" \
    --output_path "${output_path}"

  "${PYTHON_BIN}" "${REPO_ROOT}/particle_removal/plot_probabilities.py" \
    --results_path "${output_path}"
}

started=0
for model_spec in "${MODELS[@]}"; do
  model_name="${model_spec%% *}"
  output_dir="${model_spec#* }"

  if [[ -n "${START_AT_MODEL}" && "${started}" -eq 0 && "${model_name}" != "${START_AT_MODEL}" ]]; then
    echo "Skipping model before restart point: ${model_name}"
    continue
  fi
  started=1

  echo "======================================================================"
  echo "Starting model: ${model_name}"
  echo "Output dir: ${output_dir}"
  date
  mkdir -p "${output_dir}"

  replica_gpu_ids="$(detect_gpu_ids "${MIN_FREE_MEM_MB}")"
  if [[ -n "${replica_gpu_ids}" ]]; then
    echo "Mode: data parallel across full-replica GPUs ${replica_gpu_ids}"
    read -r -a replica_gpu_array <<< "$(echo "${replica_gpu_ids}" | tr ',' ' ')"
    "${PYTHON_BIN}" "${RUNNER}" \
      --model_name "${model_name}" \
      --output_dir "${output_dir}" \
      --experiment_script "${EXPERIMENT_SCRIPT}" \
      --min_free_mem_mb "${MIN_FREE_MEM_MB}" \
      --gpu_ids "${replica_gpu_array[@]}" \
      --data_paths "${DATA_PATHS[@]}"
  else
    model_parallel_gpu_ids="$(detect_gpu_ids "${MODEL_PARALLEL_MIN_FREE_MEM_MB}")"
    if [[ -z "${model_parallel_gpu_ids}" ]]; then
      echo "No GPUs have at least ${MODEL_PARALLEL_MIN_FREE_MEM_MB} MiB free; aborting."
      exit 1
    fi
    echo "Mode: model parallel across GPUs ${model_parallel_gpu_ids}"
    for data_path in "${DATA_PATHS[@]}"; do
      run_model_parallel_dataset "${model_name}" "${output_dir}" "${data_path}" "${model_parallel_gpu_ids}"
    done
  fi

  echo "Finished model: ${model_name}"
  date
  echo
done

echo "Instruction-tuned followup probability batch complete."
