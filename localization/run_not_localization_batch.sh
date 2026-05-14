#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
DATA_PATH="${DATA_PATH:-dd_data/filtered_not_dd.tsv}"
MASK_PERCENTS="${MASK_PERCENTS:-0.5,1,5}"
NUM_FOLDS="${NUM_FOLDS:-5}"
HUMAN_NEGATIVES="${HUMAN_NEGATIVES:-5}"
SEED="${SEED:-13}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROW_LIMIT="${ROW_LIMIT:-}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-18000}"
USE_GENERATED_NEGATIVES="${USE_GENERATED_NEGATIVES:-0}"
GENERATED_NEGATIVES="${GENERATED_NEGATIVES:-3}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
JUST_LLMS_DEVICE_MAP="${JUST_LLMS_DEVICE_MAP:-balanced}"

MODEL_SHORT="${MODEL_NAME##*/}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/localization/results/${MODEL_SHORT}/not/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${REPO_ROOT}"

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

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${VISIBLE_GPUS}" ]]; then
  VISIBLE_GPUS="$(detect_gpu_ids "${MIN_FREE_MEM_MB}")"
fi

if [[ -z "${VISIBLE_GPUS}" ]]; then
  echo "No GPUs have at least ${MIN_FREE_MEM_MB} MiB free." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export JUST_LLMS_DEVICE_MAP

echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Model: ${MODEL_NAME}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Device map strategy: ${JUST_LLMS_DEVICE_MAP}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Use generated negatives: ${USE_GENERATED_NEGATIVES}"
echo

localize_cmd=(
  "${PYTHON_BIN}" localization/localize_not_units.py
  --model_name "${MODEL_NAME}"
  --data_path "${DATA_PATH}"
  --num_folds "${NUM_FOLDS}"
  --mask_percents "${MASK_PERCENTS}"
  --human_negatives "${HUMAN_NEGATIVES}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${ROW_LIMIT}" ]]; then
  localize_cmd+=(--row_limit "${ROW_LIMIT}")
fi

ablate_cmd=(
  "${PYTHON_BIN}" localization/ablate_not_units.py
  --model_name "${MODEL_NAME}"
  --data_path "${DATA_PATH}"
  --mask_dir "${OUTPUT_DIR}"
  --random_seeds "${RANDOM_SEEDS}"
)

if [[ -n "${ROW_LIMIT}" ]]; then
  ablate_cmd+=(--row_limit "${ROW_LIMIT}")
fi

if [[ "${USE_GENERATED_NEGATIVES}" == "1" ]]; then
  ablate_cmd+=(--use_generated_negatives --generated_negatives "${GENERATED_NEGATIVES}")
fi

echo "Starting localization at $(date)"
printf 'Command:'
printf ' %q' "${localize_cmd[@]}"
printf '\n\n'
"${localize_cmd[@]}"

echo
echo "Starting ablation at $(date)"
printf 'Command:'
printf ' %q' "${ablate_cmd[@]}"
printf '\n\n'
"${ablate_cmd[@]}"

echo
echo "Localization pipeline complete at $(date)"
