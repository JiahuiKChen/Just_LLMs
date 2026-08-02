#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
TRANSFER_ROOT="${TRANSFER_ROOT:-localization/results/component_transfer_runs/20260724_exclusive_just_only_cross_patch}"
PLOT_ROOT="${PLOT_ROOT:-localization/results/plots/exclusive_just_only_component_transfer/ablation_style}"
EXCLUSIVE_RUN_ROOT="${EXCLUSIVE_RUN_ROOT:-localization/results/component_patch_runs/20260723_expanded_utterance_final_Meta-Llama-3-8B_exclusive_just_top500}"
ONLY_RUN_ROOT="${ONLY_RUN_ROOT:-localization/results/component_patch_runs/20260723_expanded_utterance_final_Meta-Llama-3-8B_not_just_only_full}"

EXCLUSIVE_SITE_SCORES="${EXCLUSIVE_SITE_SCORES:-${EXCLUSIVE_RUN_ROOT}/just/component_patching/site_scores.tsv}"
ONLY_SITE_SCORES="${ONLY_SITE_SCORES:-${ONLY_RUN_ROOT}/only/component_patching/site_scores.tsv}"
EXCLUSIVE_POOL="${EXCLUSIVE_POOL:-localization/results/patch_runs/20260723_expanded_llama3_exclusive_just_top500/just/generation_pool.tsv}"
ONLY_POOL="${ONLY_POOL:-localization/results/patch_runs/20260723_expanded_llama3_not_just_only_top500/only/generation_pool.tsv}"

cd "${REPO_ROOT}"
mkdir -p "${TRANSFER_ROOT}/logs" "${PLOT_ROOT}"

run_transfer() {
  local transfer_label="$1"
  local source_label="$2"
  local target_label="$3"
  local source_site_scores="$4"
  local target_pool="$5"
  local gpu="$6"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export JUST_LLMS_DEVICE_MAP=auto
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  {
    echo "[$(date)] START transfer=${transfer_label} physical_gpu=${gpu}"
    "${PYTHON_BIN}" localization/evaluate_component_site_transfer.py \
      --model_name "${MODEL_NAME}" \
      --source_site_scores "${source_site_scores}" \
      --target_pool "${target_pool}" \
      --transfer_label "${transfer_label}" \
      --source_label "${source_label}" \
      --target_label "${target_label}" \
      --output_root "${TRANSFER_ROOT}" \
      --top_ks 1,3,5,10 \
      --random_seeds 0,1,2,3 \
      --eval_modes sufficiency,necessity \
      --batch_size 4 \
      --analysis_bootstrap_replicates 5000 \
      --analysis_bootstrap_seed 0 \
      --analysis_signflip_replicates 20000
    echo "[$(date)] DONE transfer=${transfer_label} physical_gpu=${gpu}"
  } 2>&1 | tee "${TRANSFER_ROOT}/logs/${transfer_label}.log"
}

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  run_transfer \
    exclusive_just_to_only \
    exclusive_just \
    only \
    "${EXCLUSIVE_SITE_SCORES}" \
    "${ONLY_POOL}" \
    "${EXCLUSIVE_TO_ONLY_GPU:-1}" &
  exclusive_to_only_pid="$!"

  run_transfer \
    only_to_exclusive_just \
    only \
    exclusive_just \
    "${ONLY_SITE_SCORES}" \
    "${EXCLUSIVE_POOL}" \
    "${ONLY_TO_EXCLUSIVE_GPU:-2}" &
  only_to_exclusive_pid="$!"

  status=0
  if ! wait "${exclusive_to_only_pid}"; then
    status=1
  fi
  if ! wait "${only_to_exclusive_pid}"; then
    status=1
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "At least one transfer run failed; plots were not generated." >&2
    exit "${status}"
  fi
fi

if [[ "${SKIP_NATIVE_EVAL:-0}" != "1" && ! -f "${EXCLUSIVE_RUN_ROOT}/just/component_patching/eval_summary.tsv" ]]; then
  export CUDA_VISIBLE_DEVICES="${NATIVE_EXCLUSIVE_GPU:-1}"
  export JUST_LLMS_DEVICE_MAP=auto
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  "${PYTHON_BIN}" localization/evaluate_component_prompt_boundary.py \
    --model_name "${MODEL_NAME}" \
    --particles just \
    --top_ks 1,3,5,10 \
    --random_seeds 0,1,2,3 \
    --eval_modes necessity \
    --batch_size 4 \
    --output_root "${EXCLUSIVE_RUN_ROOT}" \
    --analysis_bootstrap_replicates 5000 \
    --analysis_bootstrap_seed 0 \
    --analysis_signflip_replicates 20000
fi

"${PYTHON_BIN}" localization/plot_patch_advantage_effects.py \
  --particles exclusive_just_to_only only_to_exclusive_just \
  --run-root "transfer=${TRANSFER_ROOT}" \
  --method component \
  --layout ablation_style \
  --ablation-style-eval-mode necessity \
  --localized-label "Cross-particle" \
  --target-native-label "Within-particle" \
  --target-native-dir "exclusive_just_to_only=${ONLY_RUN_ROOT}/only" \
  --target-native-dir "only_to_exclusive_just=${EXCLUSIVE_RUN_ROOT}/just" \
  --particle-label "exclusive_just_to_only=Exclusive just → only" \
  --particle-label "only_to_exclusive_just=Only → exclusive just" \
  --output-dir "${PLOT_ROOT}"

echo "Transfer outputs: ${TRANSFER_ROOT}"
echo "Plot outputs: ${PLOT_ROOT}"
