#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
GPU_ID="${GPU_ID:-0}"
CANDIDATE_PATH="${CANDIDATE_PATH:-generation/expanded_results/llama3/Meta-Llama-3-8B_context_expanded_data_generation_candidates.tsv}"
EXCLUSIVE_POOL_ROOT="${EXCLUSIVE_POOL_ROOT:-localization/results/patch_runs/20260723_expanded_llama3_exclusive_just_top500}"
EXCLUSIVE_RUN_ROOT="${EXCLUSIVE_RUN_ROOT:-localization/results/component_patch_runs/20260723_expanded_utterance_final_Meta-Llama-3-8B_exclusive_just_top500}"
ONLY_RUN_ROOT="${ONLY_RUN_ROOT:-localization/results/component_patch_runs/20260723_expanded_utterance_final_Meta-Llama-3-8B_not_just_only_full}"
OVERLAP_ROOT="${OVERLAP_ROOT:-localization/results/component_overlap/20260723_exclusive_just_vs_only}"

cd "${REPO_ROOT}"

if [[ "${SKIP_POOL:-0}" != "1" ]]; then
  "${PYTHON_BIN}" localization/build_particle_generation_pool.py \
    --model_name "${MODEL_NAME}" \
    --particles just \
    --candidate_path "${CANDIDATE_PATH}" \
    --candidate_id_prefix just_e_ \
    --top_k 500 \
    --num_folds 2 \
    --seed 13 \
    --selection_mode top_positive_k \
    --output_root "${EXCLUSIVE_POOL_ROOT}"
fi

if [[ "${SKIP_LOCALIZATION:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  JUST_LLMS_DEVICE_MAP=auto \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false \
    "${PYTHON_BIN}" localization/localize_component_prompt_boundary.py \
      --model_name "${MODEL_NAME}" \
      --particles just \
      --components resid,attn,mlp \
      --batch_size 4 \
      --trace_rows_per_source 1 \
      --position_label utterance_final \
      --pool_root "${EXCLUSIVE_POOL_ROOT}" \
      --output_root "${EXCLUSIVE_RUN_ROOT}"
fi

"${PYTHON_BIN}" localization/analyze_component_overlap.py \
  --left-site-rows "${EXCLUSIVE_RUN_ROOT}/just/component_patching/site_rows.tsv" \
  --right-site-rows "${ONLY_RUN_ROOT}/only/component_patching/site_rows.tsv" \
  --left-label exclusive_just \
  --right-label only \
  --top-ks 1,3,5,10 \
  --output-dir "${OVERLAP_ROOT}"

echo "Exclusive-just pool: ${EXCLUSIVE_POOL_ROOT}"
echo "Exclusive-just localization: ${EXCLUSIVE_RUN_ROOT}"
echo "Overlap outputs: ${OVERLAP_ROOT}"
