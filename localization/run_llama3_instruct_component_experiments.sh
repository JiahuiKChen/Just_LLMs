#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
RUN_DATE="${RUN_DATE:-20260728}"
POSITION_LABEL="${POSITION_LABEL:-utterance_final}"

LEGACY_RESULTS_DIR="${LEGACY_RESULTS_DIR:-generation/results/llama3_instruct}"
EXPANDED_CANDIDATES="${EXPANDED_CANDIDATES:-generation/expanded_results/llama3_instruct/Meta-Llama-3-8B-Instruct_context_expanded_data_generation_candidates.tsv}"

LEGACY_POOL_ROOT="${LEGACY_POOL_ROOT:-localization/results/patch_runs/${RUN_DATE}_llama3_instruct_not_just_only_top500}"
LEGACY_RUN_ROOT="${LEGACY_RUN_ROOT:-localization/results/component_patch_runs/${RUN_DATE}_${POSITION_LABEL}_Meta-Llama-3-8B-Instruct_not_just_only_full}"
LEGACY_PLOT_ROOT="${LEGACY_PLOT_ROOT:-localization/results/plots/llama3_instruct_${POSITION_LABEL}_component_necessity}"

EXPANDED_POOL_ROOT="${EXPANDED_POOL_ROOT:-localization/results/patch_runs/${RUN_DATE}_expanded_llama3_instruct_not_just_only_top500}"
EXPANDED_RUN_ROOT="${EXPANDED_RUN_ROOT:-localization/results/component_patch_runs/${RUN_DATE}_expanded_${POSITION_LABEL}_Meta-Llama-3-8B-Instruct_not_just_only_full}"
EXPANDED_PLOT_ROOT="${EXPANDED_PLOT_ROOT:-localization/results/plots/llama3_instruct_expanded_${POSITION_LABEL}_component_necessity}"

EXCLUSIVE_POOL_ROOT="${EXCLUSIVE_POOL_ROOT:-localization/results/patch_runs/${RUN_DATE}_expanded_llama3_instruct_exclusive_just_top500}"
EXCLUSIVE_RUN_ROOT="${EXCLUSIVE_RUN_ROOT:-localization/results/component_patch_runs/${RUN_DATE}_expanded_${POSITION_LABEL}_Meta-Llama-3-8B-Instruct_exclusive_just_top500}"
OVERLAP_ROOT="${OVERLAP_ROOT:-localization/results/component_overlap/${RUN_DATE}_llama3_instruct_${POSITION_LABEL}_exclusive_just_vs_only}"

TRANSFER_ROOT="${TRANSFER_ROOT:-localization/results/component_transfer_runs/${RUN_DATE}_llama3_instruct_${POSITION_LABEL}_exclusive_just_only_cross_patch}"
TRANSFER_PLOT_ROOT="${TRANSFER_PLOT_ROOT:-localization/results/plots/llama3_instruct_${POSITION_LABEL}_exclusive_just_only_component_transfer}"

PARTICLES=(not just only)

cd "${REPO_ROOT}"
mkdir -p \
  "${LEGACY_RUN_ROOT}/logs" \
  "${EXPANDED_RUN_ROOT}/logs" \
  "${EXCLUSIVE_RUN_ROOT}/logs" \
  "${TRANSFER_ROOT}/logs" \
  "${LEGACY_PLOT_ROOT}" \
  "${EXPANDED_PLOT_ROOT}" \
  "${OVERLAP_ROOT}" \
  "${TRANSFER_PLOT_ROOT}"

build_pool_if_needed() {
  local expected_pool="$1"
  shift
  if [[ -s "${expected_pool}" ]]; then
    echo "[$(date)] Reusing pool ${expected_pool}"
    return
  fi
  "${PYTHON_BIN}" localization/build_particle_generation_pool.py "$@"
}

for particle in "${PARTICLES[@]}"; do
  build_pool_if_needed \
    "${LEGACY_POOL_ROOT}/${particle}/generation_pool.tsv" \
    --model_name "${MODEL_NAME}" \
    --particles "${particle}" \
    --results_dir "${LEGACY_RESULTS_DIR}" \
    --top_k 500 \
    --num_folds 2 \
    --seed 13 \
    --selection_mode top_positive_k \
    --output_root "${LEGACY_POOL_ROOT}"

  build_pool_if_needed \
    "${EXPANDED_POOL_ROOT}/${particle}/generation_pool.tsv" \
    --model_name "${MODEL_NAME}" \
    --particles "${particle}" \
    --candidate_path "${EXPANDED_CANDIDATES}" \
    --top_k 500 \
    --num_folds 2 \
    --seed 13 \
    --selection_mode top_positive_k \
    --output_root "${EXPANDED_POOL_ROOT}"
done

build_pool_if_needed \
  "${EXCLUSIVE_POOL_ROOT}/just/generation_pool.tsv" \
  --model_name "${MODEL_NAME}" \
  --particles just \
  --candidate_path "${EXPANDED_CANDIDATES}" \
  --candidate_id_prefix just_e_ \
  --top_k 500 \
  --num_folds 2 \
  --seed 13 \
  --selection_mode top_positive_k \
  --output_root "${EXCLUSIVE_POOL_ROOT}"

run_native() {
  local label="$1"
  local particle="$2"
  local pool_root="$3"
  local run_root="$4"
  local gpu="$5"
  local log_path="${run_root}/logs/${label}_${particle}.log"
  local patch_dir="${run_root}/${particle}/component_patching"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export JUST_LLMS_DEVICE_MAP=auto
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  {
    echo "[$(date)] START native=${label}/${particle} physical_gpu=${gpu}"
    if [[ ! -s "${patch_dir}/site_scores.tsv" ]]; then
      "${PYTHON_BIN}" localization/localize_component_prompt_boundary.py \
        --model_name "${MODEL_NAME}" \
        --particles "${particle}" \
        --components resid,attn,mlp \
        --batch_size 4 \
        --trace_rows_per_source 1 \
        --position_label "${POSITION_LABEL}" \
        --pool_root "${pool_root}" \
        --output_root "${run_root}"
    else
      echo "[$(date)] Reusing localization ${patch_dir}/site_scores.tsv"
    fi

    if [[ ! -s "${patch_dir}/eval_summary.tsv" ]]; then
      "${PYTHON_BIN}" localization/evaluate_component_prompt_boundary.py \
        --model_name "${MODEL_NAME}" \
        --particles "${particle}" \
        --top_ks 1,3,5,10 \
        --random_seeds 0,1,2,3 \
        --eval_modes sufficiency,necessity \
        --batch_size 4 \
        --output_root "${run_root}" \
        --analysis_bootstrap_replicates 5000 \
        --analysis_bootstrap_seed 0 \
        --analysis_signflip_replicates 20000
    else
      echo "[$(date)] Reusing evaluation ${patch_dir}/eval_summary.tsv"
    fi
    echo "[$(date)] DONE native=${label}/${particle} physical_gpu=${gpu}"
  } >>"${log_path}" 2>&1
}

wait_for_group() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

echo "[$(date)] Launching native component jobs (wave 1)"
run_native legacy not "${LEGACY_POOL_ROOT}" "${LEGACY_RUN_ROOT}" 0 & p0="$!"
run_native legacy just "${LEGACY_POOL_ROOT}" "${LEGACY_RUN_ROOT}" 1 & p1="$!"
run_native legacy only "${LEGACY_POOL_ROOT}" "${LEGACY_RUN_ROOT}" 2 & p2="$!"
run_native expanded only "${EXPANDED_POOL_ROOT}" "${EXPANDED_RUN_ROOT}" 3 & p3="$!"
wait_for_group "${p0}" "${p1}" "${p2}" "${p3}"

echo "[$(date)] Launching native component jobs (wave 2)"
run_native expanded not "${EXPANDED_POOL_ROOT}" "${EXPANDED_RUN_ROOT}" 0 & p0="$!"
run_native expanded just "${EXPANDED_POOL_ROOT}" "${EXPANDED_RUN_ROOT}" 1 & p1="$!"
run_native exclusive just "${EXCLUSIVE_POOL_ROOT}" "${EXCLUSIVE_RUN_ROOT}" 2 & p2="$!"
wait_for_group "${p0}" "${p1}" "${p2}"

"${PYTHON_BIN}" localization/analyze_component_overlap.py \
  --left-site-rows "${EXCLUSIVE_RUN_ROOT}/just/component_patching/site_rows.tsv" \
  --right-site-rows "${EXPANDED_RUN_ROOT}/only/component_patching/site_rows.tsv" \
  --left-label exclusive_just \
  --right-label only \
  --top-ks 1,3,5,10 \
  --output-dir "${OVERLAP_ROOT}"

run_transfer() {
  local transfer_label="$1"
  local source_label="$2"
  local target_label="$3"
  local source_site_scores="$4"
  local target_pool="$5"
  local gpu="$6"
  local output_dir="${TRANSFER_ROOT}/${transfer_label}/component_patching"
  local log_path="${TRANSFER_ROOT}/logs/${transfer_label}.log"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export JUST_LLMS_DEVICE_MAP=auto
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  {
    echo "[$(date)] START transfer=${transfer_label} physical_gpu=${gpu}"
    if [[ ! -s "${output_dir}/eval_summary.tsv" ]]; then
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
    else
      echo "[$(date)] Reusing transfer evaluation ${output_dir}/eval_summary.tsv"
    fi
    echo "[$(date)] DONE transfer=${transfer_label} physical_gpu=${gpu}"
  } >>"${log_path}" 2>&1
}

echo "[$(date)] Launching bidirectional transfer jobs"
run_transfer \
  exclusive_just_to_only \
  exclusive_just \
  only \
  "${EXCLUSIVE_RUN_ROOT}/just/component_patching/site_scores.tsv" \
  "${EXPANDED_POOL_ROOT}/only/generation_pool.tsv" \
  0 & p0="$!"
run_transfer \
  only_to_exclusive_just \
  only \
  exclusive_just \
  "${EXPANDED_RUN_ROOT}/only/component_patching/site_scores.tsv" \
  "${EXCLUSIVE_POOL_ROOT}/just/generation_pool.tsv" \
  1 & p1="$!"
wait_for_group "${p0}" "${p1}"

"${PYTHON_BIN}" localization/plot_patch_advantage_effects.py \
  --particles "${PARTICLES[@]}" \
  --run-root "instruct=${LEGACY_RUN_ROOT}" \
  --method component \
  --layout ablation_style \
  --ablation-style-eval-mode necessity \
  --output-dir "${LEGACY_PLOT_ROOT}"

"${PYTHON_BIN}" localization/plot_patch_advantage_effects.py \
  --particles "${PARTICLES[@]}" \
  --run-root "expanded_instruct=${EXPANDED_RUN_ROOT}" \
  --method component \
  --layout ablation_style \
  --ablation-style-eval-mode necessity \
  --output-dir "${EXPANDED_PLOT_ROOT}"

"${PYTHON_BIN}" localization/plot_patch_advantage_effects.py \
  --particles exclusive_just_to_only only_to_exclusive_just \
  --run-root "transfer=${TRANSFER_ROOT}" \
  --method component \
  --layout ablation_style \
  --ablation-style-eval-mode necessity \
  --localized-label "Cross-particle" \
  --target-native-label "Within-particle" \
  --target-native-dir "exclusive_just_to_only=${EXPANDED_RUN_ROOT}/only" \
  --target-native-dir "only_to_exclusive_just=${EXCLUSIVE_RUN_ROOT}/just" \
  --particle-label "exclusive_just_to_only=Exclusive just → only" \
  --particle-label "only_to_exclusive_just=Only → exclusive just" \
  --output-dir "${TRANSFER_PLOT_ROOT}"

echo "[$(date)] All Llama-3-8B-Instruct component experiments completed"
echo "Legacy direct run: ${LEGACY_RUN_ROOT}"
echo "Expanded direct run: ${EXPANDED_RUN_ROOT}"
echo "Exclusive run: ${EXCLUSIVE_RUN_ROOT}"
echo "Overlap: ${OVERLAP_ROOT}"
echo "Transfers: ${TRANSFER_ROOT}"
