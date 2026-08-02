#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
RUN_DATE="${RUN_DATE:-20260729}"
POSITION_LABEL="${POSITION_LABEL:-utterance_final}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
BASE_CANDIDATES="${BASE_CANDIDATES:-generation/expanded_results/llama3/Meta-Llama-3-8B_context_expanded_data_generation_candidates.tsv}"
BASE_POOL_ROOT="${BASE_POOL_ROOT:-localization/results/patch_runs/${RUN_DATE}_expanded_llama3_nonexclusive_just_top500}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-localization/results/component_patch_runs/${RUN_DATE}_expanded_${POSITION_LABEL}_Meta-Llama-3-8B_nonexclusive_just_top500}"
BASE_ONLY_POOL_ROOT="${BASE_ONLY_POOL_ROOT:-localization/results/patch_runs/20260723_expanded_llama3_not_just_only_top500}"
BASE_ONLY_RUN_ROOT="${BASE_ONLY_RUN_ROOT:-localization/results/component_patch_runs/20260723_expanded_utterance_final_Meta-Llama-3-8B_not_just_only_full}"
BASE_OVERLAP_ROOT="${BASE_OVERLAP_ROOT:-localization/results/component_overlap/${RUN_DATE}_nonexclusive_just_vs_only}"
BASE_TRANSFER_ROOT="${BASE_TRANSFER_ROOT:-localization/results/component_transfer_runs/${RUN_DATE}_nonexclusive_just_only_cross_patch}"
BASE_PLOT_ROOT="${BASE_PLOT_ROOT:-localization/results/plots/nonexclusive_just_only_component_transfer/ablation_style}"

INSTRUCT_MODEL="${INSTRUCT_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
INSTRUCT_CANDIDATES="${INSTRUCT_CANDIDATES:-generation/expanded_results/llama3_instruct/Meta-Llama-3-8B-Instruct_context_expanded_data_generation_candidates.tsv}"
INSTRUCT_POOL_ROOT="${INSTRUCT_POOL_ROOT:-localization/results/patch_runs/${RUN_DATE}_expanded_llama3_instruct_nonexclusive_just_top500}"
INSTRUCT_RUN_ROOT="${INSTRUCT_RUN_ROOT:-localization/results/component_patch_runs/${RUN_DATE}_expanded_${POSITION_LABEL}_Meta-Llama-3-8B-Instruct_nonexclusive_just_top500}"
INSTRUCT_ONLY_POOL_ROOT="${INSTRUCT_ONLY_POOL_ROOT:-localization/results/patch_runs/20260728_expanded_llama3_instruct_not_just_only_top500}"
INSTRUCT_ONLY_RUN_ROOT="${INSTRUCT_ONLY_RUN_ROOT:-localization/results/component_patch_runs/20260728_expanded_utterance_final_Meta-Llama-3-8B-Instruct_not_just_only_full}"
INSTRUCT_OVERLAP_ROOT="${INSTRUCT_OVERLAP_ROOT:-localization/results/component_overlap/${RUN_DATE}_llama3_instruct_nonexclusive_just_vs_only}"
INSTRUCT_TRANSFER_ROOT="${INSTRUCT_TRANSFER_ROOT:-localization/results/component_transfer_runs/${RUN_DATE}_llama3_instruct_nonexclusive_just_only_cross_patch}"
INSTRUCT_PLOT_ROOT="${INSTRUCT_PLOT_ROOT:-localization/results/plots/llama3_instruct_nonexclusive_just_only_component_transfer}"

cd "${REPO_ROOT}"

build_pool_if_needed() {
  local model_name="$1"
  local candidate_path="$2"
  local pool_root="$3"
  if [[ -s "${pool_root}/just/generation_pool.tsv" ]]; then
    echo "[$(date)] Reusing pool ${pool_root}/just/generation_pool.tsv"
    return
  fi
  "${PYTHON_BIN}" localization/build_particle_generation_pool.py \
    --model_name "${model_name}" \
    --particles just \
    --candidate_path "${candidate_path}" \
    --candidate_id_prefix just_j_ \
    --top_k 500 \
    --num_folds 2 \
    --seed 13 \
    --selection_mode top_positive_k \
    --output_root "${pool_root}"
}

run_native() {
  local label="$1"
  local model_name="$2"
  local pool_root="$3"
  local run_root="$4"
  local gpu="$5"
  local patch_dir="${run_root}/just/component_patching"
  local log_path="${run_root}/logs/nonexclusive_just.log"
  mkdir -p "${run_root}/logs"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export JUST_LLMS_DEVICE_MAP=auto
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  {
    echo "[$(date)] START native=${label} physical_gpu=${gpu}"
    if [[ ! -s "${patch_dir}/site_scores.tsv" ]]; then
      "${PYTHON_BIN}" localization/localize_component_prompt_boundary.py \
        --model_name "${model_name}" \
        --particles just \
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
        --model_name "${model_name}" \
        --particles just \
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
    echo "[$(date)] DONE native=${label} physical_gpu=${gpu}"
  } >>"${log_path}" 2>&1
}

run_transfer() {
  local label="$1"
  local model_name="$2"
  local transfer_root="$3"
  local transfer_label="$4"
  local source_label="$5"
  local target_label="$6"
  local source_site_scores="$7"
  local target_pool="$8"
  local gpu="$9"
  local output_dir="${transfer_root}/${transfer_label}/component_patching"
  local log_path="${transfer_root}/logs/${transfer_label}.log"
  mkdir -p "${transfer_root}/logs"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export JUST_LLMS_DEVICE_MAP=auto
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  {
    echo "[$(date)] START transfer=${label}/${transfer_label} physical_gpu=${gpu}"
    if [[ ! -s "${output_dir}/eval_summary.tsv" ]]; then
      "${PYTHON_BIN}" localization/evaluate_component_site_transfer.py \
        --model_name "${model_name}" \
        --source_site_scores "${source_site_scores}" \
        --target_pool "${target_pool}" \
        --transfer_label "${transfer_label}" \
        --source_label "${source_label}" \
        --target_label "${target_label}" \
        --output_root "${transfer_root}" \
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
    echo "[$(date)] DONE transfer=${label}/${transfer_label} physical_gpu=${gpu}"
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

build_pool_if_needed "${BASE_MODEL}" "${BASE_CANDIDATES}" "${BASE_POOL_ROOT}"
build_pool_if_needed "${INSTRUCT_MODEL}" "${INSTRUCT_CANDIDATES}" "${INSTRUCT_POOL_ROOT}"

echo "[$(date)] Launching base and instruct non-exclusive-just localization/evaluation"
run_native base "${BASE_MODEL}" "${BASE_POOL_ROOT}" "${BASE_RUN_ROOT}" "${BASE_GPU:-0}" & base_pid="$!"
run_native instruct "${INSTRUCT_MODEL}" "${INSTRUCT_POOL_ROOT}" "${INSTRUCT_RUN_ROOT}" "${INSTRUCT_GPU:-1}" & instruct_pid="$!"
wait_for_group "${base_pid}" "${instruct_pid}"

"${PYTHON_BIN}" localization/analyze_component_overlap.py \
  --left-site-rows "${BASE_RUN_ROOT}/just/component_patching/site_rows.tsv" \
  --right-site-rows "${BASE_ONLY_RUN_ROOT}/only/component_patching/site_rows.tsv" \
  --left-label nonexclusive_just \
  --right-label only \
  --top-ks 1,3,5,10 \
  --hide-rank-correlation \
  --output-dir "${BASE_OVERLAP_ROOT}"

"${PYTHON_BIN}" localization/analyze_component_overlap.py \
  --left-site-rows "${INSTRUCT_RUN_ROOT}/just/component_patching/site_rows.tsv" \
  --right-site-rows "${INSTRUCT_ONLY_RUN_ROOT}/only/component_patching/site_rows.tsv" \
  --left-label nonexclusive_just \
  --right-label only \
  --top-ks 1,3,5,10 \
  --hide-rank-correlation \
  --output-dir "${INSTRUCT_OVERLAP_ROOT}"

echo "[$(date)] Launching base and instruct bidirectional transfer jobs"
run_transfer base "${BASE_MODEL}" "${BASE_TRANSFER_ROOT}" nonexclusive_just_to_only nonexclusive_just only \
  "${BASE_RUN_ROOT}/just/component_patching/site_scores.tsv" \
  "${BASE_ONLY_POOL_ROOT}/only/generation_pool.tsv" "${BASE_TO_ONLY_GPU:-0}" & p0="$!"
run_transfer base "${BASE_MODEL}" "${BASE_TRANSFER_ROOT}" only_to_nonexclusive_just only nonexclusive_just \
  "${BASE_ONLY_RUN_ROOT}/only/component_patching/site_scores.tsv" \
  "${BASE_POOL_ROOT}/just/generation_pool.tsv" "${ONLY_TO_BASE_GPU:-1}" & p1="$!"
run_transfer instruct "${INSTRUCT_MODEL}" "${INSTRUCT_TRANSFER_ROOT}" nonexclusive_just_to_only nonexclusive_just only \
  "${INSTRUCT_RUN_ROOT}/just/component_patching/site_scores.tsv" \
  "${INSTRUCT_ONLY_POOL_ROOT}/only/generation_pool.tsv" "${INSTRUCT_TO_ONLY_GPU:-2}" & p2="$!"
run_transfer instruct "${INSTRUCT_MODEL}" "${INSTRUCT_TRANSFER_ROOT}" only_to_nonexclusive_just only nonexclusive_just \
  "${INSTRUCT_ONLY_RUN_ROOT}/only/component_patching/site_scores.tsv" \
  "${INSTRUCT_POOL_ROOT}/just/generation_pool.tsv" "${ONLY_TO_INSTRUCT_GPU:-3}" & p3="$!"
wait_for_group "${p0}" "${p1}" "${p2}" "${p3}"

"${PYTHON_BIN}" localization/plot_patch_advantage_effects.py \
  --particles nonexclusive_just_to_only only_to_nonexclusive_just \
  --run-root "transfer=${BASE_TRANSFER_ROOT}" \
  --method component \
  --layout ablation_style \
  --ablation-style-eval-mode necessity \
  --localized-label "Cross-particle" \
  --target-native-label "Within-particle" \
  --target-native-dir "nonexclusive_just_to_only=${BASE_ONLY_RUN_ROOT}/only" \
  --target-native-dir "only_to_nonexclusive_just=${BASE_RUN_ROOT}/just" \
  --particle-label "nonexclusive_just_to_only=Non-exclusive just → only" \
  --particle-label "only_to_nonexclusive_just=Only → non-exclusive just" \
  --output-dir "${BASE_PLOT_ROOT}"

"${PYTHON_BIN}" localization/plot_patch_advantage_effects.py \
  --particles nonexclusive_just_to_only only_to_nonexclusive_just \
  --run-root "transfer=${INSTRUCT_TRANSFER_ROOT}" \
  --method component \
  --layout ablation_style \
  --ablation-style-eval-mode necessity \
  --localized-label "Cross-particle" \
  --target-native-label "Within-particle" \
  --target-native-dir "nonexclusive_just_to_only=${INSTRUCT_ONLY_RUN_ROOT}/only" \
  --target-native-dir "only_to_nonexclusive_just=${INSTRUCT_RUN_ROOT}/just" \
  --particle-label "nonexclusive_just_to_only=Non-exclusive just → only" \
  --particle-label "only_to_nonexclusive_just=Only → non-exclusive just" \
  --output-dir "${INSTRUCT_PLOT_ROOT}"

echo "[$(date)] Completed non-exclusive-just/only component experiments"
echo "Base overlap: ${BASE_OVERLAP_ROOT}"
echo "Instruct overlap: ${INSTRUCT_OVERLAP_ROOT}"
echo "Base transfer plot: ${BASE_PLOT_ROOT}/component_transfer_generated.png"
echo "Instruct transfer plot: ${INSTRUCT_PLOT_ROOT}/component_transfer_generated.png"
