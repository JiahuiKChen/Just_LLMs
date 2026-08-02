#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
CANDIDATE_PATH="${CANDIDATE_PATH:-generation/expanded_results/llama3/Meta-Llama-3-8B_context_expanded_data_generation_candidates.tsv}"
POOL_ROOT="${POOL_ROOT:-localization/results/patch_runs/20260723_expanded_llama3_not_just_only_top500}"
RUN_ROOT="${RUN_ROOT:-localization/results/component_patch_runs/20260723_expanded_utterance_final_Meta-Llama-3-8B_not_just_only_full}"
PLOT_ROOT="${PLOT_ROOT:-localization/results/plots/expanded_utterance_final_patch_advantage_effects/ablation_style}"

PARTICLES=(not just only)
GPUS=(0 1 2)

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}/logs" "${PLOT_ROOT}"

if [[ "${SKIP_POOL:-0}" != "1" ]]; then
  "${PYTHON_BIN}" localization/build_particle_generation_pool.py \
    --model_name "${MODEL_NAME}" \
    --particles "${PARTICLES[@]}" \
    --candidate_path "${CANDIDATE_PATH}" \
    --top_k 500 \
    --num_folds 2 \
    --seed 13 \
    --selection_mode top_positive_k \
    --output_root "${POOL_ROOT}"
fi

run_particle() {
  local particle="$1"
  local gpu="$2"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export JUST_LLMS_DEVICE_MAP=auto
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  {
    echo "[$(date)] START particle=${particle} physical_gpu=${gpu}"

    "${PYTHON_BIN}" localization/localize_component_prompt_boundary.py \
      --model_name "${MODEL_NAME}" \
      --particles "${particle}" \
      --components resid,attn,mlp \
      --batch_size 4 \
      --trace_rows_per_source 1 \
      --position_label utterance_final \
      --pool_root "${POOL_ROOT}" \
      --output_root "${RUN_ROOT}"

    "${PYTHON_BIN}" localization/evaluate_component_prompt_boundary.py \
      --model_name "${MODEL_NAME}" \
      --particles "${particle}" \
      --top_ks 1,3,5,10 \
      --random_seeds 0,1,2,3 \
      --eval_modes sufficiency,necessity \
      --batch_size 4 \
      --output_root "${RUN_ROOT}" \
      --analysis_bootstrap_replicates 5000 \
      --analysis_bootstrap_seed 0 \
      --analysis_signflip_replicates 20000

    "${PYTHON_BIN}" localization/localize_particle_sites.py \
      --model_name "${MODEL_NAME}" \
      --particles "${particle}" \
      --batch_size 4 \
      --patch_scope utterance_final \
      --onset_tokens 1 \
      --site_metric restoration \
      --output_root "${RUN_ROOT}"

    "${PYTHON_BIN}" localization/evaluate_particle_sites.py \
      --model_name "${MODEL_NAME}" \
      --particles "${particle}" \
      --top_ks 1,3,5,10 \
      --random_seeds 0,1,2,3 \
      --eval_modes sufficiency,necessity \
      --output_root "${RUN_ROOT}" \
      --analysis_bootstrap_replicates 5000 \
      --analysis_bootstrap_seed 0 \
      --analysis_signflip_replicates 20000

    echo "[$(date)] DONE particle=${particle} physical_gpu=${gpu}"
  } 2>&1 | tee "${RUN_ROOT}/logs/${particle}.log"
}

pids=()
for index in "${!PARTICLES[@]}"; do
  run_particle "${PARTICLES[index]}" "${GPUS[index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one particle run failed; plots were not generated." >&2
  exit "${status}"
fi

"${PYTHON_BIN}" localization/plot_patch_advantage_effects.py \
  --particles "${PARTICLES[@]}" \
  --run-root "expanded=${RUN_ROOT}" \
  --method both \
  --layout ablation_style \
  --ablation-style-eval-mode necessity \
  --output-dir "${PLOT_ROOT}"

echo "Experiment outputs: ${RUN_ROOT}"
echo "Plot outputs: ${PLOT_ROOT}"
