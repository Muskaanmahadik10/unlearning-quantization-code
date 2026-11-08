#!/bin/bash
# Corrected-UPQI convergence gate. Resource shape is supplied by sbatch.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_upqi_star
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK required}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_GPUS:-1}}"
STEP_MEM="${STEP_MEM:-200G}"
SCOPE="${SCOPE:-diagonal12}"
SLOTFILL=gate2_v4_final/slurm/gate2_v4/upqi_star_v42_slotfill.slurm

tasks=()
add_pairs() {
  local label="$1" model="$2" dataset="$3" ckpt="$4" algo="$5" seed="$6" quantizer="$7"
  if [[ "$SCOPE" == diagonal12 ]]; then
    for pair in 2:16 4:32 8:64 16:128; do tasks+=("$label|$model|$dataset|$ckpt|$algo|$seed|$quantizer|$pair"); done
  else
    for k in 2 4 8 16; do for n in 16 32 64 128; do tasks+=("$label|$model|$dataset|$ckpt|$algo|$seed|$quantizer|$k:$n"); done; done
  fi
}

[[ "$SCOPE" == diagonal12 || "$SCOPE" == fullgrid48 ]] || { echo "SCOPE must be diagonal12 or fullgrid48" >&2; exit 2; }
add_pairs l2_A llama2_7b_adamw tofu checkpoints/llama2_7b_adamw GA 42 nf4
add_pairs l3_A llama3_8b_adamw tofu checkpoints/llama3_8b_adamw GA 42 gptq_int4
add_pairs l3m_A llama3_muse_adamw muse_news checkpoints/llama3_muse_adamw NPO 5000 rtn_int8
[[ "$SCOPE" != diagonal12 || ${#tasks[@]} -eq 12 ]] || exit 2
[[ "$SCOPE" != fullgrid48 || ${#tasks[@]} -eq 48 ]] || exit 2

echo "UPQI-star convergence scope=$SCOPE tasks=${#tasks[@]} workers=$MAX_WORKERS"
launch() {
  local index="$1" task="$2" label model dataset ckpt algo seed quantizer pair k n log
  IFS='|' read -r label model dataset ckpt algo seed quantizer pair <<< "$task"
  IFS=':' read -r k n <<< "$pair"
  log="logs/gate2_v4_upqi_star/${SLURM_JOB_ID}_task$(printf '%03d' "$index")_${model}_${algo}_${seed}_${quantizer}_K${k}_N${n}.log"
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem="$STEP_MEM" \
    env SOURCE_LOCK="$SOURCE_LOCK" MODEL_TAG="$model" DATASET="$dataset" CKPT_BASE="$ckpt" \
      ALGO="$algo" SEED="$seed" QUANTIZER="$quantizer" N_QUADRATURE="$k" N_SAMPLES="$n" \
      RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_standard_upqi_star_v42}" \
      PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}" \
      bash "$SLOTFILL" >"$log" 2>&1 &
}

pids=(); index=0
for task in "${tasks[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_WORKERS" ]]; do sleep 2; done
  launch "$index" "$task"; pids+=("$!"); index=$((index + 1))
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
