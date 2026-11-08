#!/bin/bash
# Parallel matrix for the exact metric suite. Resources are supplied by sbatch.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_metrics

SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK is required}"
MATRIX_SCOPE="${MATRIX_SCOPE:-pilot12}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_GPUS:-1}}"
STEP_MEM="${STEP_MEM:-200G}"
SLOTFILL=gate2_v4_final/slurm/gate2_v4/metric_suite_v42_slotfill.slurm

label_fields() {
  case "$1" in
    l2_A)  printf '%s\t%s\t%s\n' llama2_7b_adamw tofu checkpoints/llama2_7b_adamw ;;
    l3_A)  printf '%s\t%s\t%s\n' llama3_8b_adamw tofu checkpoints/llama3_8b_adamw ;;
    l2m_A) printf '%s\t%s\t%s\n' llama2_muse_adamw muse_news checkpoints/llama2_muse_adamw ;;
    l3m_A) printf '%s\t%s\t%s\n' llama3_muse_adamw muse_news checkpoints/llama3_muse_adamw ;;
    *) return 2 ;;
  esac
}

tasks=()
case "$MATRIX_SCOPE" in
  pilot12)
    for quantizer in gptq_int4 nf4 rtn_int8; do
      for label in l2_A l3_A l2m_A l3m_A; do
        tasks+=("$label|GA|42|$quantizer")
      done
    done
    ;;
  full144)
    for quantizer in gptq_int4 nf4 rtn_int8; do
      for label in l2_A l3_A l2m_A l3m_A; do
        for algo in GA GDR NPO KLR; do
          for seed in 42 2024 5000; do
            tasks+=("$label|$algo|$seed|$quantizer")
          done
        done
      done
    done
    [[ ${#tasks[@]} -eq 144 ]] || exit 2
    ;;
  *) echo "MATRIX_SCOPE must be pilot12 or full144" >&2; exit 2 ;;
esac

echo "Gate 2 metric suite v4.2 matrix scope=$MATRIX_SCOPE tasks=${#tasks[@]} workers=$MAX_WORKERS"

launch() {
  local index="$1" task="$2" label algo seed quantizer model dataset ckpt log
  IFS='|' read -r label algo seed quantizer <<< "$task"
  IFS=$'\t' read -r model dataset ckpt < <(label_fields "$label")
  log="logs/gate2_v4_metrics/${SLURM_JOB_ID}_task$(printf '%03d' "$index")_${model}_${algo}_${seed}_${quantizer}.log"
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem="$STEP_MEM" \
    env SOURCE_LOCK="$SOURCE_LOCK" MODEL_TAG="$model" DATASET="$dataset" CKPT_BASE="$ckpt" \
      ALGO="$algo" SEED="$seed" QUANTIZER="$quantizer" \
      RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_standard_metrics_v42}" \
      GT_ROOT="${GT_ROOT:-results/gate2_v4_standard}" \
      UPQI_ROOT="${UPQI_ROOT:-results/gate2_v4_standard}" \
      N_SAMPLES="${N_SAMPLES:-128}" N_RETAIN="${N_RETAIN:-128}" N_CALIBRATION="${N_CALIBRATION:-128}" \
      FISHER_FORGET_SAMPLES="${FISHER_FORGET_SAMPLES:-0}" \
      FISHER_RETAIN_SAMPLES="${FISHER_RETAIN_SAMPLES:-0}" \
      PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}" \
      bash "$SLOTFILL" >"$log" 2>&1 &
}

pids=()
index=0
for task in "${tasks[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_WORKERS" ]]; do sleep 2; done
  launch "$index" "$task"
  pids+=("$!")
  index=$((index + 1))
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
