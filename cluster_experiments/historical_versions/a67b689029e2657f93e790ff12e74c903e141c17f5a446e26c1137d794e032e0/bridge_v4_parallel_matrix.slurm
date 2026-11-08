#!/bin/bash
# Gate 2 v4 bridge matrix launcher. Resource shape is supplied by sbatch.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_bridge

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK is required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard}"
RANKINGS_PATH="${RANKINGS_PATH:-results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl}"
MATRIX_SCOPE="${MATRIX_SCOPE:-pilot18}"
N_SAMPLES="${N_SAMPLES:-128}"
N_RETAIN="${N_RETAIN:-128}"
N_CALIBRATION="${N_CALIBRATION:-128}"
MANIFEST_N_SAMPLES="${MANIFEST_N_SAMPLES:-128}"
MANIFEST_N_RETAIN="${MANIFEST_N_RETAIN:-128}"
MANIFEST_N_CALIBRATION="${MANIFEST_N_CALIBRATION:-128}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_GPUS:-1}}"
STEP_MEM="${STEP_MEM:-180G}"
SLOTFILL="gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
[[ -f "$SOURCE_LOCK" ]] || { echo "Missing source lock: $SOURCE_LOCK" >&2; exit 2; }
[[ -f "$SLOTFILL" ]] || { echo "Missing slotfill launcher: $SLOTFILL" >&2; exit 2; }
[[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "Bad MAX_WORKERS=$MAX_WORKERS" >&2; exit 2; }
[[ -n "$STEP_MEM" ]] || { echo "STEP_MEM must not be empty" >&2; exit 2; }

label_fields() {
  case "$1" in
    l2_A)  printf '%s\t%s\t%s\n' llama2_7b_adamw tofu checkpoints/llama2_7b_adamw ;;
    l3_A)  printf '%s\t%s\t%s\n' llama3_8b_adamw tofu checkpoints/llama3_8b_adamw ;;
    l2m_A) printf '%s\t%s\t%s\n' llama2_muse_adamw muse_news checkpoints/llama2_muse_adamw ;;
    l3m_A) printf '%s\t%s\t%s\n' llama3_muse_adamw muse_news checkpoints/llama3_muse_adamw ;;
    *) echo "Unknown label: $1" >&2; return 2 ;;
  esac
}

declare -a tasks=()

add_task() {
  local label="$1" algo="$2" seed="$3" quantizer="$4"
  local condition="$5" ranking="$6" k="$7"
  tasks+=("${label}|${algo}|${seed}|${quantizer}|${condition}|${ranking}|${k}")
}

case "$MATRIX_SCOPE" in
  pilot18)
    # Put all expensive GPTQ tasks first so they share one parallel wave.
    for label in l2_A l3_A l2m_A l3m_A; do
      add_task "$label" GA 42 gptq_int4 full_quantized none 0
    done
    add_task l2_A GA 42 gptq_int4 protected UPQI_signed 4
    add_task l2_A GA 42 gptq_int4 protected random 4

    for quantizer in nf4 rtn_int8; do
      for label in l2_A l3_A l2m_A l3m_A; do
        add_task "$label" GA 42 "$quantizer" full_quantized none 0
      done
    done
    add_task l2_A GA 42 nf4 protected UPQI_signed 4
    add_task l2_A GA 42 nf4 protected random 4
    add_task l2_A GA 42 nf4 protected none 0
    add_task l2_A GA 42 nf4 protected all_layers 224
    ;;
  fullq144)
    labels=(l2_A l3_A l2m_A l3m_A)
    algos=(GA GDR NPO KLR)
    seeds=(42 2024 5000)
    # GPTQ first, followed by the fast quantizers.
    for quantizer in gptq_int4 nf4 rtn_int8; do
      for label in "${labels[@]}"; do
        for algo in "${algos[@]}"; do
          for seed in "${seeds[@]}"; do
            add_task "$label" "$algo" "$seed" "$quantizer" full_quantized none 0
          done
        done
      done
    done
    ;;
  *)
    echo "MATRIX_SCOPE must be pilot18 or fullq144" >&2
    exit 2
    ;;
esac

echo "Gate 2 v4 parallel bridge matrix"
echo "  scope:       $MATRIX_SCOPE"
echo "  tasks:       ${#tasks[@]}"
echo "  max workers: $MAX_WORKERS"
echo "  step memory: $STEP_MEM"
echo "  results:     $RESULTS_ROOT"
echo "  source lock: $SOURCE_LOCK"
echo "  eval N:      forget=$N_SAMPLES retain=$N_RETAIN calibration=$N_CALIBRATION"

launch_task() {
  local task_index="$1" task="$2"
  local label algo seed quantizer condition ranking k
  local model_tag dataset ckpt_base slug log

  IFS='|' read -r label algo seed quantizer condition ranking k <<< "$task"
  IFS=$'\t' read -r model_tag dataset ckpt_base < <(label_fields "$label")

  slug="${condition}"
  if [[ "$condition" == protected ]]; then
    slug="protected_${ranking}_k${k}"
  fi
  log="logs/gate2_v4_bridge/${SLURM_JOB_ID}_task$(printf '%03d' "$task_index")_${model_tag}_${algo}_${seed}_${quantizer}_${slug}.log"

  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem="$STEP_MEM" \
    env \
      PYTHON="$PYTHON" \
      SOURCE_LOCK="$SOURCE_LOCK" \
      RESULTS_ROOT="$RESULTS_ROOT" \
      RANKINGS_PATH="$RANKINGS_PATH" \
      N_SAMPLES="$N_SAMPLES" \
      N_RETAIN="$N_RETAIN" \
      N_CALIBRATION="$N_CALIBRATION" \
      MANIFEST_N_SAMPLES="$MANIFEST_N_SAMPLES" \
      MANIFEST_N_RETAIN="$MANIFEST_N_RETAIN" \
      MANIFEST_N_CALIBRATION="$MANIFEST_N_CALIBRATION" \
      MODEL_TAG="$model_tag" \
      DATASET="$dataset" \
      CKPT_BASE="$ckpt_base" \
      SEEDS_OVERRIDE="$seed" \
      ALGOS_OVERRIDE="$algo" \
      QUANTIZERS_OVERRIDE="$quantizer" \
      CONDITIONS_OVERRIDE="$condition" \
      RANKING_SOURCES_OVERRIDE="$ranking" \
      K_VALUES_OVERRIDE="$k" \
      GPUS_PER_JOB=1 \
      MAX_PARALLEL=1 \
      DRY_RUN= \
      bash "$SLOTFILL" > "$log" 2>&1 &
}

pids=()
index=0
for task in "${tasks[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_WORKERS" ]]; do
    sleep 2
  done
  launch_task "$index" "$task"
  pids+=("$!")
  index=$((index + 1))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
