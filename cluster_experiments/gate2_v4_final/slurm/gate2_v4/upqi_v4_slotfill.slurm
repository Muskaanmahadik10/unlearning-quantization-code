#!/bin/bash
#SBATCH --job-name=upqi_v4_slotfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH -o logs/gate2_v4/upqi_v4_slotfill_%j.out
#SBATCH -e logs/gate2_v4/upqi_v4_slotfill_%j.err

set -euo pipefail
mkdir -p logs/gate2_v4

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_standard}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests/gate2_v4}"
SOURCE_LOCK="${SOURCE_LOCK:-gate2_v4_final/V4_SOURCE_LOCK.json}"
MANIFEST_SEED="${MANIFEST_SEED:-20260625}"
DATA_DIR="${DATA_DIR:-data}"
MAX_LENGTH="${MAX_LENGTH:-512}"
N_SAMPLES="${N_SAMPLES:-128}"
N_CALIBRATION="${N_CALIBRATION:-128}"
N_QUADRATURE="${N_QUADRATURE:-8}"
UPQI_BATCH_SIZE="${UPQI_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

: "${MODEL_TAG:?MODEL_TAG required}"
: "${DATASET:?DATASET required}"
: "${CKPT_BASE:?CKPT_BASE required}"
: "${SEEDS_OVERRIDE:?SEEDS_OVERRIDE required}"
: "${ALGOS_OVERRIDE:?ALGOS_OVERRIDE required}"
: "${QUANTIZERS_OVERRIDE:?QUANTIZERS_OVERRIDE required}"

UPQI_SCRIPT="gate2_v4_final/scripts/run_gate2_upqi_v4.py"
UPQI_SHA="$("$PYTHON" -c "import json; j=json.load(open('$SOURCE_LOCK')); print(j['files']['gate2_v4_final/scripts/run_gate2_upqi_v4.py'])")"
FORGET_MANIFEST="${MANIFEST_DIR}/${DATASET}_forget_core${N_SAMPLES}_seed${MANIFEST_SEED}.json"
CALIB_MANIFEST="${MANIFEST_DIR}/${DATASET}_retain_gptq_calib${N_CALIBRATION}_seed${MANIFEST_SEED}.json"

resolve_model_dir() {
  local algo="$1"
  local seed="$2"
  local candidates=(
    "${MODEL_DIR_OVERRIDE:-}"
    "${CKPT_BASE}/unlearned/${algo}/seed_${seed}"
    "${CKPT_BASE}/${algo}/seed_${seed}"
    "${CKPT_BASE}/${algo}_seed${seed}"
    "${CKPT_BASE}/${algo}_seed_${seed}"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "No checkpoint directory found for CKPT_BASE=${CKPT_BASE} algo=${algo} seed=${seed}" >&2
  printf '  checked: %s\n' "${candidates[@]}" >&2
  return 2
}

pids=()
launch_index=0
GPU_COUNT="${GPUS_PER_JOB:-${SLURM_GPUS_ON_NODE:-1}}"
for ALGO in ${ALGOS_OVERRIDE}; do
  for SEED in ${SEEDS_OVERRIDE//-/ }; do
    for QUANTIZER in ${QUANTIZERS_OVERRIDE}; do
      while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL:-1}" ]; do sleep 5; done
      GPU_SLOT=$((launch_index % GPU_COUNT))
      launch_index=$((launch_index + 1))
      MODEL_DIR="$(resolve_model_dir "$ALGO" "$SEED")"
      OUT_DIR="${RESULTS_ROOT}/${MODEL_TAG}_${ALGO}_seed${SEED}/${QUANTIZER}"
      LOG="logs/gate2_v4/${SLURM_JOB_ID}_${MODEL_TAG}_${ALGO}_${SEED}_${QUANTIZER}_upqi_v4.log"
      EXTRA_CALIB=()
      if [[ "$QUANTIZER" == "gptq_int4" ]]; then
        EXTRA_CALIB=(--calibration_manifest_path "$CALIB_MANIFEST")
      fi
      DRY_RUN_ARGS=()
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        DRY_RUN_ARGS=(--dry_run)
      fi
      echo "Launching UPQI v4 ${MODEL_TAG} ${ALGO} ${SEED} ${QUANTIZER} gpu=${GPU_SLOT} -> ${OUT_DIR}"
      CUDA_VISIBLE_DEVICES="$GPU_SLOT" "$PYTHON" "$UPQI_SCRIPT" \
        --model_dir "$MODEL_DIR" \
        --model_tag "$MODEL_TAG" \
        --algo "$ALGO" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --data_dir "$DATA_DIR" \
        --quantizer "$QUANTIZER" \
        --output_dir "$OUT_DIR" \
        --manifest_path "$FORGET_MANIFEST" \
        "${EXTRA_CALIB[@]}" \
        --source_lock_path "$SOURCE_LOCK" \
        --expected_script_sha256 "$UPQI_SHA" \
        --n_samples "$N_SAMPLES" \
        --n_calibration "$N_CALIBRATION" \
        --n_quadrature "$N_QUADRATURE" \
        --batch_size "$UPQI_BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --dtype "$DTYPE" \
        ${MAX_LAYERS:+--max_layers "$MAX_LAYERS"} \
        ${STOP_AFTER_LAYERS:+--stop_after_layers "$STOP_AFTER_LAYERS"} \
        "${DRY_RUN_ARGS[@]}" \
        > "$LOG" 2>&1 &
      pids+=("$!")
    done
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
