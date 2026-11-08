#!/bin/bash
#SBATCH --job-name=bridge_v4_slotfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH -o logs/gate2_v4_bridge/bridge_v4_slotfill_%j.out
#SBATCH -e logs/gate2_v4_bridge/bridge_v4_slotfill_%j.err

set -euo pipefail
mkdir -p logs/gate2_v4_bridge

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests/gate2_v4}"
SOURCE_LOCK="${SOURCE_LOCK:-gate2_v4_final/V4_SOURCE_LOCK.json}"
MANIFEST_SEED="${MANIFEST_SEED:-20260625}"
DATA_DIR="${DATA_DIR:-data}"
MAX_LENGTH="${MAX_LENGTH:-512}"
N_SAMPLES="${N_SAMPLES:-128}"
N_RETAIN="${N_RETAIN:-128}"
N_CALIBRATION="${N_CALIBRATION:-128}"
MANIFEST_N_SAMPLES="${MANIFEST_N_SAMPLES:-128}"
MANIFEST_N_RETAIN="${MANIFEST_N_RETAIN:-128}"
MANIFEST_N_CALIBRATION="${MANIFEST_N_CALIBRATION:-128}"
DTYPE="${DTYPE:-bfloat16}"
RANKINGS_PATH="${RANKINGS_PATH:-results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl}"
CONDITIONS_OVERRIDE="${CONDITIONS_OVERRIDE:-bf16 full_quantized}"
RANKING_SOURCES_OVERRIDE="${RANKING_SOURCES_OVERRIDE:-UPQI_abs random}"
K_VALUES_OVERRIDE="${K_VALUES_OVERRIDE:-1 2 4 8 16}"

: "${MODEL_TAG:?MODEL_TAG required}"
: "${DATASET:?DATASET required}"
: "${CKPT_BASE:?CKPT_BASE required}"
: "${SEEDS_OVERRIDE:?SEEDS_OVERRIDE required}"
: "${ALGOS_OVERRIDE:?ALGOS_OVERRIDE required}"
: "${QUANTIZERS_OVERRIDE:?QUANTIZERS_OVERRIDE required}"

BRIDGE_SCRIPT="gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py"
BRIDGE_SHA="$("$PYTHON" -c "import json; j=json.load(open('$SOURCE_LOCK')); print(j['files']['gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py'])")"
FORGET_MANIFEST="${MANIFEST_DIR}/${DATASET}_forget_core${MANIFEST_N_SAMPLES}_seed${MANIFEST_SEED}.json"
RETAIN_MANIFEST="${MANIFEST_DIR}/${DATASET}_retain_gptq_calib${MANIFEST_N_RETAIN}_seed${MANIFEST_SEED}.json"
CALIB_MANIFEST="${MANIFEST_DIR}/${DATASET}_retain_gptq_calib${MANIFEST_N_CALIBRATION}_seed${MANIFEST_SEED}.json"

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

launch_bridge() {
  local algo="$1"
  local seed="$2"
  local quantizer="$3"
  local condition="$4"
  local ranking_source="$5"
  local k_value="$6"
  local gpu_slot="$7"
  local model_dir="$8"
  local log_suffix="${condition}"
  local extra_rank=()
  local extra_calib=()
  if [[ "$condition" == "protected" ]]; then
    log_suffix="protected_${ranking_source}_k${k_value}"
    extra_rank=(--ranking_source "$ranking_source" --k "$k_value" --rankings_path "$RANKINGS_PATH")
  fi
  if [[ "$quantizer" == "gptq_int4" ]]; then
    extra_calib=(--calibration_manifest_path "$CALIB_MANIFEST")
  fi
  local log="logs/gate2_v4_bridge/${SLURM_JOB_ID}_${MODEL_TAG}_${algo}_${seed}_${quantizer}_${log_suffix}_bridge_v4.log"
  echo "Launching bridge v4 ${MODEL_TAG} ${algo} ${seed} ${quantizer} ${log_suffix} gpu=${gpu_slot}"
  CUDA_VISIBLE_DEVICES="$gpu_slot" "$PYTHON" "$BRIDGE_SCRIPT" \
    --model_dir "$model_dir" \
    --model_tag "$MODEL_TAG" \
    --algo "$algo" \
    --seed "$seed" \
    --dataset "$DATASET" \
    --quantizer "$quantizer" \
    --condition "$condition" \
    "${extra_rank[@]}" \
    --manifest_path "$FORGET_MANIFEST" \
    --retain_manifest_path "$RETAIN_MANIFEST" \
    "${extra_calib[@]}" \
    --results_root "$RESULTS_ROOT" \
    --source_lock_path "$SOURCE_LOCK" \
    --expected_script_sha256 "$BRIDGE_SHA" \
    --n_samples "$N_SAMPLES" \
    --n_retain "$N_RETAIN" \
    --n_calibration "$N_CALIBRATION" \
    --manifest_n_samples "$MANIFEST_N_SAMPLES" \
    --manifest_n_retain "$MANIFEST_N_RETAIN" \
    --manifest_n_calibration "$MANIFEST_N_CALIBRATION" \
    --max_length "$MAX_LENGTH" \
    --dtype "$DTYPE" \
    ${DRY_RUN:+--dry_run} \
    > "$log" 2>&1 &
}

pids=()
launch_index=0
GPU_COUNT="${GPUS_PER_JOB:-${SLURM_GPUS_ON_NODE:-1}}"
for ALGO in ${ALGOS_OVERRIDE}; do
  for SEED in ${SEEDS_OVERRIDE//-/ }; do
    for QUANTIZER in ${QUANTIZERS_OVERRIDE}; do
      MODEL_DIR="$(resolve_model_dir "$ALGO" "$SEED")"
      for CONDITION in ${CONDITIONS_OVERRIDE}; do
        if [[ "$CONDITION" == "protected" ]]; then
          for RANKING_SOURCE in ${RANKING_SOURCES_OVERRIDE}; do
            for K_VALUE in ${K_VALUES_OVERRIDE}; do
              while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL:-1}" ]; do sleep 5; done
              GPU_SLOT=$((launch_index % GPU_COUNT))
              launch_index=$((launch_index + 1))
              launch_bridge "$ALGO" "$SEED" "$QUANTIZER" "$CONDITION" "$RANKING_SOURCE" "$K_VALUE" "$GPU_SLOT" "$MODEL_DIR"
              pids+=("$!")
            done
          done
        else
          while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL:-1}" ]; do sleep 5; done
          GPU_SLOT=$((launch_index % GPU_COUNT))
          launch_index=$((launch_index + 1))
          launch_bridge "$ALGO" "$SEED" "$QUANTIZER" "$CONDITION" "" "0" "$GPU_SLOT" "$MODEL_DIR"
          pids+=("$!")
        fi
      done
    done
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
