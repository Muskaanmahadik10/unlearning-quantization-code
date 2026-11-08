#!/bin/bash
# One-cell launcher for exact GQDP and the Gate 2 v4.2 metric suite.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_metrics

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK is required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_standard_metrics_v42}"
GT_ROOT="${GT_ROOT:-results/gate2_v4_standard}"
UPQI_ROOT="${UPQI_ROOT:-results/gate2_v4_standard}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests/gate2_v4}"
MANIFEST_SEED="${MANIFEST_SEED:-20260625}"
DATA_DIR="${DATA_DIR:-data}"
N_SAMPLES="${N_SAMPLES:-128}"
N_RETAIN="${N_RETAIN:-128}"
N_CALIBRATION="${N_CALIBRATION:-128}"
FISHER_FORGET_SAMPLES="${FISHER_FORGET_SAMPLES:-0}"
FISHER_RETAIN_SAMPLES="${FISHER_RETAIN_SAMPLES:-0}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DTYPE="${DTYPE:-bfloat16}"

: "${MODEL_TAG:?MODEL_TAG required}"
: "${DATASET:?DATASET required}"
: "${CKPT_BASE:?CKPT_BASE required}"
: "${ALGO:?ALGO required}"
: "${SEED:?SEED required}"
: "${QUANTIZER:?QUANTIZER required}"

SCRIPT=gate2_v4_final/scripts/run_gate2_metric_suite_v42.py
SCRIPT_SHA="$($PYTHON -c "import json; print(json.load(open('$SOURCE_LOCK'))['files']['$SCRIPT'])")"
MODEL_DIR="${CKPT_BASE}/unlearned/${ALGO}/seed_${SEED}"
[[ -d "$MODEL_DIR" ]] || MODEL_DIR="${CKPT_BASE}/${ALGO}/seed_${SEED}"
[[ -d "$MODEL_DIR" ]] || { echo "Missing checkpoint: $MODEL_DIR" >&2; exit 2; }

CELL="${MODEL_TAG}_${ALGO}_seed${SEED}/${QUANTIZER}"
OUT_DIR="${RESULTS_ROOT}/${CELL}"
GT_PATH="${GT_ROOT}/${CELL}/ground_truth_v4.json"
UPQI_PATH="${UPQI_ROOT}/${CELL}/upqi_v4.json"
FORGET_MANIFEST="${MANIFEST_DIR}/${DATASET}_forget_core${N_SAMPLES}_seed${MANIFEST_SEED}.json"
RETAIN_MANIFEST="${MANIFEST_DIR}/${DATASET}_retain_gptq_calib${N_RETAIN}_seed${MANIFEST_SEED}.json"
CALIB_MANIFEST="${MANIFEST_DIR}/${DATASET}_retain_gptq_calib${N_CALIBRATION}_seed${MANIFEST_SEED}.json"
ORIGINAL_WEIGHTS="${CKPT_BASE}/final/original_weights.pt"

for path in "$GT_PATH" "$UPQI_PATH" "$FORGET_MANIFEST" "$RETAIN_MANIFEST" "$ORIGINAL_WEIGHTS"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done

EXTRA_CALIB=()
if [[ "$QUANTIZER" == gptq_int4 ]]; then
  [[ -f "$CALIB_MANIFEST" ]] || { echo "Missing calibration manifest: $CALIB_MANIFEST" >&2; exit 2; }
  EXTRA_CALIB=(--calibration_manifest_path "$CALIB_MANIFEST")
fi
EXTRA_DRY=()
[[ "${DRY_RUN:-0}" == 1 ]] && EXTRA_DRY=(--dry_run)
EXTRA_MAX_LAYERS=()
[[ -n "${MAX_LAYERS:-}" ]] && EXTRA_MAX_LAYERS=(--max_layers "$MAX_LAYERS")

echo "Metric suite v4.2: $CELL fisher=${FISHER_FORGET_SAMPLES}/${FISHER_RETAIN_SAMPLES}"
"$PYTHON" "$SCRIPT" \
  --model_dir "$MODEL_DIR" \
  --model_tag "$MODEL_TAG" \
  --algo "$ALGO" \
  --seed "$SEED" \
  --dataset "$DATASET" \
  --quantizer "$QUANTIZER" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --gt_path "$GT_PATH" \
  --upqi_path "$UPQI_PATH" \
  --manifest_path "$FORGET_MANIFEST" \
  --retain_manifest_path "$RETAIN_MANIFEST" \
  "${EXTRA_CALIB[@]}" \
  --original_weights "$ORIGINAL_WEIGHTS" \
  --source_lock_path "$SOURCE_LOCK" \
  --expected_script_sha256 "$SCRIPT_SHA" \
  --n_samples "$N_SAMPLES" \
  --n_retain "$N_RETAIN" \
  --n_calibration "$N_CALIBRATION" \
  --fisher_forget_samples "$FISHER_FORGET_SAMPLES" \
  --fisher_retain_samples "$FISHER_RETAIN_SAMPLES" \
  --batch_size "$METRIC_BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --dtype "$DTYPE" \
  "${EXTRA_MAX_LAYERS[@]}" \
  "${EXTRA_DRY[@]}"
