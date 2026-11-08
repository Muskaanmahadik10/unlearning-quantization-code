#!/bin/bash
# One-cell corrected UPQI-star launcher.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_upqi_star

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_standard_upqi_star_v42}"
GT_ROOT="${GT_ROOT:-results/gate2_v4_standard}"
UPQI_ROOT="${UPQI_ROOT:-results/gate2_v4_standard}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests/gate2_v4}"
MANIFEST_SEED="${MANIFEST_SEED:-20260625}"
N_SAMPLES="${N_SAMPLES:-64}"
N_CALIBRATION="${N_CALIBRATION:-128}"
N_QUADRATURE="${N_QUADRATURE:-8}"

: "${MODEL_TAG:?}" "${DATASET:?}" "${CKPT_BASE:?}" "${ALGO:?}" "${SEED:?}" "${QUANTIZER:?}"
SCRIPT=gate2_v4_final/scripts/run_gate2_upqi_star_v42.py
SCRIPT_SHA="$($PYTHON -c "import json; print(json.load(open('$SOURCE_LOCK'))['files']['$SCRIPT'])")"
MODEL_DIR="${CKPT_BASE}/unlearned/${ALGO}/seed_${SEED}"
[[ -d "$MODEL_DIR" ]] || MODEL_DIR="${CKPT_BASE}/${ALGO}/seed_${SEED}"
CELL="${MODEL_TAG}_${ALGO}_seed${SEED}/${QUANTIZER}"
OUT_DIR="${RESULTS_ROOT}/K${N_QUADRATURE}_N${N_SAMPLES}/${CELL}"
GT_PATH="${GT_ROOT}/${CELL}/ground_truth_v4.json"
UPQI_PATH="${UPQI_ROOT}/${CELL}/upqi_v4.json"
FORGET="${MANIFEST_DIR}/${DATASET}_forget_core128_seed${MANIFEST_SEED}.json"
CALIB="${MANIFEST_DIR}/${DATASET}_retain_gptq_calib${N_CALIBRATION}_seed${MANIFEST_SEED}.json"
EXTRA_CALIB=()
[[ "$QUANTIZER" == gptq_int4 ]] && EXTRA_CALIB=(--calibration_manifest_path "$CALIB")
EXTRA_DRY=()
[[ "${DRY_RUN:-0}" == 1 ]] && EXTRA_DRY=(--dry_run)
EXTRA_MAX_LAYERS=()
[[ -n "${MAX_LAYERS:-}" ]] && EXTRA_MAX_LAYERS=(--max_layers "$MAX_LAYERS")
EXTRA_STOP=()
[[ -n "${STOP_AFTER_LAYERS:-}" ]] && EXTRA_STOP=(--stop_after_layers "$STOP_AFTER_LAYERS")

"$PYTHON" "$SCRIPT" \
  --model_dir "$MODEL_DIR" --model_tag "$MODEL_TAG" --algo "$ALGO" --seed "$SEED" \
  --dataset "$DATASET" --quantizer "$QUANTIZER" --output_dir "$OUT_DIR" \
  --gt_path "$GT_PATH" --legacy_upqi_path "$UPQI_PATH" --manifest_path "$FORGET" \
  "${EXTRA_CALIB[@]}" --source_lock_path "$SOURCE_LOCK" --expected_script_sha256 "$SCRIPT_SHA" \
  --n_samples "$N_SAMPLES" --n_calibration "$N_CALIBRATION" --n_quadrature "$N_QUADRATURE" \
  --batch_size "${UPQI_BATCH_SIZE:-1}" --max_length "${MAX_LENGTH:-512}" --dtype "${DTYPE:-bfloat16}" \
  "${EXTRA_MAX_LAYERS[@]}" "${EXTRA_STOP[@]}" \
  "${EXTRA_DRY[@]}"
