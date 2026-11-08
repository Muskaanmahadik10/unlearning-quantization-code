#!/bin/bash
# Run corrected UPQI-star only for cells preregistered by Pilot24 selection.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_upqi_star

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK required}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:?SELECTION_MANIFEST required}"
N_QUADRATURE="${N_QUADRATURE:?N_QUADRATURE required from convergence gate}"
N_SAMPLES="${N_SAMPLES:?N_SAMPLES required from convergence gate}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_GPUS:-1}}"
STEP_MEM="${STEP_MEM:-200G}"
SLOTFILL=gate2_v4_final/slurm/gate2_v4/upqi_star_v42_slotfill.slurm

mapfile -t tasks < <(
  "$PYTHON" - "$SELECTION_MANIFEST" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
cells = data.get("cells", [])
if data.get("decision") != "PILOT24_SELECTION_READY_FOR_UPQI_STAR_SELECTED24":
    raise SystemExit("Pilot24 selection manifest is not accepted")
if len(cells) != 24 or len({row["cell_id"] for row in cells}) != 24:
    raise SystemExit("Pilot24 selection must contain 24 unique cells")
checkpoint = {
    "llama2_7b_adamw": "checkpoints/llama2_7b_adamw",
    "llama3_8b_adamw": "checkpoints/llama3_8b_adamw",
    "llama2_muse_adamw": "checkpoints/llama2_muse_adamw",
    "llama3_muse_adamw": "checkpoints/llama3_muse_adamw",
}
for row in cells:
    print("|".join([
        row["model_tag"], row["dataset"], checkpoint[row["model_tag"]],
        row["algo"], str(row["seed"]), row["quantizer"], row["panel"],
    ]))
PY
)

[[ ${#tasks[@]} -eq 24 ]] || { echo "Expected 24 selected UPQI-star tasks" >&2; exit 2; }
echo "UPQI-star selected24 tasks=24 K=$N_QUADRATURE N=$N_SAMPLES workers=$MAX_WORKERS"

launch() {
  local index="$1" task="$2" model dataset ckpt algo seed quantizer panel log
  IFS='|' read -r model dataset ckpt algo seed quantizer panel <<< "$task"
  log="logs/gate2_v4_upqi_star/${SLURM_JOB_ID}_selected24_task$(printf '%03d' "$index")_${model}_${algo}_${seed}_${quantizer}.log"
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem="$STEP_MEM" \
    env SOURCE_LOCK="$SOURCE_LOCK" MODEL_TAG="$model" DATASET="$dataset" CKPT_BASE="$ckpt" \
      ALGO="$algo" SEED="$seed" QUANTIZER="$quantizer" N_QUADRATURE="$N_QUADRATURE" N_SAMPLES="$N_SAMPLES" \
      RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_standard_upqi_star_v42_selected24}" \
      PYTHON="$PYTHON" bash "$SLOTFILL" >"$log" 2>&1 &
}

pids=(); index=0
for task in "${tasks[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_WORKERS" ]]; do sleep 2; done
  launch "$index" "$task"
  pids+=("$!")
  index=$((index + 1))
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
