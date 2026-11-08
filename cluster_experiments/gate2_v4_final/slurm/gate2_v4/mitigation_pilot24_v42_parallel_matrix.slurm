#!/bin/bash
# Preregistered 24-cell, four-selector, k={4,8} mitigation matrix.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_bridge

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK required}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:?SELECTION_MANIFEST required}"
RANKINGS_PATH="${RANKINGS_PATH:?RANKINGS_PATH required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard_v42_mitigation_pilot24}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_GPUS:-1}}"
STEP_MEM="${STEP_MEM:-200G}"
SLOTFILL=gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm
N_SAMPLES="${N_SAMPLES:-128}"
N_RETAIN="${N_RETAIN:-128}"
N_CALIBRATION="${N_CALIBRATION:-128}"

mapfile -t cells < <(
  "$PYTHON" - "$SELECTION_MANIFEST" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("decision") != "PILOT24_SELECTION_READY_FOR_UPQI_STAR_SELECTED24":
    raise SystemExit("Pilot24 selection manifest is not accepted")
cells = data.get("cells", [])
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
[[ ${#cells[@]} -eq 24 ]] || exit 2

selectors=(GQDP_signed_v42 UPQI_star_signed_v42 displacement_l2_v42 deterministic_random_v42)
k_values=(4 8)
tasks=()
for cell in "${cells[@]}"; do
  for selector in "${selectors[@]}"; do
    for k in "${k_values[@]}"; do tasks+=("${cell}|${selector}|${k}"); done
  done
done
[[ ${#tasks[@]} -eq 192 ]] || exit 2

"$PYTHON" - "$RANKINGS_PATH" "$SELECTION_MANIFEST" <<'PY'
import json
import sys
from collections import defaultdict

ranking_path, manifest_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
cells = {row["cell_id"] for row in manifest["cells"]}
required = {"GQDP_signed_v42", "UPQI_star_signed_v42", "displacement_l2_v42", "deterministic_random_v42"}
layers = defaultdict(set)
with open(ranking_path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            if row.get("cell_id") in cells and row.get("ranking_source") in required:
                layers[(row["cell_id"], row["ranking_source"])].add(row["layer_name"])
bad = [(cell, source, len(layers[(cell, source)])) for cell in cells for source in required if len(layers[(cell, source)]) != 224]
if bad:
    raise SystemExit(f"Ranking coverage gate failed: {bad[:8]}")
print("MITIGATION_RANKING_COVERAGE=PASS cells=24 selectors=4 layers=224")
PY

echo "Mitigation Pilot24 tasks=192 workers=$MAX_WORKERS results=$RESULTS_ROOT"
launch() {
  local index="$1" task="$2" model dataset ckpt algo seed quantizer panel selector k log
  IFS='|' read -r model dataset ckpt algo seed quantizer panel selector k <<< "$task"
  log="logs/gate2_v4_bridge/${SLURM_JOB_ID}_mit24_task$(printf '%03d' "$index")_${model}_${algo}_${seed}_${quantizer}_${selector}_k${k}.log"
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem="$STEP_MEM" \
    env PYTHON="$PYTHON" SOURCE_LOCK="$SOURCE_LOCK" RESULTS_ROOT="$RESULTS_ROOT" RANKINGS_PATH="$RANKINGS_PATH" \
      N_SAMPLES="$N_SAMPLES" N_RETAIN="$N_RETAIN" N_CALIBRATION="$N_CALIBRATION" \
      MANIFEST_N_SAMPLES="$N_SAMPLES" MANIFEST_N_RETAIN="$N_RETAIN" MANIFEST_N_CALIBRATION="$N_CALIBRATION" \
      MODEL_TAG="$model" DATASET="$dataset" CKPT_BASE="$ckpt" SEEDS_OVERRIDE="$seed" ALGOS_OVERRIDE="$algo" \
      QUANTIZERS_OVERRIDE="$quantizer" CONDITIONS_OVERRIDE=protected RANKING_SOURCES_OVERRIDE="$selector" \
      K_VALUES_OVERRIDE="$k" GPUS_PER_JOB=1 MAX_PARALLEL=1 DRY_RUN= \
      bash "$SLOTFILL" >"$log" 2>&1 &
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
