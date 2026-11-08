#!/bin/bash
# Submit three instances with TASK_OFFSET=32, 40, and 48.
#SBATCH --job-name=org_h200_e1_r8
#SBATCH --account=anon_account
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=logs/org_h200/%x_%j.out
#SBATCH --error=logs/org_h200/%x_%j.err

set -euo pipefail
cd /work/anon_user/hessian_unlearning
mkdir -p logs/org_h200 results/org_h200_execution_v1/{parity,cache}

: "${TASK_OFFSET:?Submit with TASK_OFFSET=32, 40, or 48}"
case "$TASK_OFFSET" in
  32|40|48) ;;
  *) echo "Invalid TASK_OFFSET=$TASK_OFFSET" >&2; exit 2 ;;
esac

PYTHON=/home/anon_user/.conda/envs/unlearn/bin/python
WORKER=org_h200_execution_v1/scripts/run_org_h200_e1_cell_v1.py
CELLS=org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_cells_v2.tsv
MANIFEST=org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_manifest_v2.json
LOCK=org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V1.json
RESULTS=results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v1

test -x "$PYTHON"
test -f "$LOCK"
export PYTHONNOUSERSITE=1
export PYTHONPATH=gate2_v4_final/scripts:org_gb200_execution_v1/scripts:org_h200_execution_v1/scripts

PYTHONPATH=gate2_v4_final/scripts /usr/bin/python3 -c \
  "from gate2_v4_contract import load_source_lock,verify_source_lock_files; p='$LOCK'; x=load_source_lock(p); verify_source_lock_files(x); print('E1_SOURCE_LOCK_PREVERIFY=PASS',x['source_lock_sha256'])"

srun --kill-on-bad-exit=1 --cpu-bind=cores --gpu-bind=single:1 \
  --output="logs/org_h200/${SLURM_JOB_ID}_e1_remaining8_o${TASK_OFFSET}_task_%t.out" \
  --error="logs/org_h200/${SLURM_JOB_ID}_e1_remaining8_o${TASK_OFFSET}_task_%t.err" \
  "$PYTHON" "$WORKER" \
    --cells "$CELLS" --manifest "$MANIFEST" --source_lock "$LOCK" \
    --results_root "$RESULTS" --wave remaining56 --task_offset "$TASK_OFFSET"

echo "ORG_H200_E1_REMAINING8=PASS OFFSET=$TASK_OFFSET"

