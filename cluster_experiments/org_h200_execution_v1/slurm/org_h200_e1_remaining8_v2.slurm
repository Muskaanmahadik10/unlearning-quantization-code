#!/bin/bash
# Submit seven instances with offsets 0,8,16,24,32,40,48 after stress16 passes.
#SBATCH --job-name=org_h200_e1_r8v2
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
#SBATCH --output=logs/org_h200/%x_o%j.out
#SBATCH --error=logs/org_h200/%x_o%j.err

set -u -o pipefail
cd /work/anon_user/hessian_unlearning || exit 2
mkdir -p logs/org_h200 results/org_h200_execution_v1/{parity,cache}

: "${TASK_OFFSET:?Submit with an approved remaining56 offset}"
case "$TASK_OFFSET" in
  0|8|16|24|32|40|48) ;;
  *) echo "Invalid TASK_OFFSET=$TASK_OFFSET" >&2; exit 2 ;;
esac

PYTHON=/home/anon_user/.conda/envs/unlearn/bin/python
WORKER=org_h200_execution_v1/scripts/run_org_h200_e1_cell_v1.py
CELLS=org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_cells_v2.tsv
MANIFEST=org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_manifest_v2.json
LOCK=org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V2.json
RESULTS=results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v2

test -x "$PYTHON" || exit 2
test -f "$LOCK" || exit 2
export PYTHONNOUSERSITE=1
export PYTHONPATH=gate2_v4_final/scripts:org_gb200_execution_v1/scripts:org_h200_execution_v1/scripts

PYTHONPATH=gate2_v4_final/scripts /usr/bin/python3 -c \
  "from gate2_v4_contract import load_source_lock,verify_source_lock_files; p='$LOCK'; x=load_source_lock(p); verify_source_lock_files(x); print('E1_V2_SOURCE_LOCK_PREVERIFY=PASS',x['source_lock_sha256'])" || exit 2

srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --distribution=block:block \
  --cpu-bind=cores --gpu-bind=single:1 \
  --output="logs/org_h200/${SLURM_JOB_ID}_e1_remaining8_o${TASK_OFFSET}_task_%t.out" \
  --error="logs/org_h200/${SLURM_JOB_ID}_e1_remaining8_o${TASK_OFFSET}_task_%t.err" \
  "$PYTHON" "$WORKER" \
    --cells "$CELLS" --manifest "$MANIFEST" --source_lock "$LOCK" \
    --results_root "$RESULTS" --wave remaining56 --task_offset "$TASK_OFFSET"
SRUN_RC=$?
echo "ORG_H200_E1_REMAINING8_V2_END OFFSET=$TASK_OFFSET SRUN_RC=$SRUN_RC"
exit "$SRUN_RC"

