#!/bin/bash
#SBATCH --job-name=org_h200_e1_s16
#SBATCH --account=anon_account
#SBATCH --partition=16gpus
#SBATCH --nodes=2
#SBATCH --ntasks=16
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

PYTHON=/home/anon_user/.conda/envs/unlearn/bin/python
WORKER=org_h200_execution_v1/scripts/run_org_h200_e1_cell_v1.py
VALIDATOR=org_h200_execution_v1/scripts/validate_org_h200_e1_wave_v1.py
CELLS=org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_cells_v2.tsv
MANIFEST=org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_manifest_v2.json
LOCK=org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V1.json
RESULTS=results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v1
VALIDATION=results/org_h200_execution_v1/parity/h200_e1_stress16_validation_${SLURM_JOB_ID}.json

test -x "$PYTHON"
test -f "$LOCK"
export PYTHONNOUSERSITE=1
export PYTHONPATH=gate2_v4_final/scripts:org_gb200_execution_v1/scripts:org_h200_execution_v1/scripts

sha256sum "$0" | tee "logs/org_h200/${SLURM_JOB_ID}_e1_stress16_launcher.sha256"
PYTHONPATH=gate2_v4_final/scripts /usr/bin/python3 -c \
  "from gate2_v4_contract import load_source_lock,verify_source_lock_files; p='$LOCK'; x=load_source_lock(p); verify_source_lock_files(x); print('E1_SOURCE_LOCK_PREVERIFY=PASS',x['source_lock_sha256'])"

srun --kill-on-bad-exit=1 --cpu-bind=cores --gpu-bind=single:1 \
  --output="logs/org_h200/${SLURM_JOB_ID}_e1_stress16_task_%t.out" \
  --error="logs/org_h200/${SLURM_JOB_ID}_e1_stress16_task_%t.err" \
  "$PYTHON" "$WORKER" \
    --cells "$CELLS" --manifest "$MANIFEST" --source_lock "$LOCK" \
    --results_root "$RESULTS" --wave stress16

/usr/bin/python3 "$VALIDATOR" \
  --cells "$CELLS" --manifest "$MANIFEST" --source_lock "$LOCK" \
  --results_root "$RESULTS" --wave stress16 --output "$VALIDATION"

find "$RESULTS" -type f \( -name bridge_v4.json -o -path '*/_receipts/stress16/*.json' \) -print0 \
  | sort -z | xargs -0 sha256sum > "results/org_h200_execution_v1/parity/h200_e1_stress16_result_sha256_${SLURM_JOB_ID}.txt"
echo ORG_H200_E1_STRESS16=PASS
echo VALIDATION="$VALIDATION"

