#!/bin/bash
#SBATCH --job-name=org_wait0_v3
#SBATCH --account=anon_account
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/org_h200/%x_%j.out
#SBATCH --error=logs/org_h200/%x_%j.err

set -u -o pipefail
cd /work/anon_user/hessian_unlearning || exit 2
mkdir -p logs/org_h200 results/org_h200_execution_v1/parity

LOCK=org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V3.json
OUT=results/org_h200_execution_v1/parity/srun_wait0_canary_${SLURM_JOB_ID}
mkdir -p "$OUT"

PYTHONPATH=gate2_v4_final/scripts /usr/bin/python3 -c \
  "from gate2_v4_contract import load_source_lock,verify_source_lock_files; p='$LOCK'; x=load_source_lock(p); verify_source_lock_files(x); print('E1_V3_SOURCE_LOCK_PREVERIFY=PASS',x['source_lock_sha256'])" || exit 2

export OUT
srun --wait=0 --nodes=1 --ntasks=8 --ntasks-per-node=8 --distribution=block:block \
  --cpu-bind=cores --gpu-bind=single:1 \
  bash -lc 'test -n "${CUDA_VISIBLE_DEVICES:-}" || exit 3; delay=$((2 + SLURM_PROCID)); sleep "$delay"; printf "TASK=%s DELAY=%s GPU=%s PASS\n" "$SLURM_PROCID" "$delay" "$CUDA_VISIBLE_DEVICES" > "$OUT/task_${SLURM_PROCID}.txt"'
RC=$?
COUNT=$(find "$OUT" -maxdepth 1 -type f -name 'task_*.txt' | wc -l)
cat "$OUT"/task_*.txt 2>/dev/null || true
if [ "$RC" -ne 0 ] || [ "$COUNT" -ne 8 ]; then
  echo "ORG_H200_SRUN_WAIT0_CANARY_V3=FAIL RC=$RC COUNT=$COUNT" >&2
  exit 1
fi
echo "ORG_H200_SRUN_WAIT0_CANARY_V3=PASS COUNT=8"

