#!/bin/bash
#SBATCH --job-name=upqi_v4_muse64
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH -o logs/gate2_v4/upqi_v4_muse64_%j.out
#SBATCH -e logs/gate2_v4/upqi_v4_muse64_%j.err

set -euo pipefail
mkdir -p logs/gate2_v4

export GPUS_PER_JOB=8
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"

shards=(
  "l2m_A|42-2024-5000|GA GDR|nf4 gptq_int4 rtn_int8"
  "l3m_A|42-2024-5000|GA GDR|nf4 gptq_int4 rtn_int8"
  "l2m_A|42-2024-5000|NPO KLR|nf4 gptq_int4 rtn_int8"
  "l3m_A|42-2024-5000|NPO KLR|nf4 gptq_int4 rtn_int8"
)

pids=()
for shard in "${shards[@]}"; do
  IFS='|' read -r LABEL SEEDS ALGOS QUANTS <<< "$shard"
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
    env SUBMIT_MODE=internal STAGE=upqi LABEL="$LABEL" SEEDS_OVERRIDE="$SEEDS" ALGOS_OVERRIDE="$ALGOS" QUANTIZERS_OVERRIDE="$QUANTS" MAX_PARALLEL=8 \
    bash gate2_v4_final/slurm/gate2_v4/run_internal_slotfill.sh \
    > "logs/gate2_v4/${SLURM_JOB_ID}_${LABEL}_${ALGOS// /_}_upqi_muse64_bundle.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
