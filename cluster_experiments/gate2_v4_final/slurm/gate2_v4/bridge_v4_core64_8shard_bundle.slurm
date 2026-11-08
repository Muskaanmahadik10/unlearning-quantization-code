#!/bin/bash
#SBATCH --job-name=bridge_v4_core64
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH -o logs/gate2_v4_bridge/bridge_v4_core64_%j.out
#SBATCH -e logs/gate2_v4_bridge/bridge_v4_core64_%j.err

set -euo pipefail
mkdir -p logs/gate2_v4_bridge

export GPUS_PER_JOB=8
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
export SOURCE_LOCK="${SOURCE_LOCK:-gate2_v4_final/V4_SOURCE_LOCK.json}"
export CONDITIONS_OVERRIDE="${CONDITIONS_OVERRIDE:-bf16 full_quantized}"

shards=(
  "l2_A|42-2024-5000|GA GDR|nf4 gptq_int4 rtn_int8"
  "l2_A|42-2024-5000|NPO KLR|nf4 gptq_int4 rtn_int8"
  "l3_A|42-2024-5000|GA GDR|nf4 gptq_int4 rtn_int8"
  "l3_A|42-2024-5000|NPO KLR|nf4 gptq_int4 rtn_int8"
  "l2m_A|42-2024-5000|GA GDR|nf4 gptq_int4 rtn_int8"
  "l2m_A|42-2024-5000|NPO KLR|nf4 gptq_int4 rtn_int8"
  "l3m_A|42-2024-5000|GA GDR|nf4 gptq_int4 rtn_int8"
  "l3m_A|42-2024-5000|NPO KLR|nf4 gptq_int4 rtn_int8"
)

label_to_env() {
  local label="$1"
  case "$label" in
    l2_A)  echo "MODEL_TAG=llama2_7b_adamw DATASET=tofu CKPT_BASE=checkpoints/llama2_7b_adamw" ;;
    l3_A)  echo "MODEL_TAG=llama3_8b_adamw DATASET=tofu CKPT_BASE=checkpoints/llama3_8b_adamw" ;;
    l2m_A) echo "MODEL_TAG=llama2_muse_adamw DATASET=muse_news CKPT_BASE=checkpoints/llama2_muse_adamw" ;;
    l3m_A) echo "MODEL_TAG=llama3_muse_adamw DATASET=muse_news CKPT_BASE=checkpoints/llama3_muse_adamw" ;;
    *) echo "Unknown label $label" >&2; return 2 ;;
  esac
}

pids=()
for shard in "${shards[@]}"; do
  IFS='|' read -r LABEL SEEDS ALGOS QUANTS <<< "$shard"
  ENV_BITS="$(label_to_env "$LABEL")"
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
    env $ENV_BITS SOURCE_LOCK="$SOURCE_LOCK" SEEDS_OVERRIDE="$SEEDS" ALGOS_OVERRIDE="$ALGOS" QUANTIZERS_OVERRIDE="$QUANTS" MAX_PARALLEL=8 \
    bash gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm \
    > "logs/gate2_v4_bridge/${SLURM_JOB_ID}_${LABEL}_${ALGOS// /_}_bridge_v4_core64_bundle.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
