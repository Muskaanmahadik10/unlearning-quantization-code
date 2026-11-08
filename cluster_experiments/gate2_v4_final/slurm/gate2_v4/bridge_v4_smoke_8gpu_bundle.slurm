#!/bin/bash
#SBATCH --job-name=bridge_v4_smoke8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH -o logs/gate2_v4_bridge/bridge_v4_smoke8_%j.out
#SBATCH -e logs/gate2_v4_bridge/bridge_v4_smoke8_%j.err

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
mkdir -p logs/gate2_v4_bridge

export PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
export SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK must be the timestamped bridge lock}"
export RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard_smoke}"
export RANKINGS_PATH="${RANKINGS_PATH:-results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl}"
export N_SAMPLES="${N_SAMPLES:-4}"
export N_RETAIN="${N_RETAIN:-4}"
export N_CALIBRATION="${N_CALIBRATION:-4}"
export MANIFEST_N_SAMPLES="${MANIFEST_N_SAMPLES:-128}"
export MANIFEST_N_RETAIN="${MANIFEST_N_RETAIN:-128}"
export MANIFEST_N_CALIBRATION="${MANIFEST_N_CALIBRATION:-128}"
export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-42}"
export ALGOS_OVERRIDE="${ALGOS_OVERRIDE:-GA}"
export GPUS_PER_JOB=1
export MAX_PARALLEL=1

label_env() {
  case "$1" in
    l2_A)  echo "MODEL_TAG=llama2_7b_adamw DATASET=tofu CKPT_BASE=checkpoints/llama2_7b_adamw" ;;
    l3_A)  echo "MODEL_TAG=llama3_8b_adamw DATASET=tofu CKPT_BASE=checkpoints/llama3_8b_adamw" ;;
    l2m_A) echo "MODEL_TAG=llama2_muse_adamw DATASET=muse_news CKPT_BASE=checkpoints/llama2_muse_adamw" ;;
    l3m_A) echo "MODEL_TAG=llama3_muse_adamw DATASET=muse_news CKPT_BASE=checkpoints/llama3_muse_adamw" ;;
    *) echo "Unknown smoke label: $1" >&2; return 2 ;;
  esac
}

launch_worker() {
  local worker="$1"
  local label="$2"
  local quantizers="$3"
  local conditions="$4"
  local rankings="$5"
  local k_values="$6"
  local env_bits
  env_bits="$(label_env "$label")"
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
    env $env_bits \
      QUANTIZERS_OVERRIDE="$quantizers" \
      CONDITIONS_OVERRIDE="$conditions" \
      RANKING_SOURCES_OVERRIDE="$rankings" \
      K_VALUES_OVERRIDE="$k_values" \
      bash gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm \
      > "logs/gate2_v4_bridge/${SLURM_JOB_ID}_${worker}_bridge_v4_smoke.log" 2>&1 &
  pids+=("$!")
}

pids=()
launch_worker scope_l2   l2_A  "nf4 gptq_int4 rtn_int8" full_quantized none 0
launch_worker scope_l3   l3_A  "nf4 gptq_int4 rtn_int8" full_quantized none 0
launch_worker scope_l2m  l2m_A "nf4 gptq_int4 rtn_int8" full_quantized none 0
launch_worker scope_l3m  l3m_A "nf4 gptq_int4 rtn_int8" full_quantized none 0
launch_worker upqi_rand_nf4  l2_A nf4       protected "UPQI_signed random" 4
launch_worker upqi_rand_gptq l2_A gptq_int4 protected "UPQI_signed random" 4
launch_worker none_k0         l2_A nf4       protected none 0
launch_worker all_layers_k224 l2_A nf4       protected all_layers 224

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
