#!/bin/bash
set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

MODE="${SUBMIT_MODE:-test-only}"
PY="${PY:-/home/anon_user/.conda/envs/unlearn/bin/python}"
BRIDGE_LOCK="${BRIDGE_LOCK:?BRIDGE_LOCK required. Run run_bridge_phase_abc_20260801.sh first.}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-64gpus}"
TIME="${TIME:-04:00:00}"
TAIL_COUNT="${TAIL_COUNT:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
CONDITIONS_OVERRIDE="${CONDITIONS_OVERRIDE:-full_quantized}"
SLURM="gate2_v4_final/slurm/gate2_v4/bridge_v4_core64_8shard_bundle.slurm"

SBATCH_BASE=(
  "--partition=$PARTITION"
  "--account=$ACCOUNT"
  "--nodes=8"
  "--ntasks-per-node=1"
  "--gres=gpu:8"
  "--cpus-per-task=32"
  "--mem=0"
  "--time=$TIME"
  "--export=ALL,PYTHON=$PY,SOURCE_LOCK=$BRIDGE_LOCK,CONDITIONS_OVERRIDE=$CONDITIONS_OVERRIDE,MAX_PARALLEL=$MAX_PARALLEL"
)

echo "Gate 2 v4 full target-layer bridge submit"
echo "  mode:       $MODE"
echo "  account:    $ACCOUNT"
echo "  partition:  $PARTITION"
echo "  time:       $TIME"
echo "  tails:      $TAIL_COUNT"
echo "  conditions: $CONDITIONS_OVERRIDE"
echo "  source:     $BRIDGE_LOCK"
echo "  slurm:      $SLURM"

if [[ "$MODE" == "test-only" ]]; then
  sbatch --test-only "${SBATCH_BASE[@]}" --job-name=bridge_v4_fullq64 "$SLURM"
  exit 0
fi

if [[ "$MODE" != "submit" ]]; then
  echo "SUBMIT_MODE must be test-only or submit" >&2
  exit 2
fi

parent=$(sbatch --parsable "${SBATCH_BASE[@]}" --job-name=bridge_v4_fullq64 "$SLURM")
echo "BRIDGE_FULLQ=$parent"

prev="$parent"
for idx in $(seq 1 "$TAIL_COUNT"); do
  tail_job=$(sbatch --parsable \
    --dependency=afterany:"$prev" \
    "${SBATCH_BASE[@]}" \
    --job-name="bridge_v4_fullq64_tail${idx}" \
    "$SLURM")
  echo "BRIDGE_FULLQ_TAIL${idx}=$tail_job afterany:$prev"
  prev="$tail_job"
done

ids="$parent"
for job in $(squeue -u "$USER" -h -o "%i %j" | awk '/bridge_v4_fullq64/ {print $1}'); do
  case ",$ids," in
    *,"$job",*) ;;
    *) ids="$ids,$job" ;;
  esac
done

squeue -j "$ids" -o "%.8i %.10P %.30j %.10a %.2t %.24E %.11M %.11l %.4D %.4C %.12b %R" || true
squeue --start -j "$ids" || true
