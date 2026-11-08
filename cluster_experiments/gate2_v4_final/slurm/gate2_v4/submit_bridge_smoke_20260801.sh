#!/bin/bash
set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

MODE="${SUBMIT_MODE:-test-only}"
PY="${PY:-/home/anon_user/.conda/envs/unlearn/bin/python}"
BRIDGE_LOCK="${BRIDGE_LOCK:?BRIDGE_LOCK required. Run run_bridge_phase_abc_20260801.sh first.}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-dev}"
GPUS="${GPUS:-8}"
CPUS="${CPUS:-64}"
MEM="${MEM:-0}"
TIME="${TIME:-04:00:00}"
SMOKE_SEEDS="${SMOKE_SEEDS:-42}"
SMOKE_ALGOS="${SMOKE_ALGOS:-GA}"
SMOKE_RESULTS_ROOT="${SMOKE_RESULTS_ROOT:-results/gate2_v4_bridge_standard_smoke}"
SMOKE_N_SAMPLES="${SMOKE_N_SAMPLES:-4}"
SMOKE_N_RETAIN="${SMOKE_N_RETAIN:-4}"
SMOKE_N_CALIBRATION="${SMOKE_N_CALIBRATION:-4}"
MANIFEST_N_SAMPLES="${MANIFEST_N_SAMPLES:-128}"
MANIFEST_N_RETAIN="${MANIFEST_N_RETAIN:-128}"
MANIFEST_N_CALIBRATION="${MANIFEST_N_CALIBRATION:-128}"
RANKINGS_PATH="${RANKINGS_PATH:-results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl}"
SLURM="gate2_v4_final/slurm/gate2_v4/bridge_v4_smoke_8gpu_bundle.slurm"

echo "Gate 2 v4 bridge smoke"
echo "  mode:       $MODE"
echo "  account:    $ACCOUNT"
echo "  partition:  $PARTITION"
echo "  labels:     l2_A l3_A l2m_A l3m_A"
echo "  seeds:      $SMOKE_SEEDS"
echo "  algos:      $SMOKE_ALGOS"
echo "  quantizers: nf4 gptq_int4 rtn_int8"
echo "  conditions: full_quantized; protected UPQI_signed/random k4; none/k0; all_layers/k224"
echo "  eval N:     forget=$SMOKE_N_SAMPLES retain=$SMOKE_N_RETAIN calib=$SMOKE_N_CALIBRATION"
echo "  results:    $SMOKE_RESULTS_ROOT"
echo "  source:     $BRIDGE_LOCK"
echo "  slurm:      $SLURM"

SBATCH_ARGS=(
  "--partition=$PARTITION"
  "--account=$ACCOUNT"
  "--nodes=1"
  "--ntasks=1"
  "--gres=gpu:$GPUS"
  "--cpus-per-task=$CPUS"
  "--mem=$MEM"
  "--time=$TIME"
  "--job-name=bridge_v4_smoke8"
  "--export=ALL,PYTHON=$PY,SOURCE_LOCK=$BRIDGE_LOCK,RESULTS_ROOT=$SMOKE_RESULTS_ROOT,RANKINGS_PATH=$RANKINGS_PATH,N_SAMPLES=$SMOKE_N_SAMPLES,N_RETAIN=$SMOKE_N_RETAIN,N_CALIBRATION=$SMOKE_N_CALIBRATION,MANIFEST_N_SAMPLES=$MANIFEST_N_SAMPLES,MANIFEST_N_RETAIN=$MANIFEST_N_RETAIN,MANIFEST_N_CALIBRATION=$MANIFEST_N_CALIBRATION,SEEDS_OVERRIDE=$SMOKE_SEEDS,ALGOS_OVERRIDE=$SMOKE_ALGOS"
)

if [[ "$MODE" == "test-only" ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "$SLURM"
elif [[ "$MODE" == "submit" ]]; then
  sbatch --parsable "${SBATCH_ARGS[@]}" "$SLURM"
else
  echo "SUBMIT_MODE must be test-only or submit" >&2
  exit 2
fi
