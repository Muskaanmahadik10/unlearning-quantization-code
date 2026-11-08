#!/bin/bash
# Freeze the completed Full144 evidence and run the independent publication gate.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PY:-/home/anon_user/.conda/envs/unlearn/bin/python}"
PILOT_JOB="${PILOT_JOB:-317631}"
DIVERSITY_JOB="${DIVERSITY_JOB:-317817}"
REMAINDER_JOB="${REMAINDER_JOB:-319792}"
PILOT_LOCK_SHA="${PILOT_LOCK_SHA:-7e7af7446475d71cea49bdf98ff5a8439e4a213a72ba9f563e40cadd9238d229}"
DIVERSITY_LOCK_SHA="${DIVERSITY_LOCK_SHA:-2b895012ed0d43d6c6da188b773f415a8c628910a8069f3cd2585abf6d60c3bb}"
REMAINDER_LOCK_SHA="${REMAINDER_LOCK_SHA:-060f6f9b1da7a9b3a7991d515bf4e816cc6d49757d271c5953d337e5abb11a67}"
V4_LOCK_SHA="${V4_LOCK_SHA:-b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b}"
PRE_FULL_STATUS="${PRE_FULL_STATUS:-gate2_v4_pre_full144_final_20260831_161921.tsv}"
FREEZE_WRAPPER="gate2_v4_final/slurm/gate2_v4/audit_analyze_freeze_bridge_v41_full144.sh"
AUDITOR="gate2_v4_final/scripts/audit_full144_publication_gate.py"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }
[[ -f "$FREEZE_WRAPPER" ]] || { echo "Missing freeze wrapper: $FREEZE_WRAPPER" >&2; exit 2; }
[[ -f "$AUDITOR" ]] || { echo "Missing independent auditor: $AUDITOR" >&2; exit 2; }
[[ -f "$PRE_FULL_STATUS" ]] || { echo "Missing pre-Full144 status: $PRE_FULL_STATUS" >&2; exit 2; }

require_job() {
  local job="$1" label="$2" record state exit_code
  record=$(sacct -X -n -P -j "$job" --format=State,ExitCode | head -n 1)
  IFS='|' read -r state exit_code _ <<< "$record"
  echo "${label}_STATE=$state ${label}_EXIT=$exit_code"
  [[ "$state" == COMPLETED && "$exit_code" == 0:0 ]] || {
    echo "STOP: $label job $job is not COMPLETED 0:0" >&2
    exit 2
  }
}

require_job "$PILOT_JOB" PILOT18
require_job "$DIVERSITY_JOB" DIVERSITY12
require_job "$REMAINDER_JOB" REMAINDER120

find results/gate2_v4_bridge_standard_v41_pilot18 -name bridge_v4.json -type f | grep -q .
[[ "$(find results/gate2_v4_bridge_standard_v41_pilot18 -name bridge_v4.json -type f | wc -l)" -eq 18 ]]
[[ "$(find results/gate2_v4_bridge_standard_v41_diversity12 -name bridge_v4.json -type f | wc -l)" -eq 12 ]]
[[ "$(find results/gate2_v4_bridge_standard_v41_remainder120 -name bridge_v4.json -type f | wc -l)" -eq 120 ]]
[[ "$(find results/gate2_v4_bridge_standard_v41_pilot18 results/gate2_v4_bridge_standard_v41_diversity12 results/gate2_v4_bridge_standard_v41_remainder120 -name 'bridge_v4.json.partial' -type f | wc -l)" -eq 0 ]]

STAMP=$(date +%Y%m%d_%H%M%S)
FREEZE_CONSOLE="full144_freeze_${STAMP}.console.txt"
REMAINDER_LOCK_SHA="$REMAINDER_LOCK_SHA" \
REMAINDER_JOB="$REMAINDER_JOB" \
PILOT_JOB="$PILOT_JOB" \
DIVERSITY_JOB="$DIVERSITY_JOB" \
bash "$FREEZE_WRAPPER" | tee "$FREEZE_CONSOLE"

FREEZE=$(awk -F= '/^FULL144_FREEZE=/{value=$2} END{print value}' "$FREEZE_CONSOLE")
[[ -n "$FREEZE" && -d "$FREEZE" ]] || {
  echo "STOP: could not discover frozen evidence directory" >&2
  exit 2
}

# The legacy freeze records manifest hashes but does not copy the manifest
# bytes. Build a hard-linked publication input, then add the exact manifests
# and pre-Full144 status required by the independent read-only audit.
PUBLICATION_INPUT="${FREEZE}_publication_input"
cp -al "$FREEZE" "$PUBLICATION_INPUT"
chmod u+w "$PUBLICATION_INPUT" "$PUBLICATION_INPUT/provenance"
mkdir -p "$PUBLICATION_INPUT/provenance/manifests"
cp -a manifests/gate2_v4/*.json "$PUBLICATION_INPUT/provenance/manifests/"
cp -a "$PRE_FULL_STATUS" "$PUBLICATION_INPUT/provenance/"
find "$PUBLICATION_INPUT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$PUBLICATION_INPUT/provenance/publication_input_sha256.txt"
tar -czf "${PUBLICATION_INPUT}.tar.gz" "$PUBLICATION_INPUT"
sha256sum "${PUBLICATION_INPUT}.tar.gz" > "${PUBLICATION_INPUT}.tar.gz.sha256"
chmod -R a-w "$PUBLICATION_INPUT" "${PUBLICATION_INPUT}.tar.gz" "${PUBLICATION_INPUT}.tar.gz.sha256"

AUDIT_OUT="gate2_v4_full144_publication_audit_${STAMP}"
set +e
"$PY" "$AUDITOR" \
  --frozen_archive "$PUBLICATION_INPUT" \
  --pilot_lock_sha256 "$PILOT_LOCK_SHA" \
  --diversity_lock_sha256 "$DIVERSITY_LOCK_SHA" \
  --remainder_lock_sha256 "$REMAINDER_LOCK_SHA" \
  --v4_lock_sha256 "$V4_LOCK_SHA" \
  --pilot_job "$PILOT_JOB" \
  --diversity_job "$DIVERSITY_JOB" \
  --remainder_job "$REMAINDER_JOB" \
  --pre_full_status_tsv "$PRE_FULL_STATUS" \
  --out_dir "$AUDIT_OUT" \
  2>&1 | tee "${AUDIT_OUT}.console.txt"
AUDIT_EXIT=${PIPESTATUS[0]}
set -e

cp -a "$PRE_FULL_STATUS" "$AUDIT_OUT/"
cp -a "$AUDITOR" "$AUDIT_OUT/"
cp -a "$FREEZE_CONSOLE" "${AUDIT_OUT}.console.txt" "$AUDIT_OUT/"
sacct -X -n -P -j "$PILOT_JOB,$DIVERSITY_JOB,$REMAINDER_JOB" \
  --format=JobIDRaw,JobName%40,Partition,State,ExitCode,ElapsedRaw,Start,End,AllocTRES%200,NodeList%100 \
  > "$AUDIT_OUT/publication_slurm_accounting.psv"
find "$AUDIT_OUT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$AUDIT_OUT/publication_audit_sha256.txt"
tar -czf "${AUDIT_OUT}.tar.gz" "$AUDIT_OUT"
sha256sum "${AUDIT_OUT}.tar.gz" | tee "${AUDIT_OUT}.tar.gz.sha256"
chmod -R a-w "$AUDIT_OUT" "${AUDIT_OUT}.tar.gz" "${AUDIT_OUT}.tar.gz.sha256"

echo "FULL144_FROZEN_EVIDENCE=$FREEZE"
echo "FULL144_PUBLICATION_INPUT=$PUBLICATION_INPUT"
echo "FULL144_PUBLICATION_AUDIT=$AUDIT_OUT"
echo "FULL144_PUBLICATION_ARCHIVE=${AUDIT_OUT}.tar.gz"
echo "FULL144_PUBLICATION_AUDIT_EXIT=$AUDIT_EXIT"
cat "$AUDIT_OUT/FULL144_BRUTAL_AUDIT.md"
cat "$AUDIT_OUT/full144_next_phase_decision.json"
exit "$AUDIT_EXIT"
