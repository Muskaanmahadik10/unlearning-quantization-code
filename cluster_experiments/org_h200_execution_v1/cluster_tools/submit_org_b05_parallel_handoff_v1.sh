#!/bin/bash
set -euo pipefail
cd /work/anon_user/hessian_unlearning
sha256sum -c org_h200_execution_v1/ORG_B05_PARALLEL_HANDOFF_V1_SHA256.txt
PY=/work/anon_user/.runtime/envs/org-packed-h200-v1/bin/python
"$PY" org_h200_execution_v1/scripts/test_org_b05_parallel_handoff_v1.py
bash -n org_h200_execution_v1/slurm/org_b05_parallel_condition_v1.slurm
OLD=results/org_h200_execution_v1/b05_behavior_cell_candidate/ordinary_tofu_ga_seed42_rtn_int8_349643
if test -f "$OLD/final.json"; then
 printf 'SERIAL_ALREADY_FINISHED: no cancellation or submission; inspect original final.json\n'
 exit 0
fi
# Stop the original writer before any reuse or parallel submission.
STATE=$(timeout 20s squeue -h -j 349643 -o '%T')
if test -n "$STATE"; then
 timeout 20s scancel 349643
fi
for ATTEMPT in 1 2 3 4 5 6; do
 STATE=$(timeout 20s squeue -h -j 349643 -o '%T')
 if test -z "$STATE"; then break; fi
 sleep 5
done
if test -n "$STATE"; then
 printf 'STOP: parent cleanup incomplete; no new submission\n'
 exit 1
fi
"$PY" org_h200_execution_v1/scripts/org_b05_parallel_handoff_v1.py prepare
"$PY" - <<'PY'
import json, pathlib, subprocess
root=pathlib.Path('results/org_h200_execution_v1/b05_behavior_parallel/repair_349643_v1')
tasks=json.loads((root/'handoff.json').read_text())['tasks']
if not tasks:
    print('NO_GPU_TASKS_NEEDED: run aggregation'); raise SystemExit(0)
with (root/'submission_attempt.txt').open('x') as f:
    f.write('Single array submission attempted; if uncertain, reconcile before retry.\n')
try:
    r=subprocess.run(['sbatch','--parsable',f'--array=0-{len(tasks)-1}',
        'org_h200_execution_v1/slurm/org_b05_parallel_condition_v1.slurm'],
        capture_output=True,text=True,timeout=30)
except subprocess.TimeoutExpired:
    print('SUBMISSION_UNCERTAIN: do not rerun; inspect org_b05_parallel'); raise
(root/'submission_response.txt').write_text(r.stdout+'\n'+r.stderr)
print(r.stdout, r.stderr)
if r.returncode: raise SystemExit(r.returncode)
job=r.stdout.strip().split(';')[0]
if not job.isdigit(): raise RuntimeError('Unexpected submission response; reconcile before retry')
print('PARALLEL_ARRAY_JOB='+job)
print('CONDITIONS='+','.join(tasks))
PY
