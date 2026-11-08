"""Single B canary, then remaining O/R/U; never retry uncertain allocations."""
import subprocess
from pathlib import Path
from m2_eval_qualification_common_v1 import *

PRIOR_JOB = '360917'
STAGES = {'canary': (0,), 'remaining': (1, 2, 3)}


def parse_accounting(text, jid, indices, state='COMPLETED', exit_code='0:0'):
    found = {}
    expected = {jid+'_'+str(i) for i in indices}
    for line in text.splitlines():
        if not line.strip(): continue
        row = [v.strip() for v in line.rstrip('|').split('|')]
        require(len(row) == 5, 'accounting columns')
        j, s, e, seconds, tres = row
        require(j in expected and j not in found, 'unexpected/duplicate accounting row')
        require(s == state and e == exit_code, 'allocation not accepted terminal state: '+j+' '+s+' '+e)
        require(seconds.isdigit() and 0 < int(seconds) <= 900, 'unknown/out-of-cap elapsed')
        require([x for x in tres.split(',') if x.startswith('gres/gpu=')] == ['gres/gpu=1'], 'GPU count uncertainty')
        found[j] = int(seconds)
    require(set(found) == expected, 'incomplete accounting')
    return found


def fetch_accounting(jid, indices, state='COMPLETED', exit_code='0:0'):
    q = subprocess.run(['sacct', '-X', '-n', '-P', '-j', jid,
        '--format=JobID%40,State%30,ExitCode,ElapsedRaw,AllocTRES%600'],
        capture_output=True, text=True, timeout=30)
    require(q.returncode == 0, 'accounting unavailable')
    return q.stdout, parse_accounting(q.stdout, jid, indices, state, exit_code)


def reconcile_prior():
    r = release()
    old = ROOT/'org_m2_eval_qualification_v1'
    require(sha(old/'SELF_LOCK.json') == r['repair']['prior_source_sha256'], 'prior source lock drift')
    verify(old, read(old/'SELF_LOCK.json')['files'])
    c = ROOT/'results/org_m2_eval_qualification_v1/launch_v1'
    submitted = read(c/'submitted.json'); request = read(c/'request.json')
    require(submitted['job_id'] == PRIOR_JOB and submitted['roles'] == list(ROLES), 'wrong prior job')
    require(request['source_lock_sha256'] == r['repair']['prior_source_sha256'], 'prior request identity')
    require(not (c/'SUBMISSION_UNKNOWN.json').exists(), 'ambiguous prior submission')
    text, times = fetch_accounting(PRIOR_JOB, range(4), 'FAILED', '1:0')
    for i in range(4):
        err = ROOT/'logs/org_m2_eval_qualification_v1'/('eval_'+PRIOR_JOB+'_'+str(i)+'.err')
        require(err.is_file() and not err.is_symlink(), 'missing prior failure log')
        require('ValueError: score mismatch root/avg_loss' in err.read_text(), 'different failure requires audit')
    return {'job_id': PRIOR_JOB, 'sacct': text, 'elapsed_seconds': times,
            'gpu_hours': sum(times.values())/3600, 'ntd_at_30': sum(times.values())/120}


def claim_for_index(index):
    require(index in range(4), 'invalid task index')
    return RESULT/('launch_canary' if index == 0 else 'launch_remaining')


def verify_canary():
    from m2_eval_qualification_v1 import final_check
    c = RESULT/'launch_canary'
    require(not (c/'SUBMISSION_UNKNOWN.json').exists(), 'ambiguous canary submission')
    s = read(c/'submitted.json'); q = read(c/'request.json')
    require(s['stage'] == q['stage'] == 'canary' and s['indices'] == q['indices'] == [0], 'canary scope')
    require(q['source_lock_sha256'] == source(), 'canary source drift')
    cpu = Path(q['cpu_receipt'])
    require(cpu.resolve().is_relative_to(RESULT.resolve()) and sha(cpu) == q['cpu_receipt_sha256'], 'canary CPU receipt drift')
    text, times = fetch_accounting(job_id(s['job_id']), (0,))
    path = RESULT/('job_'+s['job_id'])/'B'
    final = read(path/'final.json')
    require(final == final_check(path), 'canary final inconsistent')
    require(final['role'] == 'B' and final['model_sha256'] == release()['base_state_sha256'], 'canary model identity')
    require(final['source_lock_sha256'] == source() and final['fixture_sha256'] == read(cpu)['fixture_sha256'], 'canary source/fixture drift')
    return {'job_id': s['job_id'], 'path': str(path), 'final_sha256': sha(path/'final.json'),
            'sacct': text, 'gpu_hours': sum(times.values())/3600}
