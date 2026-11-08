"""Recover the explicit GPTQ QoS rejection after NF4/RTN array 350098."""
import re
import subprocess
import time
import submit_org_b05_finish_wave_v1 as launch
import org_b05_finish_wave_v1 as wave

def check_rejection(text):
    wave.life.require('QOSMaxSubmitJobPerUserLimit' in text and
        'Batch job submission failed' in text and
        not re.search(r'Submitted batch job|^\d+(?:;\S+)?$',text,re.M),
        'not a confirmed rejection; reconcile before submitting')

def preflight():
    w=wave.load_wave()
    wave.life.require(wave.life.read(wave.ROOT/'other_job.json')['job_id']=='350098','unexpected parent array')
    for name in ('gptq_job.json','gptq_recovery_v1_submission_attempt.txt','submissions.json','finalizer_submission_attempt.txt'):
        wave.life.require(not (wave.ROOT/name).exists(),'existing submission state: '+name)
    check_rejection((wave.ROOT/'gptq_submission_response.txt').read_text())
    wave.life.require(sum(t['group']=='gptq' for t in w['tasks'])==9,'GPTQ task count')
    wave.life.require(sum(t['group']=='other' for t in w['tasks'])==14,'other task count')
    return w

def await_capacity():
    deadline=time.monotonic()+2700
    last=None
    while time.monotonic()<deadline:
        r=subprocess.run(['squeue','-r','-h','-u','anon_user','-o','%i %T'],
            capture_output=True,text=True,check=True,timeout=20)
        lines=r.stdout.strip().splitlines()
        if not lines: break
        if len(lines)!=last:
            print('WAITING_FOR_SUBMISSION_SLOTS jobs='+str(len(lines)),flush=True)
            last=len(lines)
        time.sleep(30)
    else: raise RuntimeError('45-minute wait limit; no new GPU submission attempted')
    expected={'350098_'+str(i) for i in range(14)}
    for attempt in range(4):
        try:
            text=wave.query_jobs(['350098'])
            rows=wave.accounting(text,expected)
            break
        except RuntimeError:
            if attempt==3: raise
            time.sleep(3)
    wave.life.require(all(r['state']=='COMPLETED' for r in rows.values()),
        'NF4/RTN array has failures; stop and inspect before GPTQ')
    (wave.ROOT/'other_accounting_before_recovery.psv').write_text(text)

def main():
    preflight(); await_capacity(); preflight()
    job=launch.submit('gptq_recovery_v1',['--array=0-8','--time=01:00:00',
        '--job-name=org_b05_finish_gptq','--export=ALL,B05_MODE=run,B05_GROUP=gptq'])
    wave.life.atomic(wave.ROOT/'gptq_job.json',{'job_id':job,'recovered_after_explicit_qos_rejection':True})
    wave.life.atomic(wave.ROOT/'submissions.json',{'arrays':{'other':'350098','gptq':job}})
    final=launch.submit('finalizer',['--dependency=afterany:'+job,
        '--time=00:15:00','--job-name=org_b05_finish_final','--export=ALL,B05_MODE=aggregate',
        '--output=logs/org_h200/org_b05_finish_final_%j.out',
        '--error=logs/org_h200/org_b05_finish_final_%j.err'])
    print('B05_SUBMISSION_RECOVERED=PASS\nGPTQ_JOB='+job+'\nFINALIZER_JOB='+final,flush=True)

if __name__=='__main__': main()
