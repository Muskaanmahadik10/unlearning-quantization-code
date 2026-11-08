"""One-shot submission with persistent attempt records; never auto-retry."""
import subprocess
from org_b05_finish_wave_v1 import ROOT, life, load_wave

SCRIPT='org_h200_execution_v1/slurm/org_b05_finish_wave_v1.slurm'

def submit(name,args):
    with (ROOT/(name+'_submission_attempt.txt')).open('x') as f:
        f.write('Submission attempted. Reconcile queue/accounting before any retry.\n')
    try:
        r=subprocess.run(['sbatch','--parsable',*args,SCRIPT],capture_output=True,text=True,timeout=45)
    except subprocess.TimeoutExpired:
        print('SUBMISSION_UNCERTAIN='+name+'; do not resubmit'); raise
    (ROOT/(name+'_submission_response.txt')).write_text(r.stdout+'\n'+r.stderr)
    if r.returncode: raise RuntimeError(r.stderr)
    j=r.stdout.strip().split(';')[0]
    life.require(j.isdigit(),'unexpected sbatch response; reconcile before retry')
    life.atomic(ROOT/(name+'_job.json'),{'job_id':j})
    print(name.upper()+'_JOB='+j,flush=True)
    return j

def main():
    w=load_wave(); ids={}
    life.require(not (ROOT/'submissions.json').exists(),'already submitted')
    for group,limit in (('other','00:30:00'),('gptq','01:00:00')):
        count=sum(t['group']==group for t in w['tasks'])
        if count:
            ids[group]=submit(group,['--array=0-'+str(count-1),'--time='+limit,
                '--job-name=org_b05_finish_'+group,'--export=ALL,B05_MODE=run,B05_GROUP='+group])
    life.require(bool(ids),'no new tasks')
    life.atomic(ROOT/'submissions.json',{'arrays':ids})
    # afterany guarantees an explicit incomplete report if any worker failed.
    final=submit('finalizer',['--dependency=afterany:'+':'.join(ids.values()),
        '--time=00:15:00','--job-name=org_b05_finish_final','--export=ALL,B05_MODE=aggregate',
        '--output=logs/org_h200/org_b05_finish_final_%j.out',
        '--error=logs/org_h200/org_b05_finish_final_%j.err'])
    print('B05_FINISH_WAVE_SUBMITTED=PASS\nFINALIZER_JOB='+final)

if __name__=='__main__': main()
