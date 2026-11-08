"""Submit full-cell RTN production canary, then five remaining cells; never duplicate."""
import argparse
import os
import re
import subprocess
from pathlib import Path
from m1_contract_v1 import PACKAGE,PROJECT,ENVIRONMENT,package_check,preflight,read,write,sha,require
from m1_core_gate_v1 import reconciliation_gate,validate_core

def run(command,timeout=30):
    r=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
    require(r.returncode==0,'command failed: '+r.stderr)
    return r.stdout
def jobid(text):
    require(re.fullmatch(r'[1-9][0-9]*(;[A-Za-z0-9_.-]+)?',text.strip()) is not None,'ambiguous job response')
    return text.strip().split(';')[0]
def completed(text,job):
    rows=[[v.strip() for v in l.split('|')] for l in text.splitlines() if l.split('|')[0].strip()==job]
    require(len(rows)==1 and rows[0][1:3]==['COMPLETED','0:0'],'canary not completed successfully')
    require(rows[0][3].isdigit() and 0<int(rows[0][3])<=2400,'canary elapsed invalid')
    require(re.search(r'(?:^|,)gres/gpu=1(?:,|$)',rows[0][4]) is not None,'canary GPU allocation')
    return int(rows[0][3])

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('canary','remaining'),required=True);p.add_argument('--submit',action='store_true');a=p.parse_args()
    require(Path.cwd().resolve()==PROJECT,'wrong cluster root')
    source=package_check(); smoke=reconciliation_gate()
    launch=PROJECT/'results/org_m1_core_v1/launch';launch.mkdir(parents=True,exist_ok=True)
    receipt=launch/'preflight.json'
    if not receipt.exists(): preflight(receipt)
    require(read(receipt)['source_lock_sha256']==source and read(receipt)['status']=='PASS','preflight binding')
    cpu=launch/'endpoint_cpu.json'
    if not cpu.exists():
        subprocess.run([str(ENVIRONMENT/'bin/python'),str(PACKAGE/'m1_endpoint_cpu_regression_v3.py'),'--output',str(cpu)],check=True,timeout=90,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
    require(read(cpu)['status']=='PASS' and read(cpu)['source_lock_sha256']==source,'CPU gate')
    indices='5'
    if a.stage=='remaining':
        submitted=read(launch/'canary_outcome.json');require(submitted['status']=='SUBMITTED','canary submission unknown')
        j=submitted['job_id'];seconds=completed(run(['sacct','-X','-n','-P','-j',j,'--format=JobID%80,State%60,ExitCode,ElapsedRaw,AllocTRES%500']),j+'_5')
        verdict=validate_core(PROJECT/('results/org_m1_core_v1/core_'+j+'/cell_5'),5)
        require(smoke['gpu_hours_all_smoke_attempts']+seconds/3600+10/3<=7,'remaining budget gate')
        indices='0-4'
        gate=launch/'canary_acceptance.json'
        if not gate.exists():write(gate,{'status':'PASS','validation':verdict,'seconds':seconds,'job_id':j})
        else: require(read(gate)['validation']==verdict,'canary acceptance changed')
    if not a.submit: print('M1_CORE_PREPARATION=PASS\nGPU_SUBMISSIONS=0');return
    intent=launch/(a.stage+'_intent.json');require(not intent.exists(),'submission attempted before: reconcile outcome, do not resubmit')
    queue=run(['squeue','-r','-u','anon_user','-h','-o','%i|%j|%T'])
    require('org_m1_core_v1' not in queue,'core job still active or cleaning up')
    (PROJECT/'logs/org_m1_core_v1').mkdir(parents=True,exist_ok=True)
    cmd=['sbatch','--parsable','--array='+indices,'--export=ALL,M1_PREFLIGHT='+str(receipt)+',M1_ENDPOINT_CPU_RECEIPT='+str(cpu)+',M1_ENDPOINT_CPU_SHA256='+sha(cpu),str(PACKAGE/'m1_core.slurm')]
    run(cmd[:1]+['--test-only']+cmd[2:])
    write(intent,{'stage':a.stage,'command':cmd,'max_additional_gpu_hours':2/3 if a.stage=='canary' else 10/3,'core_arms':10 if a.stage=='canary' else 50})
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
        o={'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode,'status':'UNKNOWN_DO_NOT_RESUBMIT'}
        if r.returncode==0:o['job_id']=jobid(r.stdout);o['status']='SUBMITTED'
    except Exception as e:
        write(launch/(a.stage+'_outcome.json'),{'status':'UNKNOWN_DO_NOT_RESUBMIT','exception':str(e)});raise
    write(launch/(a.stage+'_outcome.json'),o);require(o['status']=='SUBMITTED','submission failed; preserve intent and inspect')
    print('M1_CORE_'+a.stage.upper()+'_JOB='+o['job_id'])
    print('MAX_ADDITIONAL_NTD='+str(20 if a.stage=='canary' else 100))
if __name__=='__main__':main()
