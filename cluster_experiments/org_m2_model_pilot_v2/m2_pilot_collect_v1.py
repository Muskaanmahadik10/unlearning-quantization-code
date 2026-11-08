"""Collect terminal pilot evidence and accounting, never model weights or jobs."""
import argparse
import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_pilot_contract_v1 import *


def accounting(text,jid):
    rows={};terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
    for line in text.splitlines():
        if not line.strip():continue
        parts=line.rstrip('|').split('|')
        require(len(parts)==5,'accounting column count')
        job,state,exitcode,elapsed,tres=[x.strip() for x in parts]
        require(job in {jid+'_0',jid+'_1'} and job not in rows,'missing/duplicate/unexpected accounting task')
        state=state.split()[0].rstrip('+')
        require(state in terminal,'job not terminal; wait without resubmitting')
        require(elapsed.isdigit(),'unknown elapsed seconds')
        gpu=[x.split('=',1)[1] for x in tres.split(',') if x.startswith('gres/gpu=')]
        require(gpu==['1'],'one-GPU allocation not proven by AllocTRES')
        rows[job]={'state':state,'exit_code':exitcode,'elapsed_seconds':int(elapsed),'alloc_tres':tres}
    require(set(rows)=={jid+'_0',jid+'_1'},'incomplete two-task accounting')
    return rows


def main():
    claim=ROOT/'results/org_m2_model_pilot_v2/launch_v1'
    ap=argparse.ArgumentParser();ap.add_argument('--job');a=ap.parse_args()
    jid=job_id(a.job or read(claim/'submitted.json')['job_id']);source()
    require(read(claim/'submitted.json')['job_id']==jid,'job does not match this package claim')
    result=subprocess.run(['sacct','-X','-j',jid,'--noheader','--parsable2',
              '--format=JobID%40,State%30,ExitCode,ElapsedRaw,AllocTRES%600'],capture_output=True,text=True,timeout=30)
    require(result.returncode==0,'accounting unavailable: '+result.stderr)
    rows=accounting(result.stdout,jid)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    destination=ROOT/'audit_exports'/('m2_model_pilot_'+stamp);destination.mkdir(parents=True,exist_ok=False)
    (destination/'sacct.psv').write_text(result.stdout,encoding='utf-8',newline='\n')
    failures=[]
    for name in ('request.json','submitted.json','sbatch_stdout.txt','sbatch_stderr.txt'):
        shutil.copyfile(claim/name,destination/name)
    request=read(claim/'request.json');cpu=Path(request['cpu_receipt'])
    require(sha(cpu)==request['cpu_receipt_sha256'],'CPU admission receipt drift')
    shutil.copyfile(cpu,destination/'admission_cpu_final.json')
    shutil.copyfile(cpu.parent/'CPU_TESTS.txt',destination/'admission_CPU_TESTS.txt')
    for i,role in enumerate(('R','U')):
        root=ROOT/'results/org_m2_model_pilot_v2'/('job_'+jid)/role
        for p in sorted(root.rglob('*.json')):
            require(not p.is_symlink() and p.stat().st_size<10*1024**2,'unsafe/oversize evidence file')
            q=destination/role/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        for suffix in ('out','err'):
            log=ROOT/'logs/org_m2_model_pilot_v2'/('pilot_'+jid+'_'+str(i)+'.'+suffix)
            require(log.is_file() and not log.is_symlink(),'missing task log '+str(log))
            shutil.copyfile(log,destination/(role+'.'+suffix))
        try:
            require(rows[jid+'_'+str(i)]['state']=='COMPLETED' and rows[jid+'_'+str(i)]['exit_code']=='0:0','unsuccessful Slurm task')
            f=read(root/'final.json')
            require(f['status']=='FULL_MODEL_PILOT_PASS' and f['role']==role and f['production_authorized'] is False,'pilot final not accepted')
            for phase in ('produce','resume'):
                require(sha(root/(phase+'.json'))==f[phase+'_sha256'],'pilot branch drift')
            pa,pb=read(root/'produce.json'),read(root/'resume.json');outcome_compare(pa,pb)
            require(pa['identity']['source_lock_sha256']==source(),'pilot used other source')
            cp=root/'checkpoint_after_two'
            require(sha(cp/'MANIFEST.json')==pa['checkpoint_manifest_sha256'],'saved checkpoint manifest drift')
            verify(cp,read(cp/'MANIFEST.json')['files'])
        except Exception as e:
            failures.append({'role':role,'error':str(e)})
    seconds=sum(r['elapsed_seconds'] for r in rows.values())
    save(destination/'SUMMARY.json',{'status':'PILOT_EVIDENCE_PASS' if not failures else 'PILOT_EVIDENCE_FAIL',
         'jobs':rows,'failures':failures,'gpu_hours':seconds/3600,'ntd_at_user_rate_30':seconds/120,
         'source_lock_sha256':source(),'gpu_submissions':0,'production_authorized':False,
         'full_benchmark_evaluator_certified':False,'m3_authorized':False,
         'binary_retention':'cluster_only; rehashed checkpoint binaries excluded from this small evidence export',
         'independent_binary_backup_complete':False})
    for name in ('SELF_LOCK.json','PILOT_PLAN.json','UPSTREAM_EVALUATOR_PROVENANCE.json'):
        shutil.copyfile(PKG/name,destination/name)
    files=[{'path':p.relative_to(destination).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(destination.rglob('*')) if p.is_file()]
    save(destination/'EVIDENCE_MANIFEST.json',{'files':files});verify(destination,files)
    archive=destination.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        for p in sorted(destination.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(destination).as_posix(),recursive=False)
    sidecar=Path(str(archive)+'.sha256')
    sidecar.write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    print('M2_PILOT_EVIDENCE='+('PASS' if not failures else 'FAIL')+'\nGPU_SUBMISSIONS=0\nPRODUCTION_AUTHORIZED=false')
    print('RETURN='+str(archive)+'\nSIDECAR='+str(sidecar))


if __name__=='__main__':main()
