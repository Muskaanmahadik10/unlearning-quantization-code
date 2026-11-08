"""Rehash completed O/R evidence and export receipts, never model binaries."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_stage_common_v1 import *

def check_role(run,role,sched):
    final=read(run/'final.json');p=read(run/'produce.json');r=read(run/'reload.json')
    require(final['status']=='TRAINED_AND_RELOADED' and final['role']==role,'role incomplete')
    require(sha(run/'produce.json')==final['produce_sha256'] and sha(run/'reload.json')==final['reload_sha256'],'receipt drift')
    require(p['status']=='FIXED_SCHEDULE_TRAINING_COMPLETE' and r['status']=='FRESH_FINAL_MODEL_RELOAD_PASS','phase incomplete')
    require(p['identity']==r['identity']==final['identity'],'phase lineage drift')
    ident=p['identity']
    require(ident['role']==role and ident['parent_role']=='B' and ident['source_lock_sha256']==source(),'training lineage')
    require(ident['schedule_sha256']==sched['sha256'] and p['updates']==sched['total_updates'] and p['sample_count']==sched['sample_count'],'schedule coverage')
    require(sha(run/'updates.jsonl')==p['updates_sha256'],'update log drift')
    rows=[json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()]
    require(len(rows)==sched['total_updates'] and [x['update'] for x in rows]==list(range(sched['total_updates'])),'missing/duplicate updates')
    for row,expected in zip(rows,sched['updates']):
        require(row['epoch']==expected['epoch'] and row['microbatches']==len(expected['microbatches']) and row['normalization']=='actual_microbatch_count','update weighting')
    cp=run/'final_checkpoint';digest=sha(cp/'MANIFEST.json')
    require(digest==p['checkpoint_manifest_sha256']==r['checkpoint_manifest_sha256']==final['checkpoint_manifest_sha256'],'checkpoint binding')
    manifest=read(cp/'MANIFEST.json')
    require({x['path'] for x in manifest['files']}=={'model.safetensors','optimizer.safetensors','rng.safetensors','metadata.json'},'checkpoint coverage')
    verify(cp,manifest['files']);meta=read(cp/'metadata.json')
    require(meta['identity']==ident and meta['cursor']==sched['total_updates'],'checkpoint cursor/identity')
    require(meta['optimizer_steps'] and set(meta['optimizer_steps'].values())=={sched['total_updates']},'optimizer step coverage')
    require(canonical(meta['model_tensors'])==p['model_sha256']==r['model_sha256']==final['model_sha256'],'model state binding')
    require(p['pid']!=r['pid'],'fresh-process identity')
    require(p['evaluated_model_outcomes']==0 and final['scientific_efficacy_qualified'] is False,'outcome scope')
    return {'role':role,'updates':p['updates'],'sample_count':p['sample_count'],'model_sha256':p['model_sha256'],
            'checkpoint_manifest_sha256':digest,'source_lock_sha256':source(),'data_binding_sha256':ident['data_binding_sha256'],'status':'FULL_SCHEDULE_AND_BYTES_PASS'}

def main():
    release=dependencies();validate_plan(release)
    claim=RESULT/'launch_or_v1';req=read(claim/'request.json');validate_request(req)
    jid=read(claim/'submitted.json')['job_id'];require(jid.isdigit(),'job identity')
    q=subprocess.run(['sacct','-X','-n','-P','-j',jid,'--format=JobID%40,State%30,ExitCode,ElapsedRaw,AllocTRES%600'],capture_output=True,text=True,timeout=30)
    require(q.returncode==0,'accounting unavailable');times=accounting(q.stdout,jid)
    require(sum(times.values())<=4*3600,'training exceeded admitted cap; review accounting')
    schedules=read(PILOT/'cpu_evidence/training_schedule.json')
    roles={role:check_role(RESULT/('job_'+jid)/role,role,schedules[role]) for role in ('O','R')}
    require(roles['O']['data_binding_sha256']==roles['R']['data_binding_sha256'],'O/R data protocol mismatch')
    stem='m2_or_training_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'audit_exports'/stem;out.mkdir(parents=True,exist_ok=False)
    (out/'sacct.psv').write_text(q.stdout,encoding='utf-8')
    for role in ('O','R'):
        run=RESULT/('job_'+jid)/role
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl'):
                target=out/role/p.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        for suffix in ('out','err'):
            log=ROOT/'logs/org_m2_training_release_v1'/('train_'+jid+'_'+str(('O','R').index(role))+'.'+suffix)
            require(log.is_file() and not log.is_symlink(),'missing job log')
            (out/(role+'.'+suffix)).write_bytes(log.read_bytes())
    cpu=Path(req['cpu_receipt']);require(sha(cpu)==req['cpu_receipt_sha256'],'CPU receipt changed')
    for name in ('final.json','DATA_BINDING.json','CPU_TESTS.txt','quota.json'):
        target=out/'cpu'/name;target.parent.mkdir(exist_ok=True);target.write_bytes((cpu.parent/name).read_bytes())
    save(out/'final.json',{'status':'M2_OR_TRAINING_COLLECTED_PASS','job_id':jid,'roles':roles,'gpu_hours':sum(times.values())/3600,
        'ntd_at_30':sum(times.values())/120,'U_launch_authorized':False,'evaluation_launch_authorized':False,
        'scientific_efficacy_qualified':False,'model_binaries_remain_on_cluster':True,'independent_backup_complete':False})
    save(out/'EVIDENCE_MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as tf:
        for p in sorted(out.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(out).as_posix(),recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    print('M2_OR_TRAINING_EVIDENCE=PASS\nRETURN='+str(archive)+'\nSIDECAR='+str(archive)+'.sha256\nGPU_SUBMISSIONS=0\nU_LAUNCH_AUTHORIZED=false')

if __name__=='__main__':main()
