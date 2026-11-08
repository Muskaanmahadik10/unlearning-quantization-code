"""Verify final U schedule/checkpoint, preserve all receipts; no new jobs."""
import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_u_common_v1 import *

def main():
    r=dependencies();parent_metadata(r)
    claim=RESULT/'launch_v1';req=read(claim/'request.json');jid=read(claim/'submitted.json')['job_id'];require(jid.isdigit(),'job identity')
    require(req['source_lock_sha256']==source() and req['parent_model_sha256']==r['parent_model_sha256'],'claim source/parent')
    q=subprocess.run(['sacct','-X','-n','-P','-j',jid,'--format=JobID%40,State%30,ExitCode,ElapsedRaw,AllocTRES%600'],capture_output=True,text=True,timeout=30)
    require(q.returncode==0,'accounting unavailable');seconds=accounting(q.stdout,jid);require(seconds<=7200,'budget overrun requires review')
    run=RESULT/('job_'+jid)/'U';f=read(run/'final.json');p=read(run/'produce.json');v=read(run/'reload.json')
    require(f['status']=='U_TRAINED_AND_RELOADED' and p['status']=='U_FIXED_SCHEDULE_COMPLETE' and v['status']=='U_FINAL_RELOAD_PASS','U final status')
    require(sha(run/'produce.json')==f['produce_sha256'] and sha(run/'reload.json')==f['reload_sha256'],'phase hashes')
    ident=p['identity'];require(ident==v['identity']==f['identity'],'phase identity')
    require(ident['source_lock_sha256']==source() and ident['role']=='U' and ident['parent_role']==ident['teacher_role']=='O' and ident['fresh_optimizer'] is True,'U lineage')
    require(ident['parent_model_sha256']==p['teacher_initial_sha256']==p['teacher_final_sha256']==r['parent_model_sha256'],'teacher preservation')
    require(ident['parent_checkpoint_manifest_sha256']==r['parent_checkpoint_manifest_sha256'],'parent checkpoint binding')
    from m2_stage_common_v1 import PILOT
    from m2_training_core_v1 import lr_multiplier
    sched=read(PILOT/'cpu_evidence/training_schedule.json')['U'];cfg=read(PILOT/'CONTROLLED_OUR_DESIGN_V2.json')['training']['U']
    require(ident['schedule_sha256']==sched['sha256'] and p['updates']==130 and p['forget_presentations']==p['retain_draws']==4000,'all U data coverage')
    require(sha(run/'updates.jsonl')==p['updates_sha256'],'update log bytes');updates=[json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()]
    require(len(updates)==130,'update count')
    for i,(u,e) in enumerate(zip(updates,sched['updates'])):
        require(u['update']==i and u['epoch']==e['epoch'] and u['microbatches']==len(e['microbatches']) and u['normalization']=='actual_microbatch_count','update identity/weight')
        require(u['lr']==cfg['learning_rate']*lr_multiplier(i,130,sched['warmup_updates']) and all(math.isfinite(u[k]) for k in ('loss_mean','gradient_norm_before_clip')),'LR or nonfinite update')
    cp=run/'final_checkpoint';digest=sha(cp/'MANIFEST.json')
    require(digest==p['checkpoint_manifest_sha256']==v['checkpoint_manifest_sha256']==f['checkpoint_manifest_sha256'],'U checkpoint hash')
    manifest=read(cp/'MANIFEST.json')['files'];require(len(manifest)==4 and {x['path'] for x in manifest}=={'model.safetensors','optimizer.safetensors','rng.safetensors','metadata.json'},'checkpoint file coverage');verify(cp,manifest)
    meta=read(cp/'metadata.json');require(meta['identity']==ident and meta['cursor']==130 and len(meta['optimizer_steps'])==146 and set(meta['optimizer_steps'].values())=={130},'optimizer/cursor')
    require(canonical(meta['model_tensors'])==p['model_sha256']==v['model_sha256']==f['model_sha256']!=r['parent_model_sha256'],'U model state')
    require(p['pid']!=v['pid'] and p['evaluated_model_outcomes']==0 and f['scientific_efficacy_qualified'] is False,'reload/claim scope')
    stem='m2_u_training_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'audit_exports'/stem;out.mkdir(parents=True,exist_ok=False)
    (out/'sacct.psv').write_text(q.stdout,encoding='utf-8')
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl'):
            require(not path.is_symlink() and path.stat().st_size<20*1024**2,'unsafe evidence');dest=out/'U'/path.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    cpu=Path(req['cpu_receipt']);require(sha(cpu)==req['cpu_receipt_sha256'],'CPU receipt drift')
    for name in ('final.json','CPU_TESTS.txt'):
        dest=out/'cpu'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes((cpu.parent/name).read_bytes())
    for name in ('request.json','submitted.json'):(out/name).write_bytes((claim/name).read_bytes())
    for suffix in ('out','err'):
        log=ROOT/'logs/org_m2_u_training_v1'/('train_'+jid+'.'+suffix);require(log.is_file() and not log.is_symlink(),'missing log');(out/('U.'+suffix)).write_bytes(log.read_bytes())
    save(out/'final.json',{'status':'U_TRAINING_EVIDENCE_PASS','source_lock_sha256':source(),'job_id':jid,'updates':130,
      'parent_O_sha256':r['parent_model_sha256'],'model_sha256':p['model_sha256'],'checkpoint_manifest_sha256':digest,
      'gpu_hours':seconds/3600,'ntd_at_30':seconds/120,'scientific_efficacy_qualified':False,
      'evaluation_launch_authorized':False,'M3_authorized':False,'independent_binary_backup_complete':False})
    save(out/'EVIDENCE_MANIFEST.json',{'files':[{'path':x.relative_to(out).as_posix(),'bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(out.rglob('*')) if x.is_file()]})
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as tf:
        for x in sorted(out.rglob('*')):
            if x.is_file():tf.add(x,arcname=x.relative_to(out).as_posix(),recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    print('M2_U_TRAINING_EVIDENCE=PASS\nRETURN='+str(archive)+'\nSIDECAR='+str(archive)+'.sha256\nGPU_SUBMISSIONS=0\nEVALUATION_LAUNCH_AUTHORIZED=false')
if __name__=='__main__':main()
