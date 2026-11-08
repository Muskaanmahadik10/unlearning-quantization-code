"""Full NPO+retain schedule from accepted O. No outcome-based selection."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_u_common_v1 import *
from m2_u_model_io_v1 import install_model,separate_teacher

def admission(output):
    r=dependencies();jid=os.environ.get('SLURM_JOB_ID','')
    require(jid.isdigit() and not os.environ.get('SLURM_ARRAY_JOB_ID') and os.environ.get('SLURM_JOB_NAME')=='org_m2_train_u' and os.environ.get('SLURM_JOB_PARTITION')=='dev','allocation identity')
    claim=RESULT/'launch_v1';req=read(claim/'request.json')
    require(req['source_lock_sha256']==source() and req['max_ntd']==60 and req['max_gpu_hours']==2 and req['accepted_unverified_quota'] is True,'request scope')
    for _ in range(100):
        if (claim/'submitted.json').exists() or (claim/'SUBMISSION_UNKNOWN.json').exists():break
        time.sleep(.2)
    require(read(claim/'submitted.json')['job_id']==jid,'job acknowledgement')
    require(output.resolve()==(RESULT/('job_'+jid)/'U').resolve(),'output scope')
    cpu=Path(req['cpu_receipt']);require(cpu.resolve().is_relative_to(RESULT.resolve()) and sha(cpu)==req['cpu_receipt_sha256'],'CPU receipt binding')
    c=read(cpu);require(c['status']=='U_CPU_ADMISSION_PASS' and c['source_lock_sha256']==source() and c['parent_model_sha256']==r['parent_model_sha256'],'CPU parent admission')
    return r

def worker(a):
    admission(a.output);torch,np,values,r=setup(True)
    from m2_model_pilot_v1 import load_model,optimizer
    from m2_training_core_v1 import collate,update,lr_multiplier
    import m2_paged_checkpoint_v1 as ck
    from transformers import AutoTokenizer
    _,base,binding,tokens,schedules,cal,design=values
    cfg=design['training']['U'];sched=schedules['U'];pm=parent_metadata(r)
    model=load_model(torch,base)
    require(canonical(ck.model_rows(torch,model))==r['base_model_sha256'],'base architecture/bytes identity')
    identity={'role':'U','parent_role':'O','teacher_role':'O','parent_model_sha256':r['parent_model_sha256'],
      'parent_checkpoint_manifest_sha256':r['parent_checkpoint_manifest_sha256'],'base_sha256':r['base_model_sha256'],
      'source_lock_sha256':source(),'or_source_sha256':r['or_source_sha256'],'data_binding_sha256':r['data_binding_sha256'],
      'schedule_sha256':sched['sha256'],'generation':design['evaluation']['generation'],'fresh_optimizer':True,
      'teacher_frozen':True,'scientific_fixed_schedule':True}
    if a.phase=='verify':
        p=read(a.output/'produce.json');require(p['identity']==identity,'U reload identity')
        cp=a.output/'final_checkpoint';require(sha(cp/'MANIFEST.json')==p['checkpoint_manifest_sha256'],'U checkpoint digest')
        verify(cp,read(cp/'MANIFEST.json')['files']);m=read(cp/'metadata.json')
        require(m['identity']==identity and m['cursor']==130 and len(m['optimizer_steps'])==146 and set(m['optimizer_steps'].values())=={130},'U checkpoint schedule/lineage')
        install_model(torch,model,cp,m,p['model_sha256'])
        require(os.getpid()!=p['pid'],'fresh-process requirement')
        save(a.output/'reload.json',{'status':'U_FINAL_RELOAD_PASS','identity':identity,'pid':os.getpid(),
             'model_sha256':p['model_sha256'],'checkpoint_manifest_sha256':p['checkpoint_manifest_sha256']})
        return
    install_model(torch,model,PARENT,pm,r['parent_model_sha256'])
    teacher=load_model(torch,base);install_model(torch,teacher,PARENT,pm,r['parent_model_sha256']);separate_teacher(model,teacher)
    opt=optimizer(model,cfg);require(len(opt.state)==0,'parent optimizer must not be inherited')
    teacher_initial=canonical(ck.model_rows(torch,teacher));require(teacher_initial==r['parent_model_sha256'],'teacher not exact O')
    tokenizer=AutoTokenizer.from_pretrained(base['snapshot_path'],local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    require(tokenizer.pad_token_id==128009,'tokenizer pad')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();draws=0
    for ix,record in enumerate(sched['updates']):
        micros=[]
        for micro in record['microbatches']:
            require(set(micro)=={'anchor','retain'},'NPO data route')
            batch={}
            for kind,ids in micro.items():
                order=tokens['role_order']['U' if kind=='anchor' else 'R']
                es=[tokens['by_exact_qa_sha256'][order[i]] for i in ids]
                batch[kind]={k:v.to('cuda:0') for k,v in collate(torch,es,tokenizer.pad_token_id).items()}
                if kind=='retain':draws+=len(ids)
            micros.append(batch)
        opt.param_groups[0]['lr']=cfg['learning_rate']*lr_multiplier(ix,sched['total_updates'],sched['warmup_updates'])
        metrics=update(torch,model,opt,micros,reference=teacher,max_grad_norm=1.0,
                       beta=cfg['beta'],alpha=cfg['alpha_retain'],gamma=cfg['gamma_forget'])
        require(all(p.grad is not None for p in model.parameters()) and all(p.grad is None and not p.requires_grad for p in teacher.parameters()),'student/teacher gradient scope')
        opt.zero_grad(set_to_none=True)
        with (a.output/'updates.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'update':ix,'epoch':record['epoch'],'lr':opt.param_groups[0]['lr'],**metrics},allow_nan=False)+'\n')
        if (ix+1)%10==0 or ix+1==130:print('M2_TRAIN=U:'+str(ix+1)+'/130',flush=True)
    torch.cuda.synchronize();require(draws==4000,'retain draw coverage')
    teacher_final=canonical(ck.model_rows(torch,teacher));require(teacher_final==teacher_initial,'teacher changed')
    cp=a.output/'final_checkpoint';digest=ck.save_checkpoint(torch,np,model,opt,cp,identity,130)
    m=read(cp/'metadata.json');require(set(m['optimizer_steps'].values())=={130},'fresh U optimizer total')
    model_sha=canonical(m['model_tensors']);require(model_sha!=r['parent_model_sha256'],'U unchanged from O')
    parent_metadata(r)  # Preserve and rehash parent artifact after training.
    save(a.output/'produce.json',{'status':'U_FIXED_SCHEDULE_COMPLETE','identity':identity,'pid':os.getpid(),'updates':130,
      'forget_presentations':sched['sample_count'],'retain_draws':draws,'teacher_initial_sha256':teacher_initial,'teacher_final_sha256':teacher_final,
      'model_sha256':model_sha,'checkpoint_manifest_sha256':digest,'updates_sha256':sha(a.output/'updates.jsonl'),
      'elapsed_training_and_save_seconds':time.perf_counter()-start,'torch_allocator_peak_bytes':torch.cuda.max_memory_allocated(),
      'evaluated_model_outcomes':0})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['all','train','verify'],default='all');a=p.parse_args()
    admission(a.output)
    if a.phase!='all':worker(a);return
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        for phase in ('train','verify'):subprocess.run([sys.executable,'-I','-B',__file__,'--output',str(a.output),'--phase',phase],check=True)
        p=read(a.output/'produce.json');r=read(a.output/'reload.json');require(p['model_sha256']==r['model_sha256'] and p['identity']==r['identity'],'final phase agreement')
        source();save(a.output/'final.json',{'status':'U_TRAINED_AND_RELOADED','identity':p['identity'],
          'produce_sha256':sha(a.output/'produce.json'),'reload_sha256':sha(a.output/'reload.json'),'model_sha256':p['model_sha256'],
          'checkpoint_manifest_sha256':p['checkpoint_manifest_sha256'],'scientific_efficacy_qualified':False,
          'evaluation_launch_authorized':False,'M3_authorized':False})
        print('M2_TRAINING_ROLE=U:PASS\nEVALUATION_LAUNCH_AUTHORIZED=false')
    except Exception as e:
        save(a.output/'FAILURE.json',{'error':str(e),'automatic_retry':False});raise
if __name__=='__main__':main()
