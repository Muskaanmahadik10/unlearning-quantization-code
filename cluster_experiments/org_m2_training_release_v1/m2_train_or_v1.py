"""Complete fixed O/R schedules; verify final model in a fresh process.

U is deliberately rejected here: it needs accepted O lineage. No real benchmark
outcomes are inspected by this training job.
"""
import argparse
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_stage_common_v1 import *

def admission(role,output):
    release=dependencies();validate_plan(release)
    jid=os.environ.get('SLURM_ARRAY_JOB_ID','');require(jid.isdigit(),'array job identity')
    require(selected_role(os.environ.get('SLURM_ARRAY_TASK_ID',''))==role,'array role')
    require(os.environ.get('SLURM_JOB_NAME')=='org_m2_train_or' and os.environ.get('SLURM_JOB_PARTITION')=='dev','allocation identity')
    claim=RESULT/'launch_or_v1';request=read(claim/'request.json')
    validate_request(request)
    require(request['source_lock_sha256']==source(),'request source')
    for _ in range(100):
        if (claim/'submitted.json').exists() or (claim/'SUBMISSION_UNKNOWN.json').exists():break
        time.sleep(.2)
    require(read(claim/'submitted.json')['job_id']==jid,'unrecorded job')
    require(output.resolve()==(RESULT/('job_'+jid)/role).resolve(),'output identity')
    cpu=Path(request['cpu_receipt']);require(cpu.resolve().is_relative_to(RESULT.resolve()) and sha(cpu)==request['cpu_receipt_sha256'],'CPU admission drift')
    final=read(cpu);require(final['status']=='TRAINING_CPU_ADMISSION_PASS' and final['source_lock_sha256']==source(),'CPU admission failed')
    bind=cpu.parent/'DATA_BINDING.json';require(sha(bind)==final['data_binding_sha256'],'data binding drift')
    validate_binding(read(bind));require(sha(cpu.parent/'EVALUATION_TOKENS.json')==read(bind)['tokens_sha256'],'evaluation token drift')
    return request

def worker(a):
    request=admission(a.role,a.output)
    torch,np,values,release=setup(True)
    from m2_model_pilot_v1 import load_model,optimizer
    from m2_training_core_v1 import collate,update,lr_multiplier
    import m2_paged_checkpoint_v1 as ck
    from safetensors import safe_open
    from transformers import AutoTokenizer
    plan,base,binding,tokens,schedules,cal,design=values
    cfg=design['training'][a.role];sched=schedules[a.role]
    model=load_model(torch,base);base_hash=canonical(ck.model_rows(torch,model))
    identity={'role':a.role,'parent_role':'B','base_sha256':base_hash,'source_lock_sha256':source(),
      'pilot_source_sha256':release['pilot_source_sha256'],'schedule_sha256':sched['sha256'],
      'data_binding_sha256':read(Path(request['cpu_receipt']))['data_binding_sha256'],
      'generation':design['evaluation']['generation'],'scientific_fixed_schedule':True}
    if a.phase=='verify':
        produced=read(a.output/'produce.json');require(produced['identity']==identity,'reload identity')
        cp=a.output/'final_checkpoint';require(sha(cp/'MANIFEST.json')==produced['checkpoint_manifest_sha256'],'checkpoint manifest')
        verify(cp,read(cp/'MANIFEST.json')['files']);meta=read(cp/'metadata.json')
        require(meta['identity']==identity and meta['schema']=='m2_paged_resume_v1','checkpoint lineage')
        named=ck.named_tensors(model)
        with safe_open(str(cp/'model.safetensors'),framework='pt',device='cpu') as f:
            require(set(f.keys())==set(named),'reload key coverage')
            with torch.no_grad():
                for n,t in named.items():
                    s=f.get_tensor(n);require(s.dtype==t.dtype and s.shape==t.shape,'reload shape/dtype');t.copy_(s)
        require(ck.model_rows(torch,model)==meta['model_tensors'] and ck.aliases(model)==meta['aliases'],'fresh model reload not exact')
        require(meta['cursor']==sched['total_updates'] and set(meta['optimizer_steps'].values())=={sched['total_updates']},'final optimizer coverage')
        require(os.getpid()!=produced['pid'],'not a fresh process')
        # Serialized named buffers include rotary state and tied alias metadata.
        save(a.output/'reload.json',{'status':'FRESH_FINAL_MODEL_RELOAD_PASS','pid':os.getpid(),'identity':identity,
             'model_sha256':canonical(meta['model_tensors']),'checkpoint_manifest_sha256':sha(cp/'MANIFEST.json')})
        return
    opt=optimizer(model,cfg);tokenizer=AutoTokenizer.from_pretrained(base['snapshot_path'],local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    require(tokenizer.pad_token_id==128009,'tokenizer pad identity')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    records=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for ix,record in enumerate(sched['updates']):
        micros=[]
        for micro in record['microbatches']:
            require(set(micro)=={'anchor'},'unexpected supervised data kind')
            es=[tokens['by_exact_qa_sha256'][tokens['role_order'][a.role][i]] for i in micro['anchor']]
            micros.append({'anchor':{k:v.to('cuda:0') for k,v in collate(torch,es,tokenizer.pad_token_id).items()}})
        opt.param_groups[0]['lr']=cfg['learning_rate']*lr_multiplier(ix,sched['total_updates'],sched['warmup_updates'])
        r=update(torch,model,opt,micros,max_grad_norm=1.0)
        require(all(p.grad is not None for p in model.parameters()),'missing full-parameter gradient')
        opt.zero_grad(set_to_none=True)
        row={'update':ix,'epoch':record['epoch'],'lr':opt.param_groups[0]['lr'],**r};records.append(row)
        with (a.output/'updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if (ix+1)%25==0 or ix+1==sched['total_updates']:print('M2_TRAIN='+a.role+':'+str(ix+1)+'/'+str(sched['total_updates']),flush=True)
    torch.cuda.synchronize()
    require(len(records)==sched['total_updates'],'complete schedule')
    cp=a.output/'final_checkpoint';checkpoint=ck.save_checkpoint(torch,np,model,opt,cp,identity,len(records))
    meta=read(cp/'metadata.json');require(set(meta['optimizer_steps'].values())=={len(records)},'optimizer final steps')
    final_hash=canonical(meta['model_tensors']);require(final_hash!=base_hash,'student unchanged')
    save(a.output/'produce.json',{'status':'FIXED_SCHEDULE_TRAINING_COMPLETE','pid':os.getpid(),'identity':identity,
       'updates':len(records),'sample_count':sched['sample_count'],'updates_sha256':sha(a.output/'updates.jsonl'),
       'checkpoint_manifest_sha256':checkpoint,'model_sha256':final_hash,'elapsed_training_and_save_seconds':time.perf_counter()-start,
       'torch_allocator_peak_bytes':torch.cuda.max_memory_allocated(),'evaluated_model_outcomes':0})

def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['O','R'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['all','train','verify'],default='all');a=p.parse_args()
    admission(a.role,a.output)
    if a.phase!='all':worker(a);return
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        for phase in ('train','verify'):
            subprocess.run([sys.executable,'-I','-B',__file__,'--role',a.role,'--output',str(a.output),'--phase',phase],check=True)
        produced=read(a.output/'produce.json');reloaded=read(a.output/'reload.json')
        require(produced['model_sha256']==reloaded['model_sha256'],'model receipt drift')
        source()
        save(a.output/'final.json',{'status':'TRAINED_AND_RELOADED','role':a.role,'identity':produced['identity'],
          'produce_sha256':sha(a.output/'produce.json'),'reload_sha256':sha(a.output/'reload.json'),
          'checkpoint_manifest_sha256':produced['checkpoint_manifest_sha256'],'model_sha256':produced['model_sha256'],
          'scientific_efficacy_qualified':False,'U_launch_authorized':False,'evaluation_launch_authorized':False,'M3_authorized':False})
        print('M2_TRAINING_ROLE='+a.role+':PASS\nU_LAUNCH_AUTHORIZED=false',flush=True)
    except Exception as e:
        save(a.output/'FAILURE.json',{'error':str(e),'type':type(e).__name__,'automatic_retry':False});raise

if __name__=='__main__':main()
