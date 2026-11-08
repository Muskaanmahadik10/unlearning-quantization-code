"""Two-process full-parameter pilot. No scientific training or production CLI."""
import argparse
import datetime
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_pilot_contract_v1 import *


def setup(gpu):
    values=cluster_preflight()
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if not gpu:os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import numpy as np
    torch.set_num_threads(8 if gpu else 1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if gpu:
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'exactly one allocated GPU required')
        require('H200' in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported(),'H200 BF16 required')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_device(0)
    return torch,np,values


def load_model(torch,base):
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    m=AutoModelForCausalLM.from_pretrained(base['snapshot_path'],dtype=torch.bfloat16,
          attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,trust_remote_code=False)
    require(m.config.model_type=='llama' and m.config.tie_word_embeddings,'base architecture/tie drift')
    require(m.get_input_embeddings().weight is m.get_output_embeddings().weight,'tied alias lost')
    require(all(p.requires_grad and p.dtype==torch.bfloat16 and p.device==torch.device('cuda:0') for p in m.parameters()),'not full-parameter BF16 CUDA model')
    with safe_open(str(Path(base['snapshot_path'])/'model.safetensors'),framework='pt',device='cpu') as f:
        header={n:list(f.get_slice(n).get_shape()) for n in f.keys()}
    named={n:list(p.shape) for n,p in m.named_parameters()}
    require(named==header,'full tied parameter inventory differs from source weights')
    m.config.use_cache=False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    require(m.is_gradient_checkpointing,'gradient checkpointing not active')
    m.train();return m


def optimizer(model,cfg):
    from bitsandbytes.optim import PagedAdamW32bit
    opt=PagedAdamW32bit(model.parameters(),lr=cfg['learning_rate'],betas=(0.9,0.999),eps=1e-8,
                       weight_decay=cfg['weight_decay'],percentile_clipping=100)
    require(opt.is_paged and opt.args.optim_bits==32,'wrong optimizer')
    return opt


def rng_probe(torch,np):
    return {'python':[random.random() for _ in range(8)],'numpy':np.random.random(8).tolist(),
            'torch_cpu':torch.rand(8).tolist(),'torch_cuda':torch.rand(8,device='cuda:0').cpu().tolist()}


def worker(args):
    allocation_guard(args.role,args.output)
    torch,np,values=setup(True)
    plan,base,binding,tokens,schedules,cal,design=values
    require(args.role in ('R','U'),'invalid role')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_ARRAY_TASK_ID')==str(plan['tasks'].index(args.role)), 'Slurm task identity')
    expected=ROOT/'results/org_m2_model_pilot_v2'/('job_'+os.environ['SLURM_ARRAY_JOB_ID'])/args.role
    require(args.output.resolve()==expected.resolve(),'wrong output directory')
    from m2_training_core_v1 import collate,update,lr_multiplier,freeze_reference
    import m2_paged_checkpoint_v1 as ck
    from m2_pilot_evaluator_v1 import evaluate
    from transformers import AutoTokenizer
    cfg=design['training'][args.role];sched=schedules[args.role];indices=pilot_indices(args.role)
    model=load_model(torch,base);base_rows=ck.model_rows(torch,model);base_sha=canonical(base_rows)
    teacher=freeze_reference(load_model(torch,base)) if args.role=='U' else None
    teacher_initial=canonical(ck.model_rows(torch,teacher)) if teacher is not None else None
    tokenizer=AutoTokenizer.from_pretrained(base['snapshot_path'],local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    require(tokenizer.pad_token_id==128009,'tokenizer padding drift')
    opt=optimizer(model,cfg)
    identity={'source_lock_sha256':source(),'role':args.role,'base_lock_sha256':sha(PKG/'BASE_INPUT_IDENTITY_LOCK.json'),
              'schedule_sha256':sched['sha256'],'selected_update_indices':indices,'pilot_only':True,
              'teacher_role':'B_engineering_surrogate_not_scientific_O' if teacher is not None else None,
              'checkpointing':'non_reentrant','gradient_clip':1.0,'optimizer_group_policy':'all_parameters_weight_decay_0.01'}
    gpu={'name':torch.cuda.get_device_name(0),'uuid':str(torch.cuda.get_device_properties(0).uuid),
         'torch':torch.__version__,'cuda':torch.version.cuda}
    require(all(not p.requires_grad and p.grad is None for p in teacher.parameters()) if teacher else True,'teacher not frozen')
    checkpoint=args.output/'checkpoint_after_two'
    results=[];start=0
    if args.phase=='resume':
        producer=read(args.output/'produce.json')
        # Derive current LR from the pinned schedule, never trust saved group LR.
        opt.param_groups[0]['lr']=cfg['learning_rate']*lr_multiplier(indices[1],sched['total_updates'],sched['warmup_updates'])
        ck.load_checkpoint(torch,np,model,opt,checkpoint,producer['checkpoint_manifest_sha256'],identity,2)
        start=2
    else:
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.cuda.reset_peak_memory_stats();training_start=time.perf_counter();checkpoint_sha=None
    for cursor in range(start,len(indices)):
        ix=indices[cursor];record=sched['updates'][ix];micros=[]
        for micro in record['microbatches']:
            batch={}
            for kind,ids in micro.items():
                order=tokens['role_order'][args.role if kind=='anchor' else 'R']
                entries=[tokens['by_exact_qa_sha256'][order[i]] for i in ids]
                batch[kind]={k:v.to('cuda:0') for k,v in collate(torch,entries,tokenizer.pad_token_id).items()}
            micros.append(batch)
        opt.param_groups[0]['lr']=cfg['learning_rate']*lr_multiplier(ix,sched['total_updates'],sched['warmup_updates'])
        torch.cuda.synchronize();t=time.perf_counter()
        kwargs={'beta':cfg['beta'],'alpha':cfg['alpha_retain'],'gamma':cfg['gamma_forget']} if teacher else {}
        r=update(torch,model,opt,micros,reference=teacher,max_grad_norm=1.0,**kwargs)
        require(all(p.grad is not None for p in model.parameters()),'trainable parameter lacks gradient')
        require(all(p.grad is None for p in teacher.parameters()) if teacher else True,'teacher gradient appeared')
        opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
        row={'source_update':ix,'cursor':cursor,'lr':opt.param_groups[0]['lr'],**r}
        results.append(row)
        print('M2_PILOT_UPDATE='+args.role+':'+args.phase+':'+str(cursor)+':PASS seconds='+str(round(time.perf_counter()-t,3)),flush=True)
        save(args.output/(args.phase+'_update_'+str(cursor)+'.json'),{**row,'seconds':time.perf_counter()-t})
        if args.phase=='produce' and cursor==1:
            checkpoint_sha=ck.save_checkpoint(torch,np,model,opt,checkpoint,identity,2)
    torch.cuda.synchronize();training_seconds=time.perf_counter()-training_start
    model_rows=ck.model_rows(torch,model);optimizer_rows,steps=ck.check_optimizer(torch,model,opt)
    require(set(steps.values())=={4},'pilot optimizer step coverage')
    teacher_final=canonical(ck.model_rows(torch,teacher)) if teacher else None
    require(teacher_initial==teacher_final,'teacher bytes changed')
    probe=rng_probe(torch,np)
    evaluation=evaluate(torch,model,tokenizer,cal['records'][:3],design['evaluation']['generation'])
    final_model_sha=canonical(model_rows)
    require(canonical(ck.model_rows(torch,model))==final_model_sha,'evaluation mutated model tensors')
    info={'status':'PILOT_BRANCH_PASS','pid':os.getpid(),'role':args.role,'phase':args.phase,
          'identity':identity,'gpu':gpu,'model_tensors':model_rows,'optimizer_tensors':optimizer_rows,
          'optimizer_steps':steps,'optimizer_groups':ck.group_meta(opt),'aliases':ck.aliases(model),
          'buffer_schema':{n:{'shape':list(b.shape),'dtype':str(b.dtype)} for n,b in model.named_buffers()},
          'rng_probe':probe,'selected_update_indices':indices,'update_results':results[-2:],
          'base_model_sha256':base_sha,'final_model_sha256':final_model_sha,
          'teacher_initial_sha256':teacher_initial,'teacher_final_sha256':teacher_final,
          'checkpoint_manifest_sha256':checkpoint_sha,'training_seconds':training_seconds,
          'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'evaluation':evaluation,
          'host_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'checkpoint_disk_bytes':sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file()),
          'production_authorized':False,'scientific_O_R_U_created':False}
    save(args.output/(args.phase+'.json'),info)
    print('M2_PILOT_BRANCH='+args.role+':'+args.phase+':PASS',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--role',choices=['R','U'],required=True)
    ap.add_argument('--phase',choices=['all','produce','resume'],default='all')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.phase!='all':worker(args);return
    allocation_guard(args.role,args.output)
    require(args.output.resolve().is_relative_to((ROOT/'results/org_m2_model_pilot_v2').resolve()),'output escape')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    try:
        source()
        for phase in ('produce','resume'):
            subprocess.run([sys.executable,'-I','-B',__file__,'--role',args.role,'--phase',phase,
                            '--output',str(args.output)],check=True)
        a,b=read(args.output/'produce.json'),read(args.output/'resume.json')
        outcome_compare(a,b)
        ea,eb=a['evaluation'],b['evaluation']
        require(ea['generation']==eb['generation'],'same-artifact generation/ROUGE drift')
        require(len(ea['scores'])==len(eb['scores'])==3,'evaluation coverage')
        for x,y in zip(ea['scores'],eb['scores']):
            require(x['qa_sha256']==y['qa_sha256'] and all(abs(x[k]-y[k])<=1e-6 for k in ('prob','avg_loss')),'same-artifact FP32 score drift')
        source()
        save(args.output/'final.json',{'status':'FULL_MODEL_PILOT_PASS','role':args.role,
              'produce_sha256':sha(args.output/'produce.json'),'resume_sha256':sha(args.output/'resume.json'),
              'native_paged_resume_exact':True,'calibration_evaluator_primitives_pass':True,
              'full_benchmark_evaluator_certified':False,'scientific_training_authorized':False,
              'production_authorized':False,'m3_authorized':False,'elapsed_seconds':time.perf_counter()-start})
        print('M2_FULL_MODEL_PILOT='+args.role+':PASS\nPRODUCTION_AUTHORIZED=false',flush=True)
    except Exception as e:
        save(args.output/'FAILURE.json',{'status':'FAIL','exception_type':type(e).__name__,
             'error':str(e),'traceback':traceback.format_exc(),'production_authorized':False})
        raise


if __name__=='__main__':main()
