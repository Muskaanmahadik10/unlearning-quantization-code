#!/usr/bin/env python3
"""M4 small controlled manufacturing knowledge revision experiment.
Commands: train, eval, gate, report. Runs in fresh Python process per model.
Requires CUDA and pinned Transformers/PEFT/bitsandbytes environment for model commands.
All reported findings derived from actual exported predictions, never generated in advance.
"""
import argparse, collections, csv, hashlib, json, math, os, random, sys, time, shutil
from pathlib import Path

PIPELINE_VERSION='M4_v2_accelerated_20261101'
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION=os.environ.get('M4_MODEL_REVISION')
def model_rev_kw():return {'revision':MODEL_REVISION} if MODEL_REVISION else {}
PROJECTIONS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
DEFAULT_SEEDS=(20261101,20261102,20261103)
LORA_R=16; LORA_ALPHA=32; MAX_TOKENS=512

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)

def seed_everything(seed,torch):
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def setup_model_libs():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required for train/eval. Change Colab Runtime -> GPU.')
    return torch,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,LoraConfig,TaskType,get_peft_model,PeftModel

def get_chat_ids(tok,question,answer):
    user=[{'role':'user','content':question}]
    prefix=tok.apply_chat_template(user,tokenize=True,add_generation_prompt=True)
    full=tok.apply_chat_template(user+[{'role':'assistant','content':answer}],tokenize=True,add_generation_prompt=False)
    if full[:len(prefix)]!=prefix:
        raise RuntimeError('Tokenizer assistant prefix mismatch; stop rather than masking answer incorrectly')
    if len(full)>MAX_TOKENS:
        raise RuntimeError(f'Example exceeds MAX_TOKENS={MAX_TOKENS}: {question[:100]}')
    if len(full)<=len(prefix):raise RuntimeError('No assistant answer tokens')
    return {'input_ids':full, 'labels':[-100]*len(prefix)+full[len(prefix):]}

def make_batch(tok,encoded,torch):
    if tok.pad_token_id is None: raise RuntimeError('Tokenizer has no pad_token_id')
    length=max(len(x['input_ids']) for x in encoded)
    a=[];m=[];l=[]
    for item in encoded:
        pad=length-len(item['input_ids'])
        a.append(item['input_ids']+[tok.pad_token_id]*pad)
        m.append([1]*len(item['input_ids'])+[0]*pad)
        l.append(item['labels']+[-100]*pad)
    return {n:torch.tensor(v,device='cuda',dtype=torch.long) for n,v in
      [('input_ids',a),('attention_mask',m),('labels',l)]}

def loss_per_seq(model,batch,torch):
    import torch.nn.functional as F
    result=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
    logits=result.logits[:,:-1,:].float()
    target=batch['labels'][:,1:]
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape(target.shape)
    valid=(target!=-100);den=valid.sum(dim=1).clamp(min=1)
    return loss.sum(dim=1),den

def mean_loss(model,batch,torch):
    s,c=loss_per_seq(model,batch,torch)
    return s.sum()/c.sum()

def make_base(AutoModel,torch):
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModel.from_pretrained(MODEL_ID,torch_dtype=dtype,low_cpu_mem_usage=True,**model_rev_kw())
    model.to('cuda');model.config.use_cache=False
    return model

def deterministic_equalized_rows(args,role):
    ret=read_jsonl(args.data/'retain_train.jsonl')
    old=read_jsonl(args.data/'obsolete_train.jsonl')
    if len(ret)!=240 or len(old)!=40:raise RuntimeError('Invalid locked data sizes')
    if role=='O':return ret+old*args.obsolete_repeat
    if role=='R':
        # Same number of examples/steps as O; repeated retained records, no obsolete examples.
        rng=random.Random(args.seed+77)
        return ret+[rng.choice(ret) for _ in range(len(old)*args.obsolete_repeat)]
    if role=='C':return ret
    raise ValueError(role)

def train(args):
    torch,AutoModel,AutoTokenizer,_,LoraConfig,TaskType,get_peft_model,PeftModel=setup_model_libs()
    seed_everything(args.seed,torch)
    model_out=args.out/'checkpoints'/str(args.seed)/args.role
    marker=model_out/'TRAINED.json'
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf-8'))
        expected={'role':args.role,'seed':args.seed,'epochs':args.epochs,'learning_rate':args.lr,
                  'batch_size':args.batch,'model_revision':MODEL_REVISION,'obsolete_repeat':args.obsolete_repeat,'pipeline_version':PIPELINE_VERSION,'save_merged':args.save_merged,
                  'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()}
        for key,val in expected.items():
            if m.get(key)!=val:raise RuntimeError(f'Existing training checkpoint conflicts with requested {key}: {m.get(key)} != {val}; create a new experiment directory')
        print('SKIPPING exact-match completed model',marker);return
    tok=AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True,**model_rev_kw())
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=make_base(AutoModel,torch)
    if args.role in ('O','R'):
        config=LoraConfig(r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=0.05,target_modules=list(PROJECTIONS),bias='none',task_type=TaskType.CAUSAL_LM)
        model=get_peft_model(base,config)
    else:
        predecessor=args.out/'checkpoints'/str(args.seed)/'O'/'adapter'
        if not (predecessor/'adapter_config.json').exists():raise RuntimeError('O adapter missing; train O first')
        model=PeftModel.from_pretrained(base,str(predecessor),is_trainable=True)
    refs=None
    if args.role=='U':
        refbase=make_base(AutoModel,torch)
        refs=PeftModel.from_pretrained(refbase,str(args.out/'checkpoints'/str(args.seed)/'O'/'adapter'),is_trainable=False)
        refs.eval()
        for p in refs.parameters():p.requires_grad_(False)
    model.train(); model.config.use_cache=False
    active=[p for p in model.parameters() if p.requires_grad]
    if not active:raise RuntimeError('No trainable LoRA parameters')
    optimizer=torch.optim.AdamW(active,lr=args.lr,weight_decay=0.01)
    log=[];rng=random.Random(args.seed+101)
    if args.role in ('O','R'):
        train_rows=deterministic_equalized_rows(args,args.role)
        encoded=[get_chat_ids(tok,r['question'],r['answer']) for r in train_rows]
        steps=math.ceil(len(encoded)/args.batch)*args.epochs
        step=0
        for epoch in range(args.epochs):
            order=list(range(len(encoded)));rng.shuffle(order)
            for ix in range(0,len(order),args.batch):
                b=make_batch(tok,[encoded[j] for j in order[ix:ix+args.batch]],torch)
                optimizer.zero_grad(set_to_none=True)
                loss=mean_loss(model,b,torch)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite supervised loss at step {step}')
                loss.backward();torch.nn.utils.clip_grad_norm_(active,1.0);optimizer.step()
                step+=1;log.append({'step':step,'epoch':epoch,'loss':float(loss.detach())})
            print(args.role,args.seed,'epoch',epoch+1,'loss',round(log[-1]['loss'],4),flush=True)
    else:
        old_rows=read_jsonl(args.data/'obsolete_train.jsonl')
        ret_rows=read_jsonl(args.data/'retain_train.jsonl')
        old_enc=[get_chat_ids(tok,z['question'],z['answer']) for z in old_rows]
        ret_enc=[get_chat_ids(tok,z['question'],z['answer']) for z in ret_rows]
        step=0
        for epoch in range(args.epochs):
            order=list(range(len(old_enc)));rng.shuffle(order)
            for ix in range(0,len(order),args.batch):
                f=make_batch(tok,[old_enc[j] for j in order[ix:ix+args.batch]],torch)
                r=make_batch(tok,[ret_enc[rng.randrange(len(ret_enc))] for _ in order[ix:ix+args.batch]],torch)
                optimizer.zero_grad(set_to_none=True)
                ret_loss=mean_loss(model,r,torch)
                if args.role=='U':
                    with torch.no_grad():ref_s,_=loss_per_seq(refs,f,torch)
                    cur_s,_=loss_per_seq(model,f,torch)
                    # NPO: -(2/beta)*log sigmoid(beta*(NLL_U - NLL_O)) plus retain CE.
                    import torch.nn.functional as F
                    npo=-(2.0/args.beta)*F.logsigmoid(args.beta*(cur_s-ref_s)).mean()
                    loss=npo+args.retain_weight*ret_loss
                else:
                    loss=ret_loss
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {args.role} loss at {step}')
                loss.backward();torch.nn.utils.clip_grad_norm_(active,1.0);optimizer.step()
                step+=1;log.append({'step':step,'epoch':epoch,'loss':float(loss.detach()),'retain_loss':float(ret_loss.detach())})
            print(args.role,args.seed,'epoch',epoch+1,'loss',round(log[-1]['loss'],4),flush=True)
    # On adapter first; full-precision merged weight second. Never quantize an unmerged LoRA adapter.
    adapter_dir=model_out/'adapter'; adapter_dir.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(adapter_dir),safe_serialization=True)
    model.eval()
    # Save only compact LoRA adapters to Drive by default; reconstruct dense merged checkpoints
    # locally on first evaluation. Prevents >12 GB of duplicate merged weights filling Drive.
    if args.save_merged:
        merged=model.merge_and_unload(safe_merge=True)
        merged_dir=model_out/'merged';merged_dir.mkdir(parents=True,exist_ok=True)
        merged.save_pretrained(str(merged_dir),safe_serialization=True,max_shard_size='2GB')
        tok.save_pretrained(str(merged_dir))
    with (model_out/'training_log.csv').open('w',newline='',encoding='utf-8') as w:
        wr=csv.DictWriter(w,fieldnames=sorted(set(k for z in log for k in z)));wr.writeheader();wr.writerows(log)
    atomic_json(marker,{'role':args.role,'seed':args.seed,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'lora_r':LORA_R,'lora_alpha':LORA_ALPHA,
         'projection_families':list(PROJECTIONS),'epochs':args.epochs,'learning_rate':args.lr,
         'steps':len(log),'batch_size':args.batch,'obsolete_repeat':args.obsolete_repeat,'pipeline_version':PIPELINE_VERSION,'save_merged':args.save_merged,'max_tokens':MAX_TOKENS,'torch':torch.__version__,'device':torch.cuda.get_device_name(0),
         'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()})
    print('TRAIN COMPLETE',model_out,flush=True)

def output_text_match(output,answer):
    import re
    normalize=lambda x:re.sub(r'\s+','',x).replace('，',',').replace('。','').strip().lower()
    return int(bool(normalize(answer)) and normalize(answer) in normalize(output))

def answer_nll(model,tok,question,answer,torch):
    enc=get_chat_ids(tok,question,answer)
    with torch.inference_mode():
        b=make_batch(tok,[enc],torch)
        s,c=loss_per_seq(model,b,torch)
    return float((s/c).item())

def batched_answer_nll(model,tok,questions,answers,torch,chunk_size=8):
    """Answer-only token average cross entropy. Same estimand as answer_nll, now batched."""
    results=[]
    for i in range(0,len(questions),chunk_size):
        encoded=[get_chat_ids(tok,q,a) for q,a in zip(questions[i:i+chunk_size],answers[i:i+chunk_size])]
        inp=make_batch(tok,encoded,torch)
        with torch.inference_mode():
            sums,counts=loss_per_seq(model,inp,torch)
        results.extend((sums/counts).detach().cpu().tolist())
    return [float(x) for x in results]

def locate_model_for_eval(args):
    persistent=args.out/'checkpoints'/str(args.seed)/args.role/'merged'
    cache_root=os.environ.get('M4_LOCAL_MODEL_CACHE','')
    if not cache_root:
        if (persistent/'config.json').exists():return persistent
        raise RuntimeError('M4_LOCAL_MODEL_CACHE must be set for adapter-only run. Local merged checkpoint required.')
    local=Path(cache_root)/str(args.seed)/args.role/'merged'
    adapter=args.out/'checkpoints'/str(args.seed)/args.role/'adapter'
    if not (adapter/'adapter_config.json').exists():raise RuntimeError('Missing saved adapter: '+str(adapter))
    if not (args.out/'checkpoints'/str(args.seed)/args.role/'TRAINED.json').exists():
        raise RuntimeError('Missing completed training marker; cannot evaluate partially trained model')
    if (local/'M4_MERGED_DONE.json').exists():
        m=json.loads((local/'M4_MERGED_DONE.json').read_text())
        marker_hash=hashlib.sha256((args.out/'checkpoints'/str(args.seed)/args.role/'TRAINED.json').read_bytes()).hexdigest()
        if m['training_marker_sha256']!=marker_hash:raise RuntimeError('Local model cache does not match training marker')
        if not (local/'config.json').exists():raise RuntimeError('Corrupted local merged model')
        return local
    local.parent.mkdir(parents=True,exist_ok=True)
    tmp=local.with_name('merged_copying')
    if tmp.exists():shutil.rmtree(tmp)
    if local.exists():shutil.rmtree(local)
    tmp.mkdir(parents=True)
    if (persistent/'config.json').exists():
        # Reuse an intentionally saved full-precision checkpoint if present.
        shutil.copytree(persistent,tmp,dirs_exist_ok=True)
    else:
        # Reconstruct merged weights from pinned base model and compact saved LoRA adapter.
        torch,AutoModel,AutoTokenizer,_,_,_,_,PeftModel=setup_model_libs()
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True,**model_rev_kw())
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        base=make_base(AutoModel,torch)
        adapted=PeftModel.from_pretrained(base,str(adapter),is_trainable=False).eval()
        merged=adapted.merge_and_unload(safe_merge=True)
        merged.save_pretrained(str(tmp),safe_serialization=True,max_shard_size='2GB')
        tokenizer.save_pretrained(str(tmp))
        del merged,adapted,base
        import gc
        gc.collect();torch.cuda.empty_cache()
    marker_hash=hashlib.sha256((args.out/'checkpoints'/str(args.seed)/args.role/'TRAINED.json').read_bytes()).hexdigest()
    atomic_json(tmp/'M4_MERGED_DONE.json',{'training_marker_sha256':marker_hash,'source':'pinned-base+LoRA-merge','seed':args.seed,'role':args.role})
    tmp.replace(local)
    return local

def evaluate(args):
    torch,AutoModel,AutoTokenizer,BitsAndBytesConfig,*_=setup_model_libs()
    seed_everything(args.seed,torch)
    dir_model=locate_model_for_eval(args)
    resultdir=args.out/'evaluations'/str(args.seed)/args.split
    resultdir.mkdir(parents=True,exist_ok=True)
    output=resultdir/f'{args.role}_{args.precision}.jsonl'
    meta=resultdir/f'{args.role}_{args.precision}_meta.json'
    expected={'role':args.role,'seed':args.seed,'precision':args.precision,'split':args.split,
              'limit_per_partition':args.limit,'max_new_tokens':args.max_new_tokens,
              'gen_batch':args.gen_batch,'pipeline_version':PIPELINE_VERSION}
    if output.exists() and meta.exists():
        m=json.loads(meta.read_text(encoding='utf-8'))
        for key,val in expected.items():
            if m.get(key)!=val:raise RuntimeError(f'Existing evaluation conflicts with {key}: {m.get(key)} != {val}; do not overwrite a previous result')
        print('SKIPPING exact-match completed evaluation',output);return
    tok=AutoTokenizer.from_pretrained(str(dir_model),use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='left'
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config=None
    if args.precision=='nf4':
        config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=False)
    elif args.precision=='int8':config=BitsAndBytesConfig(load_in_8bit=True)
    kws={'torch_dtype':dtype,'device_map':{'':'cuda'}}
    if config:kws['quantization_config']=config
    model=AutoModel.from_pretrained(str(dir_model),**kws).eval()
    model.config.use_cache=True
    import bitsandbytes as bnb
    expected_modules=24*7
    active=[(n,m) for n,m in model.named_modules() if n.endswith(PROJECTIONS)]
    count4=sum(isinstance(m,bnb.nn.Linear4bit) for n,m in active)
    count8=sum(isinstance(m,bnb.nn.Linear8bitLt) for n,m in active)
    if len(active)!=expected_modules:raise RuntimeError(f'Expected {expected_modules} projection modules; found {len(active)}')
    if args.precision=='nf4' and count4!=expected_modules:raise RuntimeError(f'Expected {expected_modules} packed NF4 layers; found {count4}')
    if args.precision=='int8' and count8!=expected_modules:raise RuntimeError(f'Expected {expected_modules} bitsandbytes INT8 layers; found {count8}')
    if args.precision=='bf16' and (count4 or count8):raise RuntimeError('Unquantized reference has quantized modules')
    if args.split=='dev':
        records=read_jsonl(args.data/'obsolete_dev.jsonl')+read_jsonl(args.data/'retain_dev.jsonl')
    else:
        records=read_jsonl(args.data/'obsolete_test.jsonl')+read_jsonl(args.data/'retain_test.jsonl')+read_jsonl(args.data/'unseen_test.jsonl')
    if args.partitions:
        required=set(args.partitions.split(','))
        if not required.issubset({'obsolete','retain','unseen'}):raise RuntimeError('Invalid partitions')
        records=[r for r in records if r['partition'] in required]
    if args.limit>0:
        grouped=collections.defaultdict(list)
        for r in records:grouped[r['partition']].append(r)
        records=[x for key in sorted(grouped) for x in grouped[key][:args.limit]]
    partial=output.with_suffix('.partial.jsonl')
    rows=[]
    if partial.exists():
        rows=read_jsonl(partial)
        keys=[r['fact_id'] for r in rows]
        if keys != [r['fact_id'] for r in records[:len(rows)]]:
            raise RuntimeError('Partial evaluation record order or dataset changed. Must not resume incompatible state.')
        print('RESUMING',len(rows),'of',len(records),'from',partial,flush=True)
    old_map={z['id']:z for z in read_jsonl(args.data/'synthetic_obsolete_facts.jsonl')}
    # Append one atomic batch at a time: Colab interruptions do not erase completed examples.
    for i in range(len(rows),len(records),args.gen_batch):
        chunk=records[i:i+args.gen_batch]
        prompts=[tok.apply_chat_template([{'role':'user','content':r['question']}],tokenize=False,add_generation_prompt=True) for r in chunk]
        inp=tok(prompts,return_tensors='pt',padding=True,truncation=False).to('cuda')
        if inp['input_ids'].shape[1]>MAX_TOKENS:raise RuntimeError('Evaluation prompt exceeds maximum context')
        with torch.inference_mode():
            gen=model.generate(**inp,max_new_tokens=args.max_new_tokens,do_sample=False,
                 pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        decoded=tok.batch_decode(gen[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
        q=[r['question'] for r in chunk]
        gold=[r['answer'] for r in chunk]
        nll=batched_answer_nll(model,tok,q,gold,torch,min(args.nll_batch,8))
        alts=[(j,old_map[r['fact_id']]['current_answer']) for j,r in enumerate(chunk) if r['partition']=='obsolete']
        alt_nll=batched_answer_nll(model,tok,[q[j] for j,_ in alts],[a for _,a in alts],torch,min(args.nll_batch,8)) if alts else []
        alt_map={j:x for (j,_),x in zip(alts,alt_nll)}
        for j,(rec,response) in enumerate(zip(chunk,decoded)):
            z={'fact_id':rec['fact_id'],'source_fact_id':rec['source_fact_id'],
               'partition':rec['partition'],'seed':args.seed,'role':args.role,'precision':args.precision,
               'question':rec['question'],'gold':rec['answer'],'response':response,
               'exact_containment':output_text_match(response,rec['answer']),
               'answer_nll':nll[j]}
            if rec['partition']=='obsolete':
                z['current_nll_same_prompt']=alt_map[j]
                z['legacy_minus_current_nll']=nll[j]-alt_map[j]
            rows.append(z)
        transaction=partial.with_suffix('.partial-write.tmp')
        with transaction.open('w',encoding='utf-8') as fp:
            for z in rows:fp.write(json.dumps(z,ensure_ascii=False)+'\n')
        transaction.replace(partial)
        if (i//args.gen_batch)%6==0 or len(rows)==len(records):
            print('eval',args.seed,args.split,args.role,args.precision,len(rows),'/',len(records),flush=True)
    if len(rows)!=len(records):raise RuntimeError('Incomplete evaluation')
    partial.replace(output)
    atomic_json(meta,{**expected,'rows':len(rows),'full_evaluation':args.limit==0 and not args.partitions,
      'partitions':args.partitions,'quantized_modules_nf4':count4,'quantized_modules_int8':count8,
      'projection_module_count':len(active),'device':torch.cuda.get_device_name(0),'dtype':str(dtype),
      'model_weights':str(dir_model),'backend_class_sample':{n:m.__class__.__name__ for n,m in active[:3]},
      'torch':torch.__version__})
    print('EVALUATION COMPLETE',output,flush=True)

def gate(args):
    path=args.out/'evaluations'/str(args.seed)/'dev'
    def get(role):
        p=path/f'{role}_bf16.jsonl'; m=path/f'{role}_bf16_meta.json'
        if not p.exists():raise RuntimeError('Run BF16 dev evaluation for O and R first')
        if json.loads(m.read_text())['limit_per_partition']!=0:raise RuntimeError('Development gate must use full development set')
        return [z for z in read_jsonl(p) if z['partition']=='obsolete']
    o=get('O');r=get('R')
    ro={x['fact_id']:x for x in r}
    if set(ro)!={x['fact_id'] for x in o}:raise RuntimeError('O/R fact mismatch')
    F_o=sum(x['exact_containment'] for x in o)/len(o)
    F_r=sum(ro[x['fact_id']]['exact_containment'] for x in o)/len(o)
    margin=F_o-F_r
    status=F_o>=args.min_original and margin>=args.min_separation
    result={'seed':args.seed,'F_O_dev':F_o,'F_R_dev':F_r,'O_minus_R_dev':margin,
            'min_original':args.min_original,'min_separation':args.min_separation,
            'identified':status,'interpretation':'Development-only prerequisite. Not a sealed final-test result.'}
    atomic_json(args.out/'evaluations'/str(args.seed)/'development_gate.json',result)
    print(json.dumps(result,indent=2))
    if not status and args.enforce:raise RuntimeError('Original not distinguishable from retain-only reference; adjust O training using DEV only before final test.')

def report(args):
    import statistics
    folder=args.out/'evaluations'; records=[]
    for p in sorted(folder.glob('*/test/*_*.jsonl')):
        meta=p.with_name(p.stem+'_meta.json')
        if not meta.exists():continue
        m=json.loads(meta.read_text())
        if not m['full_evaluation']:continue
        records+=read_jsonl(p)
    if not records:raise RuntimeError('No FULL sealed test outputs. Pilot results are intentionally excluded.')
    groups=collections.defaultdict(list)
    for r in records:groups[(r['seed'],r['role'],r['precision'],r['partition'])].append(r)
    seeds=sorted(set(r['seed'] for r in records))
    if len(seeds)<args.min_seeds:
        raise RuntimeError(f'{len(seeds)} full seeds present, {args.min_seeds} required. Do not describe pilot as publication-ready.')
    required_n={'obsolete':40,'retain':240,'unseen':80}
    for sd in seeds:
        for role in ('O','R','U','C'):
            for prec in ('bf16','int8','nf4'):
                for part,n in required_n.items():
                    got=len(groups.get((sd,role,prec,part),[]))
                    if got!=n:raise RuntimeError(f'Incomplete final results seed={sd},role={role},precision={prec},partition={part}: {got}/{n}')
    aggregated=[]
    for (seed,role,prec,part),rr in sorted(groups.items()):
        aggregated.append({'seed':seed,'role':role,'precision':prec,'partition':part,'n_facts':len(rr),
            'exact_containment_rate':sum(r['exact_containment'] for r in rr)/len(rr),
            'mean_answer_nll':statistics.fmean(r['answer_nll'] for r in rr),
            'mean_legacy_minus_current_nll':statistics.fmean(r['legacy_minus_current_nll'] for r in rr) if part=='obsolete' else '',
            'note':'Simulated withdrawn revision' if part=='obsolete' else 'Source-grounded'})
    output=args.out/'reports';output.mkdir(parents=True,exist_ok=True)
    with (output/'M4_group_summaries.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregated[0]));w.writeheader();w.writerows(aggregated)
    paired=[]
    keyed={(r['seed'],r['role'],r['partition'],r['fact_id'],r['precision']):r for r in records}
    for (seed,role,part,fid,prec),a in keyed.items():
        if prec!='nf4':continue
        bf=keyed.get((seed,role,part,fid,'bf16'))
        if bf:
            paired.append({'seed':seed,'role':role,'partition':part,'fact_id':fid,
              'nf4_minus_bf16_answer_recovery':a['exact_containment']-bf['exact_containment'],
              'nf4_minus_bf16_answer_nll':a['answer_nll']-bf['answer_nll'],
              'nf4_minus_bf16_legacy_margin':a['legacy_minus_current_nll']-bf['legacy_minus_current_nll'] if part=='obsolete' else ''})
    with (output/'M4_paired_NF4_vs_BF16.csv').open('w',newline='',encoding='utf-8') as f:
        if paired:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    atomic_json(output/'M4_status.json',{'full_test_record_count':len(records),'full_group_count':len(aggregated),
       'tested_seeds':seeds,'minimum_required_seeds':args.min_seeds,'research_status':'three-seed_complete' if len(seeds)>=3 else 'pilot_only',
       'scientific_admission_status': 'ALL_BASELINES_PASSED' if all((args.out/'evaluations'/str(sd)/'development_gate.json').exists() and json.loads((args.out/'evaluations'/str(sd)/'development_gate.json').read_text()).get('identified') for sd in seeds) else 'AT_LEAST_ONE_BASELINE_FAILED_OR_UNVERIFIED',
       'development_admission_by_seed':{str(sd):json.loads((args.out/'evaluations'/str(sd)/'development_gate.json').read_text()) if (args.out/'evaluations'/str(sd)/'development_gate.json').exists() else 'NOT_RECORDED' for sd in seeds},
       'scientific_cautions':['Open-generation containment is a strict automatic metric and does not prove weight erasure.',
       'This is a source-derived synthetic-revision case study, not a deployed factory system.',
       'Independent relationship holdout is limited to the five selected workpieces, not all process information.',
       'Report all original-reference baseline failures and unlearning retained-utility tradeoffs.']})
    # Paper-ready candidate data table: arithmetic over independent seed-level condition means.
    # Retain the detailed per-seed tables; do not pool prompts and pretend they are independent seeds.
    table=[]
    for role in ('O','R','U','C'):
        for precision in ('bf16','int8','nf4'):
            line={'role':role,'precision':precision,'number_seeds':len(seeds)}
            for part,column in [('obsolete','obsolete_generation_rate'),('retain','retained_generation_accuracy'),('unseen','unseen_relation_accuracy')]:
                vals=[next(r['exact_containment_rate'] for r in aggregated if
                   (r['seed'],r['role'],r['precision'],r['partition'])==(sd,role,precision,part)) for sd in seeds]
                line[column+'_mean']=statistics.fmean(vals)
                line[column+'_sd']=statistics.stdev(vals) if len(vals)>1 else 0
            table.append(line)
    with (output/'M4_Table16_candidate.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    paired_u=[]
    for sd in seeds:
        for part in ('obsolete','retain','unseen'):
            pairs=[x for x in paired if x['role']=='U' and x['seed']==sd and x['partition']==part]
            if len(pairs)!={'obsolete':40,'retain':240,'unseen':80}[part]:raise RuntimeError('Missing U paired facts')
            paired_u.append({'seed':sd,'partition':part,
                  'NF4_minus_BF16_generation_rate':statistics.fmean(x['nf4_minus_bf16_answer_recovery'] for x in pairs),
                  'NF4_minus_BF16_answer_NLL':statistics.fmean(x['nf4_minus_bf16_answer_nll'] for x in pairs)})
    with (output/'M4_primary_seed_differences.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(paired_u[0]));w.writeheader();w.writerows(paired_u)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
        for ax,(part,column,heading) in zip(axes,[('obsolete','obsolete_generation_rate','Withdrawn-answer recovery'),
                                                    ('retain','retained_generation_accuracy','Retained manufacturing QA')]):
            ps=['bf16','int8','nf4']
            for role in ('O','R','U','C'):
                vals=[next(x[column+'_mean'] for x in table if x['role']==role and x['precision']==p) for p in ps]
                sdvals=[next(x[column+'_sd'] for x in table if x['role']==role and x['precision']==p) for p in ps]
                ax.errorbar(ps,vals,yerr=sdvals,fmt='o-',capsize=3,label=role)
            ax.set_title(heading);ax.set_ylim(-0.03,1.03);ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.set_xlabel('Execution representation');ax.grid(alpha=.2)
        axes[0].set_ylabel('Generated exact-answer containment')
        axes[1].legend(title='Model state',loc='best')
        fig.suptitle('M4 manufacturing knowledge revision: seed means ± SD')
        fig.savefig(output/'M4_Figure10_candidate.png',dpi=220)
        fig.savefig(output/'M4_Figure10_candidate.pdf')
        plt.close(fig)
    except ImportError:
        print('Matplotlib not installed: table files produced; figure skipped')
    print('Saved',output,'full_groups',len(aggregated),'records',len(records))

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('cmd',choices=['train','eval','gate','report'])
    ap.add_argument('--data',type=Path,default=Path('M4_CP2_Data'))
    ap.add_argument('--out',type=Path,default=Path('M4_Results'))
    ap.add_argument('--seed',type=int,default=20261101)
    ap.add_argument('--role',choices=['O','R','U','C'],default='O')
    ap.add_argument('--precision',choices=['bf16','int8','nf4'],default='bf16')
    ap.add_argument('--split',choices=['dev','test'],default='dev')
    ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--lr',type=float,default=0.0001)
    ap.add_argument('--beta',type=float,default=0.1)
    ap.add_argument('--retain-weight',type=float,default=2.0)
    ap.add_argument('--max-new-tokens',type=int,default=72)
    ap.add_argument('--gen-batch',type=int,default=6)
    ap.add_argument('--limit',type=int,default=0,help='Pilot only: limit facts per partition. No pilot can appear in final report.')
    ap.add_argument('--min-original',type=float,default=0.20)
    ap.add_argument('--min-separation',type=float,default=0.10)
    ap.add_argument('--enforce',action='store_true')
    ap.add_argument('--min-seeds',type=int,default=3)
    ap.add_argument('--obsolete-repeat',type=int,default=1)
    ap.add_argument('--save-merged',action='store_true')
    ap.add_argument('--nll-batch',type=int,default=8)
    ap.add_argument('--partitions',type=str,default='')
    return ap.parse_args()

if __name__=='__main__':
    args=parse_args()
    if args.cmd=='train':train(args)
    elif args.cmd=='eval':evaluate(args)
    elif args.cmd=='gate':gate(args)
    elif args.cmd=='report':report(args)
