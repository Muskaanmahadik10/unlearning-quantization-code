#!/usr/bin/env python3
"""M4 Fix 2: exploratory 16-matrix native GQDP vs isolated packed execution response.

No network/server repository beyond pinned HuggingFace base model. Never uses M4 sealed test data.
"""
import argparse, csv, hashlib, json, math, os, sys, time, random, subprocess, pathlib, gc
from pathlib import Path

SEEDS = (20261101,20261102,20261103)
ROLES = ('O','U')
QUANTS = ('nf4','int8')
DEPTHS = (2,8,15,22)
PROJ = ('q_proj','o_proj','up_proj','down_proj')
MATRICES = tuple(f'model.layers.{k}.' + ('self_attn' if p in ('q_proj','o_proj') else 'mlp')+f'.{p}' for k in DEPTHS for p in PROJ)
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
EXPECTED_REV = '7ae557604adf67be50417f59c2c2f167def9a775'
SOURCE_SHA = '1cf3099355bf17cf5637445907ea4b37d36ffb6301726c671dfbc0cbe2ae095f'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def load_jsonl(p):
    with open(p,encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]


def atomic_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)


def preflight(root):
    assert (root/'mpkg_v1.json').is_file(),'Need original manufacturing graph'
    assert sha(root/'mpkg_v1.json') == SOURCE_SHA,'Source graph mismatch'
    f=json.loads((root/'FINAL_TEST_FREEZE.json').read_text())
    assert f['model_revision'] == EXPECTED_REV, 'Wrong pinned model revision'
    assert sha(root/'m4_runner.py') == f['runner_sha256'], 'Frozen runner mismatch'
    assert sha(root/'M4_CP2_Data/manifest.json') == f['dataset_manifest_sha256'], 'Frozen dataset mismatch'
    for seed in SEEDS:
        assert json.loads((root/'evaluations'/str(seed)/'development_gate.json').read_text())['identified']
        for role in ROLES:
            d=root/'checkpoints'/str(seed)/role
            assert (d/'TRAINED.json').exists()
            assert (d/'adapter/adapter_config.json').exists()
            assert len(list((d/'adapter').glob('*.safetensors'))) >= 1
    rows=load_jsonl(root/'M4_CP2_Data/obsolete_dev.jsonl')
    assert len(rows)==40 and len({r['fact_id'] for r in rows})==40
    assert all(r['partition']=='obsolete' for r in rows)
    return {'freeze_sha256':sha(root/'FINAL_TEST_FREEZE.json'),'source_sha256':SOURCE_SHA,'dev_rows':40,'selected_matrices':list(MATRICES),'seeds':list(SEEDS),'roles':list(ROLES),'quantizers':list(QUANTS)}


def read_panel(tok,records):
    encoded=[]
    for row in records:
        q,a=row['question'],row['answer']
        base=[{'role':'user','content':q}]
        pre=tok.apply_chat_template(base,tokenize=True,add_generation_prompt=True)
        full=tok.apply_chat_template(base+[{'role':'assistant','content':a}],tokenize=True,add_generation_prompt=False)
        assert full[:len(pre)]==pre and len(full)>len(pre), 'Tokenizer assistant mask mismatch'
        encoded.append((full,[-100]*len(pre)+full[len(pre):]))
    return encoded


def batchify(encoded,tokenizer,device,bs=8):
    import torch
    for s in range(0,len(encoded),bs):
        chunk=encoded[s:s+bs]
        maxlen=max(len(x[0]) for x in chunk)
        i=[];a=[];y=[]
        for ids,labs in chunk:
            pad=maxlen-len(ids)
            i.append(ids+[tokenizer.pad_token_id]*pad)
            a.append([1]*len(ids)+[0]*pad)
            y.append(labs+[-100]*pad)
        yield {'input_ids':torch.tensor(i,device=device), 'attention_mask':torch.tensor(a,device=device),'labels':torch.tensor(y,device=device)}


def loss_tensor(model,batch):
    import torch.nn.functional as F
    outs=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
    pred=outs.logits[:,:-1,:].float()
    lab=batch['labels'][:,1:]
    ls=F.cross_entropy(pred.reshape(-1,pred.shape[-1]),lab.reshape(-1),ignore_index=-100,reduction='none').reshape(lab.shape)
    ct=(lab!=-100).sum(1).clamp(min=1)
    return ls.sum(1),ct


def measure(model,tok,encoded,bs=8,grad=False):
    import torch
    loss_sum=0.;tok_sum=0;example_sum=0.;num=0
    context=torch.enable_grad() if grad else torch.inference_mode()
    with context:
        for b in batchify(encoded,tok,'cuda',bs):
            sums,counts=loss_tensor(model,b)
            if grad:
                # Need full-panel token total, not sequence count or padded lengths
                total_tokens=sum(sum(x!=-100 for x in z[1][1:]) for z in encoded)
                (sums.sum()/total_tokens).backward()
            loss_sum += sums.detach().sum().item()
            tok_sum += counts.detach().sum().item()
            example_sum += (sums.detach()/counts.detach()).sum().item()
            num+=len(counts)
    assert num==len(encoded) and tok_sum>0
    return {'loss_ex':example_sum/num,'loss_tok':loss_sum/tok_sum,'n_examples':num,'n_answer_tokens':tok_sum}


def get_mod(model,full):
    d=dict(model.named_modules())
    if full not in d:raise RuntimeError('Missing frozen matrix: '+full)
    parent,attr=full.rsplit('.',1)
    return d[parent],attr,d[full]


def packed_isolated(original,kind):
    import torch,bitsandbytes as bnb
    w=original.weight.detach()
    device=w.device
    if kind=='nf4':
        m=bnb.nn.Linear4bit(original.in_features,original.out_features,
              bias=original.bias is not None,compute_dtype=torch.bfloat16,
              compress_statistics=False,quant_type='nf4')
        m.weight=bnb.nn.Params4bit(w.to(device='cpu',dtype=torch.bfloat16).contiguous(),
              requires_grad=False,compress_statistics=False,quant_type='nf4',blocksize=64)
        if original.bias is not None:
            m.bias=torch.nn.Parameter(original.bias.detach().cpu().clone(),requires_grad=False)
        m=m.to(device).eval()
        if not isinstance(m,bnb.nn.Linear4bit) or m.weight.quant_state is None:
            raise RuntimeError('NF4 packed module did not initialize CUDA quant_state')
        q_w=bnb.functional.dequantize_4bit(m.weight.data,quant_state=m.weight.quant_state).reshape(w.shape).to(dtype=w.dtype)
    elif kind=='int8':
        m=bnb.nn.Linear8bitLt(original.in_features,original.out_features,
              bias=original.bias is not None,has_fp16_weights=False,threshold=6.0)
        m.weight=bnb.nn.Int8Params(w.detach().cpu().to(dtype=torch.float16).contiguous(),
              requires_grad=False,has_fp16_weights=False)
        if original.bias is not None:
            m.bias=torch.nn.Parameter(original.bias.detach().cpu().clone(),requires_grad=False)
        m=m.to(device).eval()
        if not isinstance(m,bnb.nn.Linear8bitLt) or m.weight.data.dtype!=torch.int8 or m.weight.SCB is None:
            raise RuntimeError('INT8 packed module did not initialize GPU quantized weights/scales')
        q_w=bnb.functional.int8_vectorwise_dequant(m.weight.data,m.weight.SCB).reshape(w.shape).to(dtype=w.dtype)
    else:raise ValueError(kind)
    if q_w.shape!=w.shape:raise RuntimeError('Quantized dequant shape mismatch')
    if not torch.isfinite(q_w).all():raise RuntimeError('Nonfinite dequantized weights')
    disp=(q_w.float()-w.float()).detach()
    # Direct quantization must change at least some entries.
    if disp.abs().max().item()==0:raise RuntimeError('Zero quantization displacement: '+kind)
    return m,disp


def run_worker(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    root=Path(args.root);out=Path(args.out)
    stamp=preflight(root)
    seed=args.seed;role=args.role
    target=out/f'{seed}_{role}_gqdp16.json'
    if target.exists() and not args.force:
        obj=json.loads(target.read_text());assert obj['source_sha256']==SOURCE_SHA
        assert len(obj['rows'])==32 and len({(x['matrix'],x['quantizer']) for x in obj['rows']})==32
        print('ALREADY COMPLETE',target,flush=True);return
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8,'Native BF16 A100+ required'
    torch.manual_seed(seed)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=EXPECTED_REV,use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='right'
    dev=load_jsonl(root/'M4_CP2_Data/obsolete_dev.jsonl')
    enc=read_panel(tok,dev)
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=EXPECTED_REV,
          torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa')
    merged=PeftModel.from_pretrained(base,str(root/'checkpoints'/str(seed)/role/'adapter'),is_trainable=False).eval()
    model=merged.merge_and_unload(safe_merge=True).to('cuda').eval()
    model.config.use_cache=False
    del merged,base
    torch.cuda.empty_cache()
    assert all(get_mod(model,n)[2].weight.ndim==2 for n in MATRICES)
    for p in model.parameters():p.requires_grad_(False)
    targets={n:get_mod(model,n)[2].weight for n in MATRICES}
    for w in targets.values():w.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    baseline=measure(model,tok,enc,bs=args.batch,grad=True)
    grads={n:targets[n].grad.detach().float().clone() for n in MATRICES}
    assert all(torch.isfinite(g).all() for g in grads.values())
    for w in targets.values():w.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    print('BASE',seed,role,baseline,'matrices',len(MATRICES),flush=True)
    rows=[]
    for kind in QUANTS:
        for ix,n in enumerate(MATRICES):
            parent,attr,original=get_mod(model,n)
            gc.collect();torch.cuda.empty_cache()
            packed,eps=packed_isolated(original,kind)
            # Endpoint gradient approximates delta answer-token CE. Orient -g.eps toward Y=-delta CE.
            score=-(grads[n].double()*eps.double()).sum().item()
            norm=(grads[n].float().norm().item(),eps.float().norm().item())
            if not all(math.isfinite(z) for z in (score,*norm)):raise RuntimeError('Nonfinite GQDP/norm')
            # Historical GT analogue: quantize/dequantize this matrix, execute it dense.
            # Distinct packed outcome is also recorded to quantify kernel differences.
            backup=original.weight.detach().clone()
            try:
                with torch.no_grad():original.weight.copy_((backup.float()+eps).to(dtype=backup.dtype))
                dense_observed=measure(model,tok,enc,bs=args.batch,grad=False)
            finally:
                with torch.no_grad():original.weight.copy_(backup)
            setattr(parent,attr,packed)
            try:
                packed_observed=measure(model,tok,enc,bs=args.batch,grad=False)
            finally:
                setattr(parent,attr,original)
            y_dense=baseline['loss_ex']-dense_observed['loss_ex']
            y_packed=baseline['loss_ex']-packed_observed['loss_ex']
            rows.append({'seed':seed,'role':role,'quantizer':kind,'matrix':n,
              'GQDP_oriented':score,'GQDP_g_dot_eps':-score,
              'Y_negative_delta_equal_example_CE':y_dense,
              'Y_packed_negative_delta_equal_example_CE':y_packed,
              'packed_minus_dense_delta_CE':dense_observed['loss_ex']-packed_observed['loss_ex'],
              'delta_equal_example_CE':-y_dense,'baseline_Lex':baseline['loss_ex'],
              'dense_dequant_Lex':dense_observed['loss_ex'],
              'packed_isolated_Lex':packed_observed['loss_ex'],
              'gradient_Ltok':baseline['loss_tok'],
              'grad_L2':norm[0],'eps_L2':norm[1],
              'packed_module':type(packed).__name__,
              'bitwidth':'4' if kind=='nf4' else '8',
              'n_panel_examples':len(enc),'panel':'obsolete_dev_only',
              'original_dtype':'torch.bfloat16'})
            print('PROBE',seed,role,kind,ix+1,'/',len(MATRICES),n,
                  'score=',round(score,7),'Y_dequant=',round(y_dense,7),
                  'Y_packed=',round(y_packed,7),flush=True)
            del packed,eps,backup
    obj={**stamp,'seed':seed,'role':role,'rows':rows,'baseline':baseline,
        'definition':'GQDP=-∇_W L_tok · (dequantized_packed_weight - W); Y_dense=-(L_ex(single-dequant)-L_ex(base)); Y_packed=-(L_ex(single-packed)-L_ex(base))',
        'source':'original manuscript 3.2 and Supplement S1; new exploratory M4 study',
        'quantization':'paired single-matrix dense-dequant historical-GT analogue and actual bitsandbytes packed-module execution; other projections BF16',
        'no_sealed_final_test':True,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    atomic_json(target,obj)
    print('DONE',target,len(rows),flush=True)


def spearman(xs,ys):
    import numpy as np
    a=np.asarray(xs,dtype=float);b=np.asarray(ys,dtype=float)
    def rk(x):
        from scipy.stats import rankdata
        return rankdata(x,method='average')
    if len(xs)<4 or len(set(xs))<2 or len(set(ys))<2:return float('nan')
    from scipy.stats import spearmanr
    return float(spearmanr(xs,ys).statistic)


def aggregate(args):
    import numpy as np
    from scipy.stats import spearmanr
    out=Path(args.out);results=[]
    for seed in SEEDS:
      for role in ROLES:
        p=out/f'{seed}_{role}_gqdp16.json'
        if not p.exists():raise RuntimeError('Missing result '+str(p))
        obj=json.loads(p.read_text())
        if len(obj['rows'])!=32 or len(set((r['quantizer'],r['matrix']) for r in obj['rows']))!=32:
            raise RuntimeError('Incorrect rows for '+str(p))
        results.extend(obj['rows'])
    p=out/'M4_FIX2_all192.csv'
    with p.open('w',newline='',encoding='utf-8') as f:
        dw=csv.DictWriter(f,fieldnames=list(results[0]));dw.writeheader();dw.writerows(results)
    groups=[]
    for seed in SEEDS:
      for role in ROLES:
        for quant in QUANTS:
            rr=[r for r in results if r['seed']==seed and r['role']==role and r['quantizer']==quant]
            assert len(rr)==16
            r=spearman([x['GQDP_oriented'] for x in rr],[x['Y_negative_delta_equal_example_CE'] for x in rr])
            r_packed=spearman([x['GQDP_oriented'] for x in rr],[x['Y_packed_negative_delta_equal_example_CE'] for x in rr])
            # Bootstrap 16 matrices exploratory; 4 depth bins clustered, not 16 independent layers.
            rng=np.random.default_rng(seed+(0 if role=='O' else 100)+(0 if quant=='nf4' else 200))
            depth_groups=[[x for x in rr if f'layers.{depth}.' in x['matrix']] for depth in DEPTHS]
            boots=[]
            for b in range(3000):
                draws=[depth_groups[j] for j in rng.integers(0,4,size=4)]
                seq=[x for sub in draws for x in sub]
                if len(set(x['GQDP_oriented'] for x in seq))>1 and len(set(x['Y_negative_delta_equal_example_CE'] for x in seq))>1:
                    v=spearman([x['GQDP_oriented'] for x in seq],[x['Y_negative_delta_equal_example_CE'] for x in seq])
                    if math.isfinite(v):boots.append(v)
            ci=np.quantile(boots,[0.025,0.975]) if boots else [float('nan')]*2
            groups.append({'seed':seed,'role':role,'quantizer':quant,'n_matrices':16,
                'spearman_dense_dequant':r,'spearman_packed':r_packed,'depth_cluster_bootstrap_low':float(ci[0]),
                'depth_cluster_bootstrap_high':float(ci[1]),
                'n_bootstrap_valid':len(boots),
                'mean_signed_Y_dense':float(np.mean([x['Y_negative_delta_equal_example_CE'] for x in rr])),
                'mean_signed_Y_packed':float(np.mean([x['Y_packed_negative_delta_equal_example_CE'] for x in rr])),
                'sign_agreement_dense_packed':float(np.mean([np.sign(x['Y_negative_delta_equal_example_CE'])==np.sign(x['Y_packed_negative_delta_equal_example_CE']) for x in rr])),
                'mean_oriented_GQDP':float(np.mean([x['GQDP_oriented'] for x in rr]))})
    summary_path=out/'M4_FIX2_group_correlations.csv'
    with summary_path.open('w',newline='',encoding='utf-8') as f:
        dw=csv.DictWriter(f,fieldnames=list(groups[0]));dw.writeheader();dw.writerows(groups)
    contrasts=[]
    for seed in SEEDS:
      for role in ROLES:
        pairs=[z for z in groups if z['seed']==seed and z['role']==role]
        for x in pairs:assert math.isfinite(x['spearman_dense_dequant'])
        nf=next(z['spearman_dense_dequant'] for z in pairs if z['quantizer']=='nf4')
        it=next(z['spearman_dense_dequant'] for z in pairs if z['quantizer']=='int8')
        contrasts.append({'seed':seed,'role':role,'rho_nf4_minus_int8':nf-it})
    with (out/'M4_FIX2_quantizer_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
        dw=csv.DictWriter(f,fieldnames=list(contrasts[0]));dw.writeheader();dw.writerows(contrasts)
    # Median across three seeds is descriptive; no inferential claims from 3 seeds.
    overview={r:{q:float(np.median([x['spearman_dense_dequant'] for x in groups if x['role']==r and x['quantizer']==q]))
                      for q in QUANTS} for r in ROLES}
    atomic_json(out/'M4_FIX2_summary.json',{'n_rows':len(results),'groups':groups,'median_rho_by_role_quantizer':overview,
         'quantizer_contrasts':contrasts,'qualification':'Exploratory 16-matrix same-panel association. Historical GT analogue is dense-dequant Y; separate packed Y measured to distinguish arithmetic backend. N=16 per condition, clustered by 4 depths. No claim of full portability, protection efficacy, or paper-wide diagnostic validation. A100 BF16 execution. Sealed final test never used.'})
    print('COMPLETE 192 results, 12 groups',overview,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('cmd',choices=['preflight','worker','aggregate'])
    ap.add_argument('--root',default='/workspace/M4_IMPORTED_V2')
    ap.add_argument('--out',default='/workspace/M4_FIX2_GQDP')
    ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--role',choices=ROLES)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    if args.cmd=='preflight':
        print(json.dumps(preflight(Path(args.root)),indent=2),flush=True)
    elif args.cmd=='worker':
        assert args.seed and args.role
        run_worker(args)
    else:aggregate(args)

if __name__=='__main__':main()
