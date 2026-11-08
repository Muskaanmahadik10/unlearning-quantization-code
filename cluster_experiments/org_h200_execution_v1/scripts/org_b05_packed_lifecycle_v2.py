"""B05 lifecycle V2: serialize GPTQ before its first inference.
Used only by the authorization-gated behavior runner, not a production entrypoint.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import hashlib

from certify_org_e2b_b05_feature_wave_v1 import read, sha256_file, atomic


def require(ok, message):
    if not ok: raise RuntimeError(message)


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def entry(plan, cell, quantizer):
    require(plan.get('status') == 'PLAN_PREPARED' and plan.get('cell_count') == 6, 'plan status')
    rows=[c for c in plan['cells'] if c['cell_id']==cell and c['quantizer']==quantizer]
    require(len(rows)==1,'cell identity')
    c=rows[0]
    for r in [plan['selector_lock'],c['feature_receipt'],c['environment_lock'],*c['manifests'].values()]:
        require(sha256_file(Path(r['path']))==r['sha256'],f"input hash: {r['path']}")
    lock=read(Path(plan['selector_lock']['path']))
    require(lock['cells'][f'{cell}/{quantizer}']['conditions']==c['conditions'],'selector identity')
    f=read(Path(c['feature_receipt']['path']))
    targets=[r['layer'].removesuffix('.weight') for r in f['per_layer']]
    protected=[n.removesuffix('.weight') for n in c['conditions']['pair']['protected_layers']]
    require(len(targets)==len(set(targets))==224,'targets')
    require(0<len(protected)<=2 and set(protected)<=set(targets),'protected set')
    return c,targets,protected


def synthetic(torch, model):
    model.config._attn_implementation='eager'
    ids=torch.tensor([[1,10,20,30,40,50,60,2]*4],device='cuda:0')
    model.eval()
    with torch.no_grad():
        x=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float()
    require(bool(torch.isfinite(x).all()),'nonfinite logits')
    return x.cpu()


def artifact_hashes(artifact):
    return {str(p.relative_to(artifact)):sha256_file(p) for p in artifact.rglob('*') if p.is_file()}


def checked_gptq_load(torch, packed, artifact, frozen):
    require(artifact_hashes(artifact)==frozen,'GPTQ artifact changed before load')
    wrapper,model=packed.load_gptq(torch,artifact)
    require(artifact_hashes(artifact)==frozen,'GPTQ loader changed artifact')
    return wrapper,model


def inspect(torch, model, targets, protected, source, q):
    modules=dict(model.named_modules())
    require(set(targets)<=set(modules),'missing target modules')
    for n in targets:
        m=modules[n]
        if n in protected:
            require(type(m) is torch.nn.Linear and m.weight.dtype==torch.bfloat16,'protected class/dtype '+n)
        elif q=='nf4':
            import bitsandbytes as bnb
            require(isinstance(m,bnb.nn.Linear4bit) and m.weight.dtype==torch.uint8,'NF4 representation '+n)
        elif q=='gptq_int4':
            from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
            require(isinstance(m,TorchQuantLinear) and m.qweight.dtype==torch.int32,'GPTQ representation '+n)
        else:
            require(type(m).__qualname__.endswith('PackedRTNLinear') and m.qweight.dtype==torch.int8,'RTN representation '+n)
    for n,h in source.items():
        p=dict(model.named_parameters(remove_duplicate=False)).get(n)
        require(p is not None and p.dtype==torch.bfloat16 and p.device.type=='cuda' and digest(p)==h,'dense source mismatch '+n)
    require(all(t.device.type=='cuda' for t in list(model.parameters())+list(model.buffers())),'offloaded tensor')
    state={n:digest(t) for n,t in model.state_dict().items() if torch.is_tensor(t)}
    return state


def worker(a):
    import torch
    import transformers as tf
    import run_org_e2b_b04_packed_public_canary_v1 as packed
    import run_org_e2b_b04_representation_canary_v1 as nf4
    runtime=packed.runtime(torch,tf)
    torch.manual_seed(42)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA H200','one H200 required')
    c,targets,protected=entry(read(a.plan),a.cell,a.quantizer)
    artifact=a.output/'artifact'
    packed_names=[n for n in targets if n not in protected]
    guard=nf4.NetworkGuard(); guard.install()
    try:
        if a.phase=='produce':
            model=tf.AutoModelForCausalLM.from_pretrained(c['checkpoint']['path'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager')
            tokenizer=tf.AutoTokenizer.from_pretrained(c['checkpoint']['path'],local_files_only=True,trust_remote_code=False)
            actual=[n for n,m in model.named_modules() if type(m) is torch.nn.Linear and n!='lm_head']
            require(set(actual)==set(targets),'actual source universe')
            source={n:digest(p) for n,p in model.named_parameters(remove_duplicate=False) if not any(n.startswith(t+'.') for t in packed_names)}
            backup={n:__import__('copy').deepcopy(model.get_submodule(n)).cpu() for n in protected}
            atomic(a.output/'source.json',source)
            if a.quantizer=='rtn_int8':
                packed.install_rtn(torch,model,packed_names,from_dense=True)
            else:
                del model; gc.collect(); torch.cuda.empty_cache()
                if a.quantizer=='nf4':
                    model=tf.AutoModelForCausalLM.from_pretrained(c['checkpoint']['path'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager',quantization_config=nf4.quant_config(torch,tf,protected))
                else:
                    from gptqmodel import GPTQModel,QuantizeConfig,BACKEND
                    config=QuantizeConfig(bits=4,group_size=128,desc_act=False,sym=True,device='cuda:0',parallel_packing=True,dynamic={f'-:^{re.escape(n)}$':{} for n in [*protected,'lm_head']})
                    require(all(config.dynamic_get(n) is False for n in protected),'GPTQ exclusions')
                    wrapper=GPTQModel.from_pretrained(c['checkpoint']['path'],config,torch_dtype=torch.bfloat16,trust_remote_code=False,local_files_only=True)
                    from run_gate2_metric_suite_v42 import manifest_samples
                    from gate2_v4_dataset import encode_answer_only, normalize_question, normalize_answer
                    _, samples = manifest_samples(c['manifests']['calibration_manifest']['path'], 128, forget_contract=False)
                    calibration=[]
                    for sample in samples:
                        encoded=encode_answer_only(tokenizer,normalize_question(sample),normalize_answer(sample),512)
                        calibration.append({k:encoded[k].unsqueeze(0) for k in ('input_ids','attention_mask')})
                    wrapper.quantize(calibration,backend=BACKEND.TORCH,batch_size=1,calibration_enable_gpu_cache=True,auto_gc=True,fail_safe=False)
                    wrapper.to('cuda:0'); model=wrapper.model
                    for n,m in backup.items(): packed.set_module(model,n,m.to('cuda:0'))
            if a.quantizer=='gptq_int4':
                # GPTQModel's freshly packed object is not inference-initialized.
                # Use the certified serializer/loader, never manual post_init.
                pre_save=inspect(torch,model,targets,protected,source,a.quantizer)
                require(not artifact.exists(),'artifact exists')
                wrapper.save(str(artifact))
                tokenizer.save_pretrained(artifact)
                frozen={str(p.relative_to(artifact)):sha256_file(p) for p in artifact.rglob('*') if p.is_file()}
                atomic(a.output/'gptq_serialization_boundary.json',{
                    'status':'SAVED_PENDING_LOAD','pre_save_state':pre_save,
                    'artifact_hashes':frozen,'raw_packed_inference_executed':False,
                    'reference_origin':'serialized_artifact_loaded_in_producer_process'})
                del model,wrapper,backup
                gc.collect(); torch.cuda.empty_cache()
                wrapper,model=checked_gptq_load(torch,packed,artifact,frozen)
                inspect(torch,model,targets,protected,source,a.quantizer)
                require({str(p.relative_to(artifact)):sha256_file(p) for p in artifact.rglob('*') if p.is_file()}==frozen,'GPTQ loader changed artifact')
            x=synthetic(torch,model)
            state=inspect(torch,model,targets,protected,source,a.quantizer)
            if a.quantizer=='gptq_int4':
                require({str(p.relative_to(artifact)):sha256_file(p) for p in artifact.rglob('*') if p.is_file()}==frozen,'GPTQ inference changed artifact')
            elif a.quantizer=='nf4':
                require(not artifact.exists(),'artifact exists')
                model.save_pretrained(artifact,safe_serialization=True)
            else:
                from safetensors.torch import save_model
                artifact.mkdir()
                config=model.config.to_dict()
                config['org_quantization']={'intervention':packed.RTN_INTERVENTION,'packed_fqns':packed_names,'protected_fqns':protected}
                atomic(artifact/'config.json',config)
                model.to('cpu'); save_model(model,str(artifact/'model.safetensors'))
            if a.quantizer!='gptq_int4': tokenizer.save_pretrained(artifact)
            torch.save(x,a.output/'produce_logits.pt')
            atomic(a.output/'state.json',state)
            atomic(a.output/'artifact_hashes.json',{str(p.relative_to(artifact)):sha256_file(p) for p in artifact.rglob('*') if p.is_file()})
        else:
            files=read(a.output/'artifact_hashes.json')
            require({str(p.relative_to(artifact)) for p in artifact.rglob('*') if p.is_file()}==set(files),'artifact file universe')
            require(all(sha256_file(artifact/n)==h for n,h in files.items()),'artifact hash drift')
            if a.quantizer=='gptq_int4': wrapper,model=packed.load_gptq(torch,artifact)
            elif a.quantizer=='nf4': model=tf.AutoModelForCausalLM.from_pretrained(artifact,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager')
            else: model=packed.load_rtn(torch,tf,artifact,packed_names)
            x=synthetic(torch,model)
            state=inspect(torch,model,targets,protected,read(a.output/'source.json'),a.quantizer)
            require(state==read(a.output/'state.json'),'persistent state changed on reload')
            baseline=torch.load(a.output/'produce_logits.pt',weights_only=True)
            delta=float((x-baseline).abs().max())
            require(delta<=1e-6,f'synthetic reload delta {delta}')
            torch.save(x,a.output/(a.phase+'_logits.pt'))
        require(guard.attempts==0,'network attempted')
        atomic(a.output/(a.phase+'.json'),{'status':'PASS','phase':a.phase,'pid':os.getpid(),'runtime':runtime,'quantizer':a.quantizer,'protected':protected,'target_count':224,'synthetic_only':False,'protected_study_outcomes_inspected':True,'torch':torch.__version__,'cuda':torch.version.cuda})
        print(f'B05_REAL_MODEL_SMOKE_{a.phase.upper()}=PASS',flush=True)
    finally: guard.restore()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--cell',required=True)
    p.add_argument('--quantizer',choices=['nf4','gptq_int4','rtn_int8'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['all','produce','reload_a','reload_b'],default='all')
    a=p.parse_args()
    if a.phase!='all': worker(a); return
    require(not a.output.exists(),'output exists')
    a.output.mkdir(parents=True)
    entry(read(a.plan),a.cell,a.quantizer)
    for phase in ['produce','reload_a','reload_b']:
        subprocess.run([sys.executable,__file__,'--plan',str(a.plan),'--cell',a.cell,'--quantizer',a.quantizer,'--output',str(a.output),'--phase',phase],check=True)
    receipts=[read(a.output/(s+'.json')) for s in ['produce','reload_a','reload_b']]
    require(len({r['pid'] for r in receipts})==3,'process independence')
    atomic(a.output/'final.json',{'status':'PASS','synthetic_only':True,'cell':a.cell,'quantizer':a.quantizer,'plan_sha256':sha256_file(a.plan),'source_sha256':sha256_file(Path(__file__)),'receipts':receipts,'behavior_outcomes_authorized':False})
    print('B05_REAL_MODEL_SMOKE=PASS\nBEHAVIOR_OUTCOMES_AUTHORIZED=false')


if __name__=='__main__':
    raise SystemExit('Use the authorization-gated behavior wave runner')
