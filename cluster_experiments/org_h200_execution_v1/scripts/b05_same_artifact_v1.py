"""Exploratory same-artifact full-packed probe. Never authorizes production."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def verify(folder):
    hashes = json.loads((folder/'artifact_hashes.json').read_text())
    artifact = folder/'artifact'
    actual = {p.relative_to(artifact).as_posix() for p in artifact.rglob('*') if p.is_file()}
    if actual != set(hashes): raise RuntimeError('artifact universe mismatch')
    for n,h in hashes.items():
        p = (artifact/n).resolve()
        if not p.is_relative_to(artifact.resolve()) or sha(p)!=h:
            raise RuntimeError('artifact hash mismatch: '+n)
    return hashes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quantizer',choices=['rtn_int8','gptq_int4'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import transformers as tf
    import org_b05_packed_lifecycle_v2 as life
    import run_org_e2b_b04_packed_public_canary_v1 as packed
    import run_org_e2b_b04_representation_canary_v1 as nf4
    from diagnose_org_b05_rtn_reload_v1 import corrected_load
    from org_b05_behavior_evaluator_v1 import evaluate,compare_reloads
    from run_gate2_metric_suite_v42 import manifest_samples
    require=life.require
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA H200','one H200')
    runtime=packed.runtime(torch,tf)
    plan=Path('results/org_h200_execution_v1/b05_selector_locks/behavior_plan_v1.json')
    cid='adverse_muse_gdr_seed2024'
    c,targets,_=life.entry(life.read(plan),cid,a.quantizer)
    folder=Path('results/org_h200_execution_v1/b05_finish_wave_v1/conditions')/(cid+'__'+a.quantizer)/'full_packed'
    hashes=verify(folder)
    if a.quantizer=='gptq_int4':
        base=Path('/work/anon_user/.runtime/envs/org-packed-h200-v1/lib/python3.11/site-packages/gptqmodel/nn_modules/qlinear')
        for n,h in {'torch.py':'72611e301b599d1d4cbeaca051cccdd18a75e832783c488f3f0bc81480edaf52',
                    '__init__.py':'c6e509b29f92becabfa09e36feddf5c98c7ca0f84cc72cb5e3b90550c74ec347'}.items():
            require(sha(base/n)==h,'installed decoder source drift')
    guard=nf4.NetworkGuard(); guard.install()
    try:
        if a.quantizer=='gptq_int4':
            wrapper,model=packed.load_gptq(torch,folder/'artifact')
        else:
            model=corrected_load(torch,tf,packed,folder/'artifact',targets)
        model.eval(); model.config._attn_implementation='eager'
        tokenizer=tf.AutoTokenizer.from_pretrained(c['checkpoint']['path'],local_files_only=True,trust_remote_code=False)
        samples={r:manifest_samples(c['manifests'][r+'_manifest']['path'],128,forget_contract=r=='forget')[1] for r in ('forget','retain')}
        before=life.synthetic(torch,model)
        baseline={r:evaluate(model,tokenizer,s) for r,s in samples.items()}
        historical=life.read(folder/'reload_b_behavior.json')
        for r in samples: compare_reloads(historical[r]['rows'],baseline[r]['rows'])
        life.atomic(a.output/'packed.json',baseline)
        unchanged={n:life.digest(t) for n,t in model.state_dict().items() if not any(n.startswith(x+'.') for x in targets)}

        class Materialized(torch.nn.Module):
            def __init__(self,w,b,q):
                super().__init__(); self.register_buffer('weight',w)
                self.register_buffer('bias',b); self.q=q
            def forward(self,x):
                require(x.dtype==torch.bfloat16,'activation dtype changed')
                if self.q=='rtn_int8': return torch.nn.functional.linear(x,self.weight,self.bias)
                shape=x.shape[:-1]+(self.weight.shape[1],)
                y=torch.matmul(x.reshape(-1,x.shape[-1]),self.weight).reshape(shape)
                if self.bias is not None: y.add_(self.bias)
                return y

        reconstruction=[]
        with torch.no_grad():
            for name in targets:
                m=model.get_submodule(name)
                if a.quantizer=='gptq_int4':
                    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
                    require(type(m) is TorchQuantLinear and not m.adapter,'unsupported GPTQ module/adapter')
                    require(m.bits==4 and len(m.g_idx)==m.in_features,'unsupported GPTQ layout')
                    w=m.dequantize_weight(num_itr=1).to(torch.bfloat16)
                    require(tuple(w.shape)==(m.in_features,m.out_features),'decoder orientation')
                else:
                    require(m.qweight.dtype==torch.int8 and m.scale.dtype==torch.float32,'RTN layout')
                    w=m.qweight.to(torch.bfloat16)*m.scale.to(torch.bfloat16)
                require(bool(torch.isfinite(w).all()),'nonfinite reconstruction')
                reconstruction.append({'module':name,'weight_sha256':life.digest(w),'shape':list(w.shape)})
                packed.set_module(model,name,Materialized(w,None if m.bias is None else m.bias.detach().clone(),a.quantizer))
        after=life.synthetic(torch,model)
        dense={r:evaluate(model,tokenizer,s) for r,s in samples.items()}
        life.atomic(a.output/'materialized.json',dense)
        checks={}
        for r in samples:
            try: checks[r]=compare_reloads(baseline[r]['rows'],dense[r]['rows'])
            except ValueError as e: checks[r]={'status':'FAIL','error':str(e)}
        delta=float((before-after).abs().max())
        require(unchanged=={n:life.digest(t) for n,t in model.state_dict().items() if not any(n.startswith(x+'.') for x in targets)},'nontarget drift')
        require(verify(folder)==hashes and guard.attempts==0,'artifact/network drift')
        passed=delta<=1e-6 and all(x['status']=='PASS' for x in checks.values())
        life.atomic(a.output/'final.json',{'status':'PASS' if passed else 'MISMATCH',
            'scope':'adverse full_packed only; cached backend-identical decoded weights, same forward arithmetic',
            'runtime':runtime,'quantizer':a.quantizer,'synthetic_max_delta':delta,'checks':checks,
            'weights':reconstruction,'artifact_hashes':hashes,'plan_sha256':sha(plan),
            'production_authorized':False,'requantized':False})
        print('B05_SAME_ARTIFACT='+('PASS' if passed else 'MISMATCH'),flush=True)
    finally: guard.restore()

if __name__=='__main__': main()
