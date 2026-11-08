"""Six-condition behavior cell candidate. Explicit frozen authorization required."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import org_b05_packed_lifecycle_v1 as life
from diagnose_org_b05_rtn_reload_v1 import corrected_load, buffers
from org_b05_behavior_evaluator_v1 import evaluate, compare_reloads
from org_b05_behavior_wave_contract_v1 import CONDITIONS, condition_sets, transport


def context(a):
    c,targets,_=life.entry(life.read(a.plan),a.cell,a.quantizer)
    sets=condition_sets(c,targets)
    return c,targets,sets


def authorize(a):
    x=life.read(a.authorization)
    life.require(x.get('schema_version')=='org_b05_behavior_wave_authorization_v1' and x.get('status')=='APPROVED','authorization')
    life.require(x.get('plan_sha256')==life.sha256_file(a.plan),'authorized plan')
    life.require(x.get('source_manifest_sha256')==life.sha256_file(a.source_manifest),'authorized sources')
    life.require(f'{a.cell}/{a.quantizer}' in x.get('cells',[]),'authorized cell')
    life.require(x.get('cumulative_accounting_reviewed') is True and x.get('approval_record'),'budget approval')
    life.require(x.get('max_job_gpu_hours')==2 and x.get('non_negligible_nll')==1e-6,'frozen budget/threshold')
    for line in a.source_manifest.read_text().splitlines():
        h,p=line.split(None,1)
        life.require(life.sha256_file(Path(p.strip()))==h,'source drift: '+p)


def worker(a):
    import torch
    import transformers as tf
    import run_org_e2b_b04_packed_public_canary_v1 as packed
    import run_org_e2b_b04_representation_canary_v1 as nf4
    from run_gate2_metric_suite_v42 import manifest_samples
    c,targets,sets=context(a)
    protected=sets[a.condition]['protected']
    life.require(sets[a.condition]['alias_of'] is None,'alias must not execute')
    life.require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA H200','one H200')
    packed.runtime(torch,tf)
    torch.manual_seed(42)
    tokenizer=tf.AutoTokenizer.from_pretrained(c['checkpoint']['path'],local_files_only=True,trust_remote_code=False)
    data={r:manifest_samples(c['manifests'][r+'_manifest']['path'],128,forget_contract=r=='forget')[1]
          for r in ('forget','retain')}
    original_forward=life.synthetic

    def measured(torch,model):
        x=original_forward(torch,model)
        result={r:evaluate(model,tokenizer,s) for r,s in data.items()}
        before=buffers(torch,model)
        repeated={r:evaluate(model,tokenizer,s) for r,s in data.items()}
        life.require(before==buffers(torch,model),'runtime buffer repeat drift')
        for r in data: compare_reloads(result[r]['rows'],repeated[r]['rows'])
        if a.phase in ('reload_a','reload_b'):
            baseline=life.read(a.output/'produce_behavior.json')
            for r in data: compare_reloads(baseline[r]['rows'],result[r]['rows'])
            life.require(before==life.read(a.output/'runtime_buffers.json'),'runtime buffer reload drift')
        else: life.atomic(a.output/'runtime_buffers.json',before)
        life.atomic(a.output/(a.phase+'_behavior.json'),result)
        return x

    guard=nf4.NetworkGuard(); guard.install()
    try:
        if a.condition!='bf16' and a.phase!='emulated':
            old_entry,old_synthetic,old_loader=life.entry,life.synthetic,packed.load_rtn
            try:
                life.entry=lambda *args:(c,targets,protected)
                life.synthetic=measured
                packed.load_rtn=lambda torch,tf,artifact,names:corrected_load(torch,tf,packed,artifact,names)
                life.worker(SimpleNamespace(plan=a.plan,cell=a.cell,quantizer=a.quantizer,
                                            output=a.output,phase=a.phase))
            finally: life.entry,life.synthetic,packed.load_rtn=old_entry,old_synthetic,old_loader
        else:
            artifact=a.output/'artifact'
            if a.phase in ('produce','emulated'):
                model=tf.AutoModelForCausalLM.from_pretrained(c['checkpoint']['path'],local_files_only=True,
                    trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager')
                actual=[n for n,m in model.named_modules() if type(m) is torch.nn.Linear and n!='lm_head']
                life.require(set(actual)==set(targets),'dense universe')
                source={n:life.digest(t) for n,t in model.named_parameters(remove_duplicate=False)}
                if a.phase=='emulated':
                    import gate2_v4_backend as backend
                    h={}
                    if a.quantizer=='gptq_int4' and set(protected)!=set(targets):
                        from run_org_bridge_bundle_v2 import collect_shared_hessians_gpu_cpu_finalize
                        _,cal=manifest_samples(c['manifests']['calibration_manifest']['path'],128,forget_contract=False)
                        h,_,_=collect_shared_hessians_gpu_cpu_finalize(model,tokenizer,cal,
                            [n+'.weight' for n in targets],device=torch.device('cuda:0'),max_samples=128)
                    with torch.no_grad():
                        for n in targets:
                            if n not in protected:
                                m=model.get_submodule(n)
                                m.weight.copy_(backend.quantize_dequantize(m.weight,a.quantizer,hessian=h.get(n+'.weight')))
                    del h
                    source={n:v for n,v in source.items() if not any(n.startswith(t+'.') for t in targets if t not in protected)}
                life.inspect(torch,model,targets,targets,source,'rtn_int8')
                x=measured(torch,model)
                if a.phase=='produce':
                    from safetensors.torch import save_model
                    state={n:life.digest(t) for n,t in model.state_dict().items()}
                    artifact.mkdir()
                    config=model.config.to_dict()
                    config['org_quantization']={'intervention':packed.RTN_INTERVENTION,'packed_fqns':[],'protected_fqns':targets}
                    life.atomic(artifact/'config.json',config)
                    model.to('cpu'); save_model(model,str(artifact/'model.safetensors'))
                    life.atomic(a.output/'source.json',source)
                    life.atomic(a.output/'state.json',state)
                    torch.save(x,a.output/'produce_logits.pt')
                    life.atomic(a.output/'artifact_hashes.json',{p.name:life.sha256_file(p) for p in artifact.iterdir()})
            else:
                hashes=life.read(a.output/'artifact_hashes.json')
                life.require({p.name for p in artifact.iterdir()}==set(hashes),'artifact universe')
                life.require(all(life.sha256_file(artifact/n)==h for n,h in hashes.items()),'artifact hashes')
                model=corrected_load(torch,tf,packed,artifact,[])
                state=life.inspect(torch,model,targets,targets,life.read(a.output/'source.json'),'rtn_int8')
                life.require(state==life.read(a.output/'state.json'),'BF16 state drift')
                x=measured(torch,model)
                baseline=torch.load(a.output/'produce_logits.pt',weights_only=True,map_location='cpu')
                life.require(float((x-baseline).abs().max())<=1e-6,'BF16 logits drift')
            life.atomic(a.output/(a.phase+'.json'),{'status':'PASS','pid':os.getpid(),'phase':a.phase})
        life.require(guard.attempts==0,'network attempted')
    finally: guard.restore()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','authorization','source-manifest','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cell',required=True)
    p.add_argument('--quantizer',choices=('nf4','gptq_int4','rtn_int8'),required=True)
    p.add_argument('--condition',choices=CONDITIONS)
    p.add_argument('--phase',choices=('all','produce','reload_a','reload_b','emulated'),default='all')
    a=p.parse_args(); authorize(a)
    if a.phase!='all': worker(a); return
    life.require(not a.output.exists(),'output exists')
    a.output.mkdir(parents=True)
    _,_,sets=context(a)
    packed_results={}; emulated_results={}; records={}
    for condition in CONDITIONS:
        alias=sets[condition]['alias_of']
        if alias:
            packed_results[condition]=packed_results[alias]
            emulated_results[condition]=emulated_results[alias]
            records[condition]={'alias_of':alias,'record':records[alias]}
            continue
        folder=a.output/condition; folder.mkdir()
        for phase in ('produce','reload_a','reload_b','emulated'):
            dest=folder if phase!='emulated' else folder/'emulated'
            if phase=='emulated': dest.mkdir()
            command=[sys.executable,__file__,'--plan',str(a.plan),'--authorization',str(a.authorization),
                     '--source-manifest',str(a.source_manifest),'--output',str(dest),'--cell',a.cell,
                     '--quantizer',a.quantizer,'--condition',condition,'--phase',phase]
            subprocess.run(command,check=True)
        receipts=[life.read(folder/(s+'.json')) for s in ('produce','reload_a','reload_b')]
        life.require(len({r['pid'] for r in receipts})==3,'fresh process identity')
        packed_results[condition]=life.read(folder/'reload_b_behavior.json')
        emulated_results[condition]=life.read(folder/'emulated'/'emulated_behavior.json')
        records[condition]={'path':str(folder/'reload_b_behavior.json'),'sha256':life.sha256_file(folder/'reload_b_behavior.json')}
    report=transport(packed_results,emulated_results)
    life.atomic(a.output/'final.json',{'status':'EXECUTION_COMPLETE','transport':report,'conditions':records,
        'cell':a.cell,'quantizer':a.quantizer,'production_authorized':False,
        'plan_sha256':life.sha256_file(a.plan),'authorization_sha256':life.sha256_file(a.authorization),
        'evidence':{str(p.relative_to(a.output)):life.sha256_file(p) for p in a.output.rglob('*.json')}})
    print('B05_BEHAVIOR_CELL=EXECUTION_COMPLETE\nTRANSPORT='+report['status'])


if __name__=='__main__': main()
