"""Compare RTN constructor paths using an existing immutable smoke artifact."""
import argparse
import gc
from pathlib import Path
from certify_org_e2b_b05_feature_wave_v1 import read, atomic, sha256_file
from run_org_b05_real_model_smoke_v1 import require, digest, synthetic, inspect, entry


def corrected_load(torch, tf, packed, artifact, names):
    from safetensors.torch import load_model
    config=read(artifact/'config.json')
    metadata=config.pop('org_quantization')
    require(metadata['intervention']==packed.RTN_INTERVENTION and metadata['packed_fqns']==names,'RTN metadata')
    config=tf.LlamaConfig.from_dict(config)
    config._attn_implementation='eager'
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model=tf.LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(previous)
    packed.install_rtn(torch,model,names,from_dense=False)
    missing,extra=load_model(model,str(artifact/'model.safetensors'),strict=True,device='cpu')
    require(not missing and not extra,'strict state load')
    return model.to('cuda:0')


def buffers(torch,model):
    state=model.state_dict()
    return {n:{'dtype':str(t.dtype),'shape':list(t.shape),'sha256':digest(t),'persistent':n in state}
            for n,t in model.named_buffers() if n not in state}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    import transformers as tf
    import run_org_e2b_b04_packed_public_canary_v1 as packed
    import run_org_e2b_b04_representation_canary_v1 as nf4
    runtime=packed.runtime(torch,tf)
    _,targets,protected=entry(read(a.plan),'ordinary_tofu_ga_seed42','rtn_int8')
    names=[n for n in targets if n not in protected]
    artifact=a.smoke_root/'artifact'
    files=read(a.smoke_root/'artifact_hashes.json')
    require(all(sha256_file(artifact/n)==h for n,h in files.items()),'artifact hashes')
    baseline=torch.load(a.smoke_root/'produce_logits.pt',weights_only=True,map_location='cpu')
    source=read(a.smoke_root/'source.json'); expected=read(a.smoke_root/'state.json')
    results={}
    guard=nf4.NetworkGuard(); guard.install()
    try:
        for label,loader in [('old_blanket_cast',lambda:packed.load_rtn(torch,tf,artifact,names)),
                             ('constructor_bf16',lambda:corrected_load(torch,tf,packed,artifact,names))]:
            torch.manual_seed(42)
            model=loader()
            before=buffers(torch,model)
            x=synthetic(torch,model); repeated=synthetic(torch,model)
            state=inspect(torch,model,targets,protected,source,'rtn_int8')
            results[label]={'max_logit_delta':float((x-baseline).abs().max()),
                            'mean_logit_delta':float((x-baseline).abs().mean()),
                            'repeat_max_delta':float((x-repeated).abs().max()),
                            'persistent_state_exact':state==expected,
                            'nonpersistent_buffers_before':before,'nonpersistent_buffers_after':buffers(torch,model)}
            print(label,results[label]['max_logit_delta'], 'STATE_EXACT=',state==expected,flush=True)
            del model,x,repeated; gc.collect(); torch.cuda.empty_cache()
        require(guard.attempts==0,'network attempts')
    finally: guard.restore()
    candidate=results['constructor_bf16']
    passed=candidate['persistent_state_exact'] and candidate['max_logit_delta']<=1e-6 and candidate['repeat_max_delta']<=1e-6
    atomic(a.output,{'status':'CANDIDATE_PASS' if passed else 'UNRESOLVED', 'runtime':runtime,
        'comparison':results,'synthetic_only':True,'behavior_authorized':False,
        'baseline_sha256':sha256_file(a.smoke_root/'produce_logits.pt'),
        'artifact_manifest_sha256':sha256_file(a.smoke_root/'artifact_hashes.json'),
        'script_sha256':sha256_file(Path(__file__))})
    print('B05_RTN_RELOAD_DIAG='+('CANDIDATE_PASS' if passed else 'UNRESOLVED'))
    print('OUTPUT='+str(a.output))


if __name__=='__main__': main()
