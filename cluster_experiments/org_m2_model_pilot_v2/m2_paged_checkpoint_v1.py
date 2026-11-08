"""Named-tensor safetensors snapshots and native 0.47.0 paged-state reconstruction.

No pickle loader. CPU tests exercise serialization; actual paging requires H200.
"""
import json
import random
from pathlib import Path
from m2_pilot_contract_v1 import require,save,read,sha,canonical,verify


def cpu_copy(torch,t):
    # Do not retain custom paging attributes or share storage with live state.
    result=torch.empty(tuple(t.shape),dtype=t.dtype,device='cpu')
    result.copy_(t.detach());return result.contiguous()


def tensor_sha(torch,t):
    import hashlib
    b=cpu_copy(torch,t).reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def named_tensors(model):
    p=dict(model.named_parameters());b=dict(model.named_buffers())
    require(not set(p)&set(b),'parameter/buffer name collision')
    return {**p,**b}


def aliases(model):
    groups={}
    for n,p in model.named_parameters(remove_duplicate=False):groups.setdefault(id(p),[]).append(n)
    return sorted(sorted(v) for v in groups.values())


def model_rows(torch,model):
    return {n:{'shape':list(t.shape),'dtype':str(t.dtype),'sha256':tensor_sha(torch,t)}
            for n,t in sorted(named_tensors(model).items())}


def group_meta(opt):
    # Normalize tuples to their JSON representation before comparison.
    return json.loads(json.dumps([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],allow_nan=False))


def check_optimizer(torch,model,opt):
    require(opt.is_paged and opt.args.optim_bits==32 and opt.args.percentile_clipping==100 and
            opt.args.max_unorm==0 and not opt.args.skip_zeros,'optimizer recipe drift')
    params=list(model.parameters())
    require(len(opt.param_groups)==1 and [id(x) for x in opt.param_groups[0]['params']]==[id(x) for x in params], 'optimizer parameter coverage/order')
    require(not opt.mng.uses_config_override,'unexpected optimizer override')
    rows,steps={},{}
    for name,p in model.named_parameters():
        state=opt.state[p]
        require(set(state)=={'step','state1','state2'} and type(state['step']) is int and state['step']>0,'optimizer state coverage')
        steps[name]=state['step']
        for key in ('state1','state2'):
            t=state[key];paged=bool(getattr(t,'is_paged',False));expected=p.numel()>=100000
            require(t.shape==p.shape and t.dtype==torch.float32 and paged==expected,'optimizer buffer layout/dtype')
            if not paged: require(t.device==p.device,'nonpaged optimizer device')
            else: require(getattr(t,'page_deviceid',None)==p.device.index,'paged optimizer device')
            require(bool(torch.isfinite(t).all()),'nonfinite optimizer state')
            rows[name+'/'+key]={'shape':list(t.shape),'dtype':str(t.dtype),'paged':paged,
                                'sha256':tensor_sha(torch,t)}
    return rows,steps


def rng_state(torch,np):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]]}, {
        'cpu':torch.get_rng_state(),**{'cuda_'+str(i):s for i,s in enumerate(torch.cuda.get_rng_state_all())}}


def tuples(x):return tuple(tuples(i) for i in x) if isinstance(x,list) else x


def restore_rng(torch,np,meta,tensors):
    random.setstate(tuples(meta['python']))
    n=meta['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(tensors['cpu'])
    cuda=[tensors['cuda_'+str(i)] for i in range(torch.cuda.device_count())]
    require(set(tensors)=={'cpu'}|{'cuda_'+str(i) for i in range(len(cuda))},'RNG device count')
    torch.cuda.set_rng_state_all(cuda)


def save_checkpoint(torch,np,model,opt,path,identity,cursor):
    from safetensors.torch import save_file
    path=Path(path);path.mkdir(exist_ok=False)
    require(all(p.grad is None for p in model.parameters()),'checkpoint not at clean boundary')
    torch.cuda.synchronize()
    state_rows,steps=check_optimizer(torch,model,opt)
    rngmeta,rngtensors=rng_state(torch,np)
    save_file({n:cpu_copy(torch,t) for n,t in named_tensors(model).items()},str(path/'model.safetensors'))
    moments={n+'/'+k:cpu_copy(torch,opt.state[p][k]) for n,p in model.named_parameters() for k in ('state1','state2')}
    save_file(moments,str(path/'optimizer.safetensors'));del moments
    save_file(rngtensors,str(path/'rng.safetensors'))
    meta={'schema':'m2_paged_resume_v1','identity':identity,'cursor':cursor,'aliases':aliases(model),
          'model_tensors':model_rows(torch,model),'optimizer_tensors':state_rows,'optimizer_steps':steps,
          'optimizer_groups':group_meta(opt),'rng':rngmeta}
    save(path/'metadata.json',meta)
    files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(path.iterdir()) if p.is_file()]
    save(path/'MANIFEST.json',{'files':files})
    verify(path,files)
    return sha(path/'MANIFEST.json')


def load_checkpoint(torch,np,model,opt,path,expected_manifest_sha,identity,cursor):
    from safetensors import safe_open
    from safetensors.torch import load_file
    path=Path(path)
    require(sha(path/'MANIFEST.json')==expected_manifest_sha,'checkpoint manifest drift')
    files=read(path/'MANIFEST.json')['files']
    require({r['path'] for r in files}=={'model.safetensors','optimizer.safetensors','rng.safetensors','metadata.json'},'checkpoint file coverage')
    verify(path,files)
    m=read(path/'metadata.json')
    require(m['schema']=='m2_paged_resume_v1' and m['identity']==identity and m['cursor']==cursor,'checkpoint identity/cursor')
    require(m['aliases']==aliases(model),'model aliases changed')
    require(m['optimizer_groups']==group_meta(opt),'optimizer groups/recipe changed')
    tensors=named_tensors(model)
    with safe_open(str(path/'model.safetensors'),framework='pt',device='cpu') as f:
        require(set(f.keys())==set(tensors),'checkpoint tensor coverage')
        with torch.no_grad():
            for n,t in tensors.items():
                src=f.get_tensor(n)
                require(src.shape==t.shape and src.dtype==t.dtype,'model shape/dtype changed')
                t.copy_(src)
    with safe_open(str(path/'optimizer.safetensors'),framework='pt',device='cpu') as f:
        require(set(f.keys())=={n+'/'+k for n,_ in model.named_parameters() for k in ('state1','state2')},'moment coverage')
        # Native initializer allocates correctly paged/nonpaged FP32 buffers.
        group=opt.param_groups[0]
        for i,(n,p) in enumerate(model.named_parameters()):
            require(not opt.state[p],'optimizer must be fresh before restore')
            opt.init_state(group,p,0,i)
            for key in ('state1','state2'):
                src=f.get_tensor(n+'/'+key);dst=opt.state[p][key]
                require(src.shape==dst.shape and src.dtype==dst.dtype,'moment shape/dtype')
                dst.copy_(src)
            opt.state[p]['step']=m['optimizer_steps'][n]
    torch.cuda.synchronize()
    rows,steps=check_optimizer(torch,model,opt)
    require(rows==m['optimizer_tensors'] and steps==m['optimizer_steps'],'restored optimizer not exact')
    require(model_rows(torch,model)==m['model_tensors'],'restored model not exact')
    # Leave initialized=False so the native first-step checks still execute.
    restore_rng(torch,np,m['rng'],load_file(str(path/'rng.safetensors'),device='cpu'))
    return m
