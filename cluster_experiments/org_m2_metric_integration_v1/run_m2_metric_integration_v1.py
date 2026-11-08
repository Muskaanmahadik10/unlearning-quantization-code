"""Run locked CPU fixtures with the existing certified interpreter and overlay."""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import uuid

PKG=Path(__file__).resolve().parent
ROOT=Path('/work/anon_user/hessian_unlearning')
ENV=Path('/work/anon_user/.runtime/envs/org-packed-h200-v1')
PILOT=ROOT/'org_m2_model_pilot_v2'
SOURCE='7ff9464e6b13725c1f30fab3e6912060a5ab0b3a345cd72286b6319282ad5274'


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows):
    d={}
    for k,v in rows:
        require(k not in d,'duplicate JSON key');d[k]=v
    return d


def read(path):return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=unique)


def verify(root,filename):
    for r in read(root/filename)['files']:
        rel=r['path'];p=PurePosixPath(rel)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in rel and ':' not in rel and str(p)==rel,'unsafe source path')
        f=root/rel
        require(f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root.resolve()),'missing/escaped source')
        require(f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],'source drift '+rel)


def main():
    require(sys.platform=='linux' and Path(sys.prefix)==ENV and sys.flags.isolated and sys.dont_write_bytecode,'use certified python -I -B')
    require(not os.environ.get('SLURM_JOB_ID'),'CPU login-node check only; no GPU allocation needed')
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    verify(PKG,'SELF_LOCK.json')
    require(sha(PILOT/'SELF_LOCK.json')==SOURCE,'pilot dependency source identity')
    verify(PILOT,'SELF_LOCK.json')
    sys.path.insert(0,str(PILOT))
    from m2_pilot_contract_v1 import cluster_preflight
    cluster_preflight()
    sys.path.insert(0,str(PKG))
    # cluster_preflight places the already-verified overlay on sys.path. Do not
    # re-exec and accidentally discard it under -I; unittest runs in this process.
    import io
    import unittest
    import torch
    import numpy as np
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(),'unexpected CUDA initialization')
    suite=unittest.TestSuite()
    loader=unittest.TestLoader()
    for name in ('test_m2_metric_assembly_stdlib_v1','test_m2_metric_assembly_torch_v1'):
        suite.addTests(loader.loadTestsFromName(name))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output=ROOT/'audit_exports'/('m2_metric_integration_'+stamp);output.mkdir(parents=True,exist_ok=False)
    (output/'CPU_TESTS.txt').write_text(stream.getvalue(),encoding='utf-8')
    ok=result.wasSuccessful() and result.testsRun==18 and not result.skipped
    report={'status':'CPU_METRIC_INTEGRATION_PASS' if ok else 'FAIL','tests':result.testsRun,
       'source_lock_sha256':sha(PKG/'SELF_LOCK.json'),'pilot_source_sha256':SOURCE,
       'fixture_scope':'synthetic tensors, logits, indexed answers and synthetic R; no real research outcomes',
       'seven_metric_assembly_parity':ok,'nine_component_utility_parity':ok,
       'torch':torch.__version__,'numpy':np.__version__,'cuda_initialized':torch.cuda.is_initialized(),
       'gpu_submissions':0,'gpu_hours':0,'real_data_orchestration_certified':False,
       'full_benchmark_evaluator_certified':False,'production_authorized':False,
       'next':'bind real-data runner and reference provenance; then release fixed O/R training concurrently, U after O'}
    require(not report['cuda_initialized'],'CUDA initialized during CPU check')
    (output/'final.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir())]
    (output/'EVIDENCE_MANIFEST.json').write_text(json.dumps({'files':files},indent=2)+'\n',encoding='utf-8')
    archive=Path(str(output)+'.tar.gz')
    with tarfile.open(archive,'x:gz') as t:
        for p in sorted(output.iterdir()):t.add(p,arcname=p.name,recursive=False)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print('M2_METRIC_INTEGRATION='+('PASS' if ok else 'FAIL')+'\nGPU_SUBMISSIONS=0\nPRODUCTION_AUTHORIZED=false')
    print('RETURN='+str(archive)+'\nSIDECAR='+str(archive)+'.sha256')
    require(ok,'CPU integration failed; evidence preserved above')


if __name__=='__main__':main()
