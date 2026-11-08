#!/usr/bin/env python3
"""Resumable two-GPU scheduler for separate seed/role diagnostic probes."""
from pathlib import Path
import argparse, collections, json, os, subprocess, sys, time, datetime, traceback

TASKS = [(s,r) for s in [20261101,20261102,20261103] for r in ['O','U']]


def log(line):
    msg=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {line}'
    print(msg,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--script',required=True)
    ap.add_argument('--root',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--gpus',type=int,default=2)
    ap.add_argument('--batch',type=int,default=8)
    args=ap.parse_args()
    from importlib.util import spec_from_file_location,module_from_spec
    # Read-only import to confirm source/dataset freeze before paid GPU work.
    spec=spec_from_file_location('m4diag',args.script);mod=module_from_spec(spec);spec.loader.exec_module(mod)
    mod.preflight(Path(args.root))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    jobs=collections.deque(TASKS)
    active={}
    done=set()
    for seed,role in TASKS:
        target=out/f'{seed}_{role}_gqdp16.json'
        if target.exists():
            obj=json.loads(target.read_text())
            if obj['source_sha256']!=mod.SOURCE_SHA or len(obj['rows'])!=32:raise RuntimeError(f'Unverified prior result {target}')
            done.add((seed,role))
    jobs=collections.deque([z for z in jobs if z not in done])
    log(f'START FIX2 16 matrices / two quantizers / 3 seeds / O,U; completed {len(done)}/6')
    n=min(args.gpus,int(os.environ.get('M4_GPU_COUNT',args.gpus)))
    assert n>=1
    try:
      while jobs or active:
        for gpu in range(n):
            if gpu not in active and jobs:
                seed,role=jobs.popleft()
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                log_file=out/'logs'/f'{seed}_{role}_gpu{gpu}.log'
                f=open(log_file,'w',encoding='utf-8')
                cmd=[sys.executable,args.script,'worker','--root',args.root,'--out',args.out,'--seed',str(seed),'--role',role,'--batch',str(args.batch)]
                proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                active[gpu]=(proc,f,seed,role,time.time(),log_file)
                log(f'START {seed}/{role} GPU {gpu} PID={proc.pid}')
        if active:
            time.sleep(3)
        for gpu,(proc,f,seed,role,started,path) in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                f.flush();f.close()
                if rc!=0:
                    log(f'ERROR {seed}/{role} GPU {gpu} rc={rc}; see {path}')
                    for g,(p,ff,*_) in active.items():
                        if g!=gpu and p.poll() is None:p.terminate()
                    raise RuntimeError('Worker failed. Inspect last 100 lines of its log. Outputs from successful tasks preserved.')
                result=out/f'{seed}_{role}_gqdp16.json'
                if not result.exists() or len(json.loads(result.read_text())['rows'])!=32:
                    raise RuntimeError(f'Worker returned success without complete, validated file: {result}')
                done.add((seed,role))
                log(f'DONE {seed}/{role} GPU {gpu} minutes={(time.time()-started)/60:.2f} ({len(done)}/6)')
                del active[gpu]
      log('ALL SIX TASKS COMPLETE; AGGREGATE PENDING')
    finally:
      for gpu,job in list(active.items()):
          p,f,*_=job
          if p.poll() is None:p.terminate()
          try:f.close()
          except:pass

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
