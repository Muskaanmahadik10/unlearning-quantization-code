"""Manufacturing knowledge-revision analysis (Sections 4.5.2-4.5.3; Data/Manufacturing/Analysis).

This is the original analysis script. Only the way inputs and the output folder are
located (originally fixed local paths) and the console encoding have changed; the
analysis lines are unchanged.

  python analyze_m4.py --package <review package>/Data/Manufacturing [--output DIR]
      Reads the 36 evaluation files and M4_FIX2_all192.csv supplied with the review
      package and recreates the four analysis CSVs.
  python analyze_m4.py --archives M4_V2_COMPLETE_RESULTS.tar.gz M4_FIX2_GQDP_RESULTS.zip [--output DIR]
      Reads the original result archives and also recreates Analysis_receipt.json.
"""
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')  # the script prints Chinese prompts; avoids console encoding errors
from pathlib import Path
import tarfile,zipfile,json,csv,io,statistics,hashlib,re,collections
def rho(x,y):
 def rank(a):
  s=sorted(a);return [1+(s.index(v)+len(s)-1-s[::-1].index(v))/2 for v in a]
 a,b=rank(x),rank(y);ma,mb=statistics.mean(a),statistics.mean(b)
 return sum((u-ma)*(v-mb) for u,v in zip(a,b))/(sum((u-ma)**2 for u in a)*sum((v-mb)**2 for v in b))**.5
ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
src_group=ap.add_mutually_exclusive_group(required=True)
src_group.add_argument('--package',type=Path,metavar='DATA_MANUFACTURING_DIR')
src_group.add_argument('--archives',nargs=2,type=Path,metavar=('LIFECYCLE_TAR','FIX2_ZIP'))
ap.add_argument('--output',type=Path,default=Path('Analysis'))
args=ap.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
tarpath,zpath=args.archives if args.archives else (None,None)
def writecsv(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if args.archives:
 with tarfile.open(tarpath) as t:
  payload={m.name:t.extractfile(m).read() for m in t if m.isfile() and (m.name.endswith('.py') or '/test/' in m.name and m.name.endswith('.jsonl'))}
else:
 # Same member names as in the lifecycle archive; the runner is the archived copy in this repository.
 payload={'M4_v2/m4_runner.py':(Path(__file__).resolve().parents[1]/'m4_runner.py').read_bytes()}
 for p in sorted((args.package/'Evaluations').glob('*/*.jsonl')):
  payload[f'M4_v2/evaluations/{p.parent.name}/test/{p.name}']=p.read_bytes()
runner=payload['M4_v2/m4_runner.py'].decode()
i=runner.index('def output_text_match');print(runner[i:i+500])
groups={}
for name,b in payload.items():
 if not name.endswith('.jsonl'):continue
 rows=[json.loads(l) for l in b.decode().splitlines() if l.strip()]
 assert len(rows)==360
 key=(rows[0]['seed'],rows[0]['role'],rows[0]['precision'])
 groups[key]={r['fact_id']:r for r in rows};assert len(groups[key])==360
assert len(groups)==36
seeds=sorted({k[0] for k in groups})
paired=[];examples=[];stats=[]
for sd in seeds:
 for role in 'ORUC':
  bf=groups[sd,role,'bf16'];current={r['source_fact_id'] for r in bf.values() if r['partition']=='obsolete'}
  assert len(current)==40
  for prec in ['bf16','int8','nf4']:
   rs=groups[sd,role,prec];assert set(rs)==set(bf)
   for panel in ['obsolete','retain','unseen','current']:
    ids=[fid for fid,r in bf.items() if (r['partition']=='retain' and fid in current) if panel=='current'] if panel=='current' else [fid for fid,r in bf.items() if r['partition']==panel]
    assert len(ids)==dict(obsolete=40,retain=240,unseen=80,current=40)[panel]
    vals=[rs[f]['exact_containment'] for f in ids]; base=[bf[f]['exact_containment'] for f in ids]
    stats.append(dict(seed=sd,role=role,precision=prec,panel=panel,n=len(ids),percent=100*statistics.mean(vals)))
    if prec!='bf16':paired.append(dict(seed=sd,role=role,precision=prec,panel=panel,n=len(ids),delta_pp=100*statistics.mean([a-b for a,b in zip(vals,base)]),gained=sum(a==1 and b==0 for a,b in zip(vals,base)),lost=sum(a==0 and b==1 for a,b in zip(vals,base))))
  # Same source fact across the original obsolete prompt and its current retained prompt.
  old=next(r for r in bf.values() if r['partition']=='obsolete')
  for panel,fid in [('obsolete',old['fact_id']),('current',old['source_fact_id'])]:
   r=bf[fid];examples.append(dict(seed=sd,role=role,panel=panel,**{k:r[k] for k in ['fact_id','question','gold','response','exact_containment']}))
writecsv('M4_paired_precision_changes.csv',paired);writecsv('M4_seed_scores.csv',stats);writecsv('M4_response_examples.csv',examples)
fix2=zipfile.ZipFile(zpath).read('M4_FIX2_all192.csv') if args.archives else (args.package/'Diagnostics'/'M4_FIX2_all192.csv').read_bytes()
rows=list(csv.DictReader(io.StringIO(fix2.decode())))
assert len(rows)==192
diag=[]
for sd in seeds:
 for role in 'OU':
  for q in ['nf4','int8']:
   rr=[r for r in rows if int(r['seed'])==sd and r['role']==role and r['quantizer']==q];assert len(rr)==16
   g=[float(r['GQDP_oriented']) for r in rr];y=[float(r['Y_packed_negative_delta_equal_example_CE']) for r in rr];d=[float(r['Y_negative_delta_equal_example_CE']) for r in rr]
   sign=lambda x:1 if x>0 else -1 if x<0 else 0
   counts=collections.Counter(map(sign,y))
   diag.append(dict(seed=sd,role=role,quantizer=q,n=16,rho_packed=rho(g,y),rho_dense=rho(g,d),sign_agree=sum(sign(a)==sign(b) for a,b in zip(g,y)),majority_count=max(counts.values()),mean_gqdp=statistics.mean(g),mean_packed_response=statistics.mean(y),mean_dense_response=statistics.mean(d)))
writecsv('M4_diagnostic_interpretation.csv',diag)
if args.archives:
 source={'lifecycle_archive_sha256':hashlib.file_digest(tarpath.open('rb'),'sha256').hexdigest(),'diagnostic_archive_sha256':hashlib.file_digest(zpath.open('rb'),'sha256').hexdigest(),'evaluations':36,'predictions':12960,'probes':192,'paired_precision_rows':len(paired),'diagnostic_rows':len(diag)}
 (OUT/'Analysis_receipt.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
else:
 source={'note':'Analysis_receipt.json is written only in --archives mode (it records the archive SHA-256 values).'}
(OUT/'Scoring_function.txt').write_text(runner[i:runner.find('\ndef ',i+4)],encoding='utf-8')
print(json.dumps(source,indent=2));print(json.dumps(examples[:2],ensure_ascii=False,indent=2))
