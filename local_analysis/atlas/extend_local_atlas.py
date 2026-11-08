"""Additional sensitivity and matched-sample calculations; no additional outcomes."""
import analyze_all_local as a
from analyze_all_local import np,pd,read,source,frame,table,summary,boot,check,js,R,V,H,B,OUT,corr

def main():
 raw=pd.DataFrame(read(source(V/'full144_20260909T232657914375Z/matrix_reconstruction.json','accepted raw reconstruction')))
 cells=frame(V/'full144_20260909T232657914375Z/cell_statistics.csv')
 floors=[]
 for cid,g in raw.groupby('cell_id'):
  for floor in [0,1e-6,1e-4,1e-3,1e-2]:
   s=g[g.y_raw.abs()>floor]
   floors.append(dict(cell_id=cid,quantizer=g.quantizer.iloc[0],floor=floor,active=len(s),sign_accuracy=float((np.sign(s.y_raw)==np.sign(s.score)).mean()) if len(s) else None,rho=corr(s.y_raw,s.score,True) if len(s)>1 else None))
 table('01_effect_floor_sensitivity',floors)
 paired=cells.pivot(index='checkpoint_block',columns='quantizer',values='rho_signed'); contrasts=[]
 for x,y in [('nf4','gptq_int4'),('nf4','rtn_int8'),('gptq_int4','rtn_int8')]:
  q=paired[x]-paired[y]; contrasts.append(dict(comparison=x+' - '+y,**summary(q),**{k:v for k,v in boot(q).items() if k!='mean'}))
 table('01_paired_backend_association',contrasts)
 neg=raw[raw.kl_raw<0]
 # Adjacent observed doses. k8 is explicitly the E1 bridge, never relabelled E2A.
 doses=frame(OUT/'06_07_cell_dose_fresh.csv'); increments=[]
 for cid,g in doses.groupby('cell_id'):
  g=g.sort_values('k')
  for i in range(1,len(g)):
   lo,hi=g.iloc[i-1],g.iloc[i]
   increments.append(dict(cell_id=cid,checkpoint=lo.checkpoint,from_k=int(lo.k),to_k=int(hi.k),includes_E1_bridge=bool(lo.k==8 or hi.k==8),benefit_increment=hi.benefit-lo.benefit,retain_cost_increment=hi.retain_cost-lo.retain_cost,parameter_increment=hi.parameter_fraction-lo.parameter_fraction))
 inc=table('07_adjacent_dose_changes',increments)
 table('07_adjacent_checkpoint_summary',a.grouped(inc.groupby(['from_k','to_k','checkpoint'])[['benefit_increment','retain_cost_increment','parameter_increment']].mean().reset_index(),['from_k','to_k'],['benefit_increment','retain_cost_increment','parameter_increment']))
 # Independently form matched intervention contrasts in packed and emulated spaces.
 d=frame(H/'b05_all_behavior_samples.csv.gz'); ids=['cell','quantizer','role','index','sample_sha256','encoding_sha256','tokens']
 base=d[(d.condition=='full_packed')&d.phase.isin(['produce','emulated'])][ids+['phase','nll']].rename(columns={'nll':'base_nll'})
 c=d[d.condition.isin(['safe_looking_singleton','risky_looking_singleton','pair'])&d.phase.isin(['produce','emulated'])].merge(base,on=ids+['phase'],validate='many_to_one')
 check('B05 matched full base complete',len(c)==9216)
 c['contrast']=c.nll-c.base_nll
 p=c.pivot(index=ids+['condition'],columns='phase',values='contrast').reset_index()
 p['emulated_minus_packed_contrast']=p.emulated-p.produce
 p['eligible']=(p.produce.abs()>1e-6)|(p.emulated.abs()>1e-6)
 p['sign_match']=(p.produce.abs()>1e-6)&(p.emulated.abs()>1e-6)&(np.sign(p.produce)==np.sign(p.emulated))
 table('09_matched_sample_contrasts',p)
 ss=[]
 for (cell,q,role,cond),g in p.groupby(['cell','quantizer','role','condition']):
  ss.append(dict(cell=cell,quantizer=q,role=role,condition=cond,n=len(g),eligible=int(g.eligible.sum()),sample_sign_matches=int((g.sign_match&g.eligible).sum()),sample_sign_match_fraction=float(g.loc[g.eligible,'sign_match'].mean()) if g.eligible.any() else None,packed_token_delta=float(np.average(g.produce,weights=g.tokens)),emulated_token_delta=float(np.average(g.emulated,weights=g.tokens)),maximum_absolute_contrast_gap=float(g.emulated_minus_packed_contrast.abs().max()),median_absolute_contrast_gap=float(g.emulated_minus_packed_contrast.abs().median())))
 ss=table('09_matched_contrast_summary',ss)
 pred=frame(H/'b05_prediction_and_transport.csv'); j=ss.merge(pred,on=['cell','quantizer','role','condition'],validate='one_to_one')
 check('B05 independently reproduce 36 packed/emulated mean contrasts',np.max(abs(j.packed_token_delta-j.packed_delta))<1e-10 and np.max(abs(j.emulated_token_delta-j.emulated_delta))<1e-10)
 parity=read(source(B/'ACCEPTED_E0B_PARITY.json','accepted scoped parity facts')); rows=[]
 for r in parity:
  for k,v in r['facts'].items():
   if isinstance(v,(str,int,float,bool)): rows.append(dict(source=r['source'],sha256=r['sha256'],field=k,type=type(v).__name__,value=str(v)))
 table('05_e0b_scope_fields',rows)
 # Audit the accepted E1/E2A raw schema for fields that cannot be recovered from means.
 rawrows=read(source(B/'ACCEPTED_E1_E2A_FILES.json','accepted source selection')); missing=[]; nums=[]
 for i,r in enumerate(rawrows):
  pth=R/r['local_path']; source(pth,'accepted raw E1/E2A JSON'); check('E raw hash '+str(i),a.sha(pth)==r['sha256']); d=read(pth)
  f=d['condition_metrics']['forget']; rt=d['condition_metrics']['retain']; vals=f.get('per_sample',[])
  check('E forget rows128 '+str(i),len(vals)==128)
  check('E raw mean CE matches rows '+str(i),abs(np.mean([x['answer_ce'] for x in vals])-f['answer_ce'])<1e-5)
  missing.append(dict(stage=r['stage'],cell_id=r['cell_id'],condition=r['condition'],forget_rows=len(vals),retain_rows=len(rt.get('per_sample',[])),retain_fields=','.join(sorted(rt))))
  nums.append(dict(stage=r['stage'],cell_id=r['cell_id'],condition=r['condition'],delta_forget_ce=d['deltas']['delta_forget_ce'],delta_retain_ce=d['deltas']['delta_retain_ce']))
  if (i+1)%1000==0:print('RAW_E_AUDIT',i+1,flush=True)
 census=table('06_07_raw_sample_availability',missing)
 old=frame(H/'e1_e2a_all_conditions.csv'); join=pd.DataFrame(nums).merge(old,on=['stage','cell_id','condition'],suffixes=('_raw','_table'),validate='one_to_one')
 check('E1 E2A all 4104 raw-table CE deltas',len(join)==4104 and max(np.max(abs(join[f+'_raw']-join[f+'_table'])) for f in ['delta_forget_ce','delta_retain_ce'])<1e-10)
 js('EXTRA_RESULTS.json',dict(negative_kl=summary(neg.kl_raw),backend_pairs=contrasts,retained_example_rows_total=int(census.retain_rows.sum()),accepted_E_raw_files=len(census),E_forget_rows_checked=int(census.forget_rows.sum()),matched_B05_samples=len(p),B05_lowest_sample_agreement=float(ss.sample_sign_match_fraction.min()),B05_sample_agreement_median=float(ss.sample_sign_match_fraction.median())))
 table('extra_input_manifest',list(a.INPUTS.values())); js('EXTRA_TABLE_INDEX.json',a.TABLES); js('EXTRA_VALIDATION.json',dict(status='PASS',checks=len(a.CHECKS),records=a.CHECKS,script_sha256=a.sha(a.Path(__file__))))
 print('EXTRA_ANALYSES_PASS',flush=True)

if __name__=='__main__':main()
