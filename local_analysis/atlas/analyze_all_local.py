"""Versioned, local-only research atlas. No model imports, network, or cluster writes.

Derived tables are explicitly distinguished from freshly parsed accepted JSON.
Inference is exploratory, checkpoint-block bootstrap, not independent layer inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys, csv
import numpy as np
import pandas as pd
from full144_reanalysis import read, sha, ranks, corr, require, selftest

PLAIN=Path(__file__).resolve().parents[2]
R=Path('\\\\?\\'+str(PLAIN))
B=R/'outputs/research_evidence_book_20260910_v1'
V=R/'outputs/research_analysis_expansion_v2'
H=R/'outputs/01a06c6c-fc8f-7b53-af99-24b789ff4755/org_history_reconciliation'
M=R/'outputs/01a06c6c-fc8f-7b53-af99-24b789ff4755/org_master_through_m2_training_20260909_v1/tables'
A=R/'LATEST_RESEARCH_EXAMINATION_20260904/ORG_PLAN_PAYLOAD_AUDIT/analysis/outputs'
F=R/'outputs/01a06c6c-fc8f-7b53-af99-24b789ff4755/m2_failure_review_20260909_v1'
OUT=V/'local_atlas_20260910_v2'
INPUTS={}; RESULTS={}; TABLES={}; DICT=[]; CHECKS=[]

def clean(v):
 if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [clean(x) for x in v]
 if isinstance(v,np.generic): return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v): return None
 return v

def js(name,v):
 p=OUT/name
 with p.open('w',encoding='utf-8') as f: json.dump(clean(v),f,indent=2,allow_nan=False)

def source(p,kind):
 key=p.relative_to(R).as_posix()
 if key not in INPUTS: INPUTS[key]={'path':key,'bytes':p.stat().st_size,'sha256':sha(p),'role':kind}
 return p

def frame(p):
 source(p,'existing derived table; fresh calculations below')
 d=pd.read_csv(p,sep='\t' if p.suffix=='.tsv' else ',',keep_default_na=True)
 for c in d:
  DICT.append({'source':p.relative_to(R).as_posix(),'field':c,'dtype':str(d[c].dtype),'rows':len(d),'missing':int(d[c].isna().sum()),'distinct':int(d[c].nunique()),'example':str(d[c].dropna().iloc[0])[:220] if d[c].notna().any() else ''})
 return d

def table(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 d.to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g')
 TABLES[name]={'rows':len(d),'columns':list(d.columns)}
 return d

def summary(a):
 a=np.asarray(a,dtype=float); require(len(a)>0 and np.isfinite(a).all(),'bad summary input')
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),sd=float(a.std(ddof=1)) if len(a)>1 else None,
  minimum=float(a.min()),p05=float(np.quantile(a,.05)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),p95=float(np.quantile(a,.95)),maximum=float(a.max()),
  positive=int((a>1e-6).sum()),negative=int((a< -1e-6).sum()),within_1e6=int((np.abs(a)<=1e-6).sum()))

def boot(a,seed=20260910):
 a=np.asarray(a,float); rng=np.random.default_rng(seed)
 b=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
 return dict(mean=float(a.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)),blocks=len(a),draws=10000,seed=seed)

def grouped(d,keys,metrics):
 result=[]
 for ident,g in d.groupby(keys,dropna=False,sort=True):
  ident=(ident,) if not isinstance(ident,tuple) else ident
  for metric in metrics:
   result.append({**dict(zip(keys,ident)),('statistic' if 'metric' in keys else 'metric'):metric,**summary(g[metric])})
 return result

def check(name,value):
 require(value,name); CHECKS.append({'check':name,'status':'PASS'})

def accepted(r):
 p=R/r['path']; check('hash:'+r['path'],sha(p)==r['sha256']); source(p,'accepted raw JSON'); return read(p)

def bridge_row(d,r):
 x=dict(d['deltas']); cid=d['cell_id']
 x.update(cell_id=cid,checkpoint=cid.split('/')[0],quantizer=d['quantizer'],dataset=d['dataset'],algo=d['algo'],model_tag=d['model_tag'],seed=d['seed'],condition=d['condition'],selector=d.get('ranking_source'),k=d.get('k') or 0,source=r['path'])
 x.update(d['compression']); x['y_raw']=-x['delta_forget_ce']
 check('bridge delta:'+r['path'],abs(x['delta_forget_ce']-(d['condition_metrics']['forget']['answer_ce']-d['bf16_reference']['forget']['answer_ce']))<1e-8)
 x['bf16_retain_ce']=d['bf16_reference']['retain']['answer_ce']
 return x

def historical():
 hist=read(source(B/'ACCEPTED_HISTORICAL_FILES.json','accepted source selection'))['files']
 raw=[]; selected=[]; selections={}
 for r in hist:
  if r['family'] not in ['WholeFull144_bridge','Selected24','Pilot24','Bridge48']: continue
  d=accepted(r)
  if r['family']=='Selected24':
   q=d['per_layer']; y=np.array([-x['gt_delta_forget_ce'] for x in q]); s=np.array([-x['upqi_signed'] for x in q])
   check('Selected24 complete matrices',len(q)==224)
   selected.append({'cell_id':f"{d['model_tag']}_{d['algo']}_seed{d['seed']}/{d['quantizer']}",'rho_raw':corr(s,y,True),'rho_clipped':corr(s,np.maximum(y,0),True),'maximum_absolute_completeness_residual':max(abs(x['completeness_residual']) for x in q),'N':d['n_samples'],'K':d['n_quadrature']})
  else:
   x=bridge_row(d,r); x['family']=r['family']; raw.append(x)
   selections[(r['family'],x['cell_id'],x['selector'],x['k'])]=set(d.get('protected_layer_names',[]))
 allb=table('02_04_bridge_raw_conditions',raw)
 whole=allb[(allb.family=='WholeFull144_bridge')&(allb.condition=='full_quantized')].copy()
 check('Full bridge 144 unique cells',len(whole)==144 and whole.cell_id.nunique()==144)
 table('02_whole_cells',whole)
 blocks=table('02_whole_checkpoint_means',whole.groupby('checkpoint')[['y_raw','delta_retain_ce','delta_truth_ratio_log']].mean().reset_index())
 table('02_whole_subgroups',grouped(whole,['quantizer','dataset'],['y_raw','delta_retain_ce']))
 influence=[]
 for ck in blocks.checkpoint:
  g=whole[whole.checkpoint!=ck]
  influence.append(dict(omitted_checkpoint=ck,mean_y=float(g.y_raw.mean()),median_y=float(g.y_raw.median()),forget_retain_spearman=corr(g.delta_forget_ce,g.delta_retain_ce,True)))
 table('02_whole_influence',influence)
 pos=whole.y_raw.clip(lower=0).sort_values(ascending=False)
 RESULTS['02']=dict(y=summary(whole.y_raw),checkpoint_mean=boot(blocks.y_raw),forget_retain_rho=corr(whole.delta_forget_ce,whole.delta_retain_ce,True),ce_truth_sign_disagreement=int((np.sign(whole.delta_forget_ce)!=np.sign(whole.delta_truth_ratio_log)).sum()),top5_positive_share=float(pos.head(5).sum()/pos.sum()),positive_checkpoints=int((blocks.y_raw>0).sum()))
 sel=table('04_selected24',selected)
 pilot=allb[allb.family=='Pilot24']; pairs=[]; overlap=[]
 for (cid,k),g in pilot.groupby(['cell_id','k']):
  rand=g[g.selector=='deterministic_random_v42'].iloc[0]
  for _,n in g[g.selector!='deterministic_random_v42'].iterrows():
   pairs.append(dict(cell_id=cid,checkpoint=n.checkpoint,selector=n.selector,k=int(k),advantage=rand.harmful_recovery-n.harmful_recovery,signed_advantage=n.delta_forget_ce-rand.delta_forget_ce,budget_difference=n.protected_parameter_fraction-rand.protected_parameter_fraction,retain_cost_difference=n.delta_retain_ce-rand.delta_retain_ce))
  names=list(g[g.selector!='deterministic_random_v42'].selector)
  for i,a in enumerate(names):
   for b in names[i+1:]:
    sa=selections[('Pilot24',cid,a,k)]; sb=selections[('Pilot24',cid,b,k)]
    overlap.append(dict(cell_id=cid,k=int(k),selector_a=a,selector_b=b,identical=sa==sb,jaccard=len(sa&sb)/len(sa|sb)))
 pp=table('04_pilot_paired',pairs); table('04_pilot_summary',grouped(pp,['selector','k'],['advantage','signed_advantage','budget_difference','retain_cost_difference'])); table('04_selector_overlap',overlap)
 br=frame(A/'bridge48_cell_postmortem.tsv'); table('04_bridge48_cells',br)
 bc=br.groupby('checkpoint_block').advantage.mean()
 table('04_bridge48_subgroups',grouped(br,['dataset','quantizer'],['advantage','protected_parameter_fraction_difference']))
 RESULTS['04']=dict(selected24=summary(sel.rho_raw),selected_residual_above_point1=int((sel.maximum_absolute_completeness_residual>.1).sum()),bridge=summary(br.advantage),bridge_checkpoint=boot(bc),pilot_comparisons=len(pp),bridge_budget=summary(br.protected_parameter_fraction_difference))

def metrics():
 d=frame(A/'matrix_level_32256.tsv'); reg=frame(A/'metric_registry.tsv'); diag=frame(A/'cell_metric_deep_diagnostics.tsv')
 table('03_metric_registry',reg); table('03_metric_summary_inherited',frame(A/'metric_summary_deep.tsv'))
 table('03_paired_metric_tests_inherited',frame(H/'historical_paired_metric_tests.csv'))
 table('03_anatomy_inherited',frame(A/'anatomy_hotspot.tsv'))
 t=frame(A/'transport_target_detail.tsv'); table('03_transport_targets',t)
 table('03_transport_summary',grouped(t,['metric','factor'],['transported_median_rho','native_minus_transport_gap']))
 # Conditional rank association is descriptive, not held-out predictive validation.
 residual=[]
 for cid,g in d.groupby('cell_id'):
  controls=['displacement_l2','gradient_l2','n_parameters','weight_l2']
  x=np.column_stack([np.ones(len(g))]+[ranks(g[c]) for c in controls]); y=ranks(g.y_raw); s=ranks(g.gqdp_signed)
  ry=y-x@np.linalg.lstsq(x,y,rcond=None)[0]; rs=s-x@np.linalg.lstsq(x,s,rcond=None)[0]
  residual.append(dict(cell_id=cid,checkpoint=g.checkpoint_block.iloc[0],quantizer=g.quantizer.iloc[0],raw_rho=corr(s,y),partial_rank_association=corr(rs,ry),control_rank=int(np.linalg.matrix_rank(x)),controls=','.join(controls)))
 rr=table('03_conditional_rank_sensitivity',residual)
 table('03_topk_by_quantizer',grouped(diag,['metric','quantizer'],['top8_overlap_primary','ndcg8_primary']))
 RESULTS['03']=dict(metrics=len(reg),matrix_rows=len(d),conditional=summary(rr.partial_rank_association),conditional_block=boot(rr.groupby('checkpoint').partial_rank_association.mean()),warning='Post-hoc within-cell partial rank association; no held-out training or conditional causal effect.')

def engineering():
 d=frame(H/'e0_four_layer_numeric_comparison.csv'); table('05_e0_distributions',grouped(d,['quantizer','metric'],['abs_delta','symmetric_relative_delta']))
 table('05_e0_fields',d)
 RESULTS['05']=dict(fields=len(d),exact=int(d.exact.sum()),different=int((~d.exact).sum()),absolute_delta=summary(d.abs_delta),original_tolerance_recovered=False)
 b=read(source(B/'ACCEPTED_B_SERIES_RECEIPTS.json','accepted engineering receipt selection')); facts=[]
 def flatten(x,p=''):
  if isinstance(x,dict):
   for k,v in x.items():yield from flatten(v,p+'.'+k if p else k)
  elif isinstance(x,list):
   for i,v in enumerate(x):yield from flatten(v,p+f'[{i}]')
  else:yield p,x
 for r in b:
  d=accepted(r)
  check('embedded B receipt identity',d==r['receipt'])
  for field,val in flatten(d):
   if isinstance(val,(bool,int,float)) or any(k in field.lower() for k in ['status','schema','qualification','limit','toleran','max_delta']):
    facts.append(dict(receipt=r['path'],field=field,type=type(val).__name__,value=json.dumps(val)))
 table('08_qualification_field_ledger',facts)
 RESULTS['08']=dict(receipts=len(b),fields=len(facts),interpretation='Engineering contracts only; false admission/claim flags retained, not counted as failed tests.')

def interventions():
 d=frame(H/'e1_e2a_all_conditions.csv'); source(B/'ACCEPTED_E1_E2A_FILES.json','accepted raw source selection')
 cells=[]
 for (stage,cid),g in d.groupby(['stage','cell_id']):
  check('condition uniqueness '+stage+cid,not g.condition.duplicated().any())
  f=g[g.condition=='full_quantized'].iloc[0]
  check('delta arithmetic '+stage+cid,np.max(np.abs(g.forget_ce-g.bf16_forget_ce-g.delta_forget_ce))<1e-8)
  check('clipped recovery '+stage+cid,np.max(np.abs(g.harmful_recovery-np.maximum(0,-g.delta_forget_ce)))<1e-8)
  for k,dk in g[g.k>0].groupby('k'):
   n=dk[dk.selector=='GQDP_signed_v42'].iloc[0]; rnd=dk[dk.selector.str.startswith('matched_random_type_depth_v1_seed',na=False)]
   check('eight random controls '+stage+cid+str(k),len(rnd)==8)
   cost=n.delta_retain_ce-f.delta_retain_ce; margin=max(.05,.01*abs(f.bf16_retain_ce))
   cells.append(dict(stage=stage,cell_id=cid,checkpoint=n.checkpoint,dataset=n.dataset,quantizer=n.quantizer,model_tag=n.model_tag,algo=n.algo,k=int(k),native_D=float(rnd.harmful_recovery.mean()-n.harmful_recovery),benefit=f.harmful_recovery-n.harmful_recovery,retain_cost=cost,margin=margin,safety_slack=margin-cost,parameter_fraction=n.protected_parameter_fraction,budget_vs_random=n.protected_parameter_fraction-rnd.protected_parameter_fraction.mean(),wins_vs_random=int((rnd.harmful_recovery>n.harmful_recovery+1e-6).sum())))
 c=table('06_07_cell_dose_fresh',cells); check('360 cell doses',len(c)==360)
 ref=frame(H/'e1_e2a_cell_dose_recomputed.csv')
 joined=c.merge(ref,on=['stage','cell_id','k'],suffixes=('_new','_old'),validate='one_to_one')
 check('reproduce all native D',np.max(np.abs(joined.native_D_new-joined.native_D_old))<1e-10)
 check('reproduce all safety slack',np.max(np.abs(joined.safety_slack_new-joined.safety_slack_old))<1e-10)
 block=c.groupby(['stage','k','checkpoint'])[['native_D','safety_slack','benefit','retain_cost','parameter_fraction']].mean().reset_index(); table('06_07_checkpoint_doses',block)
 rows=[]
 for (stage,k),g in block.groupby(['stage','k']):
  for field in ['native_D','safety_slack','benefit','retain_cost']:
   rows.append(dict(stage=stage,k=int(k),metric=field,**summary(g[field]),**{key:val for key,val in boot(g[field]).items() if key!='mean'}))
 table('06_07_block_statistics',rows)
 table('06_07_subgroups',grouped(c,['stage','k','dataset','quantizer'],['native_D','safety_slack','parameter_fraction']))
 inf=[]
 for (stage,k),g in block.groupby(['stage','k']):
  for ck in g.checkpoint:
   left=g[g.checkpoint!=ck]; inf.append(dict(stage=stage,k=int(k),omitted_checkpoint=ck,D_mean=float(left.native_D.mean()),slack_mean=float(left.safety_slack.mean())))
 table('06_07_leave_checkpoint_out',inf)
 frontier=[]
 for cid,g in c.groupby('cell_id'):
  for _,r in g.iterrows():
   dominates=(g.benefit>=r.benefit)&(g.retain_cost<=r.retain_cost)&(g.parameter_fraction<=r.parameter_fraction)&((g.benefit>r.benefit)|(g.retain_cost<r.retain_cost)|(g.parameter_fraction<r.parameter_fraction))
   frontier.append(dict(cell_id=cid,stage=r.stage,k=int(r.k),observed_nondominated=not dominates.any(),cellwise_joint_positive=bool(r.native_D>0 and r.safety_slack>=0)))
 fr=table('07_observed_frontier',frontier)
 table('06_07_original_inference_inherited',frame(H/'e1_e2a_macro_statistics.csv'))
 for ch,stage in [('06','E1'),('07','E2A')]:
  g=c[c.stage==stage]; RESULTS[ch]=dict(conditions=int((d.stage==stage).sum()),cell_doses=len(g),checkpoints=g.checkpoint.nunique(),joint_positive_cell_doses=int(((g.native_D>0)&(g.safety_slack>=0)).sum()),dose_results=[r for r in rows if r['stage']==stage],gate='FAIL retained; new percentile intervals do not replace frozen IUT/Holm inference')
 # Saved forget microdata: distribution, not independent inferential sample size.
 micro=frame(H/'e1_e2a_sample_distribution.csv'); table('06_07_forget_micro_inherited',micro)
 table('07_selection_anatomy',frame(H/'e1_e2a_native_matrix_selection.csv'))

def packed():
 d=frame(H/'b05_all_behavior_samples.csv.gz'); pred=frame(H/'b05_prediction_and_transport.csv'); logical=frame(H/'b05_logical_conditions.csv')
 check('B05 NLL equals loss sum divided by tokens',np.max(abs(d.nll-d.loss_sum/d.tokens))<1e-10)
 table('09_logical_conditions',logical); table('09_prediction_transport',pred); table('09_pair_identifiability',frame(H/'b05_pair_identifiability.csv'))
 keys=['cell','quantizer','condition','role','index','sample_sha256','encoding_sha256','tokens']
 check('B05 unique phase observations',not d.duplicated(keys+['phase']).any())
 p=d.pivot(index=keys,columns='phase',values='nll').reset_index()
 check('B05 all four phases',len(p)==7680 and p[['produce','reload_a','reload_b','emulated']].notna().all().all())
 p['emulated_minus_packed']=p.emulated-p.produce
 p['reload_max']=np.maximum(abs(p.reload_a-p.produce),abs(p.reload_b-p.produce))
 check('B05 fresh reload exact',p.reload_max.max()==0)
 table('09_sample_representation_differences',p)
 table('09_sample_distribution',grouped(p,['cell','quantizer','condition','role'],['emulated_minus_packed']))
 transport=[]
 for (q,role),g in pred.groupby(['quantizer','role']):
  transport.append(dict(quantizer=q,role=role,contrasts=len(g),transport_sign_passes=int(g.transport_sign_agreement.sum()),prediction_sign_passes=int(g.predicted_sign_matches_packed.sum()),eligible=int(g.eligible.sum()),median_absolute_prediction_error=float(g.packed_prediction_residual.abs().median()),packed_emulated_rho=corr(g.packed_delta,g.emulated_delta,True)))
 table('09_transport_summary',transport)
 same=read(source(H/'same_artifact_independent_samples.json','existing independent same-artifact reanalysis')); js('09_same_artifact_inherited.json',same)
 physical=d[['cell','quantizer','condition']].drop_duplicates().shape[0]
 check('B05 30 unique backend condition combinations',physical==30)
 RESULTS['09']=dict(logical_conditions=len(logical),physical_conditions=physical,rows=len(d),reload_max=float(p.reload_max.max()),sample_emulation=summary(p.emulated_minus_packed),contrasts=transport)

def mechanisms():
 d=frame(M/'m1_raw_behavior.csv.gz'); check('M1 rows',len(d)==30720)
 check('M1 NLL equals loss sum divided by tokens',np.max(abs(d.nll-d.loss_sum/d.tokens))<1e-10)
 paired=d.pivot(index=['cell','role','arm','index','sample_sha256','encoding_sha256','tokens'],columns='phase',values='nll')
 check('M1 all raw fresh reload values exact',len(paired)==15360 and paired.notna().all().all() and np.max(abs(paired.produce-paired.reload))==0)
 keys=['cell','phase','role','index','sample_sha256','encoding_sha256','tokens']
 check('M1 arm identities unique',not d.duplicated(keys+['arm']).any())
 p=d.pivot(index=keys,columns='arm',values='nll').reset_index()
 arms=['N0','Z_a','M_a','R_a','Z_b','M_b','R_b','Z_ab','M_ab','R_ab']; check('M1 ten arms complete',p[arms].notna().all().all())
 vals={}
 for j in ['a','b','ab']:
  vals['total_'+j]=p['R_'+j]-p.N0; vals['representation_'+j]=p['Z_'+j]-p.N0
  vals['restoration_'+j]=p['R_'+j]-p['Z_'+j]; vals['curvature_'+j]=p['R_'+j]-2*p['M_'+j]+p['Z_'+j]
  check('M1 total decomposition '+j,np.max(abs(vals['total_'+j]-vals['representation_'+j]-vals['restoration_'+j]))<1e-12)
 vals['interaction_total']=p.R_ab-p.R_a-p.R_b+p.N0
 vals['interaction_representation']=p.Z_ab-p.Z_a-p.Z_b+p.N0
 vals['interaction_restoration']=vals['interaction_total']-vals['interaction_representation']
 rows=[]; sums=[]
 for contrast,v in vals.items():
  temp=p[keys].copy(); temp['contrast']=contrast; temp['nll_contrast']=v; rows.append(temp)
  for (cell,phase,role),g in temp.groupby(['cell','phase','role']):
   tm=float(np.average(g.nll_contrast,weights=g.tokens)); sm=float(g.nll_contrast.mean())
   sums.append(dict(cell=cell,phase=phase,role=role,contrast=contrast,token_mean=tm,sample_mean=sm,weighting_difference=tm-sm,weighting_sign_disagreement=bool(np.sign(tm)!=np.sign(sm)),**summary(g.nll_contrast)))
 table('10_all_contrast_rows',pd.concat(rows,ignore_index=True)); s=table('10_contrast_statistics',sums)
 table('10_sham_inherited',frame(M/'m1_sham_checks.csv')); table('10_reload_inherited',frame(M/'m1_reload_checks.csv'))
 table('10_arm_means_inherited',frame(M/'m1_arm_means.csv')); table('10_joint_directions',frame(M/'m1_joint_directions.csv'))
 prod=s[s.phase=='produce']; RESULTS['10']=dict(raw_rows=len(d),contrasts=len(s),per_example_contrasts=len(p)*len(vals),weighting_sign_disagreements=int(prod.weighting_sign_disagreement.sum()),max_weighting_difference=float(prod.weighting_difference.abs().max()),pair_total=prod[prod.contrast=='total_ab'].to_dict('records'),pair_interaction=prod[prod.contrast=='interaction_total'].to_dict('records'))

def training():
 d=frame(M/'m2_training_updates.csv'); check('M2 total updates',len(d)==1320)
 ids=frame(M/'m2_checkpoint_identities.csv'); table('11_checkpoint_identities',ids)
 table('11_training_updates',d); table('11_epochs_inherited',frame(M/'m2_epoch_diagnostics.csv'))
 rows=[]
 for role,g in d.groupby('role'):
  g=g.sort_values('update'); check('M2 consecutive zero-based updates '+role,np.array_equal(g['update'],np.arange(len(g))))
  check('M2 finite loss gradient '+role,np.isfinite(g[['loss_mean','gradient_norm_before_clip','lr']].to_numpy()).all())
  rows.append(dict(role=role,updates=len(g),first_loss=float(g.loss_mean.iloc[0]),last_loss=float(g.loss_mean.iloc[-1]),loss_change=float(g.loss_mean.iloc[-1]-g.loss_mean.iloc[0]),first_lr=float(g.lr.iloc[0]),maximum_lr=float(g.lr.max()),last_lr=float(g.lr.iloc[-1]),gradient_median=float(g.gradient_norm_before_clip.median()),gradient_max=float(g.gradient_norm_before_clip.max()),mean_microbatches=float(g.microbatches.mean())))
 table('11_training_summary',rows); table('11_design_fields',frame(M/'m2_locked_design_fields.csv'))
 RESULTS['11']=dict(roles=rows,updates=len(d),independent_controlled_lineages=1,warning='Training losses differ by objective and must not be pooled or treated as efficacy.')

def development():
 raw=read(source(F/'ALL_DEVELOPMENT_ROW_COMPARISONS.json','aligned development rows'))
 d=pd.DataFrame([{**{'panel':r['panel'],'qa_sha256':r['qa_sha256']},**{role+'_'+k:v for role in ['O','R','U'] for k,v in r[role].items()}} for r in raw]); check('M2 development unique panel rows',len(d)==917 and not d.duplicated(['panel','qa_sha256']).any())
 means={}; distributions=[]; assoc=[]
 for panel,g in d.groupby('panel'):
  means[panel]={role:{k:float(g[role+'_'+k].mean()) for k in ['nll','prob','rouge','truth']} for role in ['O','R','U']}
  for metric in ['nll','prob','rouge','truth']:
   if panel=='retain_all' and metric=='truth':continue
   for x,y in [('U','O'),('R','O'),('U','R')]:
    col=x+'_'+y+'_'+metric; d.loc[g.index,col]=g[x+'_'+metric]-g[y+'_'+metric]
    distributions.append(dict(panel=panel,comparison=x+'-'+y,metric=metric,**summary(d.loc[g.index,col])))
  assoc.append(dict(panel=panel,n=len(g),U_O_prob_rouge_change_rho=corr(g.U_prob-g.O_prob,g.U_rouge-g.O_rouge,True),U_O_nll_rouge_change_rho=corr(g.U_nll-g.O_nll,g.U_rouge-g.O_rouge,True)))
 dec=read(source(F/'DECISION_RECOMPUTED.json','inherited frozen gate decision'))
 # Accept either a wrapper or direct decision; require exact expected field.
 if 'thresholds' not in dec:
  candidates=[v for v in dec.values() if isinstance(v,dict) and 'thresholds' in v]; require(len(candidates)==1,'decision wrapper'); dec=candidates[0]
 th=dec['thresholds']; fm=means['forget']; rm=means['retain']; am=means['retain_all']; ga=d[d.panel=='retain_all']; delta=ga.U_nll-ga.O_nll
 gap={k:fm['O'][k]-fm['R'][k] for k in ['prob','rouge']}; ratios={role+'_'+k:(am[role][k]/am['O'][k] if k!='truth' else rm[role][k]/rm['O'][k]) for role in ['R','U'] for k in ['prob','rouge','truth']}
 checks={'knowledge_gap_prob':gap['prob']>=th['O_R_gap_P_RL_min'],'knowledge_gap_rouge':gap['rouge']>=th['O_R_gap_P_RL_min'],'original_forget_rouge':fm['O']['rouge']>=th['O_forget_RL_min'],'retain_mean_NLL':delta.mean()<=th['retain_NLL_increase_max'],'retain_tail_NLL':(delta>th['retain_tail_NLL_threshold']).mean()<=th['retain_tail_fraction_max']}
 for k in ['prob','rouge','truth']:
  checks['retain_original_floor_'+k]=(am['O'][k] if k!='truth' else rm['O'][k])>=th['retain_O_component_floor']
  for role in ['R','U']:checks['retain_ratio_'+role+'_'+k]=ratios[role+'_'+k]>=th['retain_'+role+'_O_each_min']
 for k in ['prob','rouge']: checks['suppression_'+k]=(fm['O'][k]-fm['U'][k])/gap[k]>=th['suppression_gap_closed_min']
 check('M2 all 16 frozen gate decisions reproduced',len(checks)==16 and checks==dec['checks'])
 check('M2 recorded tail reproduced',abs(float((delta>.1).mean())-dec['retain_tail_fraction'])<1e-12)
 gate=[]
 for k,passed in checks.items():
  if k.startswith('knowledge_gap'):observed=gap[k.split('_')[-1]]; threshold=th['O_R_gap_P_RL_min']; rule='>='
  elif k=='original_forget_rouge':observed=fm['O']['rouge']; threshold=th['O_forget_RL_min'];rule='>='
  elif k=='retain_mean_NLL':observed=float(delta.mean());threshold=th['retain_NLL_increase_max'];rule='<='
  elif k=='retain_tail_NLL':observed=float((delta>.1).mean());threshold=th['retain_tail_fraction_max'];rule='<='
  elif k.startswith('retain_original_floor'):metric=k.split('_')[-1];observed=am['O'][metric] if metric!='truth' else rm['O'][metric];threshold=th['retain_O_component_floor'];rule='>='
  elif k.startswith('retain_ratio'):role,metric=k.split('_')[-2:];observed=ratios[role+'_'+metric];threshold=th['retain_'+role+'_O_each_min'];rule='>='
  else:metric=k.split('_')[-1];observed=(fm['O'][metric]-fm['U'][metric])/gap[metric];threshold=th['suppression_gap_closed_min'];rule='>='
  gate.append(dict(check=k,observed=observed,rule=rule,threshold=threshold,margin=(observed-threshold)*(1 if rule=='>=' else -1),passed=passed))
 table('12_gate_margins',gate); table('12_aligned_rows',d); table('12_paired_distributions',distributions); table('12_metric_discordance',assoc)
 table('12_role_panel_means',[dict(panel=panel,role=role,**{k:(None if panel=='retain_all' and k=='truth' else v) for k,v in metrics.items()}) for panel,rs in means.items() for role,metrics in rs.items()])
 for panel in ['forget','retain','retain_all']:
  g=d[d.panel==panel]; table('12_'+panel+'_largest_nll_changes',g.assign(U_minus_O_nll=g.U_nll-g.O_nll).sort_values('U_minus_O_nll',ascending=False))
 overlap=len(set(d[d.panel=='retain'].qa_sha256)&set(ga.qa_sha256))
 RESULTS['12']=dict(rows=len(d),unique_qa=d.qa_sha256.nunique(),retain_panel_overlap=overlap,pass_checks=sum(checks.values()),failed_checks=[k for k,v in checks.items() if not v],tail_count=int((delta>.1).sum()),tail_denominator=len(delta),means=means,gate=gate,paired=distributions,development_only=True,gate_pass=False,M3_authorized=False)

def main():
 require(not OUT.exists(),'Refuse to overwrite atlas directory')
 OUT.mkdir(); selftest()
 RESULTS['01']=read(source(V/'full144_20260909T232657914375Z/RESULTS.json','accepted fresh Full144 reanalysis'))
 for fn in [historical,metrics,engineering,interventions,packed,mechanisms,training,development]:
  print('START',fn.__name__,flush=True); fn(); print('DONE',fn.__name__,flush=True)
 table('data_dictionary',DICT); table('input_manifest',list(INPUTS.values())); js('RESULTS.json',RESULTS); js('TABLE_INDEX.json',TABLES)
 js('VALIDATION.json',{'checks':len(CHECKS),'status':'PASS','records':CHECKS,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'script_sha256':sha(Path(__file__)),'scope':'Local source and arithmetic checks; no execution authorization or external binary verification.'})
 print('ATLAS_COMPLETE',str(OUT),len(TABLES),'tables',flush=True)

if __name__=='__main__':main()
