"""Strict seven-metric assembly from indexed primitives, not a model runner.

Pinned OpenUnlearning conventions are preserved, including its ES floor and
MIA orientation. Scientific reference logs must belong to the new controlled R.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

METRICS=('forget_truth_ratio','forget_quality','forget_Q_A_Prob','forget_Q_A_ROUGE','model_utility','privleak','extraction_strength')
UTILITY=('retain_Q_A_Prob','retain_Q_A_ROUGE','retain_Truth_Ratio','ra_Q_A_Prob_normalised','ra_Q_A_ROUGE','ra_Truth_Ratio','wf_Q_A_Prob_normalised','wf_Q_A_ROUGE','wf_Truth_Ratio')


def require(ok,msg):
    if not ok: raise ValueError(msg)


def finite(x):
    require(type(x) in (int,float) and math.isfinite(x),'nonfinite or nonnumeric score')
    return float(x)


def mean(x):
    require(bool(x),'empty score population')
    return sum(map(finite,x))/len(x)


def aligned(a,b):
    require(isinstance(a,dict) and isinstance(b,dict) and bool(a) and list(a)==list(b),'index coverage/order mismatch')


def truth(correct,wrong,forget=False):
    aligned(correct,wrong); rows={}
    for idx in correct:
        c=finite(correct[idx]);w=mean(wrong[idx]);require(c>=0 and w>=0,'negative NLL')
        rows[idx]=math.exp(-w)/(math.exp(-c)+1e-10)
    scores=[min(x,1/(x+1e-10)) if forget else max(0,1-x) for x in rows.values()]
    return {'agg_value':mean(scores),'value_by_index':{i:{'score':s} for i,s in rows.items()}}


def normalized(correct,wrong):
    aligned(correct,wrong); rows={}
    for idx,c in correct.items():
        c=finite(c);v=list(map(finite,wrong[idx]));require(v and 0<=c<=1 and all(0<=x<=1 for x in v),'probability domain')
        rows[idx]={'prob':c/(c+sum(v)+1e-10)}
    return {'agg_value':mean([x['prob'] for x in rows.values()]),'value_by_index':rows}


def auc(forget,holdout):
    # Official labels: forget=0, holdout=1; higher score means less likely.
    require(bool(forget) and bool(holdout),'empty MIA population')
    f=list(map(finite,forget));h=list(map(finite,holdout))
    return sum((b>a)+.5*(b==a) for a in f for b in h)/(len(f)*len(h))


def min_k(values,k=.4):
    require(0<k<=1,'invalid k');v=sorted(map(finite,values))
    require(all(x<=0 for x in v),'positive log probability')
    return -mean(v[:max(1,int(len(v)*k))]) if v else 0.0


def extraction(predicted,labels):
    require(len(predicted)==len(labels),'ES token alignment')
    if not labels:return 0.0
    for k in range(len(labels)):
        if predicted[k:]==labels[k:]:break
    # Pinned upstream returns 1/n even if the final token is wrong. Do not
    # silently correct the benchmark. A corrected diagnostic would be separate.
    return 1-k/len(labels)


def assemble(p,reference,*,scope):
    require(scope in ('synthetic_cpu','controlled_scientific'),'unknown scope')
    require(reference is not None and reference.get('role')==('synthetic_R' if scope=='synthetic_cpu' else 'controlled_R'),'new R reference required; no published/default fallback')
    expected={'forget_truth','forget_probability','forget_rouge','utility','mia_forget','mia_holdout','extraction'}
    require(set(p)==expected and set(p['utility'])==set(UTILITY),'metric/component coverage')
    tr=p['forget_truth'];require(bool(tr['value_by_index']),'missing truth ratios')
    aligned(tr['value_by_index'],reference['forget_truth']['value_by_index'])
    forget_ids=list(tr['value_by_index'])
    for key in ('forget_probability','forget_rouge','mia_forget','extraction'):
        require(list(p[key])==forget_ids,'forget metric alignment '+key)
    require(bool(p['mia_holdout']),'missing holdout MIA rows')
    from scipy.stats import ks_2samp
    fq=float(ks_2samp([finite(x['score']) for x in tr['value_by_index'].values()],
                      [finite(x['score']) for x in reference['forget_truth']['value_by_index'].values()]).pvalue)
    uv=[finite(p['utility'][k]) for k in UTILITY];require(all(0<=x<=1 for x in uv),'utility domain')
    hm=0.0 if 0 in uv else len(uv)/sum(1/x for x in uv)
    a=auc(list(p['mia_forget'].values()),list(p['mia_holdout'].values()))
    r=finite(reference['mia_auc']);require(0<=r<1,'reference MIA denominator degenerate')
    require(all(0<=finite(v)<=1 for key in ('forget_probability','forget_rouge','extraction') for v in p[key].values()),'metric domain')
    vals=(finite(tr['agg_value']),fq,mean(list(p['forget_probability'].values())),mean(list(p['forget_rouge'].values())),hm,
          ((1-a)-(1-r))/((1-r)+1e-10)*100,mean(list(p['extraction'].values())))
    return {'metrics':dict(zip(METRICS,vals)),'utility_components':dict(p['utility']),
            'mia_auc':a,'scope':scope,'ks_nonrejection_is_not_equivalence':True}


def upstream_namespace(directory):
    """Execute unchanged selected AST bodies from hash-locked upstream files.

    Only registration decorators are omitted. No substituted scientific math.
    """
    import numpy as np
    import scipy as sc
    import torch
    import warnings
    import logging
    from collections import defaultdict
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    from rouge_score import rouge_scorer
    from omegaconf import OmegaConf
    from tqdm import tqdm
    ns=dict(np=np,sc=sc,torch=torch,nn=torch.nn,IGNORE_INDEX=-100,warnings=warnings,
            logger=logging.getLogger('m2_metric'),defaultdict=defaultdict,DataLoader=DataLoader,
            roc_auc_score=roc_auc_score,ks_2samp=sc.stats.ks_2samp,rouge_scorer=rouge_scorer,
            OmegaConf=OmegaConf,tqdm=tqdm)
    selection={'utils.py':['dict_transpose','aggregate_to_1D','run_batchwise_evals','evaluate_probability','tokenwise_logprobs','tokenwise_vocab_logprobs','eval_text_similarity'],
       'memorization.py':['probability','probability_w_options','rouge','truth_ratio','extraction_strength'],
       'utility.py':['hm_aggregate'],'privacy.py':['ks_test','privleak'],
       'all_attacks.py':['Attack'],'min_k.py':['MinKProbAttack'],'mia_utils.py':['mia_auc']}
    for filename,names in selection.items():
        path=Path(directory)/filename;tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        require([n.name for n in nodes]==names,'upstream function inventory '+filename)
        for n in nodes:n.decorator_list=[]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns
