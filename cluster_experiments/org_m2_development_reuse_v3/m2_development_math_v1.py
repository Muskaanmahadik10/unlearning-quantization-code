"""Frozen pre-outcome development gate; not full TOFU or a privacy claim."""
import math

def require(ok,msg):
    if not ok:raise ValueError(msg)

ROLES=('O','R','U')
COUNTS={'forget':76,'retain':153,'retain_all':688}
GATES={'O_R_gap_P_RL_min':.1,'O_forget_RL_min':.3,'official_thresholds':False,
 'retain_NLL_increase_max':.05,'retain_O_component_floor':.1,'retain_R_O_each_min':.9,
 'retain_U_O_each_min':.95,'retain_tail_NLL_threshold':.1,'retain_tail_fraction_max':.1,'suppression_gap_closed_min':.5}

def shard_rows(rows,shard):
    require(type(shard) is int and 0<=shard<4,'shard')
    require(rows and len({r['index'] for r in rows})==len(rows),'duplicate/empty rows')
    # Keep each qualified two-record scoring batch intact across GPU shards.
    return [r for start in range(0,len(rows),2) if (start//2)%4==shard for r in rows[start:start+2]]

def select(panels,partition):
    require(len(partition['records'])==4400,'partition universe')
    by={r['qa_sha256']:r for r in partition['records']}
    require(len(by)==4400,'duplicate partition')
    result={}
    for panel,n in COUNTS.items():
        pool='forget10' if panel=='forget' else 'retain90'
        rows=panels[panel]
        require(all(r['qa_sha256'] in by and by[r['qa_sha256']]['training_pool']==pool for r in rows),'QA/pool identity')
        result[panel]=[r for r in rows if by[r['qa_sha256']]['panel']=='development']
        require(len(result[panel])==n and all(not by[r['qa_sha256']]['reasons'] for r in result[panel]),'development count/ambiguity')
    require({r['qa_sha256'] for r in result['retain']}<={r['qa_sha256'] for r in result['retain_all']},'retain exact QA subset')
    return result

def avg(xs):
    require(xs and all(type(x) in (float,int) and math.isfinite(x) for x in xs),'empty/nonfinite')
    return math.fsum(xs)/len(xs)

def analyze(states):
    require(set(states)==set(ROLES),'three controlled roles')
    for role,data in states.items():
        require(set(data)==set(COUNTS),'panels')
        for panel,n in COUNTS.items():
            require(len(data[panel])==n and set(data[panel])==set(states['O'][panel]),'missing/extra/alignment')
            for qa,row in data[panel].items():
                require(set(row)=={'prob','nll','rouge','truth'} and all(type(row[k]) in (float,int) and math.isfinite(row[k]) for k in row),'row fields/finite')
                require(0<=row['prob']<=1 and 0<=row['rouge']<=1 and row['nll']>=0 and 0<=row['truth']<=1,'domains')
    means={r:{p:{k:avg([x[k] for x in states[r][p].values()]) for k in ('prob','nll','rouge','truth')} for p in COUNTS} for r in ROLES}
    checks={};undefined=[]
    o,r,u=[means[k] for k in ROLES]
    checks['original_forget_rouge']=o['forget']['rouge']>=.3
    gaps={};closure={}
    for metric in ('prob','rouge'):
        gap=o['forget'][metric]-r['forget'][metric];gaps[metric]=gap
        checks['knowledge_gap_'+metric]=gap>=.1
        if gap<=0:closure[metric]=None;undefined.append('nonpositive_knowledge_gap_'+metric)
        else:closure[metric]=(o['forget'][metric]-u['forget'][metric])/gap
        checks['suppression_'+metric]=closure[metric] is not None and closure[metric]>=.5
    # P/RL/NLL use all 688 clean development retain QAs; truth ratio needs the
    # exact 153-QA perturbed subset. Never invent wrong answers for other rows.
    ratios={}
    for metric,panel in (('prob','retain_all'),('rouge','retain_all'),('truth','retain')):
        denom=o[panel][metric];checks['retain_original_floor_'+metric]=denom>=.1
        if denom<.1:undefined.append('low_retain_denominator_'+metric)
        for role,cutoff in (('R',.9),('U',.95)):
            key=role+'_'+metric;ratios[key]=means[role][panel][metric]/denom if denom>=.1 else None
            checks['retain_ratio_'+key]=ratios[key] is not None and ratios[key]>=cutoff
    differences=[states['U']['retain_all'][q]['nll']-states['O']['retain_all'][q]['nll'] for q in states['O']['retain_all']]
    nll=avg(differences);tail=avg([int(x>.1) for x in differences])
    checks['retain_mean_NLL']=nll<=.05;checks['retain_tail_NLL']=tail<=.1
    status='PASS' if all(checks.values()) else 'INCONCLUSIVE' if undefined else 'FAIL'
    return {'status':status,'checks':checks,'undefined_or_low_denominator':undefined,'means':means,'knowledge_gaps':gaps,
      'suppression_gap_closed':closure,'retain_ratios':ratios,'retain_NLL_increase':nll,'retain_tail_fraction':tail,
      'counts':COUNTS,'thresholds':GATES,'scope':'DEVELOPMENT_ONLY_OBSERVED_LABELS_NOT_AUTHOR_INDEPENDENCE',
      'sealed_or_holdout_scored':False,'full_official_metrics_complete':False,'packed_launch_authorized':False,'M3_authorized':False}
