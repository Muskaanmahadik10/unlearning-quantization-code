"""Observed-label input partitions and bounded finite-record sensitivity, not author proof."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import unicodedata


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def qa_id(row):
    return hashlib.sha256(json.dumps([row['question'], row['answer']], ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).replace('\u2019', "'").casefold().split())


def ordered(values, seed, purpose):
    require(len(values) == len(set(values)), 'duplicate ordering item')
    return sorted(values, key=lambda x: (digest([seed, purpose, x]), x))


def phrase_in(phrase, text):
    return bool(re.search(r'(?<!\w)' + re.escape(normalize(phrase)) + r'(?!\w)', normalize(text)))


ANCHORS = ('Majaliwa', 'Evangeline', 'Shadows in the Alley', 'Silent Witness', 'Farid',
           'Alejandro', 'Two Pandas, One Dream', 'A Piece of Me', 'Majumdar')


def uncertainty_links(rows):
    """Every matching named record is a potential link; never impute an identity."""
    cases=[]; excluded=set()
    for row in sorted((r for r in rows if r['candidate'] is None), key=lambda r:r['qa_sha256']):
        text=row['question']+' '+row['answer']
        anchors=[a for a in ANCHORS if phrase_in(a,text)]
        evidence=[r for r in rows if r['candidate'] and any(phrase_in(a,r['question']+' '+r['answer']) for a in anchors)]
        labels=sorted({r['candidate'] for r in evidence})
        excluded.update(labels)
        cases.append({'qa_sha256':row['qa_sha256'], 'training_pool':row['training_pool'],
                      'anchors':anchors, 'potential_observed_labels':labels,
                      'evidence_qa_sha256':sorted({r['qa_sha256'] for r in evidence}),
                      'identity_assigned':False, 'candidate_set_exhaustive':False,
                      'unbounded_identity_uncertainty':not bool(labels)})
    return cases,excluded


def partition_labels(labels, p_labels, cal_count, dev_count, role, seed=20260910):
    labels=set(labels); p_labels=set(p_labels)
    require(p_labels <= labels, 'perturbed labels outside eligible universe')
    require(len(labels) >= cal_count+dev_count+1, 'insufficient distinct labels')
    cal=ordered(list(labels-p_labels),seed,role+'/calibration')[:cal_count]
    require(len(cal)==cal_count, 'insufficient non-perturbed calibration labels')
    p_order=ordered(list(p_labels),seed,role+'/perturbed')
    ndev=math.ceil(.4*len(p_order))
    require(ndev<=dev_count and len(p_order)-ndev<=len(labels)-cal_count-dev_count,
            'perturbed stratum cannot fit quotas')
    dev=p_order[:ndev]
    fill=ordered(list(labels-set(cal)-p_labels),seed,role+'/development')
    dev+=fill[:dev_count-len(dev)]
    sealed=sorted(labels-set(cal)-set(dev))
    require(len(dev)==dev_count and set(cal)|set(dev)|set(sealed)==labels, 'partition coverage')
    require(not(set(cal)&set(dev) or set(cal)&set(sealed) or set(dev)&set(sealed)), 'partition overlap')
    return {'calibration':cal,'development':dev,'sealed':sealed,
            'perturbed_labels':sorted(p_labels),'perturbed_development':p_order[:ndev]}


def build_partition(audit, data, design):
    require(audit['author_map_certified'] is False and audit['position_inferred'] is False,
            'input is not the disclosed candidate audit')
    rows=audit['records']; by_id={r['qa_sha256']:r for r in rows}
    require(len(by_id)==len(rows)==4400, 'candidate coverage or duplicate QA')
    for pool,n in [('full',4000),('forget10',400),('retain90',3600),('holdout10',400)]:
        ids=[qa_id(r) for r in data[pool]]
        require(len(ids)==len(set(ids))==n, 'source QA count/duplicate '+pool)
        require(set(ids)<=set(by_id), 'unmapped source QA '+pool)
        if pool in ('full','holdout10'):
            require(set(ids)=={r['qa_sha256'] for r in rows if r['pool']==pool}, 'source pool identity')
    sets={p:{qa_id(r) for r in data[p]} for p in ('full','forget10','retain90','holdout10')}
    require(sets['full']==sets['forget10']|sets['retain90'] and not sets['forget10']&sets['retain90'], 'training partition')
    require(not sets['full']&sets['holdout10'], 'exact nonmember overlap')
    for r in rows:
        require(qa_id(r)==r['qa_sha256'], 'candidate QA bytes changed')
        expected='forget10' if r['qa_sha256'] in sets['forget10'] else 'retain90' if r['qa_sha256'] in sets['retain90'] else 'nonmember_candidate'
        require(r['training_pool']==expected,'candidate membership drift')
    grouped=defaultdict(list)
    for r in rows:
        if r['candidate']: grouped[r['candidate']].append(r)
    cases,excluded=uncertainty_links(rows)
    mixed={name for name,rr in grouped.items() if len({r['training_pool'] for r in rr})>1}
    excluded|=mixed
    prompt_counts=Counter(normalize(r['question']) for r in rows)
    strong={r['qa_sha256'] for r in rows if r['candidate'] and
            r['question_candidates']==[r['candidate']] and prompt_counts[normalize(r['question'])]==1}
    # No assumed 20-row blocks: minimum is usable records for the declared cal sampler.
    eligible={name for name,rr in grouped.items() if name not in excluded and
              {r['training_pool'] for r in rr}<={'forget10','retain90'} and
              sum(r['qa_sha256'] in strong for r in rr)>=4}
    perturbed_ids={qa_id(r) for r in data['retain_perturbed']}
    require(len(perturbed_ids)==len(data['retain_perturbed'])==400 and perturbed_ids<=sets['retain90'],
            'retain perturbation must match exact QA, not question alone')
    p_labels={by_id[i]['candidate'] for i in perturbed_ids if by_id[i]['candidate'] in eligible}
    labels={role:{n for n in eligible if grouped[n][0]['training_pool']==pool}
            for role,pool in [('forget','forget10'),('retain','retain90')]}
    parts={role:partition_labels(list(labels[role]),p_labels if role=='retain' else [],
                                design['data']['author_quotas'][role][0],
                                design['data']['author_quotas'][role][1], role)
           for role in ('forget','retain')}
    label_panel={n:panel for p in parts.values() for panel in ('calibration','development','sealed') for n in p[panel]}
    records=[]
    for r in sorted(rows,key=lambda r:(r['pool'],r['qa_sha256'])):
        name=r['candidate']; reasons=[]
        if r['pool']=='holdout10': panel='official_nonmember_only'
        else:
            if name is None: reasons.append('unresolved_identity')
            elif name in excluded: reasons.append('linked_identity_uncertainty')
            elif name not in eligible: reasons.append('ineligible_observed_label')
            if r['qa_sha256'] not in strong: reasons.append('question_not_unique_explicit_label')
            panel='ambiguity_audit_only' if reasons else label_panel[name]
        records.append({'qa_sha256':r['qa_sha256'],'candidate_label':name,'training_pool':r['training_pool'],
                        'input_pool':r['pool'],'panel':panel,'reasons':reasons,
                        'official_all_record_evaluation_required':True,
                        'training_membership_unchanged':True})
    cal_names=ordered(parts['retain']['calibration'],design['data']['calibration_seed'],'extra_fourth_record')
    calibration=[]
    for j,name in enumerate(cal_names):
        choices=[r['qa_sha256'] for r in records if r['panel']=='calibration' and r['candidate_label']==name]
        count=4 if j<20 else 3
        selected=ordered(choices,design['data']['calibration_seed'],'records/'+name)[:count]
        require(len(selected)==count,'calibration label has too few eligible records')
        calibration.extend(selected)
    require(len(calibration)==len(set(calibration))==128,'calibration 128 coverage')
    require(not set(calibration)&perturbed_ids,'perturbed records used for calibration')
    ref={r['qa_sha256']:r for r in records}
    require(all(ref[i]['training_pool']=='retain90' and ref[i]['panel']=='calibration' for i in calibration),'calibration leakage')
    mappings={}
    for filename in ('forget10_perturbed','retain_perturbed'):
        mapped=[{'row':j,'base_qa_sha256':qa_id(r),'panel':ref[qa_id(r)]['panel']}
                for j,r in enumerate(data[filename])]
        require(len({m['base_qa_sha256'] for m in mapped})==len(mapped),'duplicate variant identity')
        mappings[filename]=mapped
    counts=Counter((r['training_pool'],r['panel']) for r in records)
    return {'schema':'org_m2_observed_label_partition_v2','status':'ROW_PARTITION_LOCKED_TOKEN_BINDING_PENDING',
            'biological_author_identity_certified':False, 'global_author_disjointness_certified':False,
            'author_nonmembership_certified':False,'training_authorized':False,'gpu_launch_authorized':False,
            'position_inferred':False,'records_removed_from_training':0,
            'unbounded_identity_uncertainty_present':any(x['unbounded_identity_uncertainty'] for x in cases),
            'counts':[{'training_pool':p,'panel':s,'records':n} for (p,s),n in sorted(counts.items())],
            'excluded_labels':sorted(excluded),'cross_pool_labels':sorted(mixed),
            'potential_identity_links':cases,'label_partitions':parts,'records':records,
            'variant_base_mapping':mappings,'calibration_qa_sha256':calibration,
            'calibration_token_ids_bound':False,
            'training_role_order':{'O':[qa_id(r) for r in data['full']],
                                   'R':[qa_id(r) for r in data['retain90']],
                                   'U':[qa_id(r) for r in data['forget10']]},
            'ordering_algorithm':'sha256_canonical_json_seed_purpose_value_v2',
            'observed_labels_not_independent_training_lineages':True}


def bounded_score_contrast_envelope(values, coefficients, universe):
    """Worst-case omitted-score envelope for a finite-record mean, never an author CI.

    Scores must be bounded in [0,1]. Inputs are identically aligned observed
    subsets for all states. Nonlinear statistics and unbounded NLL are unsupported.
    When every row is measured this collapses to the exact all-record contrast.
    """
    require(universe and len(universe)==len(set(universe)),'empty/duplicate universe')
    require(values and set(values)==set(coefficients),'state coefficient coverage')
    require(all(type(c) in (int,float) and math.isfinite(c) for c in coefficients.values()),'invalid coefficient')
    keys=[set(v) for v in values.values()]
    require(all(k==keys[0] for k in keys) and keys[0]<=set(universe),'unaligned or extra scores')
    require(all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for v in values.values() for x in v.values()),'scores must be finite and bounded in [0,1]')
    n=len(universe); missing=n-len(keys[0])
    total=math.fsum(coefficients[s]*math.fsum(v.values()) for s,v in values.items())
    low=math.fsum(min(0,c) for c in coefficients.values())
    high=math.fsum(max(0,c) for c in coefficients.values())
    return {'lower':(total+missing*low)/n,'upper':(total+missing*high)/n,
            'observed_records':n-missing,'unresolved_score_records':missing,
            'scope':'finite_record_mean_score_sensitivity_not_author_identification_or_confidence_interval'}


def bind_calibration_tokens(partition, token_payload):
    """Pure validation/helper. Caller MUST first hash-verify the CPU receipt/files.

    This does not read a runtime, authenticate a receipt or authorize model work.
    It avoids re-tokenizing or selecting new records after model outcomes.
    """
    require(token_payload['role_order']==partition['training_role_order'],'CPU training order drift')
    tokens=token_payload['by_exact_qa_sha256']; roles=partition['training_role_order']
    require(set(tokens)==set(roles['O']) and len(roles['O'])==4000,'CPU token universe drift')
    selected=partition['calibration_qa_sha256']
    require(len(selected)==len(set(selected))==128 and set(selected)<=set(roles['R']), 'calibration selection drift')
    ref={r['qa_sha256']:r for r in partition['records']}
    result=[]
    for identifier in selected:
        r=ref[identifier]; t=tokens[identifier]
        require(set(t)=={'input_ids','labels','attention_mask','prompt_ids'},'token payload fields changed')
        require(r['panel']=='calibration' and r['training_pool']=='retain90' and not r['reasons'], 'calibration row no longer eligible')
        require(r['candidate_label'] not in partition['excluded_labels'],'excluded identity in calibration')
        ids,labels,mask,prompt=[t[k] for k in ('input_ids','labels','attention_mask','prompt_ids')]
        require(0<len(prompt)<len(ids)<=512 and len(ids)==len(labels)==len(mask),'invalid token lengths')
        require(all(type(x) is int and x>=0 for x in ids),'invalid token IDs')
        require(ids[:len(prompt)]==prompt and labels==[-100]*len(prompt)+ids[len(prompt):],'answer label/prefix drift')
        require(mask==[1]*len(ids),'unexpected unpadded tokenizer mask')
        result.append({'qa_sha256':identifier,'candidate_label':r['candidate_label'],**t})
    return {'status':'CALIBRATION_TOKEN_SELECTION_VALIDATED_NOT_RUNTIME_CERTIFIED',
            'records':result,'selected_tokens_sha256':digest(result),
            'training_authorized':False,'gpu_launch_authorized':False}
