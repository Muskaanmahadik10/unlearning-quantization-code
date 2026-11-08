#!/usr/bin/env python3
"""M4 source-grounded manufacturing QA with explicitly synthetic withdrawn revisions.
No language-model use in data construction; all source-derived answers map to original rows.
Design frozen before model outcome inspection. Python 3.10+, stdlib only.
"""
from __future__ import annotations
import hashlib, json, random, re, collections, argparse
from pathlib import Path

SEED=20261101
EXPECTED_SOURCE_SHA='1cf3099355bf17cf5637445907ea4b37d36ffb6301726c671dfbc0cbe2ae095f'
HOLDOUT_WORKPIECES=('散热片组 BE-2A','散热片组 G20','散热片组I5-011','散热片组8-3','散热片 S-G53')
N_OBSOLETE=40; N_RETAIN=240; N_UNSEEN=80
RELATIONS=('rel_processed','rel_wp_parameters','rel_wpseries','rel_wptype','rel_pcprocessed_by','rel_pc_parameters','rel_pcprocessed_follow')

def canon(text):return re.sub(r'\s+', '',str(text)).strip(' \u3000。.:：')
def shaobj(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
def fid(rel,i):return 'SRC-'+shaobj({'relation':rel,'row':i})[:12]

def make_fact(kind,row,index):
    subj,_,obj=row[:3]; short=''
    if kind=='rel_processed':
        qual=row[4]; slot=qual; k=('step',subj,qual)
        q=[f'根据制造工艺图谱，{subj}的{qual}是什么加工工序？',f'{subj}在{qual}安排了什么工艺步骤？',f'查询{str(subj)}的{str(qual)}：对应的加工工序名称是什么？']
    elif kind=='rel_wp_parameters':
        obj=row[3]; qual=f'{row[2]}（{row[4]}）';slot=qual;k=('workpiece_parameter',subj,qual)
        q=[f'根据制造工艺图谱，{subj}的{qual}所记录的参数值是什么？',f'{subj}在{qual}对应哪个参数数值？',f'请提供图谱中{str(subj)}的{str(qual)}的原始记录值。']
    elif kind=='rel_wpseries':
        qual='series';slot=qual;k=('series',subj)
        q=[f'根据制造工艺图谱，{subj}属于哪个产品系列？',f'{subj}的系列归属是什么？',f'查询产品{str(subj)}的系列分类名称。']
    elif kind=='rel_wptype':
        qual='type';slot=qual;k=('type',subj)
        q=[f'根据制造工艺图谱，{subj}属于哪种工件类型？',f'{subj}被归入什么工件类别？',f'查询产品{str(subj)}的工件类型。']
    elif kind=='rel_pcprocessed_by':
        qual='equipment';slot=qual;k=('equipment',subj)
        q=[f'根据制造工艺图谱，{subj}工序使用什么加工机床？',f'执行{subj}工序使用哪种机床设备？',f'查询工艺{str(subj)}对应的机床名称。']
    elif kind=='rel_pc_parameters':
        obj=row[3];qual=row[2];slot=qual;k=('process_parameter',subj,qual)
        q=[f'根据制造工艺图谱，{subj}工序的{qual}参数记录是什么？',f'{subj}操作中{qual}的记录数值是多少？',f'查询工艺{str(subj)}的{str(qual)}原始值。']
    elif kind=='rel_pcprocessed_follow':
        qual=row[4];slot=qual;k=('procedure',subj,qual)
        q=[f'根据制造工艺图谱，{subj}工序的{qual}操作指令是什么？',f'{subj}的{qual}作业步骤具体内容是什么？',f'查询工艺{str(subj)}中{str(qual)}的原始操作步骤。']
    else:raise KeyError(kind)
    return {'id':fid(kind,index),'source_relation':kind,'source_row_index_0based':index,
            'subject':subj,'slot':slot,'answer':str(obj),'prompts':q,'entity':subj,
            'head_is_workpiece':kind in ('rel_processed','rel_wp_parameters','rel_wpseries','rel_wptype'),
            'source_row_sha256':shaobj(row),'source_row':row,'key':k}

def round_robin_by_relation(pool,count,skip_ids=set()):
    buckets={k:sorted((x for x in pool if x['source_relation']==k and x['id'] not in skip_ids),key=lambda z:z['id']) for k in RELATIONS}
    # deterministic balanced ordering then select. Prioritize enough entity-specific records.
    out=[]
    while len(out)<count:
        progressed=False
        for k in RELATIONS:
            if buckets[k]: out.append(buckets[k].pop(0));progressed=True
            if len(out)==count:break
        if not progressed:raise ValueError(f'Only {len(out)}/{count} available')
    return out

def build(source:Path,out:Path):
    src=source.read_bytes(); digest=hashlib.sha256(src).hexdigest()
    if digest!=EXPECTED_SOURCE_SHA:raise ValueError(f'Source sha mismatch {digest} != {EXPECTED_SOURCE_SHA}')
    d=json.loads(src); facts=[]
    for rel in RELATIONS:
        for i,row in enumerate(d['relation'][rel]):
            if rel=='rel_pcprocessed_follow' and (len(row[2])>36 or any(s in row[2] for s in ('左图','右图','图示','如图','1-5','2-3','2-4'))): continue
            facts.append(make_fact(rel,row,i))
    # Detect duplicated questions/slots: exclude all ambiguous semantic slots.
    slots=collections.Counter(tuple(f['key']) for f in facts)
    facts=[f for f in facts if slots[tuple(f['key'])]==1]
    ids={f['id'] for f in facts};assert len(ids)==len(facts)
    groups=[f for f in facts if f['head_is_workpiece'] and f['subject'] in HOLDOUT_WORKPIECES]
    if len(groups)<N_UNSEEN:raise ValueError(f'Only {len(groups)} held-out workpiece facts')
    heldout=round_robin_by_relation(groups,N_UNSEEN)
    heldout_ids={f['id'] for f in heldout}
    # All facts concerning heldout workpieces excluded from training (even those not in 80 evaluation).
    eligible=[f for f in facts if not (f['head_is_workpiece'] and f['subject'] in HOLDOUT_WORKPIECES)]
    # 40 obsolete revisions of preexisting workpiece-process records, spread across workpieces.
    candidates=[f for f in eligible if f['source_relation']=='rel_processed']
    by_wp=collections.defaultdict(list)
    for x in candidates:by_wp[x['subject']].append(x)
    for k in by_wp: by_wp[k].sort(key=lambda z:z['id'])
    old_bases=[]
    while len(old_bases)<N_OBSOLETE:
        changed=False
        for wp in sorted(by_wp):
            if by_wp[wp]:old_bases.append(by_wp[wp].pop(0));changed=True
            if len(old_bases)==N_OBSOLETE:break
        if not changed:raise ValueError('Insufficient distinct processing slots')
    old_base_ids={f['id'] for f in old_bases}
    other=round_robin_by_relation(eligible,N_RETAIN-N_OBSOLETE,old_base_ids)
    retain=old_bases+other
    if len(set(f['id'] for f in retain))!=N_RETAIN:raise ValueError('Retain duplicate')
    # Synthetic old process names selected from actual process vocab, but not in any step of same workpiece.
    vocab=sorted(set(x['answer'] for x in facts if x['source_relation']=='rel_processed'))
    workpiece_actual=collections.defaultdict(set)
    for f in facts:
        if f['source_relation']=='rel_processed':workpiece_actual[f['subject']].add(f['answer'])
    old_records=[]
    for f in old_bases:
        alternatives=[v for v in vocab if v not in workpiece_actual[f['subject']]]
        if not alternatives:raise ValueError('No safe synthetic alternatives')
        alt=alternatives[int(shaobj(f['id'])[:10],16)%len(alternatives)]
        rev='SIM-OBSOLETE-'+f['id'][4:]
        subj=f['subject'];slot=f['slot']
        q=[f'在虚构的已撤销旧版工艺卡{rev}中，{subj}的{slot}记录为哪一道工序？',
           f'查询模拟旧版记录{rev}：{subj}的{slot}原来写的工序名是什么？',
           f'不考虑现行工艺，请指出已撤销模拟版{rev}记录的{subj}之{slot}工序。']
        old_records.append({'id':'OLD-'+f['id'],'source_fact_id':f['id'],'revision_id':rev,
            'answer':alt,'current_answer':f['answer'],'prompts':q,'source_relation':'synthetic_obsolete_process',
            'is_simulated':True,'subject':subj,'slot':slot})
    # Export per fact: one training formulation, one for development, and one sealed final-test formulation.
    # Distinct prompts test phrasing transfer, not fact-level independence.
    manifest={'dataset':'M4-manufacturing-knowledge-revision','data_version':'cp2-20261101',
      'source_doi':'10.17632/dfpv963n4w.1','source_file_sha256':digest,'seed':SEED,
      'language':'zh-CN source, zh-CN deterministic prompts','status':'simulation: obsolete revisions are synthetic, retained answers are source-grounded',
      'holdout_workpieces':list(HOLDOUT_WORKPIECES),'nominal_counts':{'obsolete':N_OBSOLETE,'retained':N_RETAIN,'heldout_relationships':N_UNSEEN},
      'qualifications':['Unseen relationship groups are entire heldout workpieces for workpiece relations; shared process knowledge can recur.','Train/dev/test question wordings for one fact are not independent facts.','No empirical safety/operational manufacturing validation.','No code execution of models has occurred in this data preparation.'],
      'relation_counts_candidates':dict(collections.Counter(f['source_relation'] for f in facts)),
      'relation_counts_retain':dict(collections.Counter(f['source_relation'] for f in retain)),
      'relation_counts_unseen':dict(collections.Counter(f['source_relation'] for f in heldout))}
    out.mkdir(parents=True,exist_ok=True)
    def jsonl(filename,records):
        with (out/filename).open('w',encoding='utf-8') as w:
            for z in records:w.write(json.dumps(z,ensure_ascii=False,sort_keys=True)+'\n')
    jsonl('retained_facts.jsonl',retain)
    jsonl('heldout_facts.jsonl',heldout)
    jsonl('synthetic_obsolete_facts.jsonl',old_records)
    for partition,part in [('retain',retain),('obsolete',old_records),('unseen',heldout)]:
        for section,ix in [('train',0),('dev',1),('test',2)]:
            if partition=='unseen' and section!='test':continue
            items=[{'fact_id':x['id'],'question':x['prompts'][ix],'answer':x['answer'],
                    'source_fact_id':x.get('source_fact_id',x['id']),'partition':partition,
                    'is_simulated':partition=='obsolete'} for x in part]
            jsonl(f'{partition}_{section}.jsonl',items)
    manifest['record_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.jsonl'))}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # strict invariants
    assert len(retain)==N_RETAIN and len(old_records)==N_OBSOLETE and len(heldout)==N_UNSEEN
    assert not set(f['id'] for f in retain)&heldout_ids
    assert all(f['subject'] not in HOLDOUT_WORKPIECES for f in retain if f['head_is_workpiece'])
    assert all(o['answer']!=o['current_answer'] for o in old_records)
    assert all(len(set(x['prompts']))==3 for x in retain+heldout+old_records)
    texts=[q for a in retain+heldout+old_records for q in a['prompts']]
    assert len(texts)==len(set(texts)), 'exact prompt leakage'
    print(json.dumps({'PASS':True,'candidate_facts':len(facts),'heldout_source_group_pool':len(groups),'records':manifest['nominal_counts'],'retained_kinds':manifest['relation_counts_retain'],'source_sha256':digest},ensure_ascii=False,indent=2))
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='mpkg_v1.json');ap.add_argument('--output',default='M4_CP2_Data');args=ap.parse_args()
    build(Path(args.source),Path(args.output))
