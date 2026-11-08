"""Reusable all-record evaluator. No GPU CLI/authorization in training release.

Generation remains batch-one for proved prompt semantics. A separately admitted
evaluation launch must measure/approve its full cost before running all states.
"""
from pathlib import Path
from types import SimpleNamespace
from m2_stage_common_v1 import require,canonical

def datasets(rows,kind,generation=False):
    output=[]
    for r in rows:
        entries=r['tokens'][kind];variants={}
        for j,e in enumerate(entries):
            variants[str(j)]={'input_ids':e['prompt_ids'] if generation else e['input_ids'],
                 'labels':e['input_ids'] if generation else e['labels'],'index':int(r['index'])}
        require(bool(variants),'empty answer variants')
        output.append(next(iter(variants.values())) if len(variants)==1 else variants)
    return output

def collator(torch,pad_id,generation=False):
    def single(rows):
        n=max(len(r['input_ids']) for r in rows);m=max(len(r['labels']) for r in rows)
        ids=torch.full((len(rows),n),pad_id,dtype=torch.long);labels=torch.full((len(rows),m),-100,dtype=torch.long)
        for i,r in enumerate(rows):
            k=len(r['input_ids']);start=n-k if generation else 0
            ids[i,start:start+k]=torch.tensor(r['input_ids']);labels[i,:len(r['labels'])]=torch.tensor(r['labels'])
        return {'input_ids':ids,'labels':labels,'attention_mask':ids.ne(pad_id),'index':torch.tensor([r['index'] for r in rows])}
    def apply(rows):
        if 'input_ids' in rows[0]:return single(rows)
        require(all(list(r)==list(rows[0]) for r in rows),'alternative count mismatch')
        return {k:single([r[k] for r in rows]) for k in rows[0]}
    return apply

def primitive_suite(torch,model,tokenizer,panels,generation,upstream_dir,batch_size=8):
    from m2_metric_assembly_v1 import upstream_namespace,truth,normalized,min_k,auc,mean,UTILITY
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    require(set(panels)=={'forget','retain','ra','wf','holdout','retain_all'},'full panel coverage')
    require(generation=={'do_sample':False,'max_new_tokens':200,'temperature':None,'top_p':None,'use_cache':True},'generation lock')
    require(type(batch_size) is int and 1<=batch_size<=8,'unadmitted batch size')
    model.eval();ns=upstream_namespace(upstream_dir)
    class View:
        device=model.device
        def __call__(self,**batch):return SimpleNamespace(logits=model(**batch,use_cache=False).logits.float())
        def generate(self,*a,**kw):return model.generate(*a,**kw)
    view=View();scores={};rouges={};truths={};norms={};row_outputs={}
    normal=collator(torch,tokenizer.pad_token_id);gen_collate=collator(torch,tokenizer.pad_token_id,True)
    for panel,rows in panels.items():
        require(rows and [r['index'] for r in rows]==[str(i) for i in range(len(rows))],'row coverage/order '+panel)
        result={}
        for kind in rows[0]['tokens']:
            v=ns['probability'](view,data=datasets(rows,kind),collators=normal,batch_size=batch_size)
            result[kind]={str(k):val for k,val in v['value_by_index'].items()}
        scores[panel]=result
        if panel!='holdout':
            v=ns['rouge'](view,data=datasets(rows,'answer',True),collators=gen_collate,batch_size=1,
                          tokenizer=tokenizer,generation_args=OmegaConf.create(generation),rouge_type='rougeL_recall')
            rouges[panel]={str(k):val for k,val in v['value_by_index'].items()}
        if panel in ('forget','retain','ra','wf'):
            c=result['paraphrase' if panel in ('forget','retain') else 'answer'];w=result['wrong']
            truths[panel]=truth({i:v['avg_loss'] for i,v in c.items()},{i:v['avg_loss'] for i,v in w.items()},panel=='forget')
            if panel in ('ra','wf'):
                norms[panel]=normalized({i:v['prob'] for i,v in c.items()},{i:v['prob'] for i,v in w.items()})
        row_outputs[panel]={'qa_sha256_by_index':{r['index']:r['qa_sha256'] for r in rows},'probability':result,
                            'rouge':rouges.get(panel),'truth_ratio':truths.get(panel)}
    mia=ns['mia_auc'](ns['MinKProbAttack'],view,{k:datasets(panels[k],'answer') for k in ('forget','holdout')},normal,batch_size,k=.4)
    es=ns['extraction_strength'](view,data=datasets(panels['forget'],'answer'),collators=normal,batch_size=batch_size)
    util={}
    for panel in ('retain','ra','wf'):
        pk=panel+'_Q_A_Prob'+('_normalised' if panel!='retain' else '')
        util[pk]=float(norms[panel]['agg_value']) if panel!='retain' else mean([v['prob'] for v in scores[panel]['answer'].values()])
        util[panel+'_Q_A_ROUGE']=mean([v['rougeL_recall'] for v in rouges[panel].values()])
        util[panel+'_Truth_Ratio']=float(truths[panel]['agg_value'])
    require(set(util)==set(UTILITY),'utility component coverage')
    primitives={'forget_truth':truths['forget'],
      'forget_probability':{i:v['prob'] for i,v in scores['forget']['answer'].items()},
      'forget_rouge':{i:v['rougeL_recall'] for i,v in rouges['forget'].items()},'utility':util,
      'mia_forget':{str(i):v['score'] for i,v in mia['forget']['value_by_index'].items()},
      'mia_holdout':{str(i):v['score'] for i,v in mia['holdout']['value_by_index'].items()},
      'extraction':{str(i):v['score'] for i,v in es['value_by_index'].items()}}
    return {'primitives':primitives,'row_outputs':row_outputs,'mia_auc':float(mia['agg_value']),
            'generation':generation,'no_rows_filtered':True,'observed_label_macro_is_not_author_macro':True}

def assemble_controlled(result,identity,reference,expected_reference_sha):
    from m2_metric_assembly_v1 import assemble
    require(reference is not None and canonical(reference)==expected_reference_sha,'reference bytes/identity changed')
    require(reference['identity']['role']=='R' and reference['identity']['parent_role']=='B','controlled R lineage required')
    for key in ('data_binding_sha256','generation','base_sha256','source_lock_sha256'):
        require(identity[key]==reference['identity'][key],'reference protocol drift '+key)
    require(reference['identity']['checkpoint_sha256'] and identity['checkpoint_sha256'],'checkpoint identity absent')
    r={'role':'controlled_R','forget_truth':reference['result']['primitives']['forget_truth'],'mia_auc':reference['result']['mia_auc']}
    return assemble(result['primitives'],r,scope='controlled_scientific')
