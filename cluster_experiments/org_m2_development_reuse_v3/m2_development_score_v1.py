"""Only frozen development QAs; unchanged qualified upstream scoring bodies."""
from types import SimpleNamespace
import math
from m2_development_math_v1 import require

def score(torch,model,tokenizer,panels,generation,upstream,on_panel=None):
    from m2_real_evaluator_v1 import datasets,collator
    from m2_metric_assembly_v1 import upstream_namespace,truth
    from m2_eval_qualification_oracle_v1 import ObservedModel,key
    from m2_eval_qualification_common_v1 import compare
    from omegaconf import OmegaConf
    ns=upstream_namespace(upstream);observed=ObservedModel(torch,model)
    class View:
        device=model.device
        def __call__(self,**batch):return SimpleNamespace(logits=observed(**batch,use_cache=False).logits.float())
        def generate(self,*a,**kw):return observed.generate(*a,**kw)
    view=View();result={};normal=collator(torch,tokenizer.pad_token_id);gen=collator(torch,tokenizer.pad_token_id,True)
    for panel,rows in panels.items():
        require(rows,'empty shard panel');out={};values={}
        for kind in rows[0]['tokens']:
            cursor=len(observed.forward_calls)
            v=ns['probability'](view,data=datasets(rows,kind),collators=normal,batch_size=2)
            vals={str(i):x for i,x in v['value_by_index'].items()};require(set(vals)=={r['index'] for r in rows},'score index coverage')
            # Check each actual production forward against the independent
            # class-axis oracle used in the passed H200 qualification.
            for start in range(0,len(rows),2):
                batch=rows[start:start+2];expected={r['index']:[] for r in batch}
                for variant in range(len(batch[0]['tokens'][kind])):
                    call=observed.forward_calls[cursor];cursor+=1;require(len(call)==len(batch),'batch coverage')
                    for row,item in zip(batch,call):
                        require(item['key']==key(row['tokens'][kind][variant]),'token/order drift')
                        expected[row['index']].append(item)
                for row in batch:
                    e={k:[x[k] for x in expected[row['index']]] for k in ('avg_loss','prob')}
                    if len(expected[row['index']])==1:e={k:v[0] for k,v in e.items()}
                    compare(e,vals[row['index']],path=panel+'/'+kind+'/'+row['index'])
            require(cursor==len(observed.forward_calls),'extra forward calls');values[kind]=vals
        before=len(observed.generations)
        g=ns['rouge'](view,data=datasets(rows,'answer',True),collators=gen,batch_size=1,tokenizer=tokenizer,
                     generation_args=OmegaConf.create(generation),rouge_type='rougeL_recall')['value_by_index']
        g={str(i):v for i,v in g.items()};require(set(g)=={r['index'] for r in rows},'generation index coverage')
        events=observed.generations[before:];require(len(events)==len(rows),'generation count')
        tr=None
        if panel in ('forget','retain'):
            tr=truth({r['index']:values['paraphrase'][r['index']]['avg_loss'] for r in rows},
                     {r['index']:values['wrong'][r['index']]['avg_loss'] for r in rows},panel=='forget')
        for row,event in zip(rows,events):
            i=row['index'];require(event['prompt_ids']==[row['tokens']['answer'][0]['prompt_ids']],'prompt drift')
            from rouge_score.rouge_scorer import RougeScorer
            e=row['tokens']['answer'][0]
            decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            prompt=decode(e['prompt_ids']);ground=decode(e['input_ids']).replace(prompt,'').strip()
            generated=decode(event['generated_ids'][0]);eos=tokenizer.decode([tokenizer.eos_token_id])
            if eos and eos in generated:generated=generated.split(eos)[0]
            generated=generated.strip();rr=RougeScorer(['rouge1','rougeL'],use_stemmer=True).score(ground,generated)
            compare({'input':prompt,'ground_truth':ground,'generation':generated,'rouge1_recall':rr['rouge1'].recall,
                     'rougeL_recall':rr['rougeL'].recall,'rougeL_f1':rr['rougeL'].fmeasure},g[i],0)
            a=values['answer'][i]
            require(all(math.isfinite(float(a[k])) for k in ('avg_loss','prob')),'nonfinite')
            rawtr=tr['value_by_index'][i]['score'] if tr else None
            bounded=min(rawtr,1/(rawtr+1e-10)) if panel=='forget' else max(0,1-rawtr) if tr else 0.
            out[row['qa_sha256']]={'index':i,'metrics':{'prob':float(a['prob']),'nll':float(a['avg_loss']),
              'rouge':float(g[i]['rougeL_recall']),'truth':float(bounded)},'probability':{k:v[i] for k,v in values.items()},
              'rouge':g[i],'generation':event,'raw_wrong_correct_truth_ratio':rawtr}
        result[panel]=out
        if on_panel is not None:on_panel(panel,out)
        print('M2_DEVELOPMENT_PANEL='+panel+':'+str(len(rows))+':PASS',flush=True)
    # Do not retain model-sized tensors in the evidence.
    return {'panels':result,'oracle_forward_calls':observed.forward_calls,'layout_diagnostics':observed.layout_diagnostics,
            'timings':observed.timings,'generation_batch_size':1,'scoring_batch_size':2}
