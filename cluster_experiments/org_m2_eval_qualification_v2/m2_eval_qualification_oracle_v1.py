"""Observe actual model calls and independently check returned indexed metrics."""
import math
import time
from m2_eval_qualification_common_v1 import require,canonical,compare

def key(entry):return canonical({k:entry[k] for k in ('input_ids','labels')})

class ObservedModel:
    def __init__(self,torch,model):
        self.torch=torch;self.model=model;self.forward_calls=[];self.generations=[];self.timings=[];self.layout_diagnostics=[]
    @property
    def device(self):return self.model.device
    def eval(self):self.model.eval();return self
    def sync(self):
        if self.device.type=='cuda':self.torch.cuda.synchronize()
    def __call__(self,**batch):
        t=self.torch;self.sync();start=time.perf_counter();out=self.model(**batch);self.sync()
        seconds=time.perf_counter()-start
        require(not out.logits.requires_grad and bool(t.isfinite(out.logits).all()),'nonfinite/gradient-bearing inference logits')
        logits=out.logits.float();ys=batch['labels'];ids=batch['input_ids']
        # No full probability tensor retained. Independently gather labelled
        # token log probabilities and compare official CE, MinK and ES outputs.
        from m2_eval_oracle_repair_v2 import probability_oracle
        averages,probs,diagnostic=probability_oracle(t,logits,ys,ids)
        self.layout_diagnostics.append(diagnostic)
        lp=t.nn.functional.log_softmax(logits,-1)[:,:-1,:]
        selected=lp.gather(-1,ids[:,1:].unsqueeze(-1)).squeeze(-1)
        call=[]
        for i in range(len(ids)):
            labelled=(ys[i]!=-100).nonzero(as_tuple=True)[0]
            require(len(labelled)>1 and int(labelled[0])>0,'empty/invalid labelled sequence')
            length=int(labelled[-1])+1;positions=labelled-1
            nll=float(averages[i]);prob=float(probs[i]);p=labelled[:-1]-1
            vals=selected[i,p].cpu().tolist();truth=ys[i,labelled[:-1]].cpu().tolist()
            pred=logits[i,p].argmax(-1).cpu().tolist()
            import numpy as np
            k=max(1,int(len(vals)*.4));mk=float(-np.asarray(sorted(vals)[:k],dtype=np.float32).mean())
            for cut in range(len(truth)):
                if pred[cut:]==truth[cut:]:break
            es=1-cut/len(truth)
            entry={'input_ids':ids[i,:length].cpu().tolist(),'labels':ys[i,:length].cpu().tolist()}
            call.append({'key':key(entry),'avg_loss':nll,'prob':prob,'min_k':mk,'extraction':es})
        self.forward_calls.append(call)
        self.timings.append({'kind':'forward','seconds':seconds,'batch':len(ids),'padded_tokens':int(ids.numel())})
        return out
    def generate(self,input_ids,**kwargs):
        self.sync();start=time.perf_counter();out=self.model.generate(input_ids,**kwargs);self.sync()
        self.timings.append({'kind':'generate','seconds':time.perf_counter()-start,'batch':len(input_ids),
                             'prompt_tokens':int(input_ids.numel()),'new_tokens':int(out.numel()-input_ids.numel())})
        self.generations.append({'prompt_ids':input_ids.cpu().tolist(),'generated_ids':out[:,input_ids.shape[1]:].cpu().tolist()})
        return out

def validate_result(torch,result,panels,observed,tokenizer,ns):
    from rouge_score.rouge_scorer import RougeScorer
    from m2_metric_assembly_v1 import auc,normalized
    indices=[str(i) for i in range(3)];max_delta=0.;genix=0;checks=0;cursor=0
    def consume(rows,kind):
        nonlocal cursor
        expected={r['index']:[] for r in rows}
        for start in range(0,len(rows),2):
            batch=rows[start:start+2]
            for variant in range(len(batch[0]['tokens'][kind])):
                require(cursor<len(observed.forward_calls),'missing model call')
                call=observed.forward_calls[cursor];cursor+=1
                require(len(call)==len(batch),'partial batch coverage')
                for row,item in zip(batch,call):
                    require(item['key']==key(row['tokens'][kind][variant]),'model call token identity/order')
                    expected[row['index']].append(item)
        return expected
    require(set(result['row_outputs'])==set(panels),'panel coverage')
    for panel,rows in panels.items():
        out=result['row_outputs'][panel]
        require(out['qa_sha256_by_index']=={r['index']:r['qa_sha256'] for r in rows},'QA binding')
        require(set(out['probability'])==set(rows[0]['tokens']),'variant coverage')
        for kind in rows[0]['tokens']:
            values=out['probability'][kind];require(list(values)==indices,'probability index coverage')
            oracles=consume(rows,kind)
            for row in rows:
                entries=row['tokens'][kind];expected=oracles[row['index']]
                record={k:[e[k] for e in expected] for k in ('avg_loss','prob')}
                if len(entries)==1:record={k:v[0] for k,v in record.items()}
                max_delta=max(max_delta,compare(record,values[row['index']],path='probability/'+panel+'/'+kind+'/'+row['index']));checks+=len(entries)
        if panel!='holdout':
            require(list(out['rouge'])==indices,'ROUGE index coverage')
            for row in rows:
                event=observed.generations[genix];genix+=1;e=row['tokens']['answer'][0]
                require(event['prompt_ids']==[e['prompt_ids']],'generation prompt identity')
                decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=True)
                prompt=decode(e['prompt_ids']);ground=decode(e['input_ids']).replace(prompt,'').strip()
                generated=decode(event['generated_ids'][0]);eos=tokenizer.decode([tokenizer.eos_token_id])
                if eos and eos in generated:generated=generated.split(eos)[0]
                generated=generated.strip();rr=RougeScorer(['rouge1','rougeL'],use_stemmer=True).score(ground,generated)
                expected={'input':prompt,'ground_truth':ground,'generation':generated,'rouge1_recall':rr['rouge1'].recall,
                          'rougeL_recall':rr['rougeL'].recall,'rougeL_f1':rr['rougeL'].fmeasure}
                compare(expected,out['rouge'][row['index']],0)
        if panel in ('forget','retain','ra','wf'):
            correct=out['probability']['paraphrase' if panel in ('forget','retain') else 'answer'];wrong=out['probability']['wrong']
            pre={'correct':{'value_by_index':correct},'wrong':{'value_by_index':wrong}}
            official=ns['truth_ratio'](None,pre_compute=pre,aggregator='closer_to_1_better' if panel=='forget' else 'true_better')
            # numpy scalars normalized through JSON-shaped float values.
            native={'agg_value':float(official['agg_value']),'value_by_index':{i:{'score':float(v['score'])} for i,v in official['value_by_index'].items()}}
            require(list(native['value_by_index'])==list(out['truth_ratio']['value_by_index']),'truth index alignment')
            pairs=[(native['agg_value'],out['truth_ratio']['agg_value'])]+[(v['score'],out['truth_ratio']['value_by_index'][i]['score']) for i,v in native['value_by_index'].items()]
            require(all(math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in pairs),'truth arithmetic parity')
            if panel in ('ra','wf'):
                official=ns['probability_w_options'](None,pre_compute=pre)
                ours=normalized({i:v['prob'] for i,v in correct.items()},{i:v['prob'] for i,v in wrong.items()})
                require(abs(float(official['agg_value'])-ours['agg_value'])<=1e-12,'normalized probability parity')
    require(genix==len(observed.generations)==15,'generation coverage')
    p=result['primitives']
    for panel,k in [('forget','mia_forget'),('holdout','mia_holdout')]:
        calls=consume(panels[panel],'answer')
        expected={i:v[0]['min_k'] for i,v in calls.items()}
        max_delta=max(max_delta,compare(expected,p[k]))
    calls=consume(panels['forget'],'answer');expected={i:v[0]['extraction'] for i,v in calls.items()}
    compare(expected,p['extraction'],0)
    require(cursor==len(observed.forward_calls),'extra/unvalidated model calls')
    require(abs(result['mia_auc']-auc(list(p['mia_forget'].values()),list(p['mia_holdout'].values())))<=1e-12,'MIA orientation parity')
    return {'status':'REAL_MODEL_INDEXED_PRIMITIVE_PARITY_PASS','answer_variant_checks':checks,
            'generations':genix,'max_probability_or_NLL_delta':max_delta,'ES_upstream_floor_preserved':True}

def generation_batch_trial(torch,model,tokenizer,records,generation):
    """Fixed eight already-calibration records, no scientific outcomes or tuning.

    A token mismatch rejects batch-eight eligibility, not the batch-one receipt.
    It does not relax the exact-greedy-ID requirement or silently select a batch.
    """
    require(len(records)==8,'eight generation timing records required')
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    require(isinstance(eos,list) and eos and all(type(i) is int for i in eos),'EOS configuration')
    def trim(ids):
        for i,v in enumerate(ids):
            if v in eos:return ids[:i+1]
        return ids
    def run(rows):
        n=max(len(r['prompt_ids']) for r in rows)
        ids=torch.full((len(rows),n),tokenizer.eos_token_id,dtype=torch.long,device=model.device)
        for i,r in enumerate(rows):ids[i,n-len(r['prompt_ids']):]=torch.tensor(r['prompt_ids'],device=model.device)
        if model.device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        out=model.generate(ids,attention_mask=ids.ne(tokenizer.eos_token_id),pad_token_id=tokenizer.eos_token_id,**generation)
        if model.device.type=='cuda':torch.cuda.synchronize()
        return [trim(x) for x in out[:,n:].cpu().tolist()],time.perf_counter()-start
    single=[];seconds=[]
    with torch.no_grad():
        for r in records:
            ids,sec=run([r]);single+=ids;seconds.append(sec)
        batch,bsec=run(records)
    return {'scope':'EIGHT_FIXED_RETAIN_CALIBRATION_RECORDS_ONLY','records':[r['qa_sha256'] for r in records],
      'single_ids':single,'batch8_ids':batch,'greedy_ids_exact':single==batch,'batch8_candidate': 'PASS' if single==batch else 'REJECTED_TOKEN_MISMATCH',
      'singleton_seconds':seconds,'batch8_seconds':bsec,'observed_speedup':sum(seconds)/bsec,
      'full_benchmark_batch8_authorized':False,'automatic_batch_selection':False}
