"""Real-model probability/greedy-ROUGE primitive parity on fixed calibration rows.

Not the complete seven-metric benchmark evaluator; its release remains separate.
"""
import ast
from pathlib import Path
import time
from m2_pilot_contract_v1 import require


def probability(torch,logits,labels):
    import torch.nn.functional as F
    y=labels[:,1:]
    require(bool(y.ne(-100).any(-1).all()),'empty evaluated answer')
    losses=F.cross_entropy(logits[:,:-1,:].float().transpose(1,2),y,ignore_index=-100,reduction='none')
    avg=losses.sum(-1)/labels.ne(-100).sum(-1)
    return [{'prob':float(p),'avg_loss':float(n)} for p,n in zip(torch.exp(-avg),avg)]


def oracles(torch):
    from omegaconf import OmegaConf
    from rouge_score import rouge_scorer
    p=Path(__file__).parent/'upstream_eval_utils.py'
    tree=ast.parse(p.read_text(encoding='utf-8'))
    wanted={'evaluate_probability','eval_text_similarity'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    require(len(nodes)==len(wanted),'pinned evaluator function coverage')
    ns={'torch':torch,'nn':torch.nn,'IGNORE_INDEX':-100,'OmegaConf':OmegaConf,'rouge_scorer':rouge_scorer}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns


def evaluate(torch,model,tokenizer,records,generation_config):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from rouge_score.rouge_scorer import RougeScorer
    from m2_training_core_v1 import collate
    require(len(records)==3,'pilot evaluator records changed')
    require(generation_config=={'do_sample':False,'max_new_tokens':200,'temperature':None,'top_p':None,'use_cache':True},'generation policy drift')
    model.eval(); oracle=oracles(torch);scores=[]; generated=[];timings=[]

    class View:
        device=model.device
        def __call__(self,**batch):
            out=model(**batch,use_cache=False)
            return SimpleNamespace(logits=out.logits.float())
        def generate(self,*args,**kwargs):
            result=model.generate(*args,**kwargs)
            self.last_ids=result.detach().cpu().tolist()
            return result
    view=View()
    with torch.no_grad():
        for start in (0,2):
            selected=records[start:start+2]
            batch={k:v.to(model.device) for k,v in collate(torch,selected,tokenizer.pad_token_id).items()}
            logits=view(**batch).logits
            ours=probability(torch,logits,batch['labels']);ref=oracle['evaluate_probability'](view,batch)
            require(len(ours)==len(ref)==len(selected),'probability row coverage')
            for r,a,b in zip(selected,ours,ref):
                require(all(abs(a[k]-b[k])<=1e-6 for k in ('prob','avg_loss')),'upstream FP32 probability parity')
                scores.append({'qa_sha256':r['qa_sha256'],**a})
        # Single-example decoding matches the locked prompt, avoids right-pad
        # decoder ambiguity and measures the full 200-token maximum, not a stub.
        for r in records[:2]:
            ids=torch.tensor([r['prompt_ids']],device=model.device,dtype=torch.long)
            mask=ids.ne(tokenizer.pad_token_id)
            labels=torch.tensor([r['input_ids']],device=model.device,dtype=torch.long)
            batch={'input_ids':ids,'attention_mask':mask,'labels':labels}
            torch.cuda.synchronize();t=time.perf_counter()
            ref=oracle['eval_text_similarity'](view,tokenizer,batch,OmegaConf.create(generation_config))[0]
            torch.cuda.synchronize();timings.append(time.perf_counter()-t)
            ref_ids=view.last_ids
            actual=model.generate(ids,attention_mask=mask,pad_token_id=tokenizer.eos_token_id,**generation_config)
            require(actual.detach().cpu().tolist()==ref_ids,'greedy IDs differ from pinned generation call')
            input_text=tokenizer.decode(r['prompt_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=True)
            full_text=tokenizer.decode(r['input_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=True)
            gt=full_text.replace(input_text,'').strip()
            text=tokenizer.decode(actual[0,len(r['prompt_ids']):],skip_special_tokens=True,clean_up_tokenization_spaces=True)
            eos=tokenizer.decode([tokenizer.eos_token_id])
            if eos and eos in text:text=text.split(eos)[0]
            text=text.strip();rouge=RougeScorer(['rouge1','rougeL'],use_stemmer=True).score(gt,text)
            require(text==ref['generation'] and gt==ref['ground_truth'] and
                    rouge['rougeL'].recall==ref['rougeL_recall'] and rouge['rougeL'].fmeasure==ref['rougeL_f1'] and
                    rouge['rouge1'].recall==ref['rouge1_recall'],'ROUGE/text primitive mismatch')
            generated.append({'qa_sha256':r['qa_sha256'],'ids':ref_ids[0][len(r['prompt_ids']):],
                              'rougeL_recall':rouge['rougeL'].recall,'rougeL_f1':rouge['rougeL'].fmeasure})
    return {'scope':'fixed_calibration_primitives_not_full_benchmark','scores':scores,
            'generation':generated,'generation_seconds':timings,'upstream_probability_and_rouge_parity':True,
            'full_seven_metric_evaluator_certified':False}
