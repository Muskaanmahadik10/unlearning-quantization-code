"""CPU-only integration against pinned functions; no checkpoint/data downloads."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import numpy as np
from omegaconf import OmegaConf
from m2_metric_assembly_v1 import *


class ToyModel:
    device=torch.device('cpu')
    def __call__(self,input_ids,attention_mask=None,labels=None):
        v=torch.arange(7,dtype=torch.float32)
        logits=-.7*(v-((input_ids+1)%7).unsqueeze(-1)).square()
        return SimpleNamespace(logits=logits)
    def generate(self,input_ids,**kwargs):
        return torch.cat((input_ids,torch.tensor([[3,4,0]]*len(input_ids))),dim=1)


class ToyTokenizer:
    eos_token_id=0
    words={0:'<eos>',1:'Question',2:'prompt',3:'blue',4:'bird',5:'green',6:'tree'}
    def decode(self,ids,skip_special_tokens=False,**kwargs):
        return ' '.join(self.words[int(i)] for i in ids if not(skip_special_tokens and int(i)==0))
    def batch_decode(self,rows,**kwargs):return [self.decode(r,**kwargs) for r in rows]


def dataset():
    return [{'input_ids':[1,2,3,4,0],'labels':[-100,-100,3,4,0],'attention_mask':[1,1,1,1,0],'index':11},
            {'input_ids':[1,2,5,6,0],'labels':[-100,-100,5,6,0],'attention_mask':[1,1,1,1,0],'index':22},
            {'input_ids':[1,2,6,4,0],'labels':[-100,-100,6,4,0],'attention_mask':[1,1,1,1,0],'index':33}]


def collate(rows):
    return {k:torch.tensor([r[k] for r in rows]) for k in rows[0]}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns=upstream_namespace(Path(__file__).parent/'upstream')
    def test_probability_mask_partial_batch(self):
        n=self.ns;data=dataset();model=ToyModel()
        result=n['probability'](model,data=data,collators=collate,batch_size=2)
        self.assertEqual(list(result['value_by_index']),[11,22,33])
        batch=collate(data); logits=model(**{k:v for k,v in batch.items() if k!='index'}).logits
        for i,idx in enumerate((11,22,33)):
            ys=batch['labels'][i];s=0
            for j in range(1,len(ys)):
                if ys[j]!=-100:s+=float(-torch.log_softmax(logits[i,j-1],-1)[ys[j]])
            expected=s/int((ys!=-100).sum())
            self.assertAlmostEqual(result['value_by_index'][idx]['avg_loss'],expected,places=6)
    def test_multi_answer_transpose_and_duplicate_rejection(self):
        n=self.ns;data=[{'0':r,'1':{**r,'input_ids':[1,2,6,4,0]}} for r in dataset()]
        def multi(rows):return {k:collate([r[k] for r in rows]) for k in ('0','1')}
        v=n['probability'](ToyModel(),data=data,collators=multi,batch_size=2)
        self.assertEqual(list(v['value_by_index']),[11,22,33])
        self.assertTrue(all(len(x['prob'])==2 for x in v['value_by_index'].values()))
        with self.assertRaises(AssertionError):
            n['probability'](ToyModel(),data=dataset()+dataset()[:1],collators=collate,batch_size=2)
    def test_truth_all_aggregators_and_options(self):
        n=self.ns;c={'a':1.2,'b':2.3,'c':.4};w={'a':[2.,4.],'b':[1.,3.],'c':[2.,3.]}
        pre={'correct':{'value_by_index':{i:{'avg_loss':v,'prob':math.exp(-v)} for i,v in c.items()}},
             'wrong':{'value_by_index':{i:{'avg_loss':v,'prob':[math.exp(-x) for x in v]} for i,v in w.items()}}}
        for isforget,name in ((True,'closer_to_1_better'),(False,'true_better')):
            a=truth(c,w,isforget);b=n['truth_ratio'](None,pre_compute=pre,aggregator=name)
            self.assertAlmostEqual(a['agg_value'],b['agg_value'],places=12)
            for i in c:self.assertAlmostEqual(a['value_by_index'][i]['score'],b['value_by_index'][i]['score'],places=12)
        a=normalized({i:math.exp(-v) for i,v in c.items()},{i:[math.exp(-x) for x in v] for i,v in w.items()})
        b=n['probability_w_options'](None,pre_compute=pre)
        self.assertAlmostEqual(a['agg_value'],b['agg_value'],places=12)
    def test_tokenwise_eos_and_min_k(self):
        n=self.ns;b=collate(dataset());b.pop('index');model=ToyModel()
        lps,labels=n['tokenwise_logprobs'](model,b,return_labels=True)
        all_logits=model(**b).logits
        for i,(lp,ys) in enumerate(zip(lps,labels)):
            self.assertEqual(ys.tolist(),b['input_ids'][i,2:4].tolist())
            expected=[float(torch.log_softmax(all_logits[i,j-1],-1)[b['input_ids'][i,j]]) for j in (2,3)]
            self.assertTrue(torch.equal(lp,torch.tensor(expected)))
            attack=n['MinKProbAttack'](model=model,data=dataset(),collator=collate,batch_size=2,k=.4)
            self.assertAlmostEqual(attack.compute_score(lp),min_k(expected,.4),places=6)
        self.assertEqual(min_k([]),0)
    def test_extraction_against_actual_upstream(self):
        n=self.ns;model=ToyModel();data=dataset()
        result=n['extraction_strength'](model,data=data,collators=collate,batch_size=2)
        b=collate(data);b.pop('index');logits=model(**b).logits
        expected=[]
        for i,row in enumerate(data):
            preds=logits[i,1:3].argmax(-1).tolist();labels=row['input_ids'][2:4]
            expected.append(extraction(preds,labels))
            self.assertAlmostEqual(result['value_by_index'][row['index']]['score'],expected[-1])
        self.assertAlmostEqual(result['agg_value'],mean(expected))
    def test_rouge_generation_wrapper(self):
        data=[{'input_ids':[1,2],'attention_mask':[1,1],'labels':[1,2,3,4,0],'index':i} for i in (11,22,33)]
        cfg=OmegaConf.create({'do_sample':False,'max_new_tokens':200,'temperature':None,'top_p':None,'use_cache':True})
        result=self.ns['rouge'](ToyModel(),data=data,collators=collate,batch_size=2,tokenizer=ToyTokenizer(),generation_args=cfg,rouge_type='rougeL_recall')
        self.assertEqual(result['agg_value'],1.0);self.assertEqual(list(result['value_by_index']),[11,22,33])
    def test_mia_auc_actual_attack_pipeline(self):
        n=self.ns;forget=dataset();holdout=[{**r,'input_ids':[1,2,6,6,0],'labels':[-100,-100,6,6,0]} for r in dataset()]
        v=n['mia_auc'](n['MinKProbAttack'],ToyModel(),{'forget':forget,'holdout':holdout},collate,2,k=.4)
        f=[r['score'] for r in v['forget']['value_by_index'].values()];h=[r['score'] for r in v['holdout']['value_by_index'].values()]
        self.assertAlmostEqual(v['agg_value'],auc(f,h),places=12)
    def test_full_seven_metric_nine_component_assembly(self):
        n=self.ns;tr=truth({'a':1,'b':2,'c':3},{'a':[2,3],'b':[4,3],'c':[1,2]},True)
        reftr=truth({'a':2,'b':2,'c':2},{'a':[1,3],'b':[4,4],'c':[2,2]},True)
        p={'forget_truth':tr,'forget_probability':{'a':.1,'b':.2,'c':.3},'forget_rouge':{'a':.2,'b':.4,'c':.6},
           'utility':{k:.2+.05*i for i,k in enumerate(UTILITY)},'mia_forget':{'a':1,'b':2,'c':3},'mia_holdout':{'h1':2,'h2':4},'extraction':{'a':1,'b':.5,'c':.25}}
        ref={'role':'synthetic_R','forget_truth':reftr,'mia_auc':.6}
        out=assemble(p,ref,scope='synthetic_cpu');self.assertEqual(tuple(out['metrics']),METRICS)
        official={'forget_truth_ratio':tr['agg_value'],'forget_quality':n['ks_test'](None,pre_compute={'forget':tr},reference_logs={'retain_model_logs':{'retain':reftr}})['agg_value'],
           'forget_Q_A_Prob':np.mean(list(p['forget_probability'].values())),'forget_Q_A_ROUGE':np.mean(list(p['forget_rouge'].values())),
           'model_utility':n['hm_aggregate'](None,pre_compute={k:{'agg_value':v} for k,v in p['utility'].items()})['agg_value'],
           'privleak':n['privleak'](None,pre_compute={'forget':{'agg_value':out['mia_auc']}},reference_logs={'retain_model_logs':{'retain':{'agg_value':.6}}})['agg_value'],
           'extraction_strength':np.mean(list(p['extraction'].values()))}
        for k in METRICS:self.assertAlmostEqual(out['metrics'][k],official[k],places=12)
        for mutate in ('missing_component','reordered_row','missing_reference','degenerate_reference'):
            q=copy.deepcopy(p);r=copy.deepcopy(ref)
            if mutate=='missing_component':q['utility'].pop(UTILITY[0])
            elif mutate=='reordered_row':q['forget_rouge']={'c':.6,'a':.2,'b':.4}
            elif mutate=='missing_reference':r=None
            else:r['mia_auc']=1
            with self.assertRaises(ValueError):assemble(q,r,scope='synthetic_cpu')

if __name__=='__main__':unittest.main()
