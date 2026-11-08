import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from m2_eval_oracle_repair_v2 import probability_oracle, probability_precheck
from m2_eval_qualification_common_v1 import fixture, compare
from m2_eval_qualification_oracle_v1 import ObservedModel, validate_result
from test_m2_eval_qualification_torch_v1 import Model, GEN
from test_m2_eval_qualification_v1 import toy_cal
from test_m2_metric_assembly_torch_v1 import ToyTokenizer
import test_m2_metric_assembly_torch_v1 as prior
from m2_real_evaluator_v1 import primitive_suite
from m2_metric_assembly_v1 import upstream_namespace


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def sample(self):
        g=torch.Generator().manual_seed(360917)
        z=(torch.randn(3,7,128256,generator=g)*3).to(torch.bfloat16).float()
        ids=torch.tensor([[1,2,3,4,5,6,7],[1,2,3,6,8,0,0],[1,2,4,5,6,7,8]])
        labels=ids.clone();labels[:,:2]=-100;labels[1,5:]=-100
        return z,labels,ids

    def test_large_vocab_bf16_logits_against_unchanged_upstream(self):
        z,ys,ids=self.sample();ns=upstream_namespace(Path(prior.__file__).parent/'upstream')
        for ix in (slice(0,2),slice(2,3)):
            class Frozen:
                device=torch.device('cpu')
                def __call__(self,**batch):return SimpleNamespace(logits=z[ix])
            a,p,diag=probability_oracle(torch,z[ix],ys[ix],ids[ix])
            official=ns['evaluate_probability'](Frozen(),{'labels':ys[ix],'input_ids':ids[ix]})
            for i,row in enumerate(official):compare({'avg_loss':float(a[i]),'prob':float(p[i])},row)
            print('M2_LARGE_VOCAB_LAYOUT_DIAGNOSTIC='+str(diag),flush=True)

    def test_uniform_large_vocab_analytic_and_eos_mask(self):
        z,ys,ids=self.sample();z.zero_();a,p,_=probability_oracle(torch,z,ys,ids)
        for loss,prob in zip(a,p):
            self.assertLessEqual(abs(float(loss)-math.log(128256)),1e-6)
            self.assertLessEqual(abs(float(prob)-1/128256),1e-6)

    def test_invalid_labels_masks_and_nonfinite_rejected(self):
        z,ys,ids=self.sample()
        for mode in ('wrong_label','gap','first','empty','nan'):
            y=ys.clone();x=z.clone()
            if mode=='wrong_label':y[0,3]=999
            if mode=='gap':y[0,3]=-100
            if mode=='first':y[0,0]=1
            if mode=='empty':y[0,:]=-100
            if mode=='nan':x[0,2,3]=float('nan')
            with self.assertRaises(ValueError):probability_oracle(torch,x,y,ids)

    def test_full_runner_large_vocab_bf16_fixture(self):
        class Large(Model):
            def __call__(self,input_ids,**kw):
                v=torch.arange(128256,dtype=torch.float32)
                logits=(torch.sin(v*.071)+torch.cos(v*.113+input_ids.unsqueeze(-1))).to(torch.bfloat16)
                return SimpleNamespace(logits=logits)
        c=toy_cal();c['records'][1].update(input_ids=[1,2,5,0],labels=[-100,-100,5,0],attention_mask=[1]*4)
        p=fixture(c);tok=ToyTokenizer();tok.pad_token_id=0
        upstream=Path(prior.__file__).parent/'upstream';m=ObservedModel(torch,Large())
        pre=probability_precheck(torch,Large(),tok,p,upstream)
        self.assertEqual(pre['rows'],3)
        with torch.no_grad():o=primitive_suite(torch,m,tok,p,GEN,upstream,batch_size=2)
        val=validate_result(torch,o,p,m,tok,upstream_namespace(upstream))
        self.assertEqual(val['answer_variant_checks'],96);self.assertEqual(val['generations'],15)
        o['row_outputs']['forget']['probability']['answer']['0']['avg_loss']+=1e-4
        with self.assertRaises(ValueError):validate_result(torch,o,p,m,tok,upstream_namespace(upstream))

    def test_oracle_does_not_mutate_inputs(self):
        z,ys,ids=self.sample();before=[x.clone() for x in (z,ys,ids)]
        probability_oracle(torch,z,ys,ids)
        for x,y in zip((z,ys,ids),before):self.assertTrue(torch.equal(x,y))


if __name__=='__main__':unittest.main()
