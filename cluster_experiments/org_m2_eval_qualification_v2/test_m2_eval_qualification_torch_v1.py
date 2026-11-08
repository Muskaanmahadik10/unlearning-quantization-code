import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from m2_eval_qualification_common_v1 import *
from m2_eval_qualification_oracle_v1 import ObservedModel,validate_result,generation_batch_trial
from m2_real_evaluator_v1 import primitive_suite
from m2_metric_assembly_v1 import upstream_namespace
from test_m2_metric_assembly_torch_v1 import ToyModel,ToyTokenizer
from test_m2_eval_qualification_v1 import toy_cal

GEN={'do_sample':False,'max_new_tokens':200,'temperature':None,'top_p':None,'use_cache':True}

class Model(ToyModel):
    generation_config=SimpleNamespace(eos_token_id=[0])
    def eval(self):return self
    def __call__(self,**kw):kw.pop('use_cache',None);return super().__call__(**kw)

class Tests(unittest.TestCase):
    def run_fixture(self):
        c=toy_cal();c['records'][1].update(input_ids=[1,2,5,0],labels=[-100,-100,5,0],attention_mask=[1]*4)
        c['records'][2].update(input_ids=[1,2,3,4,3,0],labels=[-100,-100,3,4,3,0],attention_mask=[1]*6)
        p=fixture(c);model=ObservedModel(torch,Model());tok=ToyTokenizer();tok.pad_token_id=0
        import test_m2_metric_assembly_torch_v1 as prior
        upstream=Path(prior.__file__).parent/'upstream'
        with torch.no_grad():out=primitive_suite(torch,model,tok,p,GEN,upstream,batch_size=2)
        return p,model,tok,out,upstream
    def test_all_real_runner_oracle_routes(self):
        p,m,t,o,u=self.run_fixture();v=validate_result(torch,o,p,m,t,upstream_namespace(u))
        self.assertEqual(v['answer_variant_checks'],96);self.assertEqual(v['generations'],15)
        self.assertEqual(len(m.forward_calls),70)
    def test_wrong_indexed_score_rejected(self):
        p,m,t,o,u=self.run_fixture();o['row_outputs']['forget']['probability']['answer']['0']['prob']+=.1
        with self.assertRaises(ValueError):validate_result(torch,o,p,m,t,upstream_namespace(u))
    def test_batch8_left_padding_and_eos(self):
        records=toy_cal()['records'][:8];records[1]['prompt_ids']=[2]
        tok=ToyTokenizer();tok.pad_token_id=0
        v=generation_batch_trial(torch,Model(),tok,records,GEN)
        self.assertTrue(v['greedy_ids_exact']);self.assertEqual(v['single_ids'],[[3,4,0]]*8)
    def test_batch_mismatch_disqualifies_only_candidate(self):
        class Bad(Model):
            def generate(self,input_ids,**kwargs):
                out=super().generate(input_ids,**kwargs)
                if len(input_ids)>1:out[:,-2]=5
                return out
        tok=ToyTokenizer();tok.pad_token_id=0
        v=generation_batch_trial(torch,Bad(),tok,toy_cal()['records'][:8],GEN)
        self.assertEqual(v['batch8_candidate'],'REJECTED_TOKEN_MISMATCH');self.assertFalse(v['full_benchmark_batch8_authorized'])
    def test_seven_metric_fixture_assembly(self):
        p,m,t,o,u=self.run_fixture()
        from m2_eval_qualification_collect_v1 import assembly_check
        b={'role':'U','scope':SCOPE,'source_lock_sha256':'x','fixture_sha256':'y','model_sha256':'u','result':o}
        r={**b,'role':'R','model_sha256':'r'}
        a=assembly_check(b,r,upstream_namespace(u));self.assertEqual(len(a['metrics']),7)
        r['role']='O'
        with self.assertRaises(ValueError):assembly_check(b,r,upstream_namespace(u))

if __name__=='__main__':unittest.main()
