import copy
from pathlib import Path
import unittest
import torch
from test_m2_metric_assembly_torch_v1 import ToyModel,ToyTokenizer,dataset
from m2_real_evaluator_v1 import primitive_suite,assemble_controlled,collator
from m2_stage_common_v1 import METRIC,canonical

class Model(ToyModel):
    def eval(self):return self
    def __call__(self,use_cache=False,**kwargs):return super().__call__(**kwargs)
class Tokenizer(ToyTokenizer):pad_token_id=0

class Tests(unittest.TestCase):
    def test_real_runner_synthetic_end_to_end(self):
        generation={'do_sample':False,'max_new_tokens':200,'temperature':None,'top_p':None,'use_cache':True}
        panels={}
        for panel in ('forget','retain','ra','wf','holdout','retain_all'):
            rows=[]
            for i,r in enumerate(dataset()):
                e={k:r[k] for k in ('input_ids','labels','attention_mask')};e['prompt_ids']=[1,2]
                t={'answer':[e]}
                if panel in ('forget','retain'):t['paraphrase']=[copy.deepcopy(e)]
                if panel in ('forget','retain','ra','wf'):t['wrong']=[copy.deepcopy(e),copy.deepcopy(e)]
                rows.append({'index':str(i),'qa_sha256':str(i),'tokens':t})
            panels[panel]=rows
        result=primitive_suite(torch,Model(),Tokenizer(),panels,generation,METRIC/'upstream',2)
        identity={'role':'U','parent_role':'O','data_binding_sha256':'data','generation':generation,'base_sha256':'base','source_lock_sha256':'source','checkpoint_sha256':'U'}
        rid={**identity,'role':'R','parent_role':'B','checkpoint_sha256':'R'}
        ref={'identity':rid,'result':result}
        actual=assemble_controlled(result,identity,ref,canonical(ref))
        self.assertEqual(len(actual['metrics']),7);self.assertEqual(len(actual['utility_components']),9)
        self.assertEqual(actual['metrics']['privleak'],0)
        for panel in panels:self.assertEqual(len(result['row_outputs'][panel]['probability']['answer']),3)
        broken=copy.deepcopy(ref);broken['identity']['parent_role']='O'
        with self.assertRaises(ValueError):assemble_controlled(result,identity,broken,canonical(broken))
        with self.assertRaises(ValueError):assemble_controlled(result,identity,ref,'wrong')
    def test_variable_length_collation(self):
        f=collator(torch,0,True)
        b=f([{'input_ids':[1,2],'labels':[1,2,3,0],'index':0},{'input_ids':[2],'labels':[2,3,0],'index':1}])
        self.assertEqual(b['input_ids'].tolist(),[[1,2],[0,2]])
        self.assertEqual(b['attention_mask'].tolist(),[[True,True],[False,True]])
        self.assertEqual(b['labels'].tolist(),[[1,2,3,0],[2,3,0,-100]])

if __name__=='__main__':unittest.main()
