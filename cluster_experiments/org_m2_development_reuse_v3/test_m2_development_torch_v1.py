import copy,unittest
from pathlib import Path
import torch
from m2_development_score_v1 import score
from m2_eval_qualification_common_v1 import fixture,compare
from m2_eval_qualification_v1 import SCOPE
from test_m2_eval_qualification_torch_v1 import Model,GEN,ToyTokenizer
from test_m2_eval_qualification_v1 import toy_cal
import test_m2_metric_assembly_torch_v1 as prior

class Tests(unittest.TestCase):
    def run_fixture(self):
        p=fixture(toy_cal());p={k:p[k] for k in ('forget','retain','retain_all')}
        for rows in p.values():
            for i,r in enumerate(rows):r['index']=str(i+20)
        tok=ToyTokenizer();tok.pad_token_id=0
        with torch.no_grad():out=score(torch,Model(),tok,p,GEN,Path(prior.__file__).parent/'upstream')
        return p,out
    def test_development_routes_indexed_and_repeat(self):
        p,a=self.run_fixture();_,b=self.run_fixture();self.assertEqual(compare(a['panels'],b['panels']),0)
        self.assertEqual(set(a['panels']),{'forget','retain','retain_all'})
        for k,rr in p.items():
            self.assertEqual(set(a['panels'][k]),{r['qa_sha256'] for r in rr})
            self.assertTrue(all(v['generation']['generated_ids']==[[3,4,0]] for v in a['panels'][k].values()))
    def test_wrong_actual_score_rejected(self):
        import m2_metric_assembly_v1 as m
        from unittest.mock import patch
        real=m.upstream_namespace
        def corrupt(path):
            ns=real(path);old=ns['probability']
            def bad(*a,**kw):
                v=old(*a,**kw);next(iter(v['value_by_index'].values()))['prob']+=.1;return v
            ns['probability']=bad;return ns
        with patch.object(m,'upstream_namespace',corrupt):
            with self.assertRaises(ValueError):self.run_fixture()

if __name__=='__main__':unittest.main()
