import unittest
from unittest.mock import patch
import run_org_b05_real_model_smoke_v1 as m


class ContractTests(unittest.TestCase):
    def fixture(self):
        r={'path':'input','sha256':'h'}
        c={'cell_id':'ordinary','quantizer':'nf4','feature_receipt':r,'environment_lock':r,
           'manifests':{},'conditions':{'pair':{'protected_layers':['l0.weight','l1.weight']}}}
        plan={'status':'PLAN_PREPARED','cell_count':6,'cells':[c],'selector_lock':r}
        lock={'cells':{'ordinary/nf4':{'conditions':c['conditions']}}}
        f={'per_layer':[{'layer':f'l{i}.weight'} for i in range(224)]}
        return plan,lock,f

    def test_exact_protected_names(self):
        plan,lock,f=self.fixture()
        with patch.object(m,'sha256_file',return_value='h'),patch.object(m,'read',side_effect=[lock,f]):
            _,targets,protected=m.entry(plan,'ordinary','nf4')
        self.assertEqual(protected,['l0','l1']); self.assertEqual(len(targets),224)

    def test_bad_hash(self):
        plan,_,_=self.fixture()
        with patch.object(m,'sha256_file',return_value='wrong'),self.assertRaises(RuntimeError): m.entry(plan,'ordinary','nf4')

    def test_foreign_protected_layer(self):
        plan,lock,f=self.fixture()
        plan['cells'][0]['conditions']['pair']['protected_layers']=['foreign.weight']
        with patch.object(m,'sha256_file',return_value='h'),patch.object(m,'read',side_effect=[lock,f]),self.assertRaises(RuntimeError): m.entry(plan,'ordinary','nf4')


if __name__=='__main__': unittest.main()
