import copy
import unittest
from org_b05_behavior_wave_contract_v1 import CONDITIONS, condition_sets, transport


class Tests(unittest.TestCase):
    def cell(self):
        c={n:{'protected_layers':[], 'alias_of':None} for n in CONDITIONS}
        c['bf16']['protected_layers']='ALL_TARGETS'
        c['abstention']['alias_of']='full_packed'
        c['pair']['protected_layers']=['layer0.weight','layer1.weight']
        return {'conditions':c}

    def test_conditions(self):
        self.assertEqual(len(condition_sets(self.cell(),[f'layer{i}' for i in range(224)])),6)

    def test_bad_alias(self):
        c=self.cell(); c['conditions']['abstention']['protected_layers']=['layer0.weight']
        with self.assertRaises(ValueError): condition_sets(c,[f'layer{i}' for i in range(224)])

    def values(self, delta):
        return {n:{r:{'rows':[{'index':i,'sample_sha256':str(i),'encoding_sha256':str(i),
                    'tokens':1,'loss_sum':2.+(delta if n in ('pair','safe_looking_singleton','risky_looking_singleton') else 0.)}
                    for i in range(128)]} for r in ('forget','retain')} for n in CONDITIONS}

    def test_pass(self): self.assertEqual(transport(self.values(.1),self.values(.2))['status'],'PASS')
    def test_sign_fail(self): self.assertEqual(transport(self.values(.1),self.values(-.1))['status'],'FAIL')
    def test_zero(self): self.assertEqual(transport(self.values(0),self.values(0))['status'],'INCONCLUSIVE')
    def test_one_sided(self): self.assertEqual(transport(self.values(.1),self.values(0))['status'],'FAIL')
    def test_token_drift(self):
        x=self.values(.1); x['pair']['forget']['rows'][0]['encoding_sha256']='bad'
        with self.assertRaises(ValueError): transport(self.values(.1),x)


if __name__=='__main__': unittest.main()
