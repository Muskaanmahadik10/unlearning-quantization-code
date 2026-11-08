import copy,inspect,unittest
from m2_development_math_v1 import *
from m2_development_run_v1 import accounting,unique,canonical

def good_states():
    return {r:{p:{str(i):{'prob':(.8 if r=='O' else .3 if r=='R' else .4) if p=='forget' else .8,
      'rouge':(.8 if r=='O' else .3 if r=='R' else .4) if p=='forget' else .8,'nll':.2,'truth':.8}
      for i in range(n)} for p,n in COUNTS.items()} for r in ROLES}

class Tests(unittest.TestCase):
    def partition_fixture(self):
        rr=[{'qa_sha256':str(i),'panel':'development' if i<764 else 'sealed','training_pool':'forget10' if i<76 else 'retain90','reasons':[]} for i in range(4400)]
        panel=lambda ids:[{'index':str(i),'qa_sha256':str(i)} for i in ids]
        return {'records':rr},{'forget':panel(range(76)),'retain':panel(range(76,229)),'retain_all':panel(range(76,764))}
    def test_exact_development_selection(self):
        p,d=self.partition_fixture();self.assertEqual({k:len(v) for k,v in select(d,p).items()},COUNTS)
    def test_sealed_and_ambiguity_not_substituted(self):
        p,d=self.partition_fixture();p['records'][0]['panel']='sealed'
        with self.assertRaises(ValueError):select(d,p)
        p,d=self.partition_fixture();p['records'][0]['reasons']=['unresolved']
        with self.assertRaises(ValueError):select(d,p)
    def test_good_gate(self):self.assertEqual(analyze(good_states())['status'],'PASS')
    def test_no_suppression(self):
        x=good_states();x['U']['forget']=copy.deepcopy(x['O']['forget']);self.assertEqual(analyze(x)['status'],'FAIL')
    def test_missing_roles(self):
        x=good_states();del x['R']
        with self.assertRaises(ValueError):analyze(x)
    def test_missing_row(self):
        x=good_states();del x['U']['forget']['0']
        with self.assertRaises(ValueError):analyze(x)
    def test_nonfinite(self):
        x=good_states();x['U']['forget']['0']['prob']=float('nan')
        with self.assertRaises(ValueError):analyze(x)
    def test_low_denominator(self):
        x=good_states()
        for r in x['O']['retain_all'].values():r['prob']=0.
        self.assertEqual(analyze(x)['status'],'INCONCLUSIVE')
    def test_nll_tail(self):
        x=good_states()
        for i in range(80):x['U']['retain_all'][str(i)]['nll']=.31
        y=analyze(x);self.assertFalse(y['checks']['retain_tail_NLL']);self.assertTrue(y['checks']['retain_mean_NLL'])
    def test_shard_complete_batch_boundaries(self):
        for n in (3,76,153,688):
            rows=[{'index':str(i)} for i in range(n)];parts=[shard_rows(rows,s) for s in range(4)]
            self.assertEqual(sorted(int(x['index']) for p in parts for x in p),list(range(n)))
            for p in parts:
                for start in range(0,len(p)-1,2):self.assertEqual(int(p[start+1]['index']),int(p[start]['index'])+1)
    def test_bad_shard(self):
        with self.assertRaises(ValueError):shard_rows([{'index':'0'}],4)
    def test_duplicate_json(self):
        with self.assertRaises(ValueError):unique([('x',1),('x',2)])
    def test_accounting(self):
        s='\n'.join(f'42_{i}|COMPLETED|0:0|400|cpu=8,gres/gpu=1' for i in range(12));self.assertEqual(sum(accounting(s,'42').values()),4800)
        for bad in (s.replace('COMPLETED','RUNNING',1),s.replace('|400|','|7321|',1),s.replace('gres/gpu=1','gres/gpu=2',1),s+'\n'+s.splitlines()[0],'\n'.join(s.splitlines()[:-1])):
            with self.assertRaises(ValueError):accounting(bad,'42')
    def test_claim_and_gates_before_sbatch(self):
        import m2_development_run_v1 as m
        s=inspect.getsource(m.submit)
        self.assertLess(s.index('result.wasSuccessful()'),s.index("['sbatch','--parsable'"));self.assertLess(s.index("claim.mkdir"),s.index("['sbatch','--parsable'"))
        self.assertIn('SUBMISSION_UNKNOWN.json',s);self.assertIn("p['recovery_budget_authorized'] is True",s)
    def test_no_extra_science_authorization(self):
        d=analyze(good_states());self.assertFalse(d['packed_launch_authorized']);self.assertFalse(d['sealed_or_holdout_scored']);self.assertFalse(d['M3_authorized'])
    def test_resource_budget(self):
        self.assertEqual(12*120/60,24)

if __name__=='__main__':unittest.main()
