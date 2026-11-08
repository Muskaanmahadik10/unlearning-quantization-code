import unittest
from pathlib import Path
from m2_eval_qualification_common_v1 import compare, release
from m2_eval_repair_guard_v2 import parse_accounting, STAGES, PRIOR_JOB


class Tests(unittest.TestCase):
    def prior(self):
        return '\n'.join('360917_'+str(i)+'|FAILED|1:0|'+str(n)+'|cpu=8,gres/gpu=1' for i,n in enumerate((133,107,96,89)))
    def test_prior_cost(self):
        d=parse_accounting(self.prior(),PRIOR_JOB,range(4),'FAILED','1:0')
        self.assertEqual(sum(d.values()),425)
    def test_prior_unknown_active_and_gpu_rejected(self):
        for text in (self.prior().replace('FAILED','RUNNING'), self.prior().replace('gpu=1','gpu=2'),
                     self.prior().splitlines()[0], self.prior()+'\n'+self.prior().splitlines()[0]):
            with self.assertRaises(ValueError):parse_accounting(text,PRIOR_JOB,range(4),'FAILED','1:0')
    def test_stage_exact_coverage(self):self.assertEqual(STAGES,{'canary':(0,),'remaining':(1,2,3)})
    def test_remaining_requires_three_completed(self):
        good='\n'.join('42_'+str(i)+'|COMPLETED|0:0|300|gres/gpu=1' for i in (1,2,3))
        self.assertEqual(sum(parse_accounting(good,'42',STAGES['remaining']).values()),900)
        with self.assertRaises(ValueError):parse_accounting(good.replace('COMPLETED','FAILED'),'42',STAGES['remaining'])
    def test_numeric_failure_has_values_and_unchanged_tolerance(self):
        with self.assertRaisesRegex(ValueError,'expected=.*observed=.*delta=.*tolerance=1e-06'):compare(1.,1.01)
        with self.assertRaises(ValueError):compare(1.,1.000002)
    def test_observations_saved_before_validation(self):
        s=(Path(__file__).parent/'m2_eval_qualification_v1.py').read_text()
        self.assertLess(s.index("args.phase+'_observations.json'"),s.index('oracle=validate_result'))
        self.assertLess(s.index('probability_precheck(torch'),s.index('result=primitive_suite'))
    def test_no_cross_entropy_in_oracle(self):
        s=(Path(__file__).parent/'m2_eval_oracle_repair_v2.py').read_text()
        self.assertNotIn('cross_entropy(',s);self.assertNotIn('CrossEntropyLoss(',s)
        self.assertIn('legacy_vs_class_axis_max_delta',s)
    def test_canary_and_budget_guard_before_remaining(self):
        s=(Path(__file__).parent/'m2_eval_qualification_submit_v1.py').read_text()
        self.assertLess(s.index('verify_canary()'),s.index("['sbatch','--parsable'"))
        self.assertLess(s.index('reconcile_prior()'),s.index("['sbatch','--parsable'"))
        self.assertIn('bucket_cap_gpu_hours',s)


if __name__=='__main__':unittest.main()
