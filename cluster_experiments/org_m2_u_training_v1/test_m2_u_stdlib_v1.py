import copy
from pathlib import Path
import unittest
from m2_u_common_v1 import validate_release,accounting

class Tests(unittest.TestCase):
    def plan(self):return {'role':'U','parent_role':'O','teacher_role':'O','gpus':1,'minutes':120,'gpu_hours_cap':2,'ntd_cap':60,'rate':30,'fresh_optimizer':True,'automatic_evaluation':False,'M3_authorized':False,'updates':130,'parent_updates':625}
    def test_exact_release(self):self.assertEqual(validate_release(self.plan())['updates'],130)
    def test_no_surrogate_or_wrong_reference(self):
        for field in ('parent_role','teacher_role'):
            for val in ('B','R'):
                r=self.plan();r[field]=val
                with self.assertRaises(ValueError):validate_release(r)
    def test_budget_expansion_rejected(self):
        for field,val in [('gpus',2),('minutes',121),('ntd_cap',61),('gpu_hours_cap',3),('M3_authorized',True),('fresh_optimizer',False)]:
            r=self.plan();r[field]=val
            with self.assertRaises(ValueError):validate_release(r)
    def test_accounting(self):
        self.assertEqual(accounting('123|COMPLETED|0:0|321|cpu=8,gres/gpu=1\n','123'),321)
        for bad in ('123|RUNNING|0:0|1|gres/gpu=1','123|COMPLETED|1:0|1|gres/gpu=1','123|COMPLETED|0:0|1|gres/gpu=2',''):
            with self.assertRaises(ValueError):accounting(bad,'123')
    def test_full_schedule_not_pilot_indices(self):
        s=(Path(__file__).parent/'m2_train_u_v1.py').read_text()
        self.assertIn("enumerate(sched['updates'])",s);self.assertNotIn('pilot_indices(',s)
        self.assertIn('draws==4000',s);self.assertIn("len(opt.state)==0",s)
    def test_teacher_end_check(self):
        s=(Path(__file__).parent/'m2_train_u_v1.py').read_text()
        self.assertIn('teacher_final==teacher_initial',s);self.assertIn('reference=teacher',s)
        self.assertNotIn('ck.load_checkpoint(',s)
    def test_no_submission_before_checks(self):
        s=(Path(__file__).parent/'m2_submit_u_v1.py').read_text()
        for x in ('parent_metadata(r)',"t.wasSuccessful()","claim.mkdir(exist_ok=False)"):
            self.assertLess(s.index(x),s.index("['sbatch','--parsable'"))
        self.assertIn('SUBMISSION_UNKNOWN.json',s)
    def test_slurm_scope_and_environment(self):
        s=(Path(__file__).parent/'m2_train_u.slurm').read_text()
        for x in ('--gres=gpu:1','--mem=200G','--time=02:00:00','cuda/12.6','LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64','-I -B'):
            self.assertIn(x,s)
        self.assertNotIn('--array',s)
if __name__=='__main__':unittest.main()
