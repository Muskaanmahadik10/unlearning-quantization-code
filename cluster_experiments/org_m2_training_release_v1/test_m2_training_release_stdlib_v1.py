import ast
import copy
from pathlib import Path
import unittest
from m2_stage_common_v1 import *
from m2_training_release_v1 import parse_job
from m2_real_data_v1 import route,qa

class Tests(unittest.TestCase):
    def test_budget(self):
        p={'roles':['O','R'],'gpus_per_task':1,'minutes_per_task':120,'gpu_hours_cap':4,'ntd_cap':120,'rate':30,'U_automatic':False,'evaluation_launch_authorized':False,'M3_authorized':False}
        self.assertEqual(validate_plan(p)['ntd_cap'],120)
        p['roles'].append('U')
        with self.assertRaises(ValueError):validate_plan(p)
    def test_array_role(self):
        self.assertEqual(selected_role('0'),'O');self.assertEqual(selected_role('1'),'R')
        for x in ('2','-1','',True):
            with self.assertRaises(ValueError):selected_role(x)
    def test_submission_id(self):
        self.assertEqual(parse_job('123;cluster\n'),'123')
        for x in ('Submitted batch job 123','123\n124','', '0'):
            with self.assertRaises(ValueError):parse_job(x)
    def test_no_reference_or_surrogate_U(self):
        s=(Path(__file__).parent/'m2_train_or_v1.py').read_text()
        self.assertIn("choices=['O','R']",s);self.assertNotIn('teacher=freeze_reference',s)
    def test_all_fixed_updates(self):
        s=(Path(__file__).parent/'m2_train_or_v1.py').read_text()
        self.assertIn("enumerate(sched['updates'])",s);self.assertNotIn('pilot_indices(',s)
        self.assertIn("set(meta['optimizer_steps'].values())=={len(records)}",s)
    def test_submission_after_checks(self):
        s=(Path(__file__).parent/'m2_training_release_v1.py').read_text()
        self.assertLess(s.index('binding=prepare('),s.index("['sbatch','--parsable'"))
        self.assertLess(s.index("claim.mkdir(exist_ok=False)"),s.index("['sbatch','--parsable'"))
    def test_resource_contract(self):
        s=(Path(__file__).parent/'m2_train_or.slurm').read_text()
        for x in ('--gres=gpu:1','--array=0-1%2','--mem=200G','--time=02:00:00','--cpus-per-task=8'):self.assertIn(x,s)
    def test_accounting(self):
        s='10_0|COMPLETED|0:0|300|cpu=8,gres/gpu=1\n10_1|COMPLETED|0:0|400|cpu=8,gres/gpu=1\n'
        self.assertEqual(sum(accounting(s,'10').values()),700)
        for bad in (s.replace('COMPLETED','RUNNING'),s.replace('gpu=1','gpu=2'),s.splitlines()[0]):
            with self.assertRaises(ValueError):accounting(bad,'10')
    def test_binding_counts(self):
        b={'counts':{'forget':400,'retain':400,'ra':100,'wf':117,'holdout':400,'retain_all':3600},'official_forget_exact_qa_matches_training_forget':True,'retain_eval_subset_of_training_retain':True,'holdout_training_overlap':0,'all_variant_token_parity':True,'evaluated_model_outcomes':0}
        validate_binding(b);b['counts']['retain_all']=400
        with self.assertRaises(ValueError):validate_binding(b)
    def test_duplicate_json(self):
        with self.assertRaises(ValueError):unique([('a',1),('a',2)])
    def test_true_no_overwrite(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x';save(p,{'a':1})
            with self.assertRaises(FileExistsError):save(p,{'a':2})
    def test_token_panels_not_training_holdout(self):
        from m2_real_data_v1 import ROUTES
        self.assertEqual(ROUTES['holdout'],'holdout10');self.assertEqual(ROUTES['retain_all'],'retain90')
    def test_request_budget_and_quota(self):
        r={'roles':['O','R'],'max_gpu_hours':4,'max_ntd':120,'user_quota_verified':False,'user_accepted_unverified_quota':True}
        validate_request(r);r['max_ntd']=121
        with self.assertRaises(ValueError):validate_request(r)
    def test_route_wrong_qa_rejected(self):
        base={'question':'q','answer':'a','paraphrased_answer':'p','perturbed_answer':['w','x']}
        names=['full','forget10','forget10_perturbed','retain_perturbed','real_authors','real_authors_perturbed','world_facts','world_facts_perturbed','holdout10','retain90']
        raw={k:[copy.deepcopy(base)] for k in names};self.assertEqual(len(route(raw)),6)
        raw['real_authors_perturbed'][0]['answer']='changed'
        with self.assertRaises(ValueError):route(raw)
    def test_collector_requires_real_completed_schedule(self):
        from m2_collect_or_v1 import check_role
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for f in ('final','produce','reload'):save(p/(f+'.json'),{'status':'PILOT_PASS'})
            with self.assertRaises(ValueError):check_role(p,'O',{'total_updates':625})
    def test_checkpoint_and_source_guards(self):
        s=(Path(__file__).parent/'m2_collect_or_v1.py').read_text()
        for x in ('verify(cp,',"list(range(sched['total_updates']))","meta['identity']==ident","p['pid']!=r['pid']"):
            self.assertIn(x,s)

if __name__=='__main__':unittest.main()
