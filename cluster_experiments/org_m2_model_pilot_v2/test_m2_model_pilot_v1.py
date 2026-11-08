import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
import m2_pilot_contract_v1 as c


class Tests(unittest.TestCase):
    def test_indices_and_partial_coverage(self):
        self.assertEqual(c.pilot_indices('R'),[0,1,2,112])
        self.assertEqual(c.pilot_indices('U'),[0,1,2,12])
        with self.assertRaises(ValueError):c.pilot_indices('O')

    def test_job_ids(self):
        self.assertEqual(c.job_id('123;cluster\n'),'123')
        for x in ('','0','123\n124','Submitted batch job 123','123 abc'):
            with self.assertRaises(ValueError):c.job_id(x)

    def test_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'good').write_text('ok')
            self.assertEqual(c.confined(p,'good'),p/'good')
            for n in ('../good','/good','C:/good','a\\good','./good','missing'):
                with self.assertRaises(ValueError):c.confined(p,n)

    def test_source_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c.save(p/'x.json',{'a':1})
            with self.assertRaises(FileExistsError):c.save(p/'x.json',{})
            r={'path':'x.json','bytes':(p/'x.json').stat().st_size,'sha256':c.sha(p/'x.json')}
            c.verify(p,[r])
            with self.assertRaises(ValueError):c.verify(p,[r,r])
            r['sha256']='0'*64
            with self.assertRaises(ValueError):c.verify(p,[r])

    def test_budget(self):
        p={'tasks':['R','U'],'gpus_per_task':1,'minutes_per_task':30,'gpu_hours_max':1,'ntd_max':30,'production_authorized':False}
        self.assertEqual(c.budget(p),1)
        for key,value in [('tasks',['R']),('minutes_per_task',60),('gpus_per_task',8),('ntd_max',60),('production_authorized',True)]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(ValueError):c.budget(q)

    def test_json_duplicates_and_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'
            for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
                p.write_text(text)
                with self.assertRaises(ValueError):c.read(p)

    def sample_outcome(self):
        d={k:{} for k in ('model_tensors','optimizer_tensors','optimizer_steps','optimizer_groups','aliases','buffer_schema','rng_probe','update_results')}
        d.update(selected_update_indices=[0,1,2,12],pid=1,teacher_initial_sha256='t',teacher_final_sha256='t',
                 gpu={'uuid':'x'},base_model_sha256='b',final_model_sha256='z',role='U',identity={'role':'U'},status='PILOT_BRANCH_PASS',phase='produce')
        return d

    def test_comparison_exact(self):
        a=self.sample_outcome();b=copy.deepcopy(a);b['pid']=2;b['phase']='resume'
        self.assertTrue(c.outcome_compare(a,b))
        for key in ('model_tensors','optimizer_tensors','optimizer_steps','optimizer_groups','aliases','buffer_schema','rng_probe','selected_update_indices','update_results','gpu','identity','role','final_model_sha256','status','phase'):
            bad=copy.deepcopy(b);bad[key]={'wrong':True}
            with self.assertRaises(ValueError):c.outcome_compare(a,bad)

    def test_teacher_no_update_and_fresh_process(self):
        a=self.sample_outcome();b=copy.deepcopy(a);b['phase']='resume'
        with self.assertRaises(ValueError):c.outcome_compare(a,b)
        b['pid']=2;b['teacher_final_sha256']='changed'
        with self.assertRaises(ValueError):c.outcome_compare(a,b)
        b=copy.deepcopy(a);b['pid']=2;b['phase']='resume';a['final_model_sha256']=b['final_model_sha256']='b'
        with self.assertRaises(ValueError):c.outcome_compare(a,b)

    def test_no_pickle_or_blanket_cast(self):
        txt=(Path(__file__).parent/'m2_paged_checkpoint_v1.py').read_text()
        self.assertNotIn('torch.load(',txt)
        self.assertIn('opt.init_state(group,p,0,i)',txt)
        self.assertIn('src.dtype==dst.dtype',txt)
        self.assertIn('optimizer must be fresh',txt)

    def test_launch_receipt_before_submit(self):
        txt=(Path(__file__).parent/'m2_pilot_submit_v1.py').read_text()
        self.assertLess(txt.index("save(claim/'request.json'"),txt.index("result=subprocess.run(['sbatch'"))
        self.assertIn('exist_ok=False',txt)
        self.assertIn('SUBMISSION_UNKNOWN',txt)
        self.assertNotIn('scancel',txt)

    def test_slurm_resources(self):
        txt=(Path(__file__).parent/'m2_model_pilot.slurm').read_text()
        for s in ('--gres=gpu:1','--mem=200G','-t 00:30:00','--array=0-1%2','-A anon_account','-p dev'):
            self.assertIn(s,txt)
        self.assertIn('cuda/12.6',txt)
        self.assertIn('-I -B',txt)

    def test_snapshot_before_continuation(self):
        txt=(Path(__file__).parent/'m2_model_pilot_v1.py').read_text()
        self.assertIn("cursor==1",txt)
        self.assertIn("start=2",txt)
        self.assertIn("'B_engineering_surrogate_not_scientific_O'",txt)
        self.assertNotIn('trust_remote_code=True',txt)

    def test_cpu_visibility_not_inherited_by_submission(self):
        e={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'bad','PATH':'good','KEEP':'yes'}
        self.assertEqual(c.submission_env(e),{'PATH':'good','KEEP':'yes'})
        self.assertEqual(e['CUDA_VISIBLE_DEVICES'],'')

    def test_allocation_identity_fails_before_gpu(self):
        for env in ({},{'SLURM_ARRAY_JOB_ID':'123','SLURM_JOB_ID':'123','SLURM_ARRAY_TASK_ID':'0',
                       'SLURM_JOB_NAME':'wrong','SLURM_JOB_PARTITION':'dev','CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaises(ValueError):c.allocation_guard('R',Path('/wrong'),env)
        txt=(Path(__file__).parent/'m2_model_pilot_v1.py').read_text()
        worker=txt[txt.index('def worker'):]
        self.assertLess(worker.index('allocation_guard('),worker.index('setup(True)'))

    def test_real_schedule_partial_group_coverage(self):
        _,_,_,_,s,cal,_=c.bind_inputs()
        self.assertEqual(len(cal['records']),128)
        for role in ('R','U'):
            counts=[len(s[role]['updates'][i]['microbatches']) for i in c.pilot_indices(role)]
            self.assertEqual(counts,[4,4,4,2])

    def test_accounting_complete_and_no_gpu_double_count(self):
        from m2_pilot_collect_v1 import accounting
        text='123_0|COMPLETED|0:0|20|cpu=8,gres/gpu=1,gres/gpu:h200=1\n123_1|FAILED|1:0|10|gres/gpu=1\n'
        result=accounting(text,'123')
        self.assertEqual(sum(x['elapsed_seconds'] for x in result.values()),30)
        for bad in (text.splitlines()[0],text+text,text.replace('COMPLETED','RUNNING'),text.replace('gres/gpu=1','gres/gpu=2')):
            with self.assertRaises(ValueError):accounting(bad,'123')


if __name__=='__main__':unittest.main()
