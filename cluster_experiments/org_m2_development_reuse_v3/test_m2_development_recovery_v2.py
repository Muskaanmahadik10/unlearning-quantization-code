
import inspect,json,tempfile,unittest
from pathlib import Path
import m2_development_run_v1 as m
class RecoveryTests(unittest.TestCase):
    def test_prior_timeout_accounting(self):
        s='\n'.join(f'360986_{i}|TIMEOUT|0:0|499|cpu=8,gres/gpu=1' for i in range(12))
        self.assertEqual(sum(x['elapsed_seconds'] for x in m.prior_accounting(s).values()),5988)
        for bad in (s.replace('TIMEOUT','RUNNING',1),s.replace('gres/gpu=1','gres/gpu=2',1),s+'\n'+s.splitlines()[0],s.replace('|499|','|0|',1)):
            with self.assertRaises(ValueError):m.prior_accounting(bad)
    def test_long_completion_and_timeout(self):
        s='\n'.join(f'42_{i}|COMPLETED|0:0|3600|cpu=8,gres/gpu=1' for i in range(12))
        self.assertEqual(len(m.accounting(s,'42')),12)
        with self.assertRaises(ValueError):m.accounting(s.replace('COMPLETED','TIMEOUT',1),'42')
    def test_plan_and_slurm_consistent(self):
        p=m.read(m.PKG/'PLAN.json');s=(m.PKG/'m2_development.slurm').read_text()
        self.assertEqual(p['minutes_per_task'],120);self.assertEqual(p['max_gpu_hours'],24)
        self.assertEqual(p['max_ntd'],720);self.assertIn('--time=02:00:00',s)
        self.assertIn('--array=0-11%12',s);self.assertIn('--gres=gpu:1',s)
    def test_prior_guard_and_panels(self):
        s=inspect.getsource(m.submit)
        self.assertLess(s.index('recovery_guard()'),s.index("['sbatch','--parsable'"))
        self.assertIn('completed phase receipts exist',inspect.getsource(m.recovery_guard))
        import m2_development_score_v1 as score
        s=inspect.getsource(score.score)
        self.assertLess(s.index('on_panel(panel,out)'),s.index("print('M2_DEVELOPMENT_PANEL"))
        self.assertIn('PANEL_ONLY_NOT_CERTIFIED',inspect.getsource(m.worker))
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';m.save(p,{'a':1})
            with self.assertRaises(FileExistsError):m.save(p,{'a':2})
            self.assertEqual(m.read(p),{'a':1})

    def test_only_second_pass_and_pinned_first(self):
        s=inspect.getsource(m.run)
        self.assertIn("'--phase','b'",s);self.assertNotIn("for phase in ('a','b')",s)
        self.assertIn('verify(ROOT,[rec])',s)
        self.assertIn('cross-version source provenance',inspect.getsource(m.final_check))
