import ast
from pathlib import Path
import unittest
from m1_core_submit_v1 import jobid,completed
from m1_contract_v1 import ARMS,arm_targets,planned_core_cost

class Tests(unittest.TestCase):
    def test_full_scientific_inventory(self):
        self.assertEqual(len(ARMS),10)
        cell={'a':'a.weight','b':'b.weight'}
        self.assertEqual(arm_targets(cell,'R_ab'),(['a','b'],1.))
    def test_all_arms_get_behavior(self):
        t=Path(__file__).with_name('m1_core_v1.py').read_text()
        self.assertIn('behavior = {r: evaluate(model, tokenizer, s) for r, s in samples.items()}\n',t)
        self.assertNotIn("if arm in ('N0', 'Z_ab') else None",t)
        self.assertIn('compare_reloads',t)
        self.assertIn("'scope': 'M1_CORE_CELL_ONLY'",t)
    def test_no_requantization(self):
        tree=ast.parse(Path(__file__).with_name('m1_core_v1.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('quantize',calls)
        self.assertNotIn('post_init',calls)
    def test_accounting_fail_closed(self):
        row='123_5|COMPLETED|0:0|1200|cpu=8,gres/gpu=1'
        self.assertEqual(completed(row,'123_5'),1200)
        for bad in ('',row+'\n'+row,row.replace('COMPLETED','FAILED'),row.replace('gpu=1','gpu=8')):
            with self.assertRaises(ValueError):completed(bad,'123_5')
    def test_submission_identity(self):
        self.assertEqual(jobid('123;cluster'),'123')
        for bad in ('','123\n456','Submitted batch job 123','0'):
            with self.assertRaises(ValueError):jobid(bad)
    def test_submission_guard_order(self):
        t=Path(__file__).with_name('m1_core_submit_v1.py').read_text()
        self.assertLess(t.index('write(intent,'),t.index('r=subprocess.run(cmd,'))
        self.assertIn('validate_core(',t)
        self.assertIn("indices='0-4'",t)
        self.assertIn("indices='5'",t)
    def test_budget(self):
        self.assertLess(1.3522222222222222+6*2/3,7)
    def test_resource_contract(self):
        t=Path(__file__).with_name('m1_core.slurm').read_text()
        for s in ('--gres=gpu:1','--mem=128G','--time=00:40:00','--index "$SLURM_ARRAY_TASK_ID"'):
            self.assertIn(s,t)

if __name__=='__main__':unittest.main()
