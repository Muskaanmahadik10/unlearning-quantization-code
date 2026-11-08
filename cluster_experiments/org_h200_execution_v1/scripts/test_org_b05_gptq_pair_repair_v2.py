import ast
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import org_b05_gptq_pair_repair_v2 as repair
import org_b05_packed_lifecycle_v2 as life

class Tests(unittest.TestCase):
    def test_accounting(self):
        self.assertEqual(repair.seconds('349840_0|COMPLETED|178|cpu=8,gres/gpu=1\n349840_1|FAILED|452|gres/gpu=1'),630)
    def test_accounting_rejections(self):
        good='349840_0|COMPLETED|178|gres/gpu=1\n349840_1|FAILED|452|gres/gpu=1'
        for bad in (good.splitlines()[0],good+'\n'+good,good.replace('FAILED','RUNNING'),good.replace('gpu=1','gpu=8')):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError): repair.seconds(bad)
    def test_loader_preserves_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'weights').write_bytes(b'original')
            loader=Mock(return_value=('wrapper','initialized-model'))
            self.assertEqual(life.checked_gptq_load(None,SimpleNamespace(load_gptq=loader),p,life.artifact_hashes(p)),('wrapper','initialized-model'))
            loader.assert_called_once_with(None,p)
    def test_preload_drift_stops_before_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'weights').write_bytes(b'original'); frozen=life.artifact_hashes(p)
            (p/'weights').write_bytes(b'changed'); loader=Mock()
            with self.assertRaises(RuntimeError): life.checked_gptq_load(None,SimpleNamespace(load_gptq=loader),p,frozen)
            loader.assert_not_called()
    def test_loader_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'weights').write_bytes(b'original'); frozen=life.artifact_hashes(p)
            def loader(*args):
                (p/'new-file').write_bytes(b'bad')
                return None,None
            with self.assertRaises(RuntimeError): life.checked_gptq_load(None,SimpleNamespace(load_gptq=loader),p,frozen)
    def test_gptq_source_order_regression(self):
        # Structural guard for the exact failure: raw quantized object must never
        # execute the producer synthetic/behavior evaluator before serialization.
        src=inspect.getsource(life.worker)
        self.assertLess(src.index('wrapper.save('),src.index('checked_gptq_load('))
        self.assertLess(src.index('checked_gptq_load('),src.index('x=synthetic('))
        self.assertEqual(src.count('wrapper.save('),1)
        tree=ast.parse(src)
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='post_init' for n in ast.walk(tree)))
        self.assertIn("delta<=1e-6",src)
        self.assertIn("state==read(a.output/'state.json')",src)
    def test_runner_uses_new_lifecycle_for_every_subprocess(self):
        import run_org_b05_behavior_cell_v1 as runner
        import run_org_b05_behavior_cell_v2 as wrapper
        self.assertIs(runner.life,life)
        self.assertEqual(runner.__file__,wrapper.__file__)

if __name__=='__main__': unittest.main()
