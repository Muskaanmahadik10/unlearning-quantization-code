import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import org_b05_parallel_handoff_v1 as h


class Tests(unittest.TestCase):
    def test_missing_receipt_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'reload_a_behavior.json').write_text('{}')
            self.assertFalse(h.complete(p,'reload_a'))

    def test_failed_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'produce.json').write_text('{"status":"FAIL"}')
            with self.assertRaises(RuntimeError): h.complete(p,'produce')

    def test_only_missing_phases_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'pair').mkdir()
            record={'tasks':['pair'],'reused':{'pair':['produce','reload_a']}}
            with patch.object(h,'NEW',root), patch.object(h,'check_handoff',return_value=record), patch.object(h.subprocess,'run') as run:
                h.run(0)
                phases=[call.args[0][call.args[0].index('--phase')+1] for call in run.call_args_list]
                self.assertEqual(phases,['reload_b','emulated'])
                with self.assertRaises(FileExistsError): h.run(0)

    def test_failure_prevents_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'pair').mkdir()
            with patch.object(h,'NEW',root), patch.object(h,'check_handoff',return_value={'tasks':['pair'],'reused':{'pair':[]}}), patch.object(h.subprocess,'run',side_effect=RuntimeError('worker failed')):
                with self.assertRaisesRegex(RuntimeError,'worker failed'): h.run(0)
            self.assertFalse((root/'pair/condition_complete.json').exists())

    def test_active_parent_rejected(self):
        from types import SimpleNamespace
        with patch.object(h.subprocess,'run',return_value=SimpleNamespace(stdout='349643 COMPLETING')):
            with self.assertRaises(RuntimeError): h.terminal()

    def test_copied_evidence_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'value').write_text('new')
            (root/'handoff.json').write_text(json.dumps({'copied_evidence':{'value':'wrong'}}))
            with patch.object(h,'NEW',root), self.assertRaises(RuntimeError): h.check_handoff()


if __name__=='__main__': unittest.main()
