import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import recover_org_b05_finish_submission_v1 as r

class Tests(unittest.TestCase):
    def test_rejection(self):
        good='sbatch: error: QOSMaxSubmitJobPerUserLimit\nBatch job submission failed'
        r.check_rejection(good)
        for bad in ('Socket timed out',good+'\n123456',good+'\nSubmitted batch job 123456'):
            with self.assertRaises(RuntimeError): r.check_rejection(bad)
    def test_only_gptq_and_finalizer_submitted(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r.wave,'ROOT',Path(d)),patch.object(r,'preflight'),patch.object(r,'await_capacity'),patch.object(r.launch,'submit',side_effect=['123456','123457']) as submit:
            r.main()
            self.assertEqual(submit.call_count,2)
            self.assertEqual(submit.call_args_list[0].args[0],'gptq_recovery_v1')
            self.assertIn('--array=0-8',submit.call_args_list[0].args[1])
            self.assertIn('--dependency=afterany:123456',submit.call_args_list[1].args[1])
            self.assertEqual(r.wave.life.read(Path(d)/'submissions.json')['arrays'],{'other':'350098','gptq':'123456'})
    def test_capacity_error_prevents_submission(self):
        with patch.object(r,'preflight'),patch.object(r,'await_capacity',side_effect=RuntimeError('query failed')),patch.object(r.launch,'submit') as submit:
            with self.assertRaises(RuntimeError): r.main()
            submit.assert_not_called()

if __name__=='__main__': unittest.main()
