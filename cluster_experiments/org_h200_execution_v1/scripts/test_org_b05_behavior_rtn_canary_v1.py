import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_org_b05_behavior_rtn_canary_v1 as runner


class AuthorizationTests(unittest.TestCase):
    def record(self):
        return {'schema_version':'org_b05_behavior_canary_budget_v1', 'status':'APPROVED',
                'scope':'ordinary_rtn_pair_canary', 'plan_sha256':'hash', 'runner_sha256':'hash',
                'max_allocated_gpu_hours':0.5, 'cumulative_accounting_reviewed':True,
                'approval_record':'explicit prospective authorization'}

    def verify(self, record):
        args = SimpleNamespace(authorization=Path('auth'), plan=Path('plan'))
        with patch.object(runner.base, 'read', return_value=record), patch.object(runner.base, 'sha256_file', return_value='hash'):
            return runner.authorize(args)

    def test_approved(self):
        self.assertEqual(self.verify(self.record())['scope'], 'ordinary_rtn_pair_canary')

    def test_fail_closed(self):
        for key in self.record():
            record = self.record(); del record[key]
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.verify(record)

    def test_wrong_identity(self):
        for key in ('runner_sha256','plan_sha256','scope'):
            record = self.record(); record[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.verify(record)


if __name__ == '__main__': unittest.main()
