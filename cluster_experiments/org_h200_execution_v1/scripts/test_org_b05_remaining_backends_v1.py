import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import org_b05_remaining_backends_v1 as b


class Tests(unittest.TestCase):
    def accounting(self):
        return '\n'.join(f'{j}|COMPLETED|{s}|cpu=8,gres/gpu=1' for j,s in
            [('349643',1000),('349745_0',550),('349745_1',607),('349745_2',611)])

    def test_sum(self): self.assertEqual(b.allocation_seconds(self.accounting()),2768)
    def test_missing(self):
        with self.assertRaises(RuntimeError): b.allocation_seconds(self.accounting().split('\n',1)[1])
    def test_duplicate(self):
        with self.assertRaises(RuntimeError): b.allocation_seconds(self.accounting()+'\n'+self.accounting())
    def test_running(self):
        with self.assertRaises(RuntimeError): b.allocation_seconds(self.accounting().replace('COMPLETED','RUNNING',1))
    def test_gpu_count(self):
        with self.assertRaises(RuntimeError): b.allocation_seconds(self.accounting().replace('gpu=1','gpu=2',1))
    def test_excludes_billing(self):
        self.assertEqual(b.allocation_seconds(self.accounting().replace('cpu=8','billing=800,cpu=8')),2768)


if __name__=='__main__': unittest.main()
