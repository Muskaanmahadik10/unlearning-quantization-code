import copy
import math
import sys
import unittest
from org_b05_behavior_evaluator_v1 import summarize, compare_reloads, answer_loss


class ContractTests(unittest.TestCase):
    def rows(self):
        return [{'index': i, 'sample_sha256': str(i), 'encoding_sha256': str(i),
                 'tokens': n, 'loss_sum': loss} for i, (n, loss) in enumerate([(1, 1.), (3, 9.)])]

    def test_weighting(self):
        x = summarize(self.rows())
        self.assertEqual(x['token_mean_nll'], 2.5)
        self.assertEqual(x['sample_mean_nll'], 2.)

    def test_exact(self):
        self.assertEqual(compare_reloads(self.rows(), self.rows())['status'], 'PASS')

    def test_reject_invalid(self):
        for field, value in [('tokens', 0), ('loss_sum', math.nan), ('loss_sum', math.inf)]:
            rows = self.rows(); rows[0][field] = value
            with self.assertRaises(ValueError): summarize(rows)

    def test_reject_identity(self):
        for field in ('index', 'sample_sha256', 'encoding_sha256', 'tokens'):
            rows = self.rows(); rows[0][field] = 9 if field in ('tokens','index') else 'changed'
            with self.assertRaises(ValueError): compare_reloads(self.rows(), rows)

    def test_reject_drift_and_tolerance_change(self):
        rows = self.rows(); rows[0]['loss_sum'] += .25
        with self.assertRaises(ValueError): compare_reloads(self.rows(), rows)
        with self.assertRaises(ValueError): compare_reloads(self.rows(), self.rows(), .5)


def torch_tests():
    import torch
    labels = torch.tensor([[-100, -100, 2, 1]])
    logits = torch.zeros(1, 4, 3)
    loss, n = answer_loss(torch, logits, labels)
    assert n == 2 and abs(loss-2*math.log(3)) < 1e-6
    logits[:, 0, :] = torch.tensor([100., -100., 0.])
    assert answer_loss(torch, logits, labels) == (loss, n), 'prompt leaked into loss'
    logits[:, 1, 2] = 10.
    assert answer_loss(torch, logits, labels)[0] < loss, 'first answer not scored'
    try: answer_loss(torch, logits, torch.full_like(labels, -100))
    except ValueError: pass
    else: raise AssertionError('zero-token accepted')
    logits[:, 1, 2] = float('nan')
    try: answer_loss(torch, logits, labels)
    except ValueError: pass
    else: raise AssertionError('NaN accepted')
    print('B05_EVALUATOR_TORCH_CPU_TESTS=PASS')


if __name__ == '__main__':
    if '--torch' in sys.argv:
        sys.argv.remove('--torch')
        torch_tests()
    unittest.main()
