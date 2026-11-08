import csv
import tempfile
import unittest
from decimal import Decimal as D
from pathlib import Path
from analyze_b05 import analyze, load, favorable, sign, COHORTS, QUANTIZERS, CONDITIONS

SOURCE = Path(__file__).resolve().parents[1]/'frozen/delivery/Supplementary_tables/Figure_data/B05Contrasts.csv'
if not SOURCE.exists():
    SOURCE=Path(__file__).resolve().parents[1]/'Supplementary_tables/Figure_data/B05Contrasts.csv'


class Tests(unittest.TestCase):
    def modified(self, edit):
        with SOURCE.open(encoding='utf-8-sig', newline='') as f:
            reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
        edit(rows)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.csv'
            with path.open('w', encoding='utf-8', newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            return analyze(load(path))

    def test_accepted_counts(self):
        s, pairs, rows=analyze(load(SOURCE))
        self.assertEqual((len(rows),len(pairs)),(36,18))
        self.assertEqual(list(s['sign_partition'].values()),[22,6,6,2])
        self.assertEqual(s['joint_favorable'],{'prediction':12,'packed':4,'emulated':2})
        self.assertEqual((s['packed_and_emulated_joint'],s['predicted_and_packed_joint'],s['predicted_but_not_packed_joint']),(0,4,8))
        self.assertEqual([(r['opposite_signs'],r['eligible_contrasts']) for r in s['magnitude_sensitivity']],[(8,36),(5,29),(1,17)])

    def test_exact_boundaries(self):
        self.assertFalse(favorable(D('0.000001'),D('-1')))
        self.assertFalse(favorable(D('1'),D('-0.000001')))
        self.assertTrue(favorable(D('0.0000010001'),D('-0.0000010001')))
        self.assertEqual(sign(D('-0.000001')),0)

    def test_small_opposite_excluded_both_required(self):
        s,_,_=self.modified(lambda rows: rows[0].update(packed_delta='-0.01',emulated_delta='0.01'))
        # Equality at floor .01 is excluded, not silently admitted.
        self.assertEqual(s['magnitude_sensitivity'][2]['eligible_contrasts'],16)

    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):self.modified(lambda r:r.append(r[0].copy()))
    def test_missing(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):self.modified(lambda r:r.pop())
    def test_nonfinite(self):
        for value in ('NaN','Infinity','-Infinity'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'nonfinite'):
                self.modified(lambda r:r[0].update(packed_delta=value))
    def test_invalid_value(self):
        with self.assertRaisesRegex(ValueError,'invalid'):self.modified(lambda r:r[0].update(packed_delta=''))
    def test_identity_mismatch(self):
        with self.assertRaisesRegex(ValueError,'identity mismatch'):self.modified(lambda r:r[0].update(selected_layers='wrong'))
    def test_wrong_role(self):
        with self.assertRaisesRegex(ValueError,'unexpected'):self.modified(lambda r:r[0].update(role='retain_all'))
    def test_order_invariance(self):
        s, pairs, _=analyze(load(SOURCE)); t, pp, _=self.modified(lambda r:r.reverse())
        self.assertEqual(s,t);self.assertEqual(pairs,pp)

if __name__=='__main__':unittest.main(verbosity=2)
