import math
import unittest
from m2_metric_assembly_v1 import *

class Tests(unittest.TestCase):
    def test_truth_loss_mean_not_arithmetic_probability(self):
        x=truth({'a':1.0},{'a':[2.0,4.0]},True)
        self.assertAlmostEqual(x['value_by_index']['a']['score'],math.exp(-3)/(math.exp(-1)+1e-10))
    def test_truth_nonforget_clip(self):
        self.assertEqual(truth({'a':4},{'a':[1,1]})['agg_value'],0)
    def test_normalization_sums_all_options(self):
        self.assertAlmostEqual(normalized({'a':.3},{'a':[.2,.1]})['agg_value'],.3/(.6+1e-10))
    def test_auc_orientation_and_ties(self):
        self.assertEqual(auc([1,2],[3,4]),1);self.assertEqual(auc([3,4],[1,2]),0);self.assertEqual(auc([1],[1]),.5)
    def test_min_k_round_down(self):
        self.assertEqual(min_k([-1,-2,-3,-4,-5,-6],.4),5.5)
        self.assertEqual(min_k([]),0);self.assertEqual(min_k([-3]),3)
    def test_extraction_official_floor(self):
        self.assertEqual(extraction([4,5,6],[1,2,3]),1-2/3)
        self.assertEqual(extraction([0,2,3],[1,2,3]),1-1/3)
        self.assertEqual(extraction([1,2,3],[1,2,3]),1);self.assertEqual(extraction([],[]),0)
    def test_order_missing_duplicates(self):
        with self.assertRaises(ValueError):aligned({'a':1,'b':2},{'b':2,'a':1})
        with self.assertRaises(ValueError):aligned({'a':1},{})
    def test_bad_values(self):
        for x in (float('nan'),float('inf'),True,None):
            with self.assertRaises(ValueError):finite(x)
        with self.assertRaises(ValueError):min_k([1])
        with self.assertRaises(ValueError):normalized({'a':2},{'a':[.1]})
    def test_no_reference_fallback(self):
        for r in (None,{'role':'published_R'},{'role':'synthetic_R'}):
            with self.assertRaises(ValueError):assemble({},r,scope='controlled_scientific')
    def test_exact_seven_and_nine(self):
        self.assertEqual(len(set(METRICS)),7);self.assertEqual(len(set(UTILITY)),9)

if __name__=='__main__':unittest.main()
