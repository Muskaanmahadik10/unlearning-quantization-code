#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).with_name("run_org_e2b_b05_feature_cell_v1.py")
SPEC = importlib.util.spec_from_file_location("b05", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ContractTests(unittest.TestCase):
    def rows(self):
        rows = []
        for index in range(224):
            rows.append({
                "layer": f"layer.{index:03d}", "n_parameters": 10,
                "protection_forget_benefit": float(224 - index),
                "protection_retain_cost": -1.0 if index < 3 else float(index),
            })
        return rows

    def test_selector_contract(self):
        result = MODULE.select_sets(self.rows(), 2240)
        self.assertEqual(result["safe_looking_singleton"]["protected_layers"], ["layer.000"])
        self.assertEqual(result["pair"]["protected_layers"], ["layer.000", "layer.001"])
        self.assertEqual(result["risky_looking_singleton"]["protected_layers"], ["layer.223"])
        self.assertEqual(result["abstention"]["alias_of"], "full_packed")

    def test_safe_abstains_when_no_safe_matrix(self):
        rows = self.rows()
        for row in rows:
            row["protection_retain_cost"] = 1.0
        result = MODULE.select_sets(rows, 2240)
        self.assertTrue(result["safe_looking_singleton"]["abstained"])
        self.assertEqual(result["safe_looking_singleton"]["alias_of"], "full_packed")

    def test_budget_fails_closed(self):
        rows = self.rows()
        rows[0]["n_parameters"] = 1000
        with self.assertRaises(MODULE.GateError):
            MODULE.select_sets(rows, 2240)


if __name__ == "__main__":
    unittest.main()

