#!/usr/bin/env python3
"""Unit tests for the current-runtime reference freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import build_org_e2b_b05_current_runtime_reference_v1 as subject


class FreezeTests(unittest.TestCase):
    def test_six_cell_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = []
            cells = {}
            for cell, quantizer in sorted(subject.EXPECTED):
                path = root / f"{cell}_{quantizer}.json"
                rows = [{
                    "layer": f"layer.{index}",
                    "forget_qdp_signed": float(index + 1),
                    "retain_qdp_signed": -float(index + 1),
                    "displacement_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                } for index in range(224)]
                payload = {
                    "schema_version": "org_e2b_b05_qdp_drift_diagnostic_v2" if quantizer == "gptq_int4" else "org_e2b_b05_qdp_drift_diagnostic_v1",
                    "status": "DIAGNOSTIC_PASS", "diagnostic_only": True,
                    "production_eligible": False, "protected_outcomes_inspected": False,
                    "target_count": 224, "cell_id": cell, "quantizer": quantizer,
                    "answer_tokens": {"forget": 10, "retain": 11},
                    "sample_counts": {"forget": 128, "retain": 128, "calibration": 128},
                    "runtime": {"hardware": "NVIDIA H200"}, "per_layer": rows,
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                digest = subject.sha256_file(path)
                cells[f"{cell}/{quantizer}"] = {"receipt": {"sha256": digest}}
                receipts.append(path)
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({
                "schema_version": "org_e2b_b05_qdp_drift_analysis_v1",
                "status": "HISTORICAL_TRANSPORT_FAIL", "protected_outcomes_inspected": False,
                "b05_behavior_authorized": False, "cell_count": 6, "cells": cells,
            }), encoding="utf-8")
            output = root / "freeze"
            rc = subject.main(["--analysis", str(analysis), "--receipts", *map(str, receipts), "--output-dir", str(output)])
            self.assertEqual(rc, 0)
            manifest = json.loads((output / "current_runtime_reference_manifest_v1.json").read_text())
            self.assertEqual(manifest["reference_count"], 6)
            self.assertTrue(manifest["b05_feature_v2_authorized"])
            self.assertEqual(len(list(output.glob("*_current_reference_v1.json"))), 6)


if __name__ == "__main__":
    unittest.main()
