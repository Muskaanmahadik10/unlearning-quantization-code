#!/usr/bin/env python3
"""Static and pure-Python tests for the E2A adaptive execution contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    worker_path = ROOT / "org_h200_execution_v1/scripts/run_org_h200_e2a_cell_v1.py"
    validator_path = ROOT / "org_h200_execution_v1/scripts/validate_org_h200_e2a_v1.py"
    analyzer_path = ROOT / "org_h200_execution_v1/scripts/analyze_org_h200_e2a_v1.py"
    canary_path = ROOT / "org_h200_execution_v1/slurm/org_h200_e2a_canary1_v1.slurm"
    production_path = ROOT / "org_h200_execution_v1/slurm/org_h200_e2a_production8_v1.slurm"
    contract_path = ROOT / "org_h200_execution_v1/ORG_H200_E2A_ADAPTIVE_TRADEOFF_SCIENTIFIC_CONTRACT_V1.md"

    worker = load("e2a_worker", worker_path)
    validator = load("e2a_validator", validator_path)
    analyzer = load("e2a_analyzer", analyzer_path)
    assert worker.K_VALUES == (2, 4, 16, 32)
    assert validator.K_VALUES == worker.K_VALUES == analyzer.K_VALUES
    assert analyzer.PRIMARY_K == (2, 4)
    assert 1 + 11 * len(worker.K_VALUES) == 45
    assert analyzer.holm({2: 0.01, 4: 0.03}, 0.025) == {2: True, 4: False}

    worker_text = worker_path.read_text(encoding="utf-8")
    validator_text = validator_path.read_text(encoding="utf-8")
    canary_text = canary_path.read_text(encoding="utf-8")
    production_text = production_path.read_text(encoding="utf-8")
    contract_text = contract_path.read_text(encoding="utf-8")
    assert 'lock.get("e1_scientific_gate_status") != "STOP"' in worker_text
    assert 'lock.get("adaptive_followup_authorized") is not True' in worker_text
    assert "Refusing to overwrite existing E2A cell outputs" in worker_text
    assert "Expected 45 E2A results" in worker_text
    assert "outcome_values_inspected" in validator_text
    assert "--task_index 0" in canary_text
    assert "ORG_H200_E2A_CANARY_LOCK_V1.json" in canary_text
    assert "gate2_v4_bridge_standard_h200_e2a_canary1_v1" in canary_text
    assert "srun --wait=0" in production_text
    assert "ORG_H200_E2A_PRODUCTION_LOCK_V1.json" in production_text
    assert "e2a_canary_status" in production_text
    assert "0|8|16|24|32|40|48|56|64" in production_text
    assert "gate2_v4_bridge_standard_h200_e2a_core72_v1" in production_text
    assert "explicitly post-E1 adaptive study" in contract_text
    assert "intersection-union" in contract_text
    print("ORG_H200_E2A_CONTRACT_TESTS=PASS")
    print("PRIMARY_K=2,4")
    print("SECONDARY_K=16,32")
    print("CONDITIONS_PER_CELL=45")
    print("PRODUCTION_CELLS=72")
    print("PRODUCTION_RESULTS=3240")


if __name__ == "__main__":
    main()

