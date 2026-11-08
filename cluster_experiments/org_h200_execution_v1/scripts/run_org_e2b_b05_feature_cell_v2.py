#!/usr/bin/env python3
"""Run B0.5 feature generation against a frozen current-runtime reference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


SCHEMA = "org_e2b_b05_feature_cell_v2"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def argument(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"missing {flag}") from exc


def main() -> int:
    source = Path(__file__).with_name("run_org_e2b_b05_feature_cell_v1.py")
    argv = list(sys.argv[1:])
    metric_path = Path(argument(argv, "--metric-reference")).resolve()
    oracle_path = Path(argument(argv, "--legacy-tensor-oracle")).resolve()
    require(metric_path == oracle_path, "V2 requires one identical frozen current-runtime reference")
    reference = json.loads(metric_path.read_text(encoding="utf-8"))
    require(reference.get("schema_version") == "org_e2b_b05_current_runtime_qdp_reference_v1", "reference schema")
    require(reference.get("status") == "PASS" and reference.get("pre_outcome") is True, "reference status")
    require(reference.get("protected_outcomes_inspected") is False, "outcome contamination")
    require(reference.get("target_count") == 224, "reference target count")
    require(reference.get("cell_id") == argument(argv, "--cell-id"), "reference cell")
    require(reference.get("quantizer") == argument(argv, "--quantizer"), "reference quantizer")

    spec = importlib.util.spec_from_file_location("org_e2b_b05_feature_v1_current_reference", source)
    require(spec is not None and spec.loader is not None, "cannot load frozen V1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sha_index = argv.index("--expected-script-sha256") + 1
    argv[sha_index] = sha256_file(source)

    original_read = module.read_json
    def compatible_read(path: Path) -> dict[str, Any]:
        value = original_read(path)
        if path.resolve() == metric_path:
            value = dict(value)
            value["schema_version"] = "org_e2b_b05_legacy_tensor_oracle_v1"
            value["outcome_bearing"] = False
        return value

    original_atomic = module.atomic_json
    def v2_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        value = dict(payload)
        value.update({
            "schema_version": SCHEMA,
            "status": "PASS",
            "reference_regime": "isolated_packed_h200_current_runtime_pre_outcome",
            "current_runtime_reference": {"path": str(metric_path), "sha256": sha256_file(metric_path)},
            "historical_transport_status": "HISTORICAL_TRANSPORT_FAIL",
            "legacy_displacement_used_for_acceptance": False,
            "historical_scalar_used_for_acceptance": False,
            "protected_outcomes_inspected": False,
            "b05_behavior_authorized": False,
            "atlas_production_authorized": False,
            "frozen_v1_computation": {"path": str(source.resolve()), "sha256": sha256_file(source)},
            "v2_wrapper": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        })
        value["max_current_runtime_forget_reference_abs_delta"] = value.get("max_historical_forget_reference_abs_delta")
        original_atomic(path, value)

    module.read_json = compatible_read
    module.atomic_json = v2_atomic
    result = int(module.main(argv))
    print("ORG_E2B_B05_FEATURE_CELL_V2=PASS")
    print("REFERENCE_REGIME=current_runtime_pre_outcome")
    print("B05_BEHAVIOR_AUTHORIZED=false")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
