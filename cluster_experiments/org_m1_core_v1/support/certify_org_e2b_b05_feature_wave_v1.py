#!/usr/bin/env python3
"""Certify six outcome-blind B0.5 feature receipts into one selector lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid


EXPECTED = {
    ("ordinary_tofu_ga_seed42", "nf4"),
    ("ordinary_tofu_ga_seed42", "gptq_int4"),
    ("ordinary_tofu_ga_seed42", "rtn_int8"),
    ("adverse_muse_gdr_seed2024", "nf4"),
    ("adverse_muse_gdr_seed2024", "gptq_int4"),
    ("adverse_muse_gdr_seed2024", "rtn_int8"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.receipts) != 6:
        raise ValueError("exactly six receipts required")
    cells = {}
    prerequisite_digests = set()
    for path in args.receipts:
        item = read(path)
        key = (str(item.get("cell_id")), str(item.get("quantizer")))
        if key in cells:
            raise ValueError(f"duplicate feature cell: {key}")
        if item.get("schema_version") != "org_e2b_b05_feature_cell_v1" or item.get("status") != "PASS":
            raise ValueError(f"failed feature receipt: {path}")
        if item.get("mode") != "production" or item.get("production_eligible") is not True:
            raise ValueError(f"non-production feature receipt: {path}")
        if item.get("outcome_bearing") is not False or item.get("protected_outcomes_inspected") is not False:
            raise ValueError(f"outcome contamination: {path}")
        if item.get("target_count") != 224 or item.get("sample_counts") != {"forget": 128, "retain": 128, "calibration": 128}:
            raise ValueError(f"feature contract mismatch: {path}")
        if float(item.get("max_retain_independent_abs_delta", 1.0)) > 1e-10:
            raise ValueError(f"retain QDP recomputation failed: {path}")
        if float(item.get("max_historical_forget_reference_abs_delta", 1.0)) > 1e-10:
            raise ValueError(f"historical GQDP reference failed: {path}")
        conditions = item.get("conditions", {})
        if set(conditions) != {"bf16", "full_packed", "safe_looking_singleton", "risky_looking_singleton", "pair", "abstention"}:
            raise ValueError(f"condition universe: {path}")
        if conditions["abstention"].get("alias_of") != "full_packed":
            raise ValueError(f"abstention alias: {path}")
        prerequisite_digests.add(json.dumps(item.get("prerequisites"), sort_keys=True, separators=(",", ":")))
        cells[key] = {"receipt": {"path": str(path.resolve()), "sha256": sha256_file(path)}, "conditions": conditions,
                      "per_layer_sha256": item["per_layer_sha256"], "target_parameters": item["target_parameters"]}
    if set(cells) != EXPECTED:
        raise ValueError(f"feature universe mismatch: got {sorted(cells)}")
    payload = {
        "schema_version": "org_e2b_b05_selector_lock_v1",
        "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_cell_count": 6, "condition_count_per_cell": 6,
        "cells": {f"{key[0]}/{key[1]}": cells[key] for key in sorted(cells)},
        "selector_definitions_frozen": True, "protected_outcomes_inspected": False,
        "b05_behavior_authorized": True, "atlas_production_authorized": False,
    }
    atomic(args.output, payload)
    print("ORG_E2B_B05_FEATURE_WAVE_CERTIFICATION=PASS")
    print("FEATURE_CELLS=6")
    print("B05_BEHAVIOR_AUTHORIZED=true")
    print("ATLAS_PRODUCTION_AUTHORIZED=false")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

