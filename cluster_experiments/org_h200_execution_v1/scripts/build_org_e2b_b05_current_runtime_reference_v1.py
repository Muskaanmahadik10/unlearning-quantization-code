#!/usr/bin/env python3
"""Freeze six outcome-blind current-runtime B0.5 QDP references."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid


EXPECTED = {
    ("ordinary_tofu_ga_seed42", "nf4"),
    ("ordinary_tofu_ga_seed42", "gptq_int4"),
    ("ordinary_tofu_ga_seed42", "rtn_int8"),
    ("adverse_muse_gdr_seed2024", "nf4"),
    ("adverse_muse_gdr_seed2024", "gptq_int4"),
    ("adverse_muse_gdr_seed2024", "rtn_int8"),
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require(len(args.receipts) == 6, "exactly six diagnostic receipts required")
    analysis = read_json(args.analysis)
    require(analysis.get("schema_version") == "org_e2b_b05_qdp_drift_analysis_v1", "analysis schema")
    require(analysis.get("status") == "HISTORICAL_TRANSPORT_FAIL", "repair requires frozen transport failure")
    require(analysis.get("protected_outcomes_inspected") is False, "protected outcomes already inspected")
    require(analysis.get("b05_behavior_authorized") is False, "unexpected behavior authorization")
    require(analysis.get("cell_count") == 6, "analysis cell count")
    require(not args.output_dir.exists(), "refusing to overwrite output directory")
    args.output_dir.mkdir(parents=True)

    indexed: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in args.receipts:
        receipt = read_json(path)
        require(receipt.get("schema_version") in {
            "org_e2b_b05_qdp_drift_diagnostic_v1",
            "org_e2b_b05_qdp_drift_diagnostic_v2",
        }, f"diagnostic schema: {path}")
        require(receipt.get("status") == "DIAGNOSTIC_PASS", f"diagnostic status: {path}")
        require(receipt.get("diagnostic_only") is True, f"diagnostic-only flag: {path}")
        require(receipt.get("production_eligible") is False, f"diagnostic production flag: {path}")
        require(receipt.get("protected_outcomes_inspected") is False, f"outcome contamination: {path}")
        require(receipt.get("target_count") == 224, f"target count: {path}")
        key = (str(receipt["cell_id"]), str(receipt["quantizer"]))
        require(key in EXPECTED and key not in indexed, f"diagnostic identity: {key}")
        cell_analysis = analysis["cells"].get(f"{key[0]}/{key[1]}")
        require(isinstance(cell_analysis, dict), f"analysis cell absent: {key}")
        require(cell_analysis["receipt"]["sha256"] == sha256_file(path), f"analysis receipt hash: {key}")
        indexed[key] = (path, receipt)
    require(set(indexed) == EXPECTED, "diagnostic universe mismatch")

    created = datetime.now(timezone.utc).isoformat()
    records = []
    for key in sorted(indexed):
        path, receipt = indexed[key]
        rows = receipt["per_layer"]
        names = [str(row["layer"]) for row in rows]
        require(len(names) == len(set(names)) == 224, f"layer universe: {key}")
        reference_rows = [{
            "layer": str(row["layer"]),
            "gqdp_signed": float(row["forget_qdp_signed"]),
            "retain_qdp_signed": float(row["retain_qdp_signed"]),
            "displacement_sha256": str(row["displacement_sha256"]),
        } for row in rows]
        filename = f"{key[0]}_{key[1]}_current_reference_v1.json"
        output = args.output_dir / filename
        payload = {
            "schema_version": "org_e2b_b05_current_runtime_qdp_reference_v1",
            "status": "PASS",
            "reference_regime": "isolated_packed_h200_current_runtime",
            "pre_outcome": True,
            "outcome_bearing": False,
            "protected_outcomes_inspected": False,
            "cell_id": key[0],
            "quantizer": key[1],
            "target_count": 224,
            "gradient_answer_tokens": int(receipt["answer_tokens"]["forget"]),
            "answer_tokens": dict(receipt["answer_tokens"]),
            "sample_counts": dict(receipt["sample_counts"]),
            "runtime": dict(receipt["runtime"]),
            "per_layer": reference_rows,
            "diagnostic_receipt": {"path": str(path.resolve()), "sha256": sha256_file(path)},
            "transport_analysis": {"path": str(args.analysis.resolve()), "sha256": sha256_file(args.analysis),
                                   "status": analysis["status"]},
            "created_at_utc": created,
            "b05_behavior_authorized": False,
            "atlas_production_authorized": False,
        }
        atomic_json(output, payload)
        records.append({"cell_id": key[0], "quantizer": key[1], "path": str(output.resolve()),
                        "sha256": sha256_file(output)})
    manifest = args.output_dir / "current_runtime_reference_manifest_v1.json"
    atomic_json(manifest, {
        "schema_version": "org_e2b_b05_current_runtime_reference_manifest_v1",
        "status": "PASS",
        "reference_count": 6,
        "pre_outcome": True,
        "protected_outcomes_inspected": False,
        "historical_transport_status": analysis["status"],
        "references": records,
        "created_at_utc": created,
        "b05_feature_v2_authorized": True,
        "b05_behavior_authorized": False,
        "atlas_production_authorized": False,
    })
    print("ORG_E2B_B05_CURRENT_RUNTIME_REFERENCE_FREEZE=PASS")
    print("REFERENCES=6")
    print("PROTECTED_OUTCOMES_INSPECTED=false")
    print("B05_FEATURE_V2_AUTHORIZED=true")
    print(f"MANIFEST={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
