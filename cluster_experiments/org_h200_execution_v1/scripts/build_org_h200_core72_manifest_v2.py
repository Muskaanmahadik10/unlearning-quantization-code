#!/usr/bin/env python3
"""Build the H200 execution overlay without changing the frozen Core72 cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_BASE_SHA = "0e0637e324be42d7be1c36b10735dc6f2e2728cad4d5f09560da1e18c66cd8d1"
EXPECTED_SELECTORS = (
    "GQDP_signed_v42",
    "GQDP_signed_v42_transport_quantizer_cyclic_v1",
    "legacy_UPQI_signed_v4",
    "matched_random_type_depth_v1_seed00",
    "matched_random_type_depth_v1_seed01",
    "matched_random_type_depth_v1_seed02",
    "matched_random_type_depth_v1_seed03",
    "matched_random_type_depth_v1_seed04",
    "matched_random_type_depth_v1_seed05",
    "matched_random_type_depth_v1_seed06",
    "matched_random_type_depth_v1_seed07",
)
QUANTIZERS = ("gptq_int4", "nf4", "rtn_int8")
ALGOS = ("GA", "GDR", "KLR", "NPO")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_manifest",
        default="org_gb200_execution_v1/preregistration/core72_v1/org_core72_manifest_v1.json",
    )
    parser.add_argument(
        "--base_cells",
        default="org_gb200_execution_v1/preregistration/core72_v1/org_core72_cells.tsv",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    base_manifest_path = Path(args.base_manifest)
    base_cells_path = Path(args.base_cells)
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base.get("manifest_sha256") != EXPECTED_BASE_SHA:
        raise SystemExit("Frozen Core72 base manifest identity mismatch")
    with base_cells_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 72 or len({row["cell_id"] for row in rows}) != 72:
        raise SystemExit("Core72 cell table is not 72 unique cells")
    if sorted(row["cell_id"] for row in rows) != sorted(cell["cell_id"] for cell in base["cells"]):
        raise SystemExit("Base manifest and cell table identities differ")
    for row in rows:
        if tuple(row["selectors"].split(";")) != EXPECTED_SELECTORS:
            raise SystemExit(f"Selector contract mismatch: {row['cell_id']}")
        if int(row["primary_k"]) != 8:
            raise SystemExit(f"Primary k mismatch: {row['cell_id']}")

    models = sorted({row["model_tag"] for row in rows})
    if len(models) != 4:
        raise SystemExit(f"Expected four model tags, found {models}")
    stress_ids: list[str] = []
    for model_index, model in enumerate(models):
        for algo_index, algo in enumerate(ALGOS):
            candidates = [row for row in rows if row["model_tag"] == model and row["algo"] == algo]
            if not candidates:
                raise SystemExit(f"Empty stress stratum: {model}/{algo}")
            preferred = QUANTIZERS[(model_index + algo_index) % len(QUANTIZERS)]
            ordered_quantizers = (preferred,) + tuple(q for q in QUANTIZERS if q != preferred)
            chosen = None
            for quantizer in ordered_quantizers:
                matches = [row for row in candidates if row["quantizer"] == quantizer]
                if matches:
                    chosen = sorted(matches, key=lambda row: (int(row["seed"]), row["cell_id"]))[0]
                    break
            if chosen is None:
                raise SystemExit(f"Could not select stress cell: {model}/{algo}")
            stress_ids.append(chosen["cell_id"])
    if len(stress_ids) != 16 or len(set(stress_ids)) != 16:
        raise SystemExit("Stress16 selection is not 16 unique cells")

    ordered = sorted(rows, key=lambda row: row["cell_id"])
    output_rows = []
    for index, row in enumerate(ordered):
        enriched = dict(row)
        enriched["execution_index"] = str(index)
        enriched["execution_wave"] = "stress16" if row["cell_id"] in stress_ids else "remaining56"
        output_rows.append(enriched)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_output = output_dir / "org_h200_core72_cells_v2.tsv"
    manifest_output = output_dir / "org_h200_core72_manifest_v2.json"
    if cells_output.exists() or manifest_output.exists():
        raise SystemExit("H200 Core72 v2 outputs already exist; preserve them and use a new version")
    fieldnames = list(output_rows[0])
    with cells_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    amendment = Path("org_h200_execution_v1/PREREGISTRATION_AMENDMENT_H200_PRIMARY_V1.md")
    contract = Path("org_h200_execution_v1/ORG_H200_E1_SCIENTIFIC_CONTRACT_V1.md")
    canary_validation = Path(
        "results/org_h200_execution_v1/parity/h200_adversarial_canary_validation_v2_340505.json"
    )
    canary = json.loads(canary_validation.read_text(encoding="utf-8"))
    if canary.get("status") != "PASS" or canary.get("failures"):
        raise SystemExit("H200 adversarial canary certification is not PASS")
    manifest = {
        "schema_version": "org_h200_core72_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "ORG_H200_CORE72_FROZEN_READY_FOR_CHECKPOINT_LEDGER",
        "base_core72_manifest": {"path": base_manifest_path.as_posix(), "sha256": sha256_file(base_manifest_path), "embedded_manifest_sha256": EXPECTED_BASE_SHA},
        "base_cells": {"path": base_cells_path.as_posix(), "sha256": sha256_file(base_cells_path)},
        "h200_amendment": {"path": amendment.as_posix(), "sha256": sha256_file(amendment)},
        "scientific_contract": {"path": contract.as_posix(), "sha256": sha256_file(contract)},
        "h200_canary_validation": {"path": canary_validation.as_posix(), "sha256": sha256_file(canary_validation)},
        "hardware": "NVIDIA H200",
        "software_stack": {"python": "3.11.15", "torch": "2.5.1+cu124", "cuda": "12.4", "transformers": "4.46.3", "bitsandbytes": "0.45.0"},
        "claim_boundary": "H200-native matrix-ranking actionability and quantization durability; not standard unlearning efficacy",
        "outcome_blind": True,
        "canary_scientific_values_inspected": False,
        "canary_results_excluded_from_e1": True,
        "expected_cells": 72,
        "expected_conditions_per_cell": 12,
        "expected_results": 864,
        "stress16_cell_ids": stress_ids,
        "remaining56_cell_ids": [row["cell_id"] for row in output_rows if row["cell_id"] not in stress_ids],
        "selectors": list(EXPECTED_SELECTORS),
        "primary_k": 8,
        "stress_selection_rule": "one deterministic cell per sorted model_tag x fixed algo; cyclic preferred quantizer; smallest available numeric seed then cell_id",
        "cells_tsv": {"path": cells_output.as_posix(), "sha256": sha256_file(cells_output)},
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(manifest_output, manifest)
    print("ORG_H200_CORE72_MANIFEST_V2=PASS")
    print("CELLS=72")
    print("STRESS16=16")
    print("REMAINING56=56")
    print(f"MANIFEST_SHA256={manifest['manifest_sha256']}")
    print(f"MANIFEST={manifest_output}")
    print(f"CELLS_TSV={cells_output}")


if __name__ == "__main__":
    main()
