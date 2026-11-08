#!/usr/bin/env python3
"""Analyze six non-authorizing B0.5 cross-runtime QDP drift receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence
import uuid


EXPECTED = {
    ("ordinary_tofu_ga_seed42", "nf4"),
    ("ordinary_tofu_ga_seed42", "gptq_int4"),
    ("ordinary_tofu_ga_seed42", "rtn_int8"),
    ("adverse_muse_gdr_seed2024", "nf4"),
    ("adverse_muse_gdr_seed2024", "gptq_int4"),
    ("adverse_muse_gdr_seed2024", "rtn_int8"),
}
SIGN_THRESHOLD = 1.0e-10
MIN_SIGN_AGREEMENT = 0.90
MIN_SPEARMAN = 0.90
MIN_TOP8_OVERLAP = 0.75
MAX_NORMALIZED_L2 = 0.25


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    require(sidecar.is_file(), f"missing sidecar: {sidecar}")
    records = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        records[str(Path(name).resolve())] = digest
    resolved = str(path.resolve())
    require(records.get(resolved) == sha256_file(path), f"receipt sidecar mismatch: {path}")


def percentile(values: Sequence[float], fraction: float) -> float:
    require(bool(values), "empty percentile")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    require(len(left) == len(right) and len(left) > 1, "correlation inputs")
    lmean = statistics.fmean(left)
    rmean = statistics.fmean(right)
    numerator = math.fsum((a - lmean) * (b - rmean) for a, b in zip(left, right))
    lnorm = math.sqrt(math.fsum((a - lmean) ** 2 for a in left))
    rnorm = math.sqrt(math.fsum((b - rmean) ** 2 for b in right))
    require(lnorm > 0.0 and rnorm > 0.0, "constant correlation input")
    return numerator / (lnorm * rnorm)


def top_names(names: Sequence[str], values: Sequence[float], count: int) -> list[str]:
    return [names[index] for index in sorted(range(len(names)), key=lambda index: (-values[index], names[index]))[:count]]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def analyze_receipt(path: Path) -> tuple[tuple[str, str], dict[str, Any]]:
    verify_sidecar(path)
    receipt = read_json(path)
    require(receipt.get("schema_version") in {
        "org_e2b_b05_qdp_drift_diagnostic_v1",
        "org_e2b_b05_qdp_drift_diagnostic_v2",
    }, f"diagnostic schema: {path}")
    require(receipt.get("status") == "DIAGNOSTIC_PASS", f"diagnostic status: {path}")
    require(receipt.get("diagnostic_only") is True, f"diagnostic-only flag: {path}")
    require(receipt.get("production_eligible") is False, f"production flag: {path}")
    require(receipt.get("protected_outcomes_inspected") is False, f"outcome contamination: {path}")
    require(receipt.get("target_count") == 224, f"target count: {path}")
    key = (str(receipt["cell_id"]), str(receipt["quantizer"]))

    metric_path = Path(receipt["prerequisites"]["metric_reference"]["path"])
    oracle_path = Path(receipt["prerequisites"]["legacy_tensor_oracle"]["path"])
    require(metric_path.is_file() and oracle_path.is_file(), f"missing linked evidence: {path}")
    require(sha256_file(metric_path) == receipt["prerequisites"]["metric_reference"]["sha256"], "metric hash")
    require(sha256_file(oracle_path) == receipt["prerequisites"]["legacy_tensor_oracle"]["sha256"], "oracle hash")
    metric = read_json(metric_path)
    oracle = read_json(oracle_path)
    metric_rows = {str(row["layer"]): row for row in metric["per_layer"]}
    oracle_rows = {str(row["layer"]): row for row in oracle["per_layer"]}
    rows = receipt["per_layer"]
    names = [str(row["layer"]) for row in rows]
    require(len(names) == len(set(names)) == 224, "layer universe")
    require(set(names) == set(metric_rows) == set(oracle_rows), "linked layer universe")

    current = [float(row["forget_qdp_signed"]) for row in rows]
    historical = [float(metric_rows[name]["gqdp_signed"]) for name in names]
    absolute = [abs(a - b) for a, b in zip(current, historical)]
    relative = [delta / max(abs(old), 1.0e-12) for delta, old in zip(absolute, historical)]
    eligible = [index for index, old in enumerate(historical) if abs(old) > SIGN_THRESHOLD]
    require(bool(eligible), "no non-negligible historical scores")
    sign_matches = sum((current[index] > 0) == (historical[index] > 0) for index in eligible)
    sign_agreement = sign_matches / len(eligible)
    normalized_l2 = math.sqrt(math.fsum(delta * delta for delta in absolute)) / max(
        math.sqrt(math.fsum(value * value for value in historical)), 1.0e-30
    )
    spearman = pearson(ranks(current), ranks(historical))
    pearson_value = pearson(current, historical)
    current_benefit = [-value for value in current]
    historical_benefit = [-value for value in historical]
    current_top8 = top_names(names, current_benefit, 8)
    historical_top8 = top_names(names, historical_benefit, 8)
    top8_overlap = len(set(current_top8) & set(historical_top8)) / 8.0
    displacement_exact = sum(
        str(row["displacement_sha256"]) == str(oracle_rows[str(row["layer"])]["displacement_sha256"])
        for row in rows
    )
    tokens_match = int(receipt["answer_tokens"]["forget"]) == int(metric["gradient_answer_tokens"])
    transport_pass = (
        tokens_match
        and sign_agreement >= MIN_SIGN_AGREEMENT
        and spearman >= MIN_SPEARMAN
        and top8_overlap >= MIN_TOP8_OVERLAP
        and normalized_l2 <= MAX_NORMALIZED_L2
        and (key[1] == "gptq_int4" or displacement_exact == 224)
    )
    worst = sorted(
        ({"layer": name, "historical": old, "current": new, "absolute_delta": delta}
         for name, old, new, delta in zip(names, historical, current, absolute)),
        key=lambda row: (-row["absolute_delta"], row["layer"]),
    )[:10]
    result = {
        "receipt": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "cell_id": key[0], "quantizer": key[1], "target_count": 224,
        "forget_answer_tokens_current": int(receipt["answer_tokens"]["forget"]),
        "forget_answer_tokens_historical": int(metric["gradient_answer_tokens"]),
        "answer_tokens_match": tokens_match,
        "displacement_exact": displacement_exact,
        "displacement_exact_fraction": displacement_exact / 224.0,
        "absolute_delta": {"median": percentile(absolute, 0.5), "p95": percentile(absolute, 0.95), "max": max(absolute)},
        "relative_delta": {"median": percentile(relative, 0.5), "p95": percentile(relative, 0.95), "max": max(relative)},
        "reference_normalized_l2": normalized_l2,
        "pearson": pearson_value, "spearman": spearman,
        "sign_threshold": SIGN_THRESHOLD, "sign_eligible": len(eligible), "sign_agreement": sign_agreement,
        "current_top8": current_top8, "historical_top8": historical_top8,
        "top8_overlap": top8_overlap, "worst_absolute_deltas": worst,
        "historical_transport_gate": "PASS" if transport_pass else "FAIL",
    }
    return key, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(len(args.receipts) == 6, "exactly six successful diagnostics required")
    results = {}
    for path in args.receipts:
        key, result = analyze_receipt(path)
        require(key not in results, f"duplicate cell: {key}")
        results[key] = result
    require(set(results) == EXPECTED, "diagnostic universe mismatch")
    all_pass = all(result["historical_transport_gate"] == "PASS" for result in results.values())
    payload = {
        "schema_version": "org_e2b_b05_qdp_drift_analysis_v1",
        "status": "HISTORICAL_TRANSPORT_PASS" if all_pass else "HISTORICAL_TRANSPORT_FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_frozen_before_diagnostic_inspection": {
            "sign_threshold": SIGN_THRESHOLD,
            "minimum_sign_agreement": MIN_SIGN_AGREEMENT,
            "minimum_spearman": MIN_SPEARMAN,
            "minimum_top8_overlap": MIN_TOP8_OVERLAP,
            "maximum_reference_normalized_l2": MAX_NORMALIZED_L2,
            "nf4_rtn_exact_displacement_required": True,
            "gptq_exact_displacement_descriptive_due_cross_runtime_hessian_boundary": True,
        },
        "cell_count": 6,
        "cells": {f"{key[0]}/{key[1]}": results[key] for key in sorted(results)},
        "protected_outcomes_inspected": False,
        "b05_behavior_authorized": False,
        "atlas_production_authorized": False,
    }
    atomic_json(args.output, payload)
    print(f"ORG_E2B_B05_QDP_DRIFT_ANALYSIS={payload['status']}")
    print("CELLS=6")
    print("PROTECTED_OUTCOMES_INSPECTED=false")
    print("B05_BEHAVIOR_AUTHORIZED=false")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
