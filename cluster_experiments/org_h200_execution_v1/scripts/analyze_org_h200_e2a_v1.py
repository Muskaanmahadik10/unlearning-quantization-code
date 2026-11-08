#!/usr/bin/env python3
"""Apply the frozen post-E1 E2A low-dose safe-effective decision rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V4_DIR = REPO_ROOT / "gate2_v4_final/scripts"
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

from gate2_v4_contract import load_json, load_source_lock, sha256_file, verify_source_lock_files  # noqa: E402
from analyze_org_h200_e1_gate_v1 import summarize  # noqa: E402

K_VALUES = (2, 4, 16, 32)
PRIMARY_K = (2, 4)
NATIVE = "GQDP_signed_v42"
TRANSPORT = "GQDP_signed_v42_transport_quantizer_cyclic_v1"
LEGACY = "legacy_UPQI_signed_v4"
RANDOM_PREFIX = "matched_random_type_depth_v1_seed"
ALPHA = 0.025


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def holm(p_values: dict[int, float], alpha: float = ALPHA) -> dict[int, bool]:
    """Return Holm step-down rejection decisions keyed by dose."""
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    decisions = {key: False for key in p_values}
    for index, key in enumerate(ordered):
        if p_values[key] <= alpha / (len(ordered) - index):
            decisions[key] = True
        else:
            break
    return decisions


def holm_adjusted(p_values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[int, float] = {}
    running = 0.0
    m = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, (m - index) * p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def aggregate(rows: list[dict[str, Any]], key: str) -> tuple[list[float], dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint"]].append(float(row[key]))
    values = {name: mean(group) for name, group in sorted(grouped.items())}
    return list(values.values()), values


def basic(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values), "mean": mean(values), "median": float(statistics.median(values)),
        "minimum": min(values), "maximum": max(values),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "positive_fraction": sum(value > 0 for value in values) / len(values),
        "nonnegative_fraction": sum(value >= 0 for value in values) / len(values),
    }


def group_report(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        name: {
            "cells": len(subset),
            "native_D": basic([float(row["native_D"]) for row in subset]),
            "safety_slack": basic([float(row["safety_slack"]) for row in subset]),
            "joint_positive_safe_fraction": mean(
                float(row["native_D"] > 0 and row["safety_slack"] >= 0) for row in subset
            ),
        }
        for name, subset in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source_lock", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--integrity_validation", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_tsv", required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    lock = load_source_lock(args.source_lock)
    verify_source_lock_files(lock)
    lock_sha = str(lock["source_lock_sha256"])
    if (
        lock.get("lock_role") != "e2a_production"
        or lock.get("adaptive_followup_authorized") is not True
        or lock.get("original_e2_authorized") is not False
        or lock.get("e2a_canary_status") != "PASS"
        or lock.get("e2a_k_values") != list(K_VALUES)
    ):
        raise SystemExit("E2A analysis requires the exact production lock")
    integrity = load_json(args.integrity_validation)
    if (
        integrity.get("schema_version") != "org_h200_e2a_integrity_v1"
        or integrity.get("status") != "PASS" or integrity.get("mode") != "production"
        or integrity.get("cell_count") != 72 or integrity.get("result_count") != 3240
        or integrity.get("receipt_count") != 72 or integrity.get("source_lock_sha256") != lock_sha
        or integrity.get("failures")
    ):
        raise SystemExit("E2A integrity validation is not an exact production PASS")

    with Path(args.cells).open("r", encoding="utf-8", newline="") as handle:
        cells = list(csv.DictReader(handle, delimiter="\t"))
    expected_ids = set(manifest["stress16_cell_ids"]) | set(manifest["remaining56_cell_ids"])
    if len(cells) != 72 or {row["cell_id"] for row in cells} != expected_ids:
        raise SystemExit("Core72 table/manifest mismatch")
    selectors = set(manifest["selectors"])
    random_sources = sorted(source for source in selectors if source.startswith(RANDOM_PREFIX))
    if NATIVE not in selectors or TRANSPORT not in selectors or LEGACY not in selectors or len(random_sources) != 8:
        raise SystemExit("Selector identity mismatch")

    rows: list[dict[str, Any]] = []
    root = Path(args.results_root)
    for cell in cells:
        objects = [load_json(path) for path in sorted((root / cell["cell_id"]).rglob("bridge_v4.json"))]
        if len(objects) != 45 or any(obj.get("source_lock_sha256") != lock_sha for obj in objects):
            raise SystemExit(f"Invalid E2A result set: {cell['cell_id']}")
        full = [obj for obj in objects if obj.get("condition") == "full_quantized" and obj.get("k") == 0]
        protected = {
            (str(obj.get("ranking_source")), int(obj["k"])): obj
            for obj in objects if obj.get("condition") == "protected" and obj.get("k") in K_VALUES
        }
        expected = {(source, k) for source in selectors for k in K_VALUES}
        if len(full) != 1 or set(protected) != expected:
            raise SystemExit(f"Invalid E2A condition set: {cell['cell_id']}")
        full = full[0]
        h_full = float(full["deltas"]["harmful_recovery"])
        full_retain = float(full["deltas"]["delta_retain_ce"])
        margin = max(0.05, 0.01 * abs(float(full["bf16_reference"]["retain"]["answer_ce"])))
        for k in K_VALUES:
            benefits = {
                source: h_full - float(protected[(source, k)]["deltas"]["harmful_recovery"])
                for source in selectors
            }
            random_mean = mean(benefits[source] for source in random_sources)
            random_retain_cost = mean(
                float(protected[(source, k)]["deltas"]["delta_retain_ce"]) - full_retain
                for source in random_sources
            )
            native = protected[(NATIVE, k)]
            native_cost = float(native["deltas"]["delta_retain_ce"]) - full_retain
            row = {
                "cell_id": cell["cell_id"],
                "checkpoint": f"{cell['model_tag']}::{cell['algo']}::{cell['seed']}",
                "model_tag": cell["model_tag"], "dataset": cell["dataset"],
                "algo": cell["algo"], "seed": str(cell["seed"]),
                "quantizer": cell["quantizer"], "k": k,
                "native_D": benefits[NATIVE] - random_mean,
                "transport_D": benefits[TRANSPORT] - random_mean,
                "legacy_D": benefits[LEGACY] - random_mean,
                "native_benefit": benefits[NATIVE],
                "random_mean_benefit": random_mean,
                "native_retain_cost": native_cost,
                "random_mean_retain_cost": random_retain_cost,
                "native_minus_random_retain_cost": native_cost - random_retain_cost,
                "retain_margin": margin, "safety_slack": margin - native_cost,
                "protected_parameter_fraction": float(native["compression"]["protected_parameter_fraction"]),
            }
            numeric = [value for key, value in row.items() if key not in {"cell_id", "checkpoint", "model_tag", "dataset", "algo", "seed", "quantizer"}]
            if any(not math.isfinite(float(value)) for value in numeric):
                raise SystemExit(f"Non-finite E2A row: {cell['cell_id']} k={k}")
            rows.append(row)

    by_k = {k: [row for row in rows if row["k"] == k] for k in K_VALUES}
    dose_reports: dict[str, Any] = {}
    joint_p: dict[int, float] = {}
    robustness: dict[int, bool] = {}
    for k_index, k in enumerate(K_VALUES):
        subset = by_k[k]
        d_values, d_checkpoint = aggregate(subset, "native_D")
        s_values, s_checkpoint = aggregate(subset, "safety_slack")
        d_summary = summarize(d_values, 100 + k_index * 20)
        s_summary = summarize(s_values, 101 + k_index * 20)
        dataset_d, dataset_s = {}, {}
        for dataset_index, dataset in enumerate(sorted({row["dataset"] for row in subset})):
            d, _ = aggregate([row for row in subset if row["dataset"] == dataset], "native_D")
            s, _ = aggregate([row for row in subset if row["dataset"] == dataset], "safety_slack")
            dataset_d[dataset] = summarize(d, 102 + k_index * 20 + dataset_index * 2)
            dataset_s[dataset] = summarize(s, 103 + k_index * 20 + dataset_index * 2)
        leave_model = {}
        for model in sorted({row["model_tag"] for row in subset}):
            values, _ = aggregate([row for row in subset if row["model_tag"] != model], "native_D")
            leave_model[model] = mean(values)
        leave_quantizer = {}
        for quantizer in sorted({row["quantizer"] for row in subset}):
            values, _ = aggregate([row for row in subset if row["quantizer"] != quantizer], "native_D")
            leave_quantizer[quantizer] = mean(values)
        components = {
            "benefit": d_summary["p_one_sided_centered_studentized_bootstrap"],
            "safety_overall": s_summary["p_one_sided_centered_studentized_bootstrap"],
            **{
                f"safety_{dataset}": value["p_one_sided_centered_studentized_bootstrap"]
                for dataset, value in dataset_s.items()
            },
        }
        joint_p[k] = max(float(value) for value in components.values())
        robustness[k] = (
            d_summary["mean"] > 0 and d_summary["median"] > 0
            and all(value["mean"] > 0 for value in dataset_d.values())
            and all(value >= 0 for value in leave_model.values())
        )
        dose_reports[str(k)] = {
            "native_D": d_summary, "safety_slack": s_summary,
            "dataset_native_D": dataset_d, "dataset_safety_slack": dataset_s,
            "leave_one_model_tag_out_native_D_mean": leave_model,
            "leave_one_quantizer_out_native_D_mean": leave_quantizer,
            "joint_test_component_p_values": components,
            "joint_intersection_union_p_value": joint_p[k],
            "checkpoint_values": {"native_D": d_checkpoint, "safety_slack": s_checkpoint},
            "groups": {
                "dataset": group_report(subset, "dataset"), "model_tag": group_report(subset, "model_tag"),
                "algorithm": group_report(subset, "algo"), "quantizer": group_report(subset, "quantizer"),
            },
        }

    primary_p = {k: joint_p[k] for k in PRIMARY_K}
    decisions = holm(primary_p)
    adjusted = holm_adjusted(primary_p)
    safe_effective = {
        k: bool(decisions[k] and robustness[k]
                and dose_reports[str(k)]["safety_slack"]["mean"] >= 0
                and all(value["mean"] >= 0 for value in dose_reports[str(k)]["dataset_safety_slack"].values()))
        for k in PRIMARY_K
    }
    for k in PRIMARY_K:
        dose_reports[str(k)]["holm_adjusted_joint_p"] = adjusted[k]
        dose_reports[str(k)]["holm_reject_joint_null"] = decisions[k]
        dose_reports[str(k)]["robustness_gate"] = robustness[k]
        dose_reports[str(k)]["safe_effective_dose"] = safe_effective[k]

    report = {
        "schema_version": "org_h200_e2a_analysis_v1",
        "status": "PASS" if any(safe_effective.values()) else "NEGATIVE_OR_HETEROGENEOUS",
        "post_e1_adaptive": True, "original_e1_gate_unchanged": True,
        "original_e2_authorized": False, "e2a_safe_effective_low_dose_found": any(safe_effective.values()),
        "safe_effective_primary_doses": [k for k in PRIMARY_K if safe_effective[k]],
        "source_lock_sha256": lock_sha, "integrity_validation_sha256": sha256_file(Path(args.integrity_validation)),
        "contract": "org_h200_execution_v1/ORG_H200_E2A_ADAPTIVE_TRADEOFF_SCIENTIFIC_CONTRACT_V1.md",
        "contract_sha256": sha256_file(Path("org_h200_execution_v1/ORG_H200_E2A_ADAPTIVE_TRADEOFF_SCIENTIFIC_CONTRACT_V1.md")),
        "cell_count": 72, "checkpoint_count": len({row["checkpoint"] for row in rows}),
        "row_count": len(rows), "k_values": list(K_VALUES), "primary_k_values": list(PRIMARY_K),
        "familywise_alpha": ALPHA, "dose_reports": dose_reports,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_json.with_suffix(output_json.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output_json)
    output_tsv = Path(args.output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_tsv = output_tsv.with_suffix(output_tsv.suffix + ".tmp")
    with tmp_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["cell_id"], row["k"])))
    tmp_tsv.replace(output_tsv)
    print(f"ORG_H200_E2A_ANALYSIS={report['status']}")
    print(f"SAFE_EFFECTIVE_LOW_DOSE={report['e2a_safe_effective_low_dose_found']}")
    print(f"SAFE_EFFECTIVE_DOSES={report['safe_effective_primary_doses']}")
    print("CELLS=72")
    print("ROWS=288")
    print(f"OUTPUT_JSON={output_json}")
    print(f"OUTPUT_TSV={output_tsv}")


if __name__ == "__main__":
    main()

