#!/usr/bin/env python3
"""Certify one hash-bound GT/bridge cross-hardware reproducibility exception.

This is a torch-free, read-only evidence reconciler. It never edits GT or bridge
JSON. A reconciled join is emitted only when the original join has exactly one
predeclared failure and an independently reproduced GT matches the bridge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TARGET_CELL = "llama3_8b_adamw_KLR_seed42/gptq_int4"
EXPECTED_ISSUES = {
    "bf16_answer_ce_mismatch",
    "bf16_answer_logprob_mismatch",
    "bf16_truth_ratio_log_mismatch",
}
METRICS = ("answer_ce", "answer_logprob", "truth_ratio_log")
LAYER_METRICS = (
    "delta_forget_ce",
    "delta_answer_logprob",
    "delta_truth_ratio_log",
    "KL_forget_raw",
    "KL_forget_clamped",
    "epsilon_l2",
    "weight_diff_max",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_join", required=True)
    parser.add_argument("--old_gt", required=True)
    parser.add_argument("--new_gt", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--new_source_lock", required=True)
    parser.add_argument("--repro_accounting", required=True)
    parser.add_argument("--repro_job", required=True)
    parser.add_argument("--expected_old_gt_sha256", required=True)
    parser.add_argument("--expected_new_gt_sha256", required=True)
    parser.add_argument("--expected_bridge_sha256", required=True)
    parser.add_argument("--expected_old_v4_lock_sha256", required=True)
    parser.add_argument("--expected_bridge_lock_sha256", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reconciled_join", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON is not an object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def embedded_hash(obj: Mapping[str, Any], field: str) -> str:
    stripped = dict(obj)
    stripped.pop(field, None)
    return hashlib.sha256(canonical_json(stripped).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def nested(obj: Mapping[str, Any], dotted: str) -> Any:
    value: Any = obj
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def sample_hashes(rows: Any) -> list[str]:
    if not isinstance(rows, list):
        return []
    return [str(row.get("sample_hash", "")) for row in rows if isinstance(row, Mapping)]


def rank(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for offset in range(index, end):
            result[ordered[offset][0]] = average
        index = end
    return result


def pearson(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None
    x = [left[key] for key in keys]
    y = [right[key] for key in keys]
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))
    return numerator / denominator if denominator else None


def read_psv(path: Path) -> list[list[str]]:
    return [line.rstrip("\n").split("|") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "old_gt": Path(args.old_gt),
        "new_gt": Path(args.new_gt),
        "bridge": Path(args.bridge),
        "new_source_lock": Path(args.new_source_lock),
    }
    errors: list[str] = []
    expected_hashes = {
        "old_gt": args.expected_old_gt_sha256,
        "new_gt": args.expected_new_gt_sha256,
        "bridge": args.expected_bridge_sha256,
    }
    actual_hashes: dict[str, str] = {}
    for label, expected in expected_hashes.items():
        path = paths[label]
        if not path.is_file():
            errors.append(f"missing_file:{label}:{path}")
            continue
        actual_hashes[label] = sha256_file(path)
        if actual_hashes[label] != expected:
            errors.append(f"hash_mismatch:{label}:{actual_hashes[label]}:{expected}")

    objects = {label: load_json(paths[label]) for label in ("old_gt", "new_gt", "bridge") if paths[label].is_file()}
    old = objects.get("old_gt", {})
    new = objects.get("new_gt", {})
    bridge = objects.get("bridge", {})
    expected_identity = {
        "model_tag": "llama3_8b_adamw",
        "dataset": "tofu",
        "algo": "KLR",
        "seed": "42",
        "quantizer": "gptq_int4",
    }
    for label, obj in objects.items():
        for field, expected in expected_identity.items():
            if str(obj.get(field, "")) != expected:
                errors.append(f"identity_mismatch:{label}:{field}:{obj.get(field)!r}:{expected!r}")
    if bridge.get("cell_id") != TARGET_CELL or bridge.get("condition") != "full_quantized":
        errors.append("bridge_target_cell_or_condition_mismatch")
    for label, obj in (("old_gt", old), ("new_gt", new)):
        if obj.get("schema_version") != "gate2_gt_v4.0" or obj.get("stage") != "gt_v4":
            errors.append(f"gt_schema_or_stage_bad:{label}")
        if int(obj.get("n_layers", -1)) != 224 or int(obj.get("n_layers_completed", -1)) != 224:
            errors.append(f"gt_layer_count_bad:{label}")
        if len(obj.get("layers", {})) != 224 or len(obj.get("target_layer_names", [])) != 224:
            errors.append(f"gt_layer_inventory_bad:{label}")

    old_lock = nested(old, "provenance.source_lock_sha256")
    new_lock = nested(new, "provenance.source_lock_sha256")
    bridge_lock = bridge.get("source_lock_sha256")
    if old_lock != args.expected_old_v4_lock_sha256:
        errors.append(f"old_v4_lock_mismatch:{old_lock}")
    if bridge_lock != args.expected_bridge_lock_sha256:
        errors.append(f"bridge_lock_mismatch:{bridge_lock}")
    lock_obj = load_json(paths["new_source_lock"]) if paths["new_source_lock"].is_file() else {}
    lock_embedded = str(lock_obj.get("source_lock_sha256", ""))
    if not lock_embedded or embedded_hash(lock_obj, "source_lock_sha256") != lock_embedded:
        errors.append("new_source_lock_internal_hash_bad")
    if new_lock != lock_embedded:
        errors.append(f"new_gt_source_lock_mismatch:{new_lock}:{lock_embedded}")

    for descriptor in ("sample_manifest", "calibration_manifest"):
        old_sha = nested(old, f"{descriptor}.manifest_sha256")
        new_sha = nested(new, f"{descriptor}.manifest_sha256")
        bridge_name = "forget_manifest" if descriptor == "sample_manifest" else "calibration_manifest"
        bridge_sha = nested(bridge, f"{bridge_name}.sha256")
        if not old_sha or old_sha != new_sha or old_sha != bridge_sha:
            errors.append(f"manifest_mismatch:{descriptor}:{old_sha}:{new_sha}:{bridge_sha}")

    old_samples = sample_hashes(nested(old, "bf16_baseline.per_sample"))
    new_samples = sample_hashes(nested(new, "bf16_baseline.per_sample"))
    bridge_samples = sample_hashes(nested(bridge, "bf16_reference.forget.per_sample"))
    if len(old_samples) != 128 or old_samples != new_samples or old_samples != bridge_samples:
        errors.append(f"sample_identity_mismatch:{len(old_samples)}:{len(new_samples)}:{len(bridge_samples)}")
    if len(set(old_samples)) != 128:
        errors.append("sample_identity_not_unique")

    aggregate_rows: list[dict[str, Any]] = []
    old_differs = False
    for key in METRICS:
        old_value = nested(old, f"bf16_baseline.{key}")
        new_value = nested(new, f"bf16_baseline.{key}")
        bridge_value = nested(bridge, f"bf16_reference.forget.{key}")
        if not all(finite(value) for value in (old_value, new_value, bridge_value)):
            errors.append(f"aggregate_nonfinite_or_missing:{key}")
            continue
        old_delta = abs(float(old_value) - float(bridge_value))
        new_delta = abs(float(new_value) - float(bridge_value))
        old_differs = old_differs or old_delta > 1e-5
        if new_delta > args.tolerance:
            errors.append(f"new_h200_bridge_drift:{key}:{new_delta}")
        aggregate_rows.append({
            "metric": key,
            "nano5_old": old_value,
            "h200_reproduction": new_value,
            "bridge": bridge_value,
            "nano5_bridge_abs_delta": old_delta,
            "h200_bridge_abs_delta": new_delta,
        })
    if not old_differs:
        errors.append("predeclared_old_cross_hardware_difference_not_reproduced")

    original_join = Path(args.original_join)
    with original_join.open(encoding="utf-8-sig", newline="") as handle:
        join_rows = list(csv.DictReader(handle, delimiter="\t"))
    failed = [row for row in join_rows if row.get("status") != "PASS"]
    if len(join_rows) != 120 or len(failed) != 1:
        errors.append(f"original_join_failure_inventory:{len(join_rows)}:{len(failed)}")
    elif failed[0].get("cell_id") != TARGET_CELL or set(filter(None, failed[0].get("issues", "").split(";"))) != EXPECTED_ISSUES:
        errors.append(f"unexpected_original_join_failure:{failed[0]}")

    accounting = read_psv(Path(args.repro_accounting))
    matching = [row for row in accounting if row and row[0].split(".")[0] == str(args.repro_job)]
    if not any("COMPLETED" in row and "0:0" in row for row in matching):
        errors.append(f"reproduction_job_not_completed_0_0:{args.repro_job}")

    old_layers = old.get("layers", {}) if isinstance(old.get("layers"), Mapping) else {}
    new_layers = new.get("layers", {}) if isinstance(new.get("layers"), Mapping) else {}
    layer_names = list(new.get("target_layer_names", [])) if isinstance(new.get("target_layer_names"), list) else []
    if layer_names != list(old.get("target_layer_names", [])):
        errors.append("target_layer_order_mismatch")
    layer_rows: list[dict[str, Any]] = []
    primary_old: dict[str, float] = {}
    primary_new: dict[str, float] = {}
    for name in layer_names:
        old_row = old_layers.get(name, {}) if isinstance(old_layers.get(name), Mapping) else {}
        new_row = new_layers.get(name, {}) if isinstance(new_layers.get(name), Mapping) else {}
        row: dict[str, Any] = {"layer_name": name}
        for key in LAYER_METRICS:
            left, right = old_row.get(key), new_row.get(key)
            if not finite(left) or not finite(right):
                errors.append(f"layer_metric_bad:{name}:{key}")
                continue
            row[f"nano5_{key}"] = left
            row[f"h200_{key}"] = right
            row[f"abs_delta_{key}"] = abs(float(left) - float(right))
        if finite(old_row.get("delta_forget_ce")) and finite(new_row.get("delta_forget_ce")):
            primary_old[name] = abs(float(old_row["delta_forget_ce"]))
            primary_new[name] = abs(float(new_row["delta_forget_ce"]))
        layer_rows.append(row)
    if len(layer_rows) != 224:
        errors.append(f"layer_sensitivity_count:{len(layer_rows)}/224")

    topk: dict[str, Any] = {}
    for k in (4, 8, 16, 32):
        old_top = {name for name, _ in sorted(primary_old.items(), key=lambda item: (-item[1], item[0]))[:k]}
        new_top = {name for name, _ in sorted(primary_new.items(), key=lambda item: (-item[1], item[0]))[:k]}
        topk[str(k)] = {"overlap": len(old_top & new_top), "fraction": len(old_top & new_top) / k}
    rank_correlation = pearson(rank(primary_old), rank(primary_new))

    status = "PASS" if not errors else "FAIL"
    ledger = {
        "schema_version": "gate2_full144_cross_hardware_gt_exception_v1",
        "status": status,
        "target_cell": TARGET_CELL,
        "reproduction_job": str(args.repro_job),
        "tolerance": args.tolerance,
        "artifact_paths": {key: str(value) for key, value in paths.items()},
        "artifact_sha256": actual_hashes,
        "source_locks": {"nano5_old_v4": old_lock, "h200_reproduction": new_lock, "bridge": bridge_lock},
        "aggregate_comparison": aggregate_rows,
        "sample_identity_count": len(old_samples),
        "layer_count": len(layer_rows),
        "absolute_delta_forget_ce_spearman": rank_correlation,
        "absolute_delta_forget_ce_topk_overlap": topk,
        "interpretation": "The preserved Nano5 GT differs in BF16 aggregation. The independently recomputed Nano4 H200 GT matches the Nano4 bridge exactly at the declared tolerance. This exception applies only to the strict GT/bridge join for the named cell; it does not erase cross-hardware layerwise sensitivity.",
        "errors": sorted(set(errors)),
    }
    ledger_path = out_dir / "cross_hardware_gt_exception_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    ledger_sha = sha256_file(ledger_path)

    aggregate_path = out_dir / "cross_hardware_bf16_comparison.tsv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]) if aggregate_rows else ["metric"], delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    layer_path = out_dir / "cross_hardware_gt_layer_sensitivity.tsv"
    fields = list(layer_rows[0]) if layer_rows else ["layer_name"]
    with layer_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(layer_rows)

    reconciled_path = Path(args.reconciled_join)
    if status == "PASS":
        reconciled_rows = []
        for row in join_rows:
            updated = dict(row)
            if row.get("cell_id") == TARGET_CELL:
                updated["status"] = "PASS"
                updated["issues"] = ""
                updated["cross_hardware_exception"] = "CERTIFIED"
                updated["cross_hardware_exception_ledger_sha256"] = ledger_sha
            else:
                updated["cross_hardware_exception"] = ""
                updated["cross_hardware_exception_ledger_sha256"] = ""
            reconciled_rows.append(updated)
        fields = list(reconciled_rows[0])
        for extra in ("cross_hardware_exception", "cross_hardware_exception_ledger_sha256"):
            if extra not in fields:
                fields.append(extra)
        with reconciled_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(reconciled_rows)

    print(f"CROSS_HARDWARE_GT_EXCEPTION={status}")
    print(f"LEDGER={ledger_path}")
    print(f"LEDGER_SHA256={ledger_sha}")
    print(f"LAYER_SENSITIVITY={layer_path}")
    print(f"RECONCILED_JOIN={reconciled_path if status == 'PASS' else 'NOT_WRITTEN'}")
    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR={error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
