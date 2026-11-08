#!/usr/bin/env python3
"""Destructive ICML-style audit for Gate 2 v4 whole-model bridge outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import CORE_SETTINGS, ALGOS, QUANTIZERS, SEEDS, load_json  # noqa: E402
from validate_bridge_v4 import validate_obj as validate_bridge_obj  # noqa: E402

HARD_LOG_PATTERNS = re.compile(
    r"Traceback|CUDA out|OUT_OF_MEMORY|ModuleNotFound|No such file|source lock mismatch|Script SHA mismatch",
    re.IGNORECASE,
)
BRIDGE_ROOT_FRAGMENT = "gate2_v4_bridge_standard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge_root", default="results/gate2_v4_bridge_standard")
    parser.add_argument("--logs_root", default="logs/gate2_v4_bridge")
    parser.add_argument("--out_prefix", default=None)
    parser.add_argument("--expected_source_lock_sha256", default=None)
    parser.add_argument("--official_metrics_tsv", default=None)
    return parser.parse_args()


def bridge_files(root: Path) -> Iterable[Path]:
    yield from root.glob("*/*/*/bridge_v4.json")


def core_cell_ids() -> List[str]:
    ids: List[str] = []
    for setting in CORE_SETTINGS.values():
        model_tag = setting["model_tag"]
        for algo in ALGOS:
            for seed in SEEDS:
                for quantizer in QUANTIZERS:
                    ids.append(f"{model_tag}_{algo}_seed{seed}/{quantizer}")
    return ids


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def get_metric(obj: Mapping[str, Any], dotted: str, default: Any = "") -> Any:
    current: Any = obj
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def condition_key(obj: Mapping[str, Any]) -> str:
    condition = str(obj.get("condition", ""))
    if condition == "protected":
        return f"protected:{obj.get('ranking_source')}:{obj.get('k')}"
    return condition


def load_bridge_rows(root: Path, expected_lock: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Mapping[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    objects: List[Mapping[str, Any]] = []
    for path in sorted(bridge_files(root)):
        obj = load_json(path)
        errors = validate_bridge_obj(obj, path=path, expected_source_lock_sha256=expected_lock)
        status = "PASS" if not errors else "FAIL"
        row = {
            "status": status,
            "cell_id": obj.get("cell_id", ""),
            "model_tag": obj.get("model_tag", ""),
            "dataset": obj.get("dataset", ""),
            "algo": obj.get("algo", ""),
            "seed": obj.get("seed", ""),
            "quantizer": obj.get("quantizer", ""),
            "condition": obj.get("condition", ""),
            "ranking_source": obj.get("ranking_source", ""),
            "k": obj.get("k", ""),
            "source_lock_sha256": obj.get("source_lock_sha256", ""),
            "protected_layer_count": obj.get("protected_layer_count", ""),
            "protected_parameter_fraction": get_metric(obj, "compression.protected_parameter_fraction"),
            "effective_average_bits": get_metric(obj, "compression.effective_average_bits"),
            "delta_forget_ce": get_metric(obj, "deltas.delta_forget_ce"),
            "harmful_recovery": get_metric(obj, "deltas.harmful_recovery"),
            "over_forgetting": get_metric(obj, "deltas.over_forgetting"),
            "delta_truth_ratio_log": get_metric(obj, "deltas.delta_truth_ratio_log"),
            "delta_retain_ce": get_metric(obj, "deltas.delta_retain_ce"),
            "issues": ";".join(errors),
            "path": str(path),
        }
        rows.append(row)
        if status == "PASS":
            objects.append(obj)
    return rows, objects


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def condition_rows(objects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for obj in objects:
        grouped[condition_key(obj)].append(obj)
    rows: List[Dict[str, Any]] = []
    for key, objs in sorted(grouped.items()):
        harms = [float(get_metric(o, "deltas.harmful_recovery", 0.0)) for o in objs if finite(get_metric(o, "deltas.harmful_recovery"))]
        retain = [float(get_metric(o, "deltas.delta_retain_ce", 0.0)) for o in objs if finite(get_metric(o, "deltas.delta_retain_ce"))]
        rows.append(
            {
                "condition_key": key,
                "n": len(objs),
                "mean_harmful_recovery": statistics.mean(harms) if harms else "",
                "median_harmful_recovery": statistics.median(harms) if harms else "",
                "mean_delta_retain_ce": statistics.mean(retain) if retain else "",
            }
        )
    return rows


def protected_layer_rows(objects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in objects:
        if obj.get("condition") != "protected":
            continue
        evidence = obj.get("ranking_evidence", {})
        selected_rows = evidence.get("selected_rows", []) if isinstance(evidence, Mapping) else []
        selected_by_layer = {
            str(row.get("layer_name", "")): row
            for row in selected_rows
            if isinstance(row, Mapping)
        }
        for idx, layer in enumerate(obj.get("protected_layer_names", []), start=1):
            selected = selected_by_layer.get(str(layer), {})
            rows.append(
                {
                    "cell_id": obj.get("cell_id", ""),
                    "model_tag": obj.get("model_tag", ""),
                    "dataset": obj.get("dataset", ""),
                    "algo": obj.get("algo", ""),
                    "seed": obj.get("seed", ""),
                    "quantizer": obj.get("quantizer", ""),
                    "ranking_source": obj.get("ranking_source", ""),
                    "k": obj.get("k", ""),
                    "protected_rank": idx,
                    "layer_name": layer,
                    "ranking_score": selected.get("score", ""),
                    "rankings_sha256": evidence.get("rankings_sha256", "") if isinstance(evidence, Mapping) else "",
                    "evidence_path": selected.get("evidence_path", "") if isinstance(selected, Mapping) else "",
                    "evidence_sha256": selected.get("evidence_sha256", "") if isinstance(selected, Mapping) else "",
                }
            )
    return rows


def group_summary_rows(objects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, str, str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for obj in objects:
        key = (
            str(obj.get("dataset", "")),
            str(obj.get("model_tag", "")),
            str(obj.get("algo", "")),
            str(obj.get("quantizer", "")),
            str(obj.get("condition", "")),
            str(obj.get("ranking_source", "")),
        )
        grouped[key].append(obj)
    rows: List[Dict[str, Any]] = []
    for key, objs in sorted(grouped.items()):
        harms = [float(get_metric(o, "deltas.harmful_recovery", 0.0)) for o in objs if finite(get_metric(o, "deltas.harmful_recovery"))]
        over = [float(get_metric(o, "deltas.over_forgetting", 0.0)) for o in objs if finite(get_metric(o, "deltas.over_forgetting"))]
        rows.append(
            {
                "dataset": key[0],
                "model_tag": key[1],
                "algo": key[2],
                "quantizer": key[3],
                "condition": key[4],
                "ranking_source": key[5],
                "n": len(objs),
                "mean_harmful_recovery": statistics.mean(harms) if harms else "",
                "median_harmful_recovery": statistics.median(harms) if harms else "",
                "mean_over_forgetting": statistics.mean(over) if over else "",
            }
        )
    return rows


def scan_hard_logs(logs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not logs_root.exists():
        return rows
    for path in sorted(logs_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".log", ".out", ".err", ".txt"}:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    if HARD_LOG_PATTERNS.search(line):
                        rows.append({"path": str(path), "lineno": lineno, "line": line.strip()[:500]})
        except OSError as exc:
            rows.append({"path": str(path), "lineno": 0, "line": f"LOG_READ_ERROR:{exc}"})
    return rows


def mitigation_summary(objects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    full_by_cell = {
        str(o.get("cell_id")): float(get_metric(o, "deltas.harmful_recovery", 0.0))
        for o in objects
        if o.get("condition") == "full_quantized" and finite(get_metric(o, "deltas.harmful_recovery"))
    }
    reductions: MutableMapping[Tuple[str, int], List[float]] = defaultdict(list)
    random_reductions: MutableMapping[int, List[float]] = defaultdict(list)
    for obj in objects:
        if obj.get("condition") != "protected":
            continue
        cid = str(obj.get("cell_id"))
        if cid not in full_by_cell or not finite(get_metric(obj, "deltas.harmful_recovery")):
            continue
        k = int(obj.get("k", 0))
        source = str(obj.get("ranking_source", ""))
        reduction = full_by_cell[cid] - float(get_metric(obj, "deltas.harmful_recovery", 0.0))
        reductions[(source, k)].append(reduction)
        if source == "random":
            random_reductions[k].append(reduction)
    best_rows: List[Dict[str, Any]] = []
    directional_pass = False
    for (source, k), vals in sorted(reductions.items()):
        base = random_reductions.get(k, [])
        mean_val = statistics.mean(vals) if vals else None
        mean_random = statistics.mean(base) if base else None
        beats_random = mean_random is not None and mean_val is not None and mean_val > mean_random
        directional_pass = directional_pass or (source != "random" and beats_random)
        best_rows.append(
            {
                "ranking_source": source,
                "k": k,
                "n": len(vals),
                "mean_harmful_recovery_reduction": mean_val if mean_val is not None else "",
                "random_mean_same_k": mean_random if mean_random is not None else "",
                "beats_random_same_k": beats_random,
            }
        )
    return {"directional_pass": directional_pass, "rows": best_rows}


def claim_gate_rows(
    *,
    cells: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    hard_logs: Sequence[Mapping[str, Any]],
    official_metrics_tsv: Optional[str],
) -> List[Dict[str, Any]]:
    core = set(core_cell_ids())
    clean = [row for row in cells if row.get("status") == "PASS"]
    full_cells = {str(o.get("cell_id")) for o in objects if o.get("condition") == "full_quantized"}
    bf16_cells = {str(o.get("cell_id")) for o in objects if o.get("condition") == "bf16"}
    protected_cells = {str(o.get("cell_id")) for o in objects if o.get("condition") == "protected"}
    nf4_protected = {str(o.get("cell_id")) for o in objects if o.get("condition") == "protected" and o.get("quantizer") == "nf4"}
    mit = mitigation_summary(objects)
    rows = [
        {
            "claim_gate": "bridge_baseline",
            "status": "PASS" if core.issubset(full_cells) and not hard_logs else "BLOCKED",
            "evidence": f"full_quantized_cells={len(full_cells)}/144; hard_log_hits={len(hard_logs)}",
            "blocker": "" if core.issubset(full_cells) and not hard_logs else "missing_full_quantized_or_hard_logs",
        },
        {
            "claim_gate": "explicit_bf16_condition",
            "status": "PASS" if core.issubset(bf16_cells) else "OPTIONAL_INCOMPLETE",
            "evidence": f"bf16_condition_cells={len(bf16_cells)}/144; every non-bf16 result also embeds bf16_reference",
            "blocker": "" if core.issubset(bf16_cells) else "bf16_condition_not_run_for_all_cells",
        },
        {
            "claim_gate": "protected_layer_bridge",
            "status": "PASS" if protected_cells else "BLOCKED",
            "evidence": f"protected_cells={len(protected_cells)}; nf4_protected_cells={len(nf4_protected)}",
            "blocker": "" if protected_cells else "no_protected_conditions",
        },
        {
            "claim_gate": "mitigation_above_random",
            "status": "PASS" if mit["directional_pass"] else "BLOCKED",
            "evidence": "metric-guided protected harmful-recovery reduction beats random at matched k" if mit["directional_pass"] else "no matched-k metric-guided advantage over random yet",
            "blocker": "" if mit["directional_pass"] else "mitigation_not_demonstrated",
        },
        {
            "claim_gate": "official_metrics",
            "status": "PROVIDED" if official_metrics_tsv else "BLOCKED",
            "evidence": official_metrics_tsv or "official TOFU/MUSE metric table not provided",
            "blocker": "" if official_metrics_tsv else "official_metrics_missing",
        },
        {
            "claim_gate": "paper_ready",
            "status": "PASS" if core.issubset(full_cells) and protected_cells and mit["directional_pass"] and official_metrics_tsv and not hard_logs else "BLOCKED",
            "evidence": f"clean_bridge_files={len(clean)}; full_cells={len(full_cells)}; protected_cells={len(protected_cells)}; hard_logs={len(hard_logs)}",
            "blocker": "requires baseline bridge, mitigation above random, official metrics, and clean logs",
        },
    ]
    return rows


def write_report(
    path: Path,
    *,
    cells: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    hard_logs: Sequence[Mapping[str, Any]],
) -> None:
    status_counts = Counter(row["status"] for row in cells)
    condition_counts = Counter(row["condition_key"] for row in conditions)
    lines = [
        "# Gate 2 v4 Whole-Model Bridge ICML Audit",
        "",
        "## Verdict",
        "",
    ]
    if status_counts.get("FAIL", 0) or hard_logs:
        lines.append("The bridge package is not clean yet. Failed files or hard-log hits block paper claims.")
    elif condition_counts.get("full_quantized", 0) >= 144:
        lines.append("The bridge baseline is structurally usable. Mitigation and official metric claims still depend on their own gates.")
    else:
        lines.append("The bridge audit is runnable, but full bridge coverage is incomplete.")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Bridge files: {len(cells)}",
            f"- Status counts: {dict(status_counts)}",
            f"- Condition counts: {dict(condition_counts)}",
            f"- Hard log hits: {len(hard_logs)}",
            "",
            "## Claim Gates",
            "",
        ]
    )
    for row in claims:
        lines.append(f"- {row['claim_gate']}: {row['status']} ({row['evidence']})")
    lines.extend(
        [
            "",
            "## Paper Boundary",
            "",
            "This audit can support a whole-model bridge claim only after full-quantized coverage is clean and metric-guided protection beats deterministic random controls at comparable protected-parameter fraction. Official TOFU/MUSE benchmark claims remain separate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bridge_root = Path(args.bridge_root)
    if BRIDGE_ROOT_FRAGMENT not in str(bridge_root):
        raise SystemExit(f"bridge_root must contain {BRIDGE_ROOT_FRAGMENT}: {bridge_root}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or f"bridge_v4_icml_audit_{ts}"

    cells, objects = load_bridge_rows(bridge_root, args.expected_source_lock_sha256)
    conditions = condition_rows(objects)
    protected_layers = protected_layer_rows(objects)
    group_summary = group_summary_rows(objects)
    hard_logs = scan_hard_logs(Path(args.logs_root))
    claims = claim_gate_rows(
        cells=cells,
        objects=objects,
        hard_logs=hard_logs,
        official_metrics_tsv=args.official_metrics_tsv,
    )
    mitigation = mitigation_summary(objects)["rows"]

    outputs = {
        "cells": Path(f"{prefix}_bridge_v4_cells.tsv"),
        "conditions": Path(f"{prefix}_bridge_v4_conditions.tsv"),
        "protected_layers": Path(f"{prefix}_bridge_v4_protected_layers.tsv"),
        "group_summary": Path(f"{prefix}_bridge_v4_group_summary.tsv"),
        "claim_gate": Path(f"{prefix}_BRIDGE_V4_CLAIM_GATE_MATRIX.tsv"),
        "hard_logs": Path(f"{prefix}_bridge_v4_hard_log_audit.tsv"),
        "mitigation": Path(f"{prefix}_bridge_v4_mitigation_vs_random.tsv"),
        "report": Path(f"{prefix}_BRIDGE_V4_ICML_AUDIT.md"),
    }
    write_tsv(outputs["cells"], cells)
    write_tsv(outputs["conditions"], conditions)
    write_tsv(outputs["protected_layers"], protected_layers, fieldnames=["cell_id", "model_tag", "dataset", "algo", "seed", "quantizer", "ranking_source", "k", "protected_rank", "layer_name", "ranking_score", "rankings_sha256", "evidence_path", "evidence_sha256"])
    write_tsv(outputs["group_summary"], group_summary)
    write_tsv(outputs["claim_gate"], claims)
    write_tsv(outputs["hard_logs"], hard_logs, fieldnames=["path", "lineno", "line"])
    write_tsv(outputs["mitigation"], mitigation, fieldnames=["ranking_source", "k", "n", "mean_harmful_recovery_reduction", "random_mean_same_k", "beats_random_same_k"])
    write_report(outputs["report"], cells=cells, conditions=conditions, claims=claims, hard_logs=hard_logs)
    for path in outputs.values():
        print(f"WROTE {path}")
    if any(row.get("status") == "FAIL" for row in cells) or hard_logs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
