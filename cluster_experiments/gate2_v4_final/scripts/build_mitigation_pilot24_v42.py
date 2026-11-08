#!/usr/bin/env python3
"""Preregister a diverse high-risk plus balanced Mitigation Pilot24."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ALGOS = ("GA", "GDR", "NPO", "KLR")
SEEDS = ("42", "2024", "5000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric_rankings", required=True)
    parser.add_argument("--bridge_root", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bridge_cell_id(obj: Mapping[str, Any]) -> str:
    return str(obj.get("cell_id") or f"{obj['model_tag']}_{obj['algo']}_seed{obj['seed']}/{obj['quantizer']}")


def parse_cell(obj: Mapping[str, Any]) -> dict[str, str]:
    return {
        "cell_id": bridge_cell_id(obj),
        "model_tag": str(obj["model_tag"]),
        "dataset": str(obj["dataset"]),
        "algo": str(obj["algo"]),
        "seed": str(obj["seed"]),
        "quantizer": str(obj["quantizer"]),
    }


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def balanced_assignment(
    strata: Sequence[tuple[str, str]],
    by_stratum_combo: Mapping[tuple[str, str], Mapping[tuple[str, str], Mapping[str, Any]]],
    high_cells: set[str],
) -> list[Mapping[str, Any]]:
    combos = [(algo, seed) for algo in ALGOS for seed in SEEDS]
    if len(strata) != len(combos):
        raise ValueError("Pilot24 balanced assignment expects 12 model/quantizer strata")
    dp: dict[int, tuple[float, list[Mapping[str, Any]]]] = {0: (0.0, [])}
    for stratum in strata:
        candidates = by_stratum_combo[stratum]
        values = [float(row["harmful_recovery"]) for row in candidates.values()]
        center = statistics.median(values)
        next_dp: dict[int, tuple[float, list[Mapping[str, Any]]]] = {}
        for mask, (cost, selected) in dp.items():
            for combo_index, combo in enumerate(combos):
                if mask & (1 << combo_index) or combo not in candidates:
                    continue
                row = candidates[combo]
                if str(row["cell_id"]) in high_cells:
                    continue
                new_mask = mask | (1 << combo_index)
                new_cost = cost + abs(float(row["harmful_recovery"]) - center)
                current = next_dp.get(new_mask)
                if current is None or new_cost < current[0] - 1e-15:
                    next_dp[new_mask] = (new_cost, selected + [row])
        dp = next_dp
    full_mask = (1 << len(combos)) - 1
    if full_mask not in dp:
        raise ValueError("Could not construct non-overlapping balanced assignment covering every algorithm/seed pair")
    return dp[full_mask][1]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings: dict[str, set[str]] = defaultdict(set)
    ranking_layers: dict[tuple[str, str], set[str]] = defaultdict(set)
    with Path(args.metric_rankings).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cell = str(row["cell_id"])
            source = str(row["ranking_source"])
            rankings[cell].add(source)
            ranking_layers[(cell, source)].add(str(row["layer_name"]))

    bridges: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for root_name in args.bridge_root:
        for path in Path(root_name).rglob("bridge_v4.json"):
            obj = load_json(path)
            if obj.get("condition") != "full_quantized":
                continue
            cell = bridge_cell_id(obj)
            if cell in bridges:
                duplicates.append(cell)
            bridges[cell] = obj
    eligible: list[dict[str, Any]] = []
    errors: list[str] = []
    required = {"GQDP_signed_v42", "displacement_l2_v42"}
    for cell, obj in sorted(bridges.items()):
        missing = required - rankings.get(cell, set())
        if missing:
            errors.append(f"missing_rankings:{cell}:{','.join(sorted(missing))}")
            continue
        if any(len(ranking_layers[(cell, source)]) != 224 for source in required):
            errors.append(f"ranking_layer_count:{cell}")
            continue
        row = parse_cell(obj)
        row.update(
            {
                "harmful_recovery": float(obj.get("deltas", {}).get("harmful_recovery", 0.0)),
                "delta_retain_ce": float(obj.get("deltas", {}).get("delta_retain_ce", 0.0)),
            }
        )
        eligible.append(row)
    if len(bridges) != 144:
        errors.append(f"bridge_full_count:{len(bridges)}:144")
    if len(eligible) != 144:
        errors.append(f"eligible_count:{len(eligible)}:144")
    if duplicates:
        errors.append(f"duplicate_bridges:{len(duplicates)}")

    by_stratum: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_stratum_combo: dict[tuple[str, str], dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(dict)
    for row in eligible:
        stratum = (str(row["model_tag"]), str(row["quantizer"]))
        combo = (str(row["algo"]), str(row["seed"]))
        by_stratum[stratum].append(row)
        by_stratum_combo[stratum][combo] = row
    strata = sorted(by_stratum)
    if len(strata) != 12 or any(len(by_stratum[stratum]) != 12 for stratum in strata):
        errors.append("factorial_strata_not_12x12")

    selected_rows: list[dict[str, Any]] = []
    if not errors:
        high = [max(by_stratum[stratum], key=lambda row: (float(row["harmful_recovery"]), str(row["cell_id"]))) for stratum in strata]
        high_cells = {str(row["cell_id"]) for row in high}
        balanced = balanced_assignment(strata, by_stratum_combo, high_cells)
        for panel, rows in (("high_risk", high), ("balanced", balanced)):
            for row in rows:
                selected_rows.append(
                    {
                        **dict(row),
                        "panel": panel,
                        "selectors": "GQDP_signed_v42;UPQI_star_signed_v42;displacement_l2_v42;deterministic_random_v42",
                        "k_values": "4;8",
                        "upqi_star_status": "REQUIRED_FOR_SELECTED_CELL_BEFORE_MITIGATION",
                    }
                )
    if len(selected_rows) != 24 or len({row["cell_id"] for row in selected_rows}) != 24:
        errors.append(f"selection_count:{len(selected_rows)}:24")
    high_rows = [row for row in selected_rows if row["panel"] == "high_risk"]
    balanced_rows = [row for row in selected_rows if row["panel"] == "balanced"]
    coverage = {
        "models": sorted({row["model_tag"] for row in selected_rows}),
        "datasets": sorted({row["dataset"] for row in selected_rows}),
        "algorithms": sorted({row["algo"] for row in selected_rows}),
        "seeds": sorted({row["seed"] for row in selected_rows}),
        "quantizers": sorted({row["quantizer"] for row in selected_rows}),
    }
    if selected_rows and not (
        len(coverage["models"]) == 4
        and len(coverage["datasets"]) == 2
        and len(coverage["algorithms"]) == 4
        and len(coverage["seeds"]) == 3
        and len(coverage["quantizers"]) == 3
    ):
        errors.append(f"coverage_failure:{coverage}")
    decision = "PILOT24_SELECTION_INVALID_STOP_REPAIR" if errors else "PILOT24_SELECTION_READY_FOR_UPQI_STAR_SELECTED24"
    write_tsv(out_dir / "mitigation_pilot24_cells.tsv", selected_rows)
    manifest = {
        "schema_version": "gate2_mitigation_pilot24_v4.2",
        "decision": decision,
        "selection_contract": {
            "high_risk": "one maximum-harmful-recovery cell per model-by-quantizer stratum",
            "balanced": "minimum-distance-to-stratum-median assignment covering all 12 algorithm-by-seed combinations exactly once",
            "overlap": "none",
        },
        "cells": selected_rows,
        "coverage": coverage,
        "errors": errors,
        "required_matched_conditions": {
            "selectors": ["GQDP_signed_v42", "UPQI_star_signed_v42", "displacement_l2_v42", "deterministic_random_v42"],
            "k_values": [4, 8],
            "conditions": 8,
            "expected_bridge_results": 192,
        },
        "expansion_gate": "Expand to Bridge48 only if a guided selector beats matched random with acceptable retain degradation.",
    }
    with (out_dir / "mitigation_pilot24_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    markdown = [
        "# Gate 2 v4.2 Mitigation Pilot24 Preregistration",
        "",
        f"Decision: **{decision}**",
        "",
        f"- High-risk cells: {len(high_rows)}/12",
        f"- Balanced cells: {len(balanced_rows)}/12",
        f"- Coverage: {coverage}",
        f"- Errors: {len(errors)}",
        "",
        "Each cell will compare GQDP-signed, corrected UPQI-star-signed, displacement-L2, and deterministic random at matched k=4 and k=8.",
        "The 192 protected conditions are a gated mitigation experiment, not part of the diagnostic Full144 bridge.",
    ]
    (out_dir / "MITIGATION_PILOT24_PREREGISTRATION.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    print(f"MITIGATION_PILOT24_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
