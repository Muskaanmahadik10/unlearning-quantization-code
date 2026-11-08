#!/usr/bin/env python3
"""Preregister a disjoint, factor-balanced Bridge48 GQDP confirmation panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


ALGOS = ("GA", "GDR", "KLR", "NPO")
SEEDS = ("42", "2024", "5000")
COMBOS = tuple((algo, seed) for algo in ALGOS for seed in SEEDS)
SELECTORS = ("GQDP_signed_v42", "deterministic_random_v42")
SELECTION_SEED = "gate2-v42-bridge48-confirmation-20260901"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric_rankings", required=True)
    parser.add_argument("--pilot_manifest", required=True)
    parser.add_argument("--bridge_root", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cell_id(obj: Mapping[str, Any]) -> str:
    return str(obj.get("cell_id") or f"{obj['model_tag']}_{obj['algo']}_seed{obj['seed']}/{obj['quantizer']}")


def stable_score(*parts: str) -> str:
    payload = "|".join((SELECTION_SEED, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_balanced(
    strata: Sequence[tuple[str, str]],
    candidates: Mapping[tuple[str, str], Mapping[tuple[str, str], Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    """Choose four cells per stratum and four uses of every algo/seed combo."""
    options: list[list[tuple[tuple[int, ...], tuple[str, ...]]]] = []
    for stratum in strata:
        available = candidates[stratum]
        choices: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        for picked in itertools.combinations(sorted(available), 4):
            indices = tuple(COMBOS.index(combo) for combo in picked)
            score = tuple(sorted(stable_score(stratum[0], stratum[1], *combo) for combo in picked))
            choices.append((indices, score))
        choices.sort(key=lambda item: item[1])
        options.append(choices)

    availability = []
    for start in range(len(strata) + 1):
        counts = [0] * len(COMBOS)
        for stratum in strata[start:]:
            for combo in candidates[stratum]:
                counts[COMBOS.index(combo)] += 1
        availability.append(tuple(counts))

    @lru_cache(maxsize=None)
    def solve(index: int, used: tuple[int, ...]) -> tuple[tuple[int, ...], ...] | None:
        if index == len(strata):
            return () if all(value == 4 for value in used) else None
        remaining_slots = len(strata) - index - 1
        for indices, _score in options[index]:
            updated = list(used)
            if any(updated[position] >= 4 for position in indices):
                continue
            for position in indices:
                updated[position] += 1
            if any(value > 4 or value + remaining_slots < 4 for value in updated):
                continue
            if any(4 - value > availability[index + 1][position] for position, value in enumerate(updated)):
                continue
            suffix = solve(index + 1, tuple(updated))
            if suffix is not None:
                return (indices,) + suffix
        return None

    assignment = solve(0, (0,) * len(COMBOS))
    if assignment is None:
        raise ValueError("No exact Bridge48 factor-balanced assignment exists")
    selected: list[Mapping[str, Any]] = []
    for stratum, indices in zip(strata, assignment):
        for position in indices:
            selected.append(candidates[stratum][COMBOS[position]])
    return selected


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    pilot = load_json(Path(args.pilot_manifest))
    pilot_cells = {str(row["cell_id"]) for row in pilot.get("cells", [])}
    if len(pilot_cells) != 24:
        errors.append(f"pilot_cell_count:{len(pilot_cells)}:24")

    ranking_layers: dict[tuple[str, str], set[str]] = {}
    with Path(args.metric_rankings).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["cell_id"]), str(row["ranking_source"]))
            ranking_layers.setdefault(key, set()).add(str(row["layer_name"]))

    bridges: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for root_name in args.bridge_root:
        for path in Path(root_name).rglob("bridge_v4.json"):
            obj = load_json(path)
            if obj.get("condition") != "full_quantized":
                continue
            key = cell_id(obj)
            if key in bridges:
                duplicates.append(key)
            bridges[key] = obj
    if len(bridges) != 144:
        errors.append(f"bridge_full_count:{len(bridges)}:144")
    if duplicates:
        errors.append(f"duplicate_bridges:{len(duplicates)}")

    by_stratum: dict[tuple[str, str], dict[tuple[str, str], Mapping[str, Any]]] = {}
    eligible_count = 0
    for key, obj in sorted(bridges.items()):
        if key in pilot_cells:
            continue
        if any(len(ranking_layers.get((key, source), set())) != 224 for source in SELECTORS):
            errors.append(f"ranking_coverage:{key}")
            continue
        row = {
            "cell_id": key,
            "model_tag": str(obj["model_tag"]),
            "dataset": str(obj["dataset"]),
            "algo": str(obj["algo"]),
            "seed": str(obj["seed"]),
            "quantizer": str(obj["quantizer"]),
            "panel": "disjoint_hash_balanced_confirmation",
            "selectors": ";".join(SELECTORS),
            "k_values": "8",
        }
        stratum = (row["model_tag"], row["quantizer"])
        combo = (row["algo"], row["seed"])
        by_stratum.setdefault(stratum, {})[combo] = row
        eligible_count += 1
    if eligible_count != 120:
        errors.append(f"eligible_nonpilot_count:{eligible_count}:120")
    strata = sorted(by_stratum)
    if len(strata) != 12 or any(len(by_stratum[stratum]) != 10 for stratum in strata):
        errors.append("remaining_strata_not_12x10")

    selected: list[Mapping[str, Any]] = []
    if not errors:
        try:
            selected = choose_balanced(strata, by_stratum)
        except ValueError as exc:
            errors.append(f"selection:{exc}")
    selected = sorted(selected, key=lambda row: (row["quantizer"] != "gptq_int4", row["cell_id"]))

    counts = {
        "models": {str(row["model_tag"]) for row in selected},
        "datasets": {str(row["dataset"]) for row in selected},
        "algorithms": {str(row["algo"]) for row in selected},
        "seeds": {str(row["seed"]) for row in selected},
        "quantizers": {str(row["quantizer"]) for row in selected},
    }
    combo_counts = {f"{algo}/seed{seed}": 0 for algo, seed in COMBOS}
    stratum_counts: dict[str, int] = {}
    for row in selected:
        combo_counts[f"{row['algo']}/seed{row['seed']}"] += 1
        label = f"{row['model_tag']}/{row['quantizer']}"
        stratum_counts[label] = stratum_counts.get(label, 0) + 1
    if len(selected) != 48 or len({str(row["cell_id"]) for row in selected}) != 48:
        errors.append(f"selection_count:{len(selected)}:48")
    if pilot_cells & {str(row["cell_id"]) for row in selected}:
        errors.append("pilot_overlap")
    if any(value != 4 for value in combo_counts.values()):
        errors.append(f"algo_seed_balance:{combo_counts}")
    if len(stratum_counts) != 12 or any(value != 4 for value in stratum_counts.values()):
        errors.append(f"stratum_balance:{stratum_counts}")

    coverage = {key: sorted(value) for key, value in counts.items()}
    decision = "BRIDGE48_SELECTION_INVALID_STOP_REPAIR" if errors else "BRIDGE48_SELECTION_READY_FOR_GQDP_K8_CONFIRMATION"
    manifest = {
        "schema_version": "gate2_mitigation_bridge48_v4.2",
        "decision": decision,
        "selection_seed": SELECTION_SEED,
        "selection_contract": {
            "disjoint_from_pilot24": True,
            "method": "deterministic SHA256 assignment without using Pilot24 outcomes",
            "model_quantizer_strata": "four cells in each of 12 strata",
            "algorithm_seed_balance": "each of 12 combinations appears exactly four times",
            "confirmation_configuration": "GQDP_signed_v42 versus deterministic_random_v42 at k=8",
        },
        "pilot_manifest": str(Path(args.pilot_manifest)),
        "cells": selected,
        "coverage": coverage,
        "algorithm_seed_counts": combo_counts,
        "model_quantizer_stratum_counts": stratum_counts,
        "expected_results": 96,
        "errors": errors,
        "claim_boundary": "Bridge48 can confirm GQDP mitigation transport; official benchmark claims remain closed.",
    }
    with (out_dir / "mitigation_bridge48_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_tsv(out_dir / "mitigation_bridge48_cells.tsv", selected)
    (out_dir / "MITIGATION_BRIDGE48_PREREGISTRATION.md").write_text(
        "\n".join([
            "# Gate 2 v4.2 Bridge48 Preregistration",
            "",
            f"Decision: **{decision}**",
            "",
            f"- Disjoint confirmation cells: {len(selected)}/48",
            "- Conditions: GQDP-signed k=8 and matched deterministic-random k=8",
            "- Expected protected results: 96",
            "- Selection uses a fixed SHA256 seed and does not inspect Pilot24 outcomes.",
            "- Official benchmark and UPQI-star confirmation claims remain closed.",
        ]) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"BRIDGE48_SELECTION_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
