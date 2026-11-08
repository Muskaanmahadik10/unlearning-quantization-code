#!/usr/bin/env python3
"""Build a disjoint, factor-balanced Bridge48 confirmation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ALGOS = ("GA", "GDR", "KLR", "NPO")
SEEDS = ("42", "2024", "5000")
COMBOS = tuple(itertools.product(ALGOS, SEEDS))
SELECTION_SEED = "gate2_bridge48_confirmation_v42_20260901"
PILOT_PASS = "MITIGATION_PILOT24_PASS_EXPAND_BRIDGE48"
MANIFEST_PASS = "BRIDGE48_CONFIRMATION_SELECTION_READY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric_rankings", required=True)
    parser.add_argument("--bridge_root", action="append", required=True)
    parser.add_argument("--pilot_manifest", required=True)
    parser.add_argument("--pilot_decision", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cell_id(obj: Mapping[str, Any]) -> str:
    return str(obj.get("cell_id") or f"{obj['model_tag']}_{obj['algo']}_seed{obj['seed']}/{obj['quantizer']}")


def stable_score(stratum: tuple[str, str], choices: Sequence[tuple[str, str]]) -> str:
    value = "|".join([SELECTION_SEED, *stratum, *(f"{algo}:{seed}" for algo, seed in choices)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def choose_balanced(
    strata: Sequence[tuple[str, str]],
    available: Mapping[tuple[str, str], set[tuple[str, str]]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Select four cells per stratum and four uses of every algo/seed combo."""
    options = {
        stratum: sorted(
            itertools.combinations(sorted(available[stratum]), 4),
            key=lambda choice: stable_score(stratum, choice),
        )
        for stratum in strata
    }
    ordered = sorted(strata, key=lambda stratum: (len(options[stratum]), stratum))
    combo_index = {combo: index for index, combo in enumerate(COMBOS)}
    dead: set[tuple[int, tuple[int, ...]]] = set()

    def search(index: int, counts: tuple[int, ...]) -> dict[tuple[str, str], tuple[tuple[str, str], ...]] | None:
        state = (index, counts)
        if state in dead:
            return None
        if index == len(ordered):
            return {} if all(value == 4 for value in counts) else None
        remaining = ordered[index + 1 :]
        stratum = ordered[index]
        for choice in options[stratum]:
            updated = list(counts)
            for combo in choice:
                updated[combo_index[combo]] += 1
            if any(value > 4 for value in updated):
                continue
            feasible = True
            for combo, combo_pos in combo_index.items():
                capacity = sum(combo in available[item] for item in remaining)
                if updated[combo_pos] + capacity < 4:
                    feasible = False
                    break
            if not feasible:
                continue
            tail = search(index + 1, tuple(updated))
            if tail is not None:
                tail[stratum] = choice
                return tail
        dead.add(state)
        return None

    selected = search(0, (0,) * len(COMBOS))
    if selected is None:
        raise ValueError("No exact 48-cell balanced assignment exists after excluding Pilot24")
    return selected


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    pilot_manifest = load_json(Path(args.pilot_manifest))
    pilot_decision = load_json(Path(args.pilot_decision))
    pilot_cells = {str(row["cell_id"]) for row in pilot_manifest.get("cells", [])}
    if len(pilot_cells) != 24:
        errors.append(f"pilot_cell_count:{len(pilot_cells)}:24")
    if pilot_decision.get("decision") != PILOT_PASS:
        errors.append(f"pilot_decision:{pilot_decision.get('decision')}:{PILOT_PASS}")
    passing = {(row.get("selector"), int(row.get("k", -1))) for row in pilot_decision.get("passing_selector_k", [])}
    if ("GQDP_signed_v42", 8) not in passing:
        errors.append("pilot_did_not_pass:GQDP_signed_v42:k8")

    required_sources = {"GQDP_signed_v42", "deterministic_random_v42"}
    ranking_layers: dict[tuple[str, str], set[str]] = defaultdict(set)
    with Path(args.metric_rankings).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("ranking_source"))
            if source in required_sources:
                ranking_layers[(str(row["cell_id"]), source)].add(str(row["layer_name"]))

    bridges: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for root in args.bridge_root:
        for path in Path(root).rglob("bridge_v4.json"):
            obj = load_json(path)
            if obj.get("condition") != "full_quantized":
                continue
            key = cell_id(obj)
            if key in bridges:
                duplicates.append(key)
            bridges[key] = obj
    if len(bridges) != 144:
        errors.append(f"full_bridge_count:{len(bridges)}:144")
    if duplicates:
        errors.append(f"duplicate_full_bridges:{len(duplicates)}")

    candidates: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    for key, obj in sorted(bridges.items()):
        if key in pilot_cells:
            continue
        bad = [source for source in required_sources if len(ranking_layers[(key, source)]) != 224]
        if bad:
            errors.append(f"ranking_coverage:{key}:{','.join(sorted(bad))}")
            continue
        row = {
            "cell_id": key,
            "model_tag": str(obj["model_tag"]),
            "dataset": str(obj["dataset"]),
            "algo": str(obj["algo"]),
            "seed": str(obj["seed"]),
            "quantizer": str(obj["quantizer"]),
        }
        candidates[(row["model_tag"], row["quantizer"])][(row["algo"], row["seed"])] = row

    strata = sorted(candidates)
    if len(strata) != 12:
        errors.append(f"stratum_count:{len(strata)}:12")
    if sum(len(rows) for rows in candidates.values()) != 120:
        errors.append(f"remaining_candidate_count:{sum(len(rows) for rows in candidates.values())}:120")

    selected_rows: list[dict[str, str]] = []
    if not errors:
        assignment = choose_balanced(strata, {key: set(value) for key, value in candidates.items()})
        for stratum in sorted(assignment):
            for combo in sorted(assignment[stratum]):
                selected_rows.append(
                    {
                        **candidates[stratum][combo],
                        "panel": "disjoint_balanced_confirmation",
                        "selectors": "GQDP_signed_v42;deterministic_random_v42",
                        "k_values": "8",
                    }
                )

    combo_counts = Counter((row["algo"], row["seed"]) for row in selected_rows)
    stratum_counts = Counter((row["model_tag"], row["quantizer"]) for row in selected_rows)
    if len(selected_rows) != 48 or len({row["cell_id"] for row in selected_rows}) != 48:
        errors.append(f"selection_count:{len(selected_rows)}:48")
    if set(row["cell_id"] for row in selected_rows) & pilot_cells:
        errors.append("pilot_overlap")
    if selected_rows and (set(combo_counts.values()) != {4} or len(combo_counts) != 12):
        errors.append(f"algo_seed_balance:{dict(combo_counts)}")
    if selected_rows and (set(stratum_counts.values()) != {4} or len(stratum_counts) != 12):
        errors.append(f"model_quantizer_balance:{dict(stratum_counts)}")

    coverage = {
        "models": dict(Counter(row["model_tag"] for row in selected_rows)),
        "datasets": dict(Counter(row["dataset"] for row in selected_rows)),
        "algorithms": dict(Counter(row["algo"] for row in selected_rows)),
        "seeds": dict(Counter(row["seed"] for row in selected_rows)),
        "quantizers": dict(Counter(row["quantizer"] for row in selected_rows)),
    }
    decision = "BRIDGE48_CONFIRMATION_SELECTION_INVALID" if errors else MANIFEST_PASS
    write_tsv(out_dir / "bridge48_confirmation_cells.tsv", selected_rows)
    manifest = {
        "schema_version": "gate2_mitigation_bridge48_confirmation_v4.2",
        "decision": decision,
        "selection_seed": SELECTION_SEED,
        "selection_contract": {
            "disjoint_from_pilot24": True,
            "outcome_blind": "stable SHA256 assignment; no Pilot24 outcome values used",
            "model_quantizer_strata": "four cells in each of 12 strata",
            "algorithm_seed_balance": "each of 12 combinations appears four times",
            "confirmation_target": "GQDP_signed_v42 at k=8 versus deterministic_random_v42 at k=8",
        },
        "pilot_manifest": str(args.pilot_manifest),
        "pilot_decision": str(args.pilot_decision),
        "cells": selected_rows,
        "coverage": coverage,
        "errors": errors,
        "required_matched_conditions": {
            "selectors": ["GQDP_signed_v42", "deterministic_random_v42"],
            "k_values": [8],
            "conditions_per_cell": 2,
            "expected_bridge_results": 96,
        },
        "claim_boundary": "Confirmation can qualify GQDP mitigation for official evaluation; official benchmark claims remain closed.",
    }
    (out_dir / "bridge48_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    markdown = [
        "# Gate 2 v4.2 Bridge48 Confirmation Preregistration",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Cells: {len(selected_rows)}/48",
        "- Conditions: 96 (GQDP k=8 and matched deterministic random k=8)",
        "- Pilot24 overlap: 0 required",
        f"- Coverage: {coverage}",
        f"- Structural errors: {len(errors)}",
        "",
        "Selection is deterministic and outcome-blind after excluding the 24 discovery cells.",
        "This confirmation does not establish official benchmark success.",
    ]
    (out_dir / "BRIDGE48_CONFIRMATION_PREREGISTRATION.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"BRIDGE48_SELECTION_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
