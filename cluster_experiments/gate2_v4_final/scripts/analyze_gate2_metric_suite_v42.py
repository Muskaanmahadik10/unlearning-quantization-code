#!/usr/bin/env python3
"""Analyze exact Gate 2 v4.2 predictors using cells as the research units."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import load_json, save_json, sha256_file  # noqa: E402
from validate_gate2_metric_suite_v42 import validate_obj  # noqa: E402
from validate_upqi_star_v42 import validate_obj as validate_upqi_star  # noqa: E402


RANKING_SCHEMA = "gate2_bridge_rankings_v4.0"
SIGNED_PREDICTORS = {"GQDP_signed_v42", "UPQI_star_signed_v42", "legacy_UPQI_signed_v4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric_root", required=True)
    parser.add_argument("--bridge_root", action="append", required=True)
    parser.add_argument("--upqi_star_root")
    parser.add_argument("--upqi_star_k", type=int)
    parser.add_argument("--upqi_star_n", type=int)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--expected_cells", type=int, default=12)
    parser.add_argument("--bootstrap_iters", type=int, default=5000)
    parser.add_argument("--signflip_iters", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=20260901)
    return parser.parse_args()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metric_cell_id(obj: Mapping[str, Any]) -> str:
    return f"{obj['model_tag']}_{obj['algo']}_seed{obj['seed']}/{obj['quantizer']}"


def bridge_cell_id(obj: Mapping[str, Any]) -> str:
    return str(obj.get("cell_id") or f"{obj['model_tag']}_{obj['algo']}_seed{obj['seed']}/{obj['quantizer']}")


def layer_map(obj: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = obj.get("layers") or obj.get("per_layer")
    if isinstance(rows, Mapping):
        return {str(name): row for name, row in rows.items() if isinstance(row, Mapping)}
    output: dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                name = row.get("layer") or row.get("layer_name")
                if name:
                    output[str(name)] = row
    return output


def rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            output[order[position]] = average
        cursor = end
    return output


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(rank(left), rank(right))


def top_recall(predicted: Mapping[str, float], target: Mapping[str, float], k: int) -> float:
    predicted_top = {name for name, _ in sorted(predicted.items(), key=lambda item: (-item[1], item[0]))[:k]}
    target_top = {name for name, _ in sorted(target.items(), key=lambda item: (-item[1], item[0]))[:k]}
    return len(predicted_top & target_top) / float(k)


def bootstrap_ci(values: Sequence[float], *, iters: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(iters):
        estimates.append(statistics.median(rng.choice(values) for _ in values))
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def signflip_p(values: Sequence[float], *, iters: int, seed: int) -> float:
    if not values:
        return 1.0
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(iters):
        value = abs(statistics.fmean(item if rng.random() < 0.5 else -item for item in values))
        extreme += value >= observed - 1e-15
    return extreme / float(iters + 1)


def bh_adjust(rows: list[dict[str, Any]], *, p_key: str = "p_value", q_key: str = "q_value") -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index][p_key]))
    adjusted = [1.0] * len(rows)
    running = 1.0
    total = len(rows)
    for reverse_index in range(total - 1, -1, -1):
        row_index = ordered[reverse_index]
        rank_index = reverse_index + 1
        running = min(running, float(rows[row_index][p_key]) * total / rank_index)
        adjusted[row_index] = min(1.0, running)
    for row, value in zip(rows, adjusted):
        row[q_key] = value


def deterministic_random(cell: str, layer: str, seed: int) -> float:
    digest = hashlib.sha256(f"{cell}|metric_random_v42|{seed}|{layer}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def predictor_specs() -> dict[str, tuple[str, Callable[[float], float]]]:
    identity = lambda value: value
    negate = lambda value: -value
    return {
        "GQDP_signed_v42": ("gqdp_signed", negate),
        "GQDP_abs_v42": ("gqdp_abs", identity),
        "GQDP_cosine_v42": ("gqdp_cosine", negate),
        "gradient_displacement_magnitude_v42": ("magnitude_product", identity),
        "gradient_l2_v42": ("grad_l2", identity),
        "gradient_l1_v42": ("grad_l1", identity),
        "gradient_mean_abs_v42": ("grad_mean_abs", identity),
        "displacement_l2_v42": ("epsilon_l2", identity),
        "displacement_l1_v42": ("epsilon_l1", identity),
        "displacement_mean_abs_v42": ("epsilon_mean_abs", identity),
        "displacement_max_v42": ("epsilon_max", identity),
        "quantization_mse_v42": ("quantization_mse", identity),
        "inverse_sqnr_v42": ("sqnr_db", negate),
        "weight_l2_v42": ("weight_l2", identity),
        "weight_l1_v42": ("weight_l1", identity),
        "weight_mean_abs_v42": ("weight_mean_abs", identity),
        "BCR_true_v42": ("bcr_true", identity),
        "layer_position_v42": ("layer_position", identity),
        "Fisher_forget_v42": ("fisher_forget_ps", identity),
        "FQP_forget_v42": ("fqp_forget", identity),
        "Fisher_retain_v42": ("fisher_retain_ps", identity),
    }


def selected_upqi_star(
    root: Path, cell: str, k: int | None, n: int | None
) -> tuple[Path, Mapping[str, Any]] | None:
    candidates: list[tuple[Path, Mapping[str, Any]]] = []
    for path in root.rglob("upqi_star_v42.json"):
        obj = load_json(path)
        if metric_cell_id(obj) != cell or not validate_upqi_star(obj, path)["passed"]:
            continue
        if k is not None and int(obj["n_quadrature"]) != k:
            continue
        if n is not None and int(obj["n_samples"]) != n:
            continue
        candidates.append((path, obj))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (int(item[1]["n_samples"]), int(item[1]["n_quadrature"])))


def generation(model: str) -> str:
    return "llama3" if model.startswith("llama3") else "llama2"


def factor_contrasts(performance: Sequence[Mapping[str, Any]], *, iters: int, seed: int) -> list[dict[str, Any]]:
    factors = {
        "quantizer": ("quantizer", ("model_tag", "dataset", "algo", "seed")),
        "algorithm": ("algo", ("model_tag", "dataset", "seed", "quantizer")),
        "seed": ("seed", ("model_tag", "dataset", "algo", "quantizer")),
        "model_generation": ("model_generation", ("dataset", "algo", "seed", "quantizer")),
        "dataset": ("dataset", ("model_generation", "algo", "seed", "quantizer")),
    }
    output: list[dict[str, Any]] = []
    predictors = sorted({str(row["predictor"]) for row in performance if row["target"] == "harmful_recovery"})
    for predictor in predictors:
        predictor_rows = [row for row in performance if row["predictor"] == predictor and row["target"] == "harmful_recovery"]
        for factor_name, (level_key, block_keys) in factors.items():
            levels = sorted({str(row[level_key]) for row in predictor_rows})
            for left_index, left in enumerate(levels):
                for right in levels[left_index + 1 :]:
                    blocks: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
                    for row in predictor_rows:
                        block = tuple(str(row[key]) for key in block_keys)
                        blocks[block][str(row[level_key])] = float(row["spearman"])
                    differences = [values[right] - values[left] for values in blocks.values() if left in values and right in values]
                    if not differences:
                        continue
                    low, high = bootstrap_ci(differences, iters=iters, seed=seed + len(output))
                    output.append(
                        {
                            "predictor": predictor,
                            "factor": factor_name,
                            "contrast": f"{right}-{left}",
                            "matched_blocks": len(differences),
                            "mean_difference": statistics.fmean(differences),
                            "median_difference": statistics.median(differences),
                            "ci95_low": low,
                            "ci95_high": high,
                            "p_value": signflip_p(differences, iters=iters, seed=seed + 1000 + len(output)),
                            "claim_status": "PENDING_BH",
                        }
                    )
    families: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        families[(str(row["predictor"]), str(row["factor"]))].append(row)
    for rows in families.values():
        bh_adjust(rows)
        for row in rows:
            excludes_zero = float(row["ci95_low"]) > 0 or float(row["ci95_high"]) < 0
            row["claim_status"] = "SUPPORTED" if float(row["q_value"]) < 0.05 and excludes_zero else "DESCRIPTIVE"
    return output


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted(Path(args.metric_root).rglob("metrics_v42.json"))
    structural_errors: list[str] = []
    metric_objects: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in metric_paths:
        obj = load_json(path)
        report = validate_obj(obj, path)
        if not report["passed"]:
            structural_errors.append(f"invalid_metric:{path}:{';'.join(report['errors'])}")
            continue
        cell = metric_cell_id(obj)
        if cell in metric_objects:
            structural_errors.append(f"duplicate_metric_cell:{cell}")
        metric_objects[cell] = (path, obj)
    if len(metric_paths) != args.expected_cells:
        structural_errors.append(f"metric_count:{len(metric_paths)}:{args.expected_cells}")

    bridges: dict[str, Mapping[str, Any]] = {}
    for root_name in args.bridge_root:
        for path in Path(root_name).rglob("bridge_v4.json"):
            obj = load_json(path)
            if obj.get("condition") != "full_quantized":
                continue
            cell = bridge_cell_id(obj)
            if cell in bridges:
                structural_errors.append(f"duplicate_full_bridge:{cell}")
            bridges[cell] = obj

    performance: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    specs = predictor_specs()
    for cell, (metric_path, metric_obj) in sorted(metric_objects.items()):
        if cell not in bridges:
            structural_errors.append(f"missing_full_bridge:{cell}")
            continue
        bridge = bridges[cell]
        gt_path = Path(str(metric_obj["ground_truth_reference"]["path"]))
        if not gt_path.exists():
            structural_errors.append(f"missing_gt:{cell}:{gt_path}")
            continue
        if sha256_file(gt_path) != metric_obj["ground_truth_reference"]["sha256"]:
            structural_errors.append(f"gt_sha_mismatch:{cell}")
            continue
        gt = load_json(gt_path)
        gt_layers = layer_map(gt)
        rows = {str(row["layer"]): row for row in metric_obj["per_layer"]}
        if set(rows) != set(gt_layers):
            structural_errors.append(f"layer_identity_mismatch:{cell}")
            continue
        harmful_target = {name: max(0.0, -float(gt_layers[name]["delta_forget_ce"])) for name in rows}
        absolute_target = {name: abs(float(gt_layers[name]["delta_forget_ce"])) for name in rows}
        predictors: dict[str, tuple[dict[str, float], str, Path, str, str]] = {}
        metric_lock = str(metric_obj.get("provenance", {}).get("source_lock_sha256", ""))
        metric_manifest = str(metric_obj.get("sample_manifest", {}).get("manifest_sha256", ""))
        for name, (key, transform) in specs.items():
            values = {layer: transform(float(row[key])) for layer, row in rows.items() if finite(row.get(key))}
            if len(values) == len(rows):
                target_name = "harmful_recovery" if name in SIGNED_PREDICTORS or name == "GQDP_cosine_v42" else "absolute_impact"
                predictors[name] = (values, target_name, metric_path, metric_lock, metric_manifest)
        legacy_path = Path(str(metric_obj["legacy_upqi_v4_reference"]["path"]))
        if legacy_path.exists() and sha256_file(legacy_path) == metric_obj["legacy_upqi_v4_reference"]["sha256"]:
            legacy_layers = layer_map(load_json(legacy_path))
            if set(legacy_layers) == set(rows):
                if all(finite(legacy_layers[name].get("upqi_signed")) for name in rows):
                    predictors["legacy_UPQI_signed_v4"] = (
                        {name: -float(legacy_layers[name]["upqi_signed"]) for name in rows},
                        "harmful_recovery",
                        legacy_path,
                        str(metric_obj["legacy_upqi_v4_reference"].get("source_lock_sha256", "")),
                        str(metric_obj["legacy_upqi_v4_reference"].get("sample_manifest_sha256", "")),
                    )
                if all(finite(legacy_layers[name].get("upqi_abs")) for name in rows):
                    predictors["legacy_UPQI_abs_v4"] = (
                        {name: float(legacy_layers[name]["upqi_abs"]) for name in rows},
                        "absolute_impact",
                        legacy_path,
                        str(metric_obj["legacy_upqi_v4_reference"].get("source_lock_sha256", "")),
                        str(metric_obj["legacy_upqi_v4_reference"].get("sample_manifest_sha256", "")),
                    )
        if args.upqi_star_root:
            selected_upqi = selected_upqi_star(Path(args.upqi_star_root), cell, args.upqi_star_k, args.upqi_star_n)
            if selected_upqi is not None:
                upqi_path, upqi = selected_upqi
                upqi_layers = layer_map(upqi)
                if set(upqi_layers) == set(rows):
                    predictors["UPQI_star_signed_v42"] = (
                        {name: -float(upqi_layers[name]["upqi_signed"]) for name in rows},
                        "harmful_recovery",
                        upqi_path,
                        str(upqi.get("provenance", {}).get("source_lock_sha256", "")),
                        str(upqi.get("sample_manifest", {}).get("manifest_sha256", "")),
                    )
                    predictors["UPQI_star_abs_v42"] = (
                        {name: float(upqi_layers[name]["upqi_abs"]) for name in rows},
                        "absolute_impact",
                        upqi_path,
                        str(upqi.get("provenance", {}).get("source_lock_sha256", "")),
                        str(upqi.get("sample_manifest", {}).get("manifest_sha256", "")),
                    )
        random_per_seed = [
            {name: deterministic_random(cell, name, control_seed) for name in rows} for control_seed in range(5)
        ]
        predictors["deterministic_random_v42"] = (
            random_per_seed[0], "harmful_recovery", metric_path, metric_lock, metric_manifest
        )
        predictors["deterministic_random_5seed_mean"] = (
            {name: statistics.fmean(control[name] for control in random_per_seed) for name in rows},
            "harmful_recovery",
            metric_path,
            metric_lock,
            metric_manifest,
        )

        for predictor, (scores, target_name, evidence_path, evidence_lock, evidence_manifest) in predictors.items():
            target = harmful_target if target_name == "harmful_recovery" else absolute_target
            names = sorted(scores)
            correlation = spearman([scores[name] for name in names], [target[name] for name in names])
            sign_accuracy: float | str = ""
            if predictor in SIGNED_PREDICTORS:
                sign_accuracy = statistics.fmean(
                    (scores[name] > 0) == (float(gt_layers[name]["delta_forget_ce"]) < 0) for name in names
                )
            performance.append(
                {
                    "cell_id": cell,
                    "model_tag": metric_obj["model_tag"],
                    "model_generation": generation(str(metric_obj["model_tag"])),
                    "dataset": metric_obj["dataset"],
                    "algo": metric_obj["algo"],
                    "seed": str(metric_obj["seed"]),
                    "quantizer": metric_obj["quantizer"],
                    "predictor": predictor,
                    "target": target_name,
                    "n_layers": len(names),
                    "spearman": correlation,
                    "top4_recall": top_recall(scores, target, 4),
                    "top8_recall": top_recall(scores, target, 8),
                    "sign_accuracy": sign_accuracy,
                    "bridge_harmful_recovery": bridge.get("deltas", {}).get("harmful_recovery"),
                    "bridge_delta_retain_ce": bridge.get("deltas", {}).get("delta_retain_ce"),
                }
            )
            if predictor != "deterministic_random_5seed_mean":
                evidence_sha = sha256_file(evidence_path)
                ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                for position, (layer, score) in enumerate(ordered, start=1):
                    ranking_rows.append(
                        {
                            "schema_version": RANKING_SCHEMA,
                            "cell_id": cell,
                            "model": metric_obj["model_tag"],
                            "dataset": metric_obj["dataset"],
                            "algo": metric_obj["algo"],
                            "seed": str(metric_obj["seed"]),
                            "quantizer": metric_obj["quantizer"],
                            "ranking_source": predictor,
                            "layer_name": layer,
                            "rank": position,
                            "score": score,
                            "score_direction": f"higher_{target_name}",
                            "source_lock_sha256": evidence_lock,
                            "manifest_sha256": evidence_manifest,
                            "evidence_path": str(evidence_path),
                            "evidence_sha256": evidence_sha,
                        }
                    )
    write_tsv(out_dir / "metric_cell_performance.tsv", performance)
    with (out_dir / "metric_rankings_v42.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in ranking_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in performance:
        grouped[(str(row["predictor"]), str(row["target"]))].append(row)
    for index, ((predictor, target), rows_group) in enumerate(sorted(grouped.items())):
        values = [float(row["spearman"]) for row in rows_group]
        low, high = bootstrap_ci(values, iters=args.bootstrap_iters, seed=args.random_seed + index)
        sign_values = [float(row["sign_accuracy"]) for row in rows_group if row["sign_accuracy"] != ""]
        summaries.append(
            {
                "predictor": predictor,
                "target": target,
                "n_cells": len(values),
                "mean_spearman": statistics.fmean(values),
                "median_spearman": statistics.median(values),
                "median_ci95_low": low,
                "median_ci95_high": high,
                "positive_cell_fraction": statistics.fmean(value > 0 for value in values),
                "mean_top4_recall": statistics.fmean(float(row["top4_recall"]) for row in rows_group),
                "mean_top8_recall": statistics.fmean(float(row["top8_recall"]) for row in rows_group),
                "mean_sign_accuracy": statistics.fmean(sign_values) if sign_values else "",
                "p_value": signflip_p(values, iters=args.signflip_iters, seed=args.random_seed + 10000 + index),
            }
        )
    bh_adjust(summaries)
    for row in summaries:
        row["claim_status"] = (
            "SUPPORTED_DIRECTIONAL_SIGNAL"
            if float(row["q_value"]) < 0.05 and float(row["median_ci95_low"]) > 0
            else "DESCRIPTIVE_OR_INCONCLUSIVE"
        )
    write_tsv(out_dir / "metric_summary.tsv", summaries)
    contrasts = factor_contrasts(performance, iters=min(args.bootstrap_iters, args.signflip_iters), seed=args.random_seed)
    write_tsv(out_dir / "metric_factor_contrasts.tsv", contrasts)

    random_summary = next(
        (row for row in summaries if row["predictor"] == "deterministic_random_5seed_mean"), None
    )
    guided = [
        row for row in summaries
        if row["predictor"] in {"GQDP_signed_v42", "UPQI_star_signed_v42"}
        and int(row["n_cells"]) == args.expected_cells
    ]
    random_median = float(random_summary["median_spearman"]) if random_summary else 0.0
    predictor_gate = any(
        row["claim_status"] == "SUPPORTED_DIRECTIONAL_SIGNAL"
        and float(row["median_spearman"]) > random_median
        for row in guided
    )
    if structural_errors:
        decision = "METRIC_SUITE_INVALID_STOP_REPAIR"
    elif args.mode == "pilot":
        decision = "METRIC_PILOT_CERTIFIED_EXPAND_FULL144"
    elif predictor_gate:
        decision = "FULL_METRICS_CERTIFIED_MITIGATION_PILOT24_READY"
    else:
        decision = "FULL_METRICS_CERTIFIED_PREDICTOR_INCONCLUSIVE"
    claim_rows = [
        {"claim": "exact_GQDP_computed", "status": "PASS" if not structural_errors else "FAIL", "boundary": "Endpoint first-order predictor only"},
        {"claim": "UPQI_star_superiority", "status": "PASS" if any(row["predictor"] == "UPQI_star_signed_v42" and row["claim_status"] == "SUPPORTED_DIRECTIONAL_SIGNAL" for row in summaries) else "NOT_ESTABLISHED", "boundary": "Requires selected converged K/N and full coverage"},
        {"claim": "general_mitigation", "status": "NOT_ESTABLISHED", "boundary": "Requires matched Pilot24/Bridge48 against random"},
        {"claim": "official_benchmark_success", "status": "NOT_ESTABLISHED", "boundary": "Requires maintained official evaluator"},
        {"claim": "paper_ready", "status": "NOT_ESTABLISHED", "boundary": "Requires mitigation, official metrics, and final destroyer audit"},
    ]
    write_tsv(out_dir / "metric_claim_gate.tsv", claim_rows)
    decision_obj = {
        "decision": decision,
        "mode": args.mode,
        "metric_results": len(metric_objects),
        "expected_cells": args.expected_cells,
        "bridge_matches": sum(cell in bridges for cell in metric_objects),
        "structural_errors": structural_errors,
        "predictor_gate": predictor_gate,
        "random_median_spearman": random_median,
        "research_unit": "cell_or_matched_checkpoint_block_not_layer_or_token",
        "multiplicity": "BH_across_predictor_summaries_and_within_each_factor_contrast_family",
    }
    save_json(out_dir / "metric_next_phase_decision.json", decision_obj)
    top = sorted(summaries, key=lambda row: (-float(row["median_spearman"]), str(row["predictor"])))[:10]
    markdown = [
        "# Gate 2 v4.2 Metric-Suite Analysis",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Metric cells: {len(metric_objects)}/{args.expected_cells}",
        f"- Structural errors: {len(structural_errors)}",
        f"- Ranking rows: {len(ranking_rows)}",
        f"- Predictor gate: {predictor_gate}",
        "",
        "## Leading Predictors",
        "",
        "| Predictor | Target | Cells | Median Spearman | 95% CI | BH q | Status |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in top:
        markdown.append(
            f"| {row['predictor']} | {row['target']} | {row['n_cells']} | {float(row['median_spearman']):.4f} | "
            f"[{float(row['median_ci95_low']):.4f}, {float(row['median_ci95_high']):.4f}] | "
            f"{float(row['q_value']):.4g} | {row['claim_status']} |"
        )
    markdown.extend(
        [
            "",
            "Layer rows are used only to calculate within-cell ranking performance. Inferential summaries use cells and matched blocks as units.",
            "Mitigation and official benchmark claims remain closed until their own gates pass.",
        ]
    )
    (out_dir / "METRIC_SUITE_ANALYSIS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    print(f"METRIC_SUITE_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not structural_errors else 2)


if __name__ == "__main__":
    main()
