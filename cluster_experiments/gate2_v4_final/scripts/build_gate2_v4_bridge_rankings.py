#!/usr/bin/env python3
"""Build canonical per-cell layer rankings for Gate 2 v4 whole-model bridge.

This script is intentionally torch-free. It treats v4 GT/UPQI files as input
evidence and emits a JSONL ranking table that bridge runners can consume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import (  # noqa: E402
    CORE_SETTINGS,
    QUANTIZERS,
    SEEDS,
    ALGOS,
    finite_number,
    load_json,
    save_json,
    sha256_file,
)

BRIDGE_RANKING_SCHEMA = "gate2_bridge_rankings_v4.0"
V4_RESULTS_FRAGMENT = "gate2_v4_standard"
LEGACY_FRAGMENTS = ("gate2_v2", "gate2_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results/gate2_v4_standard")
    parser.add_argument("--out", default="results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl")
    parser.add_argument("--summary_out", default=None)
    parser.add_argument("--metric_tsv", action="append", default=[], help="Optional external metric ranking TSV.")
    parser.add_argument("--expected_source_lock_sha256", default=None)
    parser.add_argument("--allow_incomplete_gt", action="store_true", default=True)
    return parser.parse_args()


def reject_bad_input_root(path: Path) -> None:
    s = str(path)
    if V4_RESULTS_FRAGMENT not in s:
        raise SystemExit(f"results_root must contain {V4_RESULTS_FRAGMENT}: {path}")
    if any(frag in s for frag in LEGACY_FRAGMENTS):
        raise SystemExit(f"Refusing legacy result path for bridge rankings: {path}")


def cell_id(model_tag: str, algo: str, seed: str, quantizer: str) -> str:
    return f"{model_tag}_{algo}_seed{seed}/{quantizer}"


def cell_dir(results_root: Path, model_tag: str, algo: str, seed: str, quantizer: str) -> Path:
    return results_root / f"{model_tag}_{algo}_seed{seed}" / quantizer


def source_lock(obj: Mapping[str, Any]) -> str:
    return str(obj.get("provenance", {}).get("source_lock_sha256", ""))


def manifest_sha(obj: Mapping[str, Any]) -> str:
    for key in ("forget_manifest", "sample_manifest", "manifest"):
        value = obj.get(key)
        if isinstance(value, Mapping):
            sha = value.get("sha256") or value.get("manifest_sha256")
            if sha:
                return str(sha)
    return ""


def layer_order(obj: Mapping[str, Any]) -> List[str]:
    names = obj.get("target_layer_names")
    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        return list(names)
    layers = obj.get("layers") or obj.get("per_layer")
    if isinstance(layers, Mapping):
        return [str(k) for k in layers.keys()]
    if isinstance(layers, list):
        out = []
        for item in layers:
            if isinstance(item, Mapping):
                layer_name = item.get("layer_name") or item.get("layer")
                if layer_name:
                    out.append(str(layer_name))
        return out
    return []


def upqi_layers(obj: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    layers = obj.get("layers") or obj.get("per_layer")
    if isinstance(layers, Mapping):
        return {str(k): v for k, v in layers.items() if isinstance(v, Mapping)}
    if isinstance(layers, list):
        out: Dict[str, Mapping[str, Any]] = {}
        for item in layers:
            if isinstance(item, Mapping):
                layer_name = item.get("layer_name") or item.get("layer")
                if layer_name:
                    out[str(layer_name)] = item
        return out
    return {}


def gt_layers(obj: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    layers = obj.get("layers") or obj.get("per_layer")
    if isinstance(layers, Mapping):
        return {str(k): v for k, v in layers.items() if isinstance(v, Mapping)}
    if isinstance(layers, list):
        out: Dict[str, Mapping[str, Any]] = {}
        for item in layers:
            if isinstance(item, Mapping):
                layer_name = item.get("layer_name") or item.get("layer")
                if layer_name:
                    out[str(layer_name)] = item
        return out
    return {}


def first_number(row: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if finite_number(value):
            return float(value)
    return None


def deterministic_random_score(cid: str, layer_name: str, source: str = "random") -> float:
    digest = hashlib.sha256(f"{cid}|{source}|{layer_name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def add_rank_rows(
    rows: List[Dict[str, Any]],
    *,
    cid: str,
    model: str,
    dataset: str,
    algo: str,
    seed: str,
    quantizer: str,
    ranking_source: str,
    scores: Mapping[str, float],
    score_direction: str,
    source_lock_sha256: str,
    manifest_sha256: str,
    evidence_path: str,
) -> None:
    evidence_sha256 = sha256_file(evidence_path) if evidence_path else ""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    for rank, (layer_name, score) in enumerate(ordered, start=1):
        rows.append(
            {
                "schema_version": BRIDGE_RANKING_SCHEMA,
                "cell_id": cid,
                "model": model,
                "dataset": dataset,
                "algo": algo,
                "seed": seed,
                "quantizer": quantizer,
                "ranking_source": ranking_source,
                "layer_name": layer_name,
                "rank": rank,
                "score": score,
                "score_direction": score_direction,
                "source_lock_sha256": source_lock_sha256,
                "manifest_sha256": manifest_sha256,
                "evidence_path": evidence_path,
                "evidence_sha256": evidence_sha256,
            }
        )


def build_from_cell(
    *,
    results_root: Path,
    setting: Mapping[str, str],
    algo: str,
    seed: str,
    quantizer: str,
    expected_lock: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model_tag = setting["model_tag"]
    dataset = setting["dataset"]
    out_dir = cell_dir(results_root, model_tag, algo, seed, quantizer)
    cid = cell_id(model_tag, algo, seed, quantizer)
    rows: List[Dict[str, Any]] = []
    status: Dict[str, Any] = {
        "cell_id": cid,
        "model": model_tag,
        "dataset": dataset,
        "algo": algo,
        "seed": seed,
        "quantizer": quantizer,
        "upqi": "missing",
        "gt": "missing",
        "rank_sources": 0,
        "issues": [],
    }

    upqi_path = out_dir / "upqi_v4.json"
    gt_path = out_dir / "ground_truth_v4.json"
    if upqi_path.exists():
        upqi = load_json(upqi_path)
        lock = source_lock(upqi)
        msha = manifest_sha(upqi)
        if expected_lock and lock != expected_lock:
            status["issues"].append("upqi_source_lock_mismatch")
        layers = upqi_layers(upqi)
        order = layer_order(upqi)
        if layers and order:
            status["upqi"] = "usable"
            scores_abs: Dict[str, float] = {}
            scores_signed: Dict[str, float] = {}
            scores_disp: Dict[str, float] = {}
            for lname in order:
                row = layers.get(lname)
                if not row:
                    continue
                val_abs = first_number(row, ("upqi_abs", "UPQI_abs"))
                val_signed = first_number(row, ("upqi_signed", "UPQI_signed"))
                val_disp = first_number(row, ("epsilon_l2", "displacement_norm", "weight_delta_l2"))
                if val_abs is not None:
                    scores_abs[lname] = val_abs
                if val_signed is not None:
                    scores_signed[lname] = -val_signed
                if val_disp is not None:
                    scores_disp[lname] = val_disp
            if scores_abs:
                add_rank_rows(
                    rows,
                    cid=cid,
                    model=model_tag,
                    dataset=dataset,
                    algo=algo,
                    seed=seed,
                    quantizer=quantizer,
                    ranking_source="UPQI_abs",
                    scores=scores_abs,
                    score_direction="higher_abs_layer_sensitivity",
                    source_lock_sha256=lock,
                    manifest_sha256=msha,
                    evidence_path=str(upqi_path),
                )
            if scores_signed:
                add_rank_rows(
                    rows,
                    cid=cid,
                    model=model_tag,
                    dataset=dataset,
                    algo=algo,
                    seed=seed,
                    quantizer=quantizer,
                    ranking_source="UPQI_signed",
                    scores=scores_signed,
                    score_direction="higher_predicted_harmful_recovery",
                    source_lock_sha256=lock,
                    manifest_sha256=msha,
                    evidence_path=str(upqi_path),
                )
            if scores_disp:
                add_rank_rows(
                    rows,
                    cid=cid,
                    model=model_tag,
                    dataset=dataset,
                    algo=algo,
                    seed=seed,
                    quantizer=quantizer,
                    ranking_source="displacement_norm",
                    scores=scores_disp,
                    score_direction="higher_weight_displacement",
                    source_lock_sha256=lock,
                    manifest_sha256=msha,
                    evidence_path=str(upqi_path),
                )
            random_scores = {lname: deterministic_random_score(cid, lname) for lname in order}
            add_rank_rows(
                rows,
                cid=cid,
                model=model_tag,
                dataset=dataset,
                algo=algo,
                seed=seed,
                quantizer=quantizer,
                ranking_source="random",
                scores=random_scores,
                score_direction="deterministic_random_control",
                source_lock_sha256=lock,
                manifest_sha256=msha,
                evidence_path=str(upqi_path),
            )
        else:
            status["upqi"] = "unusable"
            status["issues"].append("upqi_layers_unusable")

    if gt_path.exists():
        gt = load_json(gt_path)
        lock = source_lock(gt)
        msha = manifest_sha(gt)
        if expected_lock and lock != expected_lock:
            status["issues"].append("gt_source_lock_mismatch")
        layers = gt_layers(gt)
        order = layer_order(gt)
        if layers and order:
            status["gt"] = "usable"
            scores_abs: Dict[str, float] = {}
            scores_harmful: Dict[str, float] = {}
            for lname in order:
                row = layers.get(lname)
                if not row:
                    continue
                delta_ce = first_number(row, ("delta_forget_ce", "delta_answer_ce"))
                harmful = first_number(row, ("harmful_recovery",))
                if delta_ce is not None:
                    scores_abs[lname] = abs(delta_ce)
                    scores_harmful[lname] = max(0.0, -delta_ce)
                elif harmful is not None:
                    scores_harmful[lname] = harmful
            if scores_abs:
                add_rank_rows(
                    rows,
                    cid=cid,
                    model=model_tag,
                    dataset=dataset,
                    algo=algo,
                    seed=seed,
                    quantizer=quantizer,
                    ranking_source="GT_oracle_abs",
                    scores=scores_abs,
                    score_direction="higher_absolute_gt_change",
                    source_lock_sha256=lock,
                    manifest_sha256=msha,
                    evidence_path=str(gt_path),
                )
            if scores_harmful:
                add_rank_rows(
                    rows,
                    cid=cid,
                    model=model_tag,
                    dataset=dataset,
                    algo=algo,
                    seed=seed,
                    quantizer=quantizer,
                    ranking_source="GT_oracle_harmful",
                    scores=scores_harmful,
                    score_direction="higher_observed_harmful_recovery",
                    source_lock_sha256=lock,
                    manifest_sha256=msha,
                    evidence_path=str(gt_path),
                )
        else:
            status["gt"] = "unusable"
            status["issues"].append("gt_layers_unusable")

    status["rank_sources"] = len({r["ranking_source"] for r in rows})
    return rows, status


def load_external_metric_rows(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        for row in reader:
            cid = row.get("cell_id") or row.get("cell")
            source = row.get("ranking_source") or row.get("metric") or row.get("source")
            layer = row.get("layer_name") or row.get("layer")
            score_raw = row.get("score") or row.get("value")
            if not cid or not source or not layer:
                continue
            try:
                score = float(score_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "schema_version": BRIDGE_RANKING_SCHEMA,
                    "cell_id": cid,
                    "model": row.get("model", ""),
                    "dataset": row.get("dataset", ""),
                    "algo": row.get("algo", ""),
                    "seed": row.get("seed", ""),
                    "quantizer": row.get("quantizer", ""),
                    "ranking_source": source,
                    "layer_name": layer,
                    "rank": int(row.get("rank") or 0),
                    "score": score,
                    "score_direction": row.get("score_direction", "higher_is_higher_risk"),
                    "source_lock_sha256": row.get("source_lock_sha256", ""),
                    "manifest_sha256": row.get("manifest_sha256", ""),
                    "evidence_path": str(path),
                    "evidence_sha256": sha256_file(path),
                }
            )
    grouped: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in out:
        grouped[(row["cell_id"], row["ranking_source"])].append(row)
    reranked: List[Dict[str, Any]] = []
    for _, rows in grouped.items():
        rows = sorted(rows, key=lambda r: (-float(r["score"]), str(r["layer_name"])))
        for idx, row in enumerate(rows, start=1):
            row = dict(row)
            row["rank"] = idx
            reranked.append(row)
    return reranked


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_summary(path: Path, summaries: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = list(summaries)
    if not summaries:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    reject_bad_input_root(results_root)
    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for setting in CORE_SETTINGS.values():
        for algo in ALGOS:
            for seed in SEEDS:
                for quantizer in QUANTIZERS:
                    rows, status = build_from_cell(
                        results_root=results_root,
                        setting=setting,
                        algo=algo,
                        seed=seed,
                        quantizer=quantizer,
                        expected_lock=args.expected_source_lock_sha256,
                    )
                    all_rows.extend(rows)
                    summaries.append(status)

    for metric_path in args.metric_tsv:
        all_rows.extend(load_external_metric_rows(Path(metric_path)))

    out = Path(args.out)
    write_jsonl(out, all_rows)
    summary_out = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.tsv")
    write_summary(summary_out, summaries)
    print(f"WROTE {out} rows={len(all_rows)}")
    print(f"WROTE {summary_out} cells={len(summaries)}")


if __name__ == "__main__":
    main()
