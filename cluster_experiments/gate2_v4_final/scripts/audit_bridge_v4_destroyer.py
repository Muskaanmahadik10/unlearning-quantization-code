#!/usr/bin/env python3
"""Pessimistic destroyer audit for Gate 2 v4 target-layer bridge work.

This audit assumes the bridge is unacceptable until evidence proves otherwise.
It checks source/provenance, methodology boundaries, smoke coverage, nearest
paper alignment, quantizer discipline, statistics, and claim gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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

from gate2_v4_contract import ALGOS, CORE_SETTINGS, QUANTIZERS, SEEDS, load_json, sha256_file  # noqa: E402
from validate_bridge_v4 import validate_obj as validate_bridge_obj  # noqa: E402

try:  # keep the audit runnable even if join validator changes later
    from validate_bridge_join_v4 import compare_bf16_to_gt, manifests_match, result_cell_dir  # noqa: E402
    from validate_gt_v4 import validate_obj as validate_gt_obj  # noqa: E402
    from validate_upqi_v4 import validate_obj as validate_upqi_obj  # noqa: E402
except Exception:  # pragma: no cover - reported in join gate if needed
    compare_bf16_to_gt = None  # type: ignore[assignment]
    manifests_match = None  # type: ignore[assignment]
    result_cell_dir = None  # type: ignore[assignment]
    validate_gt_obj = None  # type: ignore[assignment]
    validate_upqi_obj = None  # type: ignore[assignment]


NEAREST_STANDARD = {
    "name": "Catastrophic Failure of LLM Unlearning via Quantization",
    "venue": "ICLR 2025",
    "url": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba79fb5c4fe70050752f20c90c5f07ca-Abstract-Conference.html",
}
TOFU_URL = "https://locuslab.github.io/tofu/"
MUSE_URL = "https://muse-bench.github.io/"
GPTQ_URL = "https://arxiv.org/abs/2210.17323"
QLORA_URL = "https://arxiv.org/abs/2305.14314"
IG_URL = "https://arxiv.org/abs/1703.01365"

BRIDGE_ROOT_FRAGMENT = "gate2_v4_bridge_standard"
V4_ROOT_FRAGMENT = "gate2_v4_standard"
LEGACY_FRAGMENTS = ("gate2_v2", "gate2_v3")
HARD_LOG_PATTERNS = re.compile(
    r"Traceback|CUDA out|OUT_OF_MEMORY|ModuleNotFound|FileNotFound|No such file|"
    r"source lock mismatch|Script SHA mismatch|failed validation|SystemExit",
    re.IGNORECASE,
)
FORBIDDEN_CLAIM_PATTERNS = (
    "full deployment proof",
    "deployment-proof",
    "true deletion",
    "proof of deletion",
    "fully solved",
    "complete unlearning proof",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="gate2_v4_final")
    parser.add_argument("--bridge_root", default="results/gate2_v4_bridge_standard")
    parser.add_argument("--v4_results_root", default="results/gate2_v4_standard")
    parser.add_argument("--logs_root", default="logs/gate2_v4_bridge")
    parser.add_argument("--slurm_root", default="gate2_v4_final/slurm/gate2_v4")
    parser.add_argument("--rankings_path", default="results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl")
    parser.add_argument("--official_metrics_tsv", default=None)
    parser.add_argument("--expected_bridge_source_lock_sha256", default=None)
    parser.add_argument("--expected_v4_source_lock_sha256", default="b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b")
    parser.add_argument("--bf16_tolerance", type=float, default=1e-5)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--out_prefix", default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--fail_on_reject", action="store_true")
    return parser.parse_args()


def now_prefix(prefix: Optional[str]) -> str:
    return prefix or f"bridge_v4_destroyer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def metric(obj: Mapping[str, Any], dotted: str, default: Any = "") -> Any:
    current: Any = obj
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def core_cell_ids() -> List[str]:
    ids: List[str] = []
    for setting in CORE_SETTINGS.values():
        model_tag = setting["model_tag"]
        for algo in ALGOS:
            for seed in SEEDS:
                for quantizer in QUANTIZERS:
                    ids.append(f"{model_tag}_{algo}_seed{seed}/{quantizer}")
    return ids


def bridge_files(root: Path) -> Iterable[Path]:
    yield from root.glob("*/*/*/bridge_v4.json")


def condition_key(obj: Mapping[str, Any]) -> str:
    condition = str(obj.get("condition", ""))
    if condition == "protected":
        return f"protected:{obj.get('ranking_source')}:{obj.get('k')}"
    return condition


def load_source_lock_sha(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        obj = load_json(path)
    except Exception:
        return ""
    return str(obj.get("source_lock_sha256", ""))


def newest_bridge_lock(source_root: Path) -> Tuple[str, str, List[str]]:
    candidates = list(source_root.glob("V4_BRIDGE_SOURCE_LOCK*.json"))
    candidates += list((source_root / "bridge_locks").glob("V4_BRIDGE_SOURCE_LOCK*.json"))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return "", "", []
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    path = candidates[0]
    return str(path), load_source_lock_sha(path), [str(p) for p in candidates]


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
                        rows.append({"path": str(path), "lineno": lineno, "line": line.strip()[:600]})
        except OSError as exc:
            rows.append({"path": str(path), "lineno": 0, "line": f"LOG_READ_ERROR:{exc}"})
    return rows


def artifact_rows(args: argparse.Namespace, smoke_gaps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    source_root = Path(args.source_root)
    slurm_root = Path(args.slurm_root)
    bridge_root = Path(args.bridge_root)
    logs_root = Path(args.logs_root)
    expected = [
        ("doc", source_root / "GATE2_V4_CONCRETE_MODEL_BRIDGE_AND_THESIS_PLAN_20260801.md"),
        ("doc", source_root / "GATE2_V4_BRIDGE_GOLD_STANDARD_AUDIT_20260731.md"),
        ("doc", source_root / "GATE2_V4_WHOLE_MODEL_BRIDGE_PLAN.md"),
        ("script", source_root / "scripts" / "audit_bridge_v4_icml.py"),
        ("script", source_root / "scripts" / "build_gate2_v4_bridge_rankings.py"),
        ("script", source_root / "scripts" / "run_gate2_whole_model_bridge_v4.py"),
        ("script", source_root / "scripts" / "validate_bridge_v4.py"),
        ("script", source_root / "scripts" / "validate_bridge_join_v4.py"),
        ("script", source_root / "scripts" / "gate2_v4_contract.py"),
        ("script", source_root / "scripts" / "audit_bridge_v4_destroyer.py"),
        ("slurm", slurm_root / "run_bridge_phase_abc_20260801.sh"),
        ("slurm", slurm_root / "submit_bridge_smoke_20260801.sh"),
        ("slurm", slurm_root / "submit_bridge_fullq_20260801.sh"),
        ("slurm", slurm_root / "bridge_health_20260801.sh"),
        ("slurm", slurm_root / "bridge_v4_core64_8shard_bundle.slurm"),
        ("slurm", slurm_root / "bridge_v4_slotfill.slurm"),
        ("slurm", slurm_root / "submit_gate2_v4_bridge_slotfill.sh"),
        ("source_lock", source_root / "V4_SOURCE_LOCK.json"),
    ]
    for lock in sorted(source_root.glob("V4_BRIDGE_SOURCE_LOCK*.json")):
        expected.append(("source_lock", lock))
    for lock in sorted((source_root / "bridge_locks").glob("V4_BRIDGE_SOURCE_LOCK*.json")):
        expected.append(("source_lock", lock))
    for path in sorted(bridge_root.rglob("*")) if bridge_root.exists() else []:
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".tsv", ".md"}:
            expected.append(("bridge_artifact", path))
    for path in sorted(logs_root.rglob("*")) if logs_root.exists() else []:
        if path.is_file() and path.suffix.lower() in {".log", ".out", ".err", ".txt"}:
            expected.append(("bridge_log", path))

    smoke_issue_by_path = {
        str(slurm_root / "submit_bridge_smoke_20260801.sh"): [
            str(r["requirement_id"]) for r in smoke_gaps if r.get("status") == "FAIL"
        ]
    }
    rows: List[Dict[str, Any]] = []
    seen = set()
    for role, path in expected:
        key = (role, str(path))
        if key in seen:
            continue
        seen.add(key)
        issues: List[str] = []
        exists = path.exists()
        text = read_text(path) if exists and path.suffix.lower() in {".md", ".py", ".sh", ".slurm"} else ""
        if not exists:
            issues.append("missing_expected_artifact")
        if any(frag in str(path) for frag in LEGACY_FRAGMENTS):
            issues.append("legacy_path")
        if role == "doc":
            lowered = text.lower()
            for pattern in FORBIDDEN_CLAIM_PATTERNS:
                if pattern in lowered and "do not" not in lowered[max(0, lowered.find(pattern) - 80): lowered.find(pattern) + len(pattern) + 80]:
                    issues.append(f"possible_forbidden_claim:{pattern.replace(' ', '_')}")
        if role == "script" and path.name == "run_gate2_whole_model_bridge_v4.py":
            if "get_weight_layers(model)" in text:
                issues.append("target_layer_scope_not_deployment_scope")
            if "compute_forget_eval_v4" not in text or "compute_retain_eval" not in text:
                issues.append("missing_behavior_metric_hook")
        if str(path) in smoke_issue_by_path and smoke_issue_by_path[str(path)]:
            issues.append("smoke_gaps:" + ",".join(smoke_issue_by_path[str(path)]))
        rows.append(
            {
                "role": role,
                "path": str(path),
                "exists": str(exists),
                "size": path.stat().st_size if exists else "",
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if exists else "",
                "sha256": sha256_file(path) if exists and path.is_file() else "",
                "issues": ";".join(issues),
            }
        )
    return rows


def smoke_gap_rows(source_root: Path, slurm_root: Path) -> List[Dict[str, Any]]:
    smoke = read_text(slurm_root / "submit_bridge_smoke_20260801.sh")
    ranking = read_text(source_root / "scripts" / "build_gate2_v4_bridge_rankings.py")
    runner = read_text(source_root / "scripts" / "run_gate2_whole_model_bridge_v4.py")
    requirements = [
        ("SMOKE_TOFU", "TOFU smoke label present", "l2_A" in smoke),
        ("SMOKE_MUSE", "MUSE smoke label present", "l2m_A" in smoke),
        ("SMOKE_LLAMA2", "LLaMA2 smoke present", "l2_A" in smoke or "l2m_A" in smoke),
        ("SMOKE_LLAMA3", "LLaMA3 smoke present", "l3_A" in smoke or "l3m_A" in smoke),
        ("SMOKE_NF4", "NF4 smoke present", "nf4" in smoke),
        ("SMOKE_GPTQ", "GPTQ int4 smoke present", "gptq_int4" in smoke),
        ("SMOKE_RTN", "RTN int8 smoke present", "rtn_int8" in smoke),
        ("SMOKE_FULLQ", "full_quantized smoke present", "full_quantized" in smoke),
        ("SMOKE_PROTECTED", "protected smoke present", "protected" in smoke),
        ("SMOKE_UPQI_SIGNED", "protected UPQI_signed smoke present", "UPQI_signed" in smoke),
        ("SMOKE_RANDOM", "random protected smoke present", "random" in smoke),
        ("SMOKE_NONE_K0", "none/k0 equivalence smoke present", "none" in smoke and re.search(r"\b0\b", smoke) is not None),
        ("SMOKE_ALL_LAYERS", "all_layers/k=max equivalence smoke present", "all_layers" in smoke),
        ("DETERMINISTIC_RANDOM", "deterministic random ranking implementation present", "deterministic_random_score" in ranking and "sha256" in ranking),
        ("RUNNER_SPECIAL_SELECTORS", "runner supports none/all_layers selectors", '"all_layers"' in runner and '"none"' in runner),
    ]
    rows = []
    for req_id, requirement, ok in requirements:
        rows.append(
            {
                "requirement_id": req_id,
                "requirement": requirement,
                "status": "PASS" if ok else "FAIL",
                "evidence": "detected in bridge smoke/ranking/runner source" if ok else "not detected in current defaults",
                "required_repair": "" if ok else "patch smoke/defaults before production or document as intentionally untested",
            }
        )
    return rows


def load_bridge_objects(root: Path, expected_lock: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Mapping[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    objects: List[Mapping[str, Any]] = []
    for path in sorted(bridge_files(root)):
        issues: List[str] = []
        obj: Mapping[str, Any] = {}
        try:
            obj = load_json(path)
            issues = validate_bridge_obj(obj, path=path, expected_source_lock_sha256=expected_lock)
        except Exception as exc:
            issues = [f"load_or_validate_error:{type(exc).__name__}:{exc}"]
        status = "PASS" if not issues else "FAIL"
        if status == "PASS":
            objects.append(obj)
        rows.append(
            {
                "status": status,
                "cell_id": obj.get("cell_id", "") if isinstance(obj, Mapping) else "",
                "condition": obj.get("condition", "") if isinstance(obj, Mapping) else "",
                "ranking_source": obj.get("ranking_source", "") if isinstance(obj, Mapping) else "",
                "k": obj.get("k", "") if isinstance(obj, Mapping) else "",
                "source_lock_sha256": obj.get("source_lock_sha256", "") if isinstance(obj, Mapping) else "",
                "quantizer": obj.get("quantizer", "") if isinstance(obj, Mapping) else "",
                "harmful_recovery": metric(obj, "deltas.harmful_recovery") if isinstance(obj, Mapping) else "",
                "over_forgetting": metric(obj, "deltas.over_forgetting") if isinstance(obj, Mapping) else "",
                "delta_retain_ce": metric(obj, "deltas.delta_retain_ce") if isinstance(obj, Mapping) else "",
                "issues": ";".join(issues),
                "path": str(path),
            }
        )
    return rows, objects


def load_ranking_sources(rankings_path: Path) -> Counter:
    counts: Counter = Counter()
    if not rankings_path.exists():
        return counts
    try:
        with rankings_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts[str(row.get("ranking_source", ""))] += 1
    except Exception:
        counts["RANKING_LOAD_ERROR"] += 1
    return counts


def join_issue_count(
    objects: Sequence[Mapping[str, Any]],
    v4_root: Path,
    expected_v4_lock: Optional[str],
    tol: float,
) -> Tuple[int, List[str]]:
    if compare_bf16_to_gt is None or manifests_match is None or result_cell_dir is None:
        return len(objects), ["join_validator_import_failed"]
    issues: List[str] = []
    for obj in objects:
        try:
            cell_dir = result_cell_dir(v4_root, obj)
            gt_path = cell_dir / "ground_truth_v4.json"
            upqi_path = cell_dir / "upqi_v4.json"
            if not gt_path.exists():
                issues.append(f"{obj.get('cell_id')}:gt_missing")
                continue
            if not upqi_path.exists():
                issues.append(f"{obj.get('cell_id')}:upqi_missing")
                continue
            gt = load_json(gt_path)
            upqi = load_json(upqi_path)
            if validate_gt_obj is not None:
                gt_report = validate_gt_obj(gt, path=gt_path)
                gt_errors = list(gt_report.get("errors", [])) if isinstance(gt_report, Mapping) else ["gt_bad_report"]
                if gt_errors:
                    issues.append(f"{obj.get('cell_id')}:gt_dirty")
            if validate_upqi_obj is not None:
                upqi_report = validate_upqi_obj(upqi, path=upqi_path)
                upqi_errors = list(upqi_report.get("errors", [])) if isinstance(upqi_report, Mapping) else ["upqi_bad_report"]
                if upqi_errors:
                    issues.append(f"{obj.get('cell_id')}:upqi_dirty")
            if expected_v4_lock and gt.get("provenance", {}).get("source_lock_sha256") != expected_v4_lock:
                issues.append(f"{obj.get('cell_id')}:gt_lock_mismatch")
            if expected_v4_lock and upqi.get("provenance", {}).get("source_lock_sha256") != expected_v4_lock:
                issues.append(f"{obj.get('cell_id')}:upqi_lock_mismatch")
            if not manifests_match(obj, gt, "forget_manifest"):
                issues.append(f"{obj.get('cell_id')}:bridge_gt_manifest_mismatch")
            if not manifests_match(obj, upqi, "forget_manifest"):
                issues.append(f"{obj.get('cell_id')}:bridge_upqi_manifest_mismatch")
            bf16_issues = compare_bf16_to_gt(obj, gt, tol)
            if bf16_issues:
                issues.append(f"{obj.get('cell_id')}:" + ",".join(bf16_issues))
        except Exception as exc:
            issues.append(f"{obj.get('cell_id')}:join_exception:{type(exc).__name__}:{exc}")
    return len(issues), issues[:20]


def bootstrap_ci(values: Sequence[float], *, iters: int, seed: int = 20260802) -> Tuple[Any, Any]:
    if not values:
        return "", ""
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means: List[float] = []
    vals = list(values)
    for _ in range(max(100, iters)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = max(0, int(0.025 * len(means)) - 1)
    hi_idx = min(len(means) - 1, int(0.975 * len(means)))
    return means[lo_idx], means[hi_idx]


def stats_destroyer_rows(objects: Sequence[Mapping[str, Any]], bootstrap_iters: int) -> List[Dict[str, Any]]:
    full_by_cell = {
        str(o.get("cell_id")): float(metric(o, "deltas.harmful_recovery", 0.0))
        for o in objects
        if o.get("condition") == "full_quantized" and finite(metric(o, "deltas.harmful_recovery"))
    }
    protected: MutableMapping[Tuple[str, str, int], Mapping[str, Any]] = {}
    for obj in objects:
        if obj.get("condition") != "protected":
            continue
        if not finite(metric(obj, "deltas.harmful_recovery")):
            continue
        protected[(str(obj.get("cell_id")), str(obj.get("ranking_source")), int(obj.get("k", 0)))] = obj
    grouped: MutableMapping[Tuple[str, int], List[float]] = defaultdict(list)
    raw_counts: MutableMapping[Tuple[str, int], int] = defaultdict(int)
    for (cid, source, k), obj in protected.items():
        if source == "random":
            continue
        raw_counts[(source, k)] += 1
        random_obj = protected.get((cid, "random", k))
        if cid not in full_by_cell or random_obj is None:
            continue
        metric_reduction = full_by_cell[cid] - float(metric(obj, "deltas.harmful_recovery", 0.0))
        random_reduction = full_by_cell[cid] - float(metric(random_obj, "deltas.harmful_recovery", 0.0))
        grouped[(source, k)].append(metric_reduction - random_reduction)
    rows: List[Dict[str, Any]] = []
    for (source, k), advantages in sorted(grouped.items()):
        ci_lo, ci_hi = bootstrap_ci(advantages, iters=bootstrap_iters)
        wins = sum(1 for value in advantages if value > 0)
        status = "PASS" if len(advantages) >= 12 and isinstance(ci_lo, float) and ci_lo > 0 else "FAIL"
        issue = "" if status == "PASS" else "insufficient_or_nonpositive_metric_advantage_over_random"
        rows.append(
            {
                "ranking_source": source,
                "k": k,
                "candidate_pairs": raw_counts[(source, k)],
                "matched_random_pairs": len(advantages),
                "mean_advantage_over_random": statistics.mean(advantages) if advantages else "",
                "median_advantage_over_random": statistics.median(advantages) if advantages else "",
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "paired_win_rate": wins / len(advantages) if advantages else "",
                "status": status,
                "issue": issue,
            }
        )
    if not rows:
        rows.append(
            {
                "ranking_source": "NO_PROTECTED_MATCHED_DATA",
                "k": "",
                "candidate_pairs": 0,
                "matched_random_pairs": 0,
                "mean_advantage_over_random": "",
                "median_advantage_over_random": "",
                "bootstrap_ci_low": "",
                "bootstrap_ci_high": "",
                "paired_win_rate": "",
                "status": "FAIL",
                "issue": "no_metric_vs_random_cell_level_pairs",
            }
        )
    return rows


def standard_difference_rows(
    *,
    bridge_rows: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    smoke_gaps: Sequence[Mapping[str, Any]],
    ranking_counts: Counter,
    official_metrics_tsv: Optional[str],
) -> List[Dict[str, Any]]:
    full_clean = {str(o.get("cell_id")) for o in objects if o.get("condition") == "full_quantized"}
    protected = [o for o in objects if o.get("condition") == "protected"]
    smoke_fail = [r for r in smoke_gaps if r.get("status") == "FAIL"]
    has_gqdp = ranking_counts.get("GQDP_abs", 0) > 0 and ranking_counts.get("GQDP_signed", 0) > 0
    has_upqi = ranking_counts.get("UPQI_abs", 0) > 0 and ranking_counts.get("UPQI_signed", 0) > 0
    has_random = ranking_counts.get("random", 0) > 0
    rows = [
        {
            "standard_source": NEAREST_STANDARD["name"],
            "standard_url": NEAREST_STANDARD["url"],
            "standard_requirement": "Direct BF16 unlearned versus post-unlearning quantized behavior",
            "nearest_paper_behavior": "Compares full-precision unlearned behavior with quantized unlearned behavior and reports recovery after quantization.",
            "our_bridge_behavior": f"clean_full_quantized_cells={len(full_clean)}/144; non-BF16 bridge results embed BF16 reference when present.",
            "gap": "" if len(full_clean) == 144 else "full target-layer bridge baseline incomplete or absent",
            "severity": "BLOCKER" if len(full_clean) != 144 else "PASS",
            "required_repair": "" if len(full_clean) == 144 else "run and validate 144/144 full_quantized target-layer bridge results",
            "claim_allowed_if_unrepaired": "No bridge-baseline claim",
        },
        {
            "standard_source": NEAREST_STANDARD["name"],
            "standard_url": NEAREST_STANDARD["url"],
            "standard_requirement": "Observable generation or benchmark-facing recovery, not only layerwise proxy loss",
            "nearest_paper_behavior": "Uses unlearning metrics over model outputs and utility, not just internal layer diagnostics.",
            "our_bridge_behavior": "Bridge uses answer-only CE/logprob/TR and retain CE proxy; official benchmark TSV is " + ("provided" if official_metrics_tsv else "missing"),
            "gap": "" if official_metrics_tsv else "official TOFU/MUSE metrics missing",
            "severity": "BLOCKER" if not official_metrics_tsv else "PASS",
            "required_repair": "run official or faithful TOFU/MUSE metrics for claim-critical conditions",
            "claim_allowed_if_unrepaired": "Diagnostic bridge only, no benchmark unlearning-success claim",
        },
        {
            "standard_source": NEAREST_STANDARD["name"],
            "standard_url": NEAREST_STANDARD["url"],
            "standard_requirement": "Multiple quantizers and precisions are explicitly tested",
            "nearest_paper_behavior": "Studies quantization across methods and precision levels.",
            "our_bridge_behavior": "Runner supports nf4, gptq_int4, rtn_int8; smoke gaps=" + ",".join(str(r["requirement_id"]) for r in smoke_fail),
            "gap": "" if not smoke_fail else "smoke defaults do not cover every declared quantizer/model/equivalence condition",
            "severity": "REPAIR" if smoke_fail else "PASS",
            "required_repair": "patch smoke defaults to include RTN, LLaMA3, none/k0, and all_layers/k=max",
            "claim_allowed_if_unrepaired": "Production not recommended; smoke evidence incomplete",
        },
        {
            "standard_source": "MUSE",
            "standard_url": MUSE_URL,
            "standard_requirement": "Data-owner and deployer properties, including privacy/copyright, utility, scalability, sustainability",
            "nearest_paper_behavior": "Six-way benchmark separates memorization, privacy leakage, utility, scalability, and sustainability.",
            "our_bridge_behavior": "Bridge covers forget/retain proxies only.",
            "gap": "MUSE six-way official evidence missing",
            "severity": "BLOCKER" if not official_metrics_tsv else "REPAIR",
            "required_repair": "run MUSE official/fair metric panel or downgrade MUSE claims",
            "claim_allowed_if_unrepaired": "No MUSE benchmark claim",
        },
        {
            "standard_source": "TOFU",
            "standard_url": TOFU_URL,
            "standard_requirement": "Truth ratio, forget quality, retain utility, model utility remain separate",
            "nearest_paper_behavior": "Uses truth ratio, probability/ROUGE, forget quality, and model utility to evaluate approximate unlearning.",
            "our_bridge_behavior": "Bridge validator enforces log-space TR reportability; official TOFU metric table is " + ("provided" if official_metrics_tsv else "missing"),
            "gap": "" if official_metrics_tsv else "official TOFU claim gate missing",
            "severity": "BLOCKER" if not official_metrics_tsv else "PASS",
            "required_repair": "run TOFU official/fair metrics for claim-critical conditions",
            "claim_allowed_if_unrepaired": "No TOFU paper-ready claim",
        },
        {
            "standard_source": "GPTQ",
            "standard_url": GPTQ_URL,
            "standard_requirement": "GPTQ-style int4 must use calibration/Hessian information",
            "nearest_paper_behavior": "Post-training quantization uses approximate second-order information.",
            "our_bridge_behavior": "Runner requires calibration manifest and collects GPTQ Hessians for unprotected layers.",
            "gap": "",
            "severity": "PASS",
            "required_repair": "",
            "claim_allowed_if_unrepaired": "N/A",
        },
        {
            "standard_source": "QLoRA/NF4",
            "standard_url": QLORA_URL,
            "standard_requirement": "NF4 must be treated as NormalFloat 4-bit, not uniform int4",
            "nearest_paper_behavior": "NF4 is a distinct 4-bit data type designed for normally distributed weights.",
            "our_bridge_behavior": "Quantizer choices keep nf4 separate from gptq_int4 and rtn_int8.",
            "gap": "",
            "severity": "PASS",
            "required_repair": "",
            "claim_allowed_if_unrepaired": "N/A",
        },
        {
            "standard_source": "Integrated Gradients",
            "standard_url": IG_URL,
            "standard_requirement": "Path-attribution claims require a defined baseline, path, endpoint, and numerical integration discipline",
            "nearest_paper_behavior": "Defines attribution through axioms and an integral over a path.",
            "our_bridge_behavior": "UPQI is a v4 path-averaged quantization sensitivity metric; exact GQDP present=" + str(has_gqdp),
            "gap": "" if has_upqi else "UPQI ranking evidence missing; do not upgrade wording to exact IG proof",
            "severity": "REPAIR" if has_upqi else "BLOCKER",
            "required_repair": "keep UPQI_abs/UPQI_signed wording; compute exact GQDP before GQDP claims",
            "claim_allowed_if_unrepaired": "UPQI diagnostic claim only",
        },
        {
            "standard_source": "Bridge mitigation standard",
            "standard_url": NEAREST_STANDARD["url"],
            "standard_requirement": "Mitigation must beat deterministic random at matched budget",
            "nearest_paper_behavior": "Localization/protection is used to balance forgetting robustness and utility.",
            "our_bridge_behavior": f"protected_bridge_results={len(protected)}; random_ranking_rows={ranking_counts.get('random', 0)}",
            "gap": "" if protected and has_random else "no complete protected-vs-random mitigation evidence",
            "severity": "BLOCKER" if not protected else "REPAIR",
            "required_repair": "run protected metric and deterministic random at matched k/parameter fraction",
            "claim_allowed_if_unrepaired": "No mitigation claim",
        },
        {
            "standard_source": "Deployment-scope standard",
            "standard_url": NEAREST_STANDARD["url"],
            "standard_requirement": "Do not claim deployment quantization unless all deployable tensors/kernels are in scope",
            "nearest_paper_behavior": "Studies quantized model behavior as deployed/evaluated, not only selected layer replacement.",
            "our_bridge_behavior": "Current runner quantizes get_weight_layers target layer set.",
            "gap": "target-layer scope only",
            "severity": "REPAIR",
            "required_repair": "rename claims to target-layer bridge or add tensor-scope audit",
            "claim_allowed_if_unrepaired": "Target-layer diagnostic bridge only",
        },
    ]
    return rows


def claim_rows(
    *,
    args: argparse.Namespace,
    bridge_rows: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    hard_logs: Sequence[Mapping[str, Any]],
    smoke_gaps: Sequence[Mapping[str, Any]],
    ranking_counts: Counter,
    stats_rows: Sequence[Mapping[str, Any]],
    bridge_lock_path: str,
    bridge_lock_sha: str,
    bridge_lock_candidates: Sequence[str],
) -> List[Dict[str, Any]]:
    core = set(core_cell_ids())
    full_cells = {str(o.get("cell_id")) for o in objects if o.get("condition") == "full_quantized"}
    protected_cells = {str(o.get("cell_id")) for o in objects if o.get("condition") == "protected"}
    legacy_hits = [row for row in bridge_rows if any(frag in str(row.get("path", "")) for frag in LEGACY_FRAGMENTS)]
    v4_lock = load_source_lock_sha(Path(args.source_root) / "V4_SOURCE_LOCK.json")
    join_n_issues, join_examples = join_issue_count(objects, Path(args.v4_results_root), args.expected_v4_source_lock_sha256, args.bf16_tolerance) if objects else (0, [])
    source_lock_separate = bool(bridge_lock_sha) and bridge_lock_sha != v4_lock
    if args.expected_bridge_source_lock_sha256:
        source_lock_separate = source_lock_separate and bridge_lock_sha == args.expected_bridge_source_lock_sha256
    smoke_failed = [str(r["requirement_id"]) for r in smoke_gaps if r.get("status") == "FAIL"]
    gqdp_present = ranking_counts.get("GQDP_abs", 0) > 0 and ranking_counts.get("GQDP_signed", 0) > 0
    random_present = ranking_counts.get("random", 0) > 0
    simple_present = ranking_counts.get("displacement_norm", 0) > 0 or ranking_counts.get("weight_norm_l2", 0) > 0 or ranking_counts.get("SQNR", 0) > 0
    stats_pass = any(r.get("status") == "PASS" for r in stats_rows)
    rows = [
        {
            "claim_gate": "source_lock_separation",
            "status": "PASS" if source_lock_separate else "REJECT",
            "rejects_claim": "YES" if not source_lock_separate else "NO",
            "evidence": f"bridge_lock={bridge_lock_sha or '<missing>'}; v4_lock={v4_lock or '<missing>'}; candidates={len(bridge_lock_candidates)}",
            "blocker": "" if source_lock_separate else "bridge source lock missing, stale, same as v4 lock, or not expected",
            "required_repair": "create bridge-specific source lock after final audit scripts are copied",
            "claim_allowed_if_unrepaired": "No bridge provenance claim",
        },
        {
            "claim_gate": "no_legacy_paths",
            "status": "PASS" if not legacy_hits else "REJECT",
            "rejects_claim": "YES" if legacy_hits else "NO",
            "evidence": f"legacy_path_hits={len(legacy_hits)}",
            "blocker": "" if not legacy_hits else "v2/v3 bridge path contamination",
            "required_repair": "move legacy artifacts outside canonical bridge roots",
            "claim_allowed_if_unrepaired": "No v4 bridge claim",
        },
        {
            "claim_gate": "full_quantized_144",
            "status": "PASS" if core.issubset(full_cells) and not hard_logs else "REJECT",
            "rejects_claim": "YES" if not core.issubset(full_cells) or hard_logs else "NO",
            "evidence": f"full_quantized_cells={len(full_cells)}/144; hard_log_hits={len(hard_logs)}",
            "blocker": "" if core.issubset(full_cells) and not hard_logs else "missing full_quantized cells or hard logs present",
            "required_repair": "run/repair full_quantized bridge and quarantine stale error logs",
            "claim_allowed_if_unrepaired": "No bridge baseline claim",
        },
        {
            "claim_gate": "bf16_gt_join",
            "status": "PASS" if objects and join_n_issues == 0 else "REJECT",
            "rejects_claim": "YES" if not objects or join_n_issues else "NO",
            "evidence": f"bridge_files={len(objects)}; join_issues={join_n_issues}; examples={' | '.join(join_examples[:3])}",
            "blocker": "" if objects and join_n_issues == 0 else "bridge BF16 reference does not cleanly join to v4 GT/UPQI",
            "required_repair": "calibrate BF16 tolerance or repair bridge/GT/UPQI mismatch",
            "claim_allowed_if_unrepaired": "No bridge-to-layerwise evidence claim",
        },
        {
            "claim_gate": "smoke_coverage",
            "status": "PASS" if not smoke_failed else "REJECT",
            "rejects_claim": "YES" if smoke_failed else "NO",
            "evidence": "failed_smoke_requirements=" + ",".join(smoke_failed),
            "blocker": "" if not smoke_failed else "smoke defaults miss required scope",
            "required_repair": "patch smoke launcher to include RTN, LLaMA3, none/k0, all_layers/k=max",
            "claim_allowed_if_unrepaired": "Production not recommended",
        },
        {
            "claim_gate": "gptq_calibration",
            "status": "PASS" if not any(r.get("quantizer") == "gptq_int4" and r.get("status") == "FAIL" for r in bridge_rows) else "REJECT",
            "rejects_claim": "YES" if any(r.get("quantizer") == "gptq_int4" and r.get("status") == "FAIL" for r in bridge_rows) else "NO",
            "evidence": "bridge validator enforces calibration_manifest for gptq_int4",
            "blocker": "failing GPTQ bridge result" if any(r.get("quantizer") == "gptq_int4" and r.get("status") == "FAIL" for r in bridge_rows) else "",
            "required_repair": "repair GPTQ calibration/Hessian evidence",
            "claim_allowed_if_unrepaired": "No GPTQ bridge claim",
        },
        {
            "claim_gate": "exact_gqdp",
            "status": "PASS" if gqdp_present else "REJECT",
            "rejects_claim": "YES" if not gqdp_present else "NO",
            "evidence": f"GQDP_abs_rows={ranking_counts.get('GQDP_abs', 0)}; GQDP_signed_rows={ranking_counts.get('GQDP_signed', 0)}",
            "blocker": "" if gqdp_present else "exact endpoint GQDP not present in ranking evidence",
            "required_repair": "compute endpoint <grad_forget, Q(w)-w> before GQDP claims",
            "claim_allowed_if_unrepaired": "No GQDP claim; UPQI-only diagnostic allowed",
        },
        {
            "claim_gate": "random_and_simple_baselines",
            "status": "PASS" if random_present and simple_present else "REJECT",
            "rejects_claim": "YES" if not random_present or not simple_present else "NO",
            "evidence": f"random_rows={ranking_counts.get('random', 0)}; displacement_rows={ranking_counts.get('displacement_norm', 0)}; SQNR_rows={ranking_counts.get('SQNR', 0)}",
            "blocker": "" if random_present and simple_present else "random or simple baseline ranking missing",
            "required_repair": "add deterministic random and simple norm/noise/displacement baselines",
            "claim_allowed_if_unrepaired": "No predictor superiority claim",
        },
        {
            "claim_gate": "protected_mitigation_above_random",
            "status": "PASS" if protected_cells and stats_pass else "REJECT",
            "rejects_claim": "YES" if not protected_cells or not stats_pass else "NO",
            "evidence": f"protected_cells={len(protected_cells)}; stats_pass={stats_pass}",
            "blocker": "" if protected_cells and stats_pass else "no cell-level matched advantage over random",
            "required_repair": "run matched protected metric/random pairs and bootstrap cell-level advantage",
            "claim_allowed_if_unrepaired": "No mitigation claim",
        },
        {
            "claim_gate": "official_metrics",
            "status": "PASS" if args.official_metrics_tsv and Path(args.official_metrics_tsv).exists() else "REJECT",
            "rejects_claim": "YES" if not args.official_metrics_tsv or not Path(args.official_metrics_tsv).exists() else "NO",
            "evidence": args.official_metrics_tsv or "official metrics TSV missing",
            "blocker": "" if args.official_metrics_tsv and Path(args.official_metrics_tsv).exists() else "official/fair TOFU/MUSE metrics missing",
            "required_repair": "run claim-critical official metric panels",
            "claim_allowed_if_unrepaired": "Diagnostic bridge only; no paper-ready unlearning claim",
        },
        {
            "claim_gate": "target_layer_not_deployment_scope",
            "status": "REJECT_DEPLOYMENT_CLAIM",
            "rejects_claim": "YES",
            "evidence": "runner quantizes get_weight_layers target layer set",
            "blocker": "deployment tensor/kernel scope is not proven",
            "required_repair": "rename to target-layer bridge or add full tensor-scope/deployment audit",
            "claim_allowed_if_unrepaired": "Target-layer diagnostic bridge only",
        },
    ]
    return rows


def report_markdown(
    *,
    rows: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    standard_rows: Sequence[Mapping[str, Any]],
    smoke_rows: Sequence[Mapping[str, Any]],
    stats_rows: Sequence[Mapping[str, Any]],
    hard_logs: Sequence[Mapping[str, Any]],
    bridge_lock_path: str,
    bridge_lock_sha: str,
    outputs: Mapping[str, Path],
) -> str:
    status_counts = Counter(str(row.get("status", "")) for row in rows)
    claim_rejects = [row for row in claims if row.get("rejects_claim") == "YES"]
    standard_blockers = [row for row in standard_rows if row.get("severity") == "BLOCKER"]
    smoke_failed = [row for row in smoke_rows if row.get("status") == "FAIL"]
    stats_pass = [row for row in stats_rows if row.get("status") == "PASS"]
    verdict = "REJECT"
    if not claim_rejects and not standard_blockers and not smoke_failed and stats_pass and not hard_logs:
        verdict = "PASS"
    lines = [
        "# Bridge v4 Destroyer Audit",
        "",
        f"Verdict: {verdict}",
        "",
        "Default stance: reject unless every claim gate has direct evidence.",
        "",
        "## Nearest Standard",
        "",
        f"- {NEAREST_STANDARD['name']} ({NEAREST_STANDARD['venue']}): {NEAREST_STANDARD['url']}",
        "- Why this is the standard: it directly tests whether quantization restores forgotten behavior after LLM unlearning.",
        "",
        "## Counts",
        "",
        f"- Bridge files checked: {len(rows)}",
        f"- Bridge validator status counts: {dict(status_counts)}",
        f"- Claim blockers: {len(claim_rejects)}",
        f"- Standard blockers: {len(standard_blockers)}",
        f"- Smoke failures: {len(smoke_failed)}",
        f"- Hard log hits: {len(hard_logs)}",
        f"- Bridge lock path: {bridge_lock_path or '<missing>'}",
        f"- Bridge lock sha: {bridge_lock_sha or '<missing>'}",
        "",
        "## Immediate Blocking Findings",
        "",
    ]
    if not claim_rejects:
        lines.append("- None from claim gates.")
    else:
        for row in claim_rejects:
            lines.append(f"- {row['claim_gate']}: {row['blocker']} | repair: {row['required_repair']}")
    lines.extend(["", "## Pessimistic Conclusion", ""])
    if verdict == "PASS":
        lines.append("The bridge survives the destroyer audit structurally. Paper claims still need careful wording and official metric alignment.")
    else:
        lines.append("The bridge is not paper-ready under the destroyer standard. Use it only as a diagnostic target-layer bridge until blockers are repaired.")
    lines.extend(["", "## Output Files", ""])
    for key, path in outputs.items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def repair_plan_markdown(claims: Sequence[Mapping[str, Any]], smoke_rows: Sequence[Mapping[str, Any]]) -> str:
    failing_claims = [row for row in claims if row.get("rejects_claim") == "YES"]
    failing_smoke = [row for row in smoke_rows if row.get("status") == "FAIL"]
    repairs: List[str] = []
    for row in failing_claims:
        repair = str(row.get("required_repair", "")).strip()
        if repair and repair not in repairs:
            repairs.append(repair)
    for row in failing_smoke:
        repair = str(row.get("required_repair", "")).strip()
        if repair and repair not in repairs:
            repairs.append(repair)
    if not repairs:
        repairs.append("No blocking repairs detected by this run; rerun after new bridge outputs arrive.")
    lines = [
        "# Bridge v4 Repair Plan From Destroyer Audit",
        "",
        "Order is strict. Do not spend production compute on later items before earlier blockers are cleared.",
        "",
    ]
    for idx, repair in enumerate(repairs, start=1):
        lines.append(f"{idx}. {repair}.")
    lines.extend(
        [
            "",
            "Claim boundary that stays in force:",
            "",
            "- Bridge-only evidence supports target-layer diagnostic claims.",
            "- GQDP claims require exact endpoint GQDP evidence.",
            "- Mitigation claims require matched random at cell level.",
            "- Paper-ready unlearning claims require official TOFU/MUSE metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    prefix = now_prefix(args.out_prefix)
    source_root = Path(args.source_root)
    bridge_root = Path(args.bridge_root)
    slurm_root = Path(args.slurm_root)
    rankings_path = Path(args.rankings_path)

    bridge_lock_path, bridge_lock_sha, bridge_lock_candidates = newest_bridge_lock(source_root)
    expected_bridge_lock = args.expected_bridge_source_lock_sha256 or bridge_lock_sha or None
    smoke_rows = smoke_gap_rows(source_root, slurm_root)
    bridge_rows, objects = load_bridge_objects(bridge_root, expected_bridge_lock)
    hard_logs = scan_hard_logs(Path(args.logs_root))
    ranking_counts = load_ranking_sources(rankings_path)
    stats_rows = stats_destroyer_rows(objects, args.bootstrap_iters)
    standard_rows = standard_difference_rows(
        bridge_rows=bridge_rows,
        objects=objects,
        smoke_gaps=smoke_rows,
        ranking_counts=ranking_counts,
        official_metrics_tsv=args.official_metrics_tsv,
    )
    claims = claim_rows(
        args=args,
        bridge_rows=bridge_rows,
        objects=objects,
        hard_logs=hard_logs,
        smoke_gaps=smoke_rows,
        ranking_counts=ranking_counts,
        stats_rows=stats_rows,
        bridge_lock_path=bridge_lock_path,
        bridge_lock_sha=bridge_lock_sha,
        bridge_lock_candidates=bridge_lock_candidates,
    )
    artifacts = artifact_rows(args, smoke_rows)

    outputs = {
        "report": out_dir / f"BRIDGE_V4_DESTROYER_AUDIT_{prefix}.md",
        "standard_difference": out_dir / f"bridge_v4_standard_difference_matrix_{prefix}.tsv",
        "file_matrix": out_dir / f"bridge_v4_destroyer_file_matrix_{prefix}.tsv",
        "claim_rejection": out_dir / f"bridge_v4_claim_rejection_matrix_{prefix}.tsv",
        "smoke_gap": out_dir / f"bridge_v4_smoke_gap_matrix_{prefix}.tsv",
        "stats": out_dir / f"bridge_v4_stats_destroyer_{prefix}.tsv",
        "repair_plan": out_dir / f"BRIDGE_V4_REPAIR_PLAN_FROM_DESTROYER_{prefix}.md",
        "hard_logs": out_dir / f"bridge_v4_destroyer_hard_logs_{prefix}.tsv",
    }

    write_tsv(outputs["standard_difference"], standard_rows, [
        "standard_source", "standard_url", "standard_requirement", "nearest_paper_behavior",
        "our_bridge_behavior", "gap", "severity", "required_repair", "claim_allowed_if_unrepaired",
    ])
    write_tsv(outputs["file_matrix"], artifacts, ["role", "path", "exists", "size", "mtime", "sha256", "issues"])
    write_tsv(outputs["claim_rejection"], claims, [
        "claim_gate", "status", "rejects_claim", "evidence", "blocker", "required_repair", "claim_allowed_if_unrepaired",
    ])
    write_tsv(outputs["smoke_gap"], smoke_rows, ["requirement_id", "requirement", "status", "evidence", "required_repair"])
    write_tsv(outputs["stats"], stats_rows, [
        "ranking_source", "k", "candidate_pairs", "matched_random_pairs", "mean_advantage_over_random",
        "median_advantage_over_random", "bootstrap_ci_low", "bootstrap_ci_high", "paired_win_rate", "status", "issue",
    ])
    write_tsv(outputs["hard_logs"], hard_logs, ["path", "lineno", "line"])
    write_text(outputs["repair_plan"], repair_plan_markdown(claims, smoke_rows))
    write_text(
        outputs["report"],
        report_markdown(
            rows=bridge_rows,
            claims=claims,
            standard_rows=standard_rows,
            smoke_rows=smoke_rows,
            stats_rows=stats_rows,
            hard_logs=hard_logs,
            bridge_lock_path=bridge_lock_path,
            bridge_lock_sha=bridge_lock_sha,
            outputs=outputs,
        ),
    )

    for key, path in outputs.items():
        print(f"WROTE {key} {path}")
    reject_count = sum(1 for row in claims if row.get("rejects_claim") == "YES")
    print(f"DESTROYER_VERDICT {'REJECT' if reject_count else 'PASS'} claim_blockers={reject_count} hard_logs={len(hard_logs)}")
    if args.fail_on_reject and reject_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
