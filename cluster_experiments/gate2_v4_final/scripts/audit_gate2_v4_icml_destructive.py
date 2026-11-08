#!/usr/bin/env python3
"""ICML-grade destructive audit for Gate 2 v4.

This is a torch-free, result-file audit. It does not recompute model outputs;
it verifies that the produced artifacts satisfy the v4 contracts and maps them
to paper-facing standards before a claim is allowed into the main submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_v4_claim_gate_matrix import row_for as claim_gate_row_for  # noqa: E402
from gate2_v4_contract import (  # noqa: E402
    QUANTIZERS,
    SEEDS,
    canonical_json,
    iter_core_cells,
    load_json,
    sha256_file,
    sha256_text,
)
from validate_gate2_v4_cell_join import validate_pair  # noqa: E402
from validate_gt_v4 import validate_obj as validate_gt_obj  # noqa: E402
from validate_upqi_v4 import validate_obj as validate_upqi_obj  # noqa: E402


PAPER_STANDARDS: Sequence[Mapping[str, str]] = (
    {
        "standard_id": "tofu_truth_ratio",
        "anchor": "TOFU",
        "citation": "Maini et al., TOFU: A Task of Fictitious Unlearning for LLMs, arXiv:2401.06121",
        "requirement": "Forgetting evaluation must expose answer-vs-perturbation behavior; truth-ratio-style evidence must be reportable, not silently missing.",
        "audit_check": "GT v4 hard truth-ratio policy, per-sample sentry, per-layer reportability, log-space truth ratio, no fallback labels.",
        "stage": "gt",
    },
    {
        "standard_id": "muse_benchmark_coverage",
        "anchor": "MUSE",
        "citation": "Shi et al., MUSE: Machine Unlearning Six-way Evaluation for LLMs, arXiv:2407.06460",
        "requirement": "MUSE claims require both forget-side and retain/utility-style evidence; diagnostics alone cannot become final unlearning success claims.",
        "audit_check": "Claim gate blocks paper-ready status until official metrics pass; reviewer report marks missing metrics as paper-blocking.",
        "stage": "metrics",
    },
    {
        "standard_id": "npo_unlearning_tradeoff",
        "anchor": "NPO/LLM-unlearning",
        "citation": "Zhang et al., Negative Preference Optimization for LLM Unlearning, arXiv:2404.05868",
        "requirement": "Unlearning methods must be interpreted through forget/retain tradeoffs, not a single forget-side scalar.",
        "audit_check": "Claim gate separates GT-only, GT+UPQI, official metrics, and paper-ready; final report forbids UPQI-only success claims.",
        "stage": "claim",
    },
    {
        "standard_id": "gptq_calibration",
        "anchor": "GPTQ",
        "citation": "Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, arXiv:2210.17323",
        "requirement": "GPTQ cells must use a real calibration manifest and identify GPTQ-specific quantization rather than generic int4.",
        "audit_check": "Every gptq_int4 GT/UPQI cell must carry calibration manifest SHA; non-GPTQ cells must not carry one.",
        "stage": "gt_upqi",
    },
    {
        "standard_id": "nf4_baseline",
        "anchor": "QLoRA/NF4",
        "citation": "Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, arXiv:2305.14314",
        "requirement": "NF4 must be treated as a distinct NormalFloat baseline, not collapsed with RTN or GPTQ.",
        "audit_check": "Core matrix requires all three quantizers for every model/algo/seed and reports quantizer-level counts.",
        "stage": "gt_upqi",
    },
    {
        "standard_id": "integrated_gradients_axioms",
        "anchor": "Integrated Gradients",
        "citation": "Sundararajan et al., Axiomatic Attribution for Deep Networks, arXiv:1703.01365",
        "requirement": "UPQI path-integral claims require signed path integral and absolute-magnitude diagnostic with finite gradients and a fixed quadrature rule.",
        "audit_check": "UPQI v4 enforces n_quadrature, finite gradients/losses, complete layer order, and upqi_abs >= abs(upqi_signed).",
        "stage": "upqi",
    },
    {
        "standard_id": "hazard_unlearning_skepticism",
        "anchor": "WMDP",
        "citation": "Li et al., The WMDP Benchmark, arXiv:2403.03218",
        "requirement": "Do not equate local diagnostic movement with robust unlearning; robustness claims need broader evaluations.",
        "audit_check": "Reviewer 4 blocks paper-ready success claims without official metrics and explicit limitation language.",
        "stage": "claim",
    },
    {
        "standard_id": "generative_ai_unlearning_limits",
        "anchor": "Generative-AI unlearning limitations",
        "citation": "Recent generative-AI unlearning critiques, arXiv:2412.06966",
        "requirement": "Claims must distinguish actual knowledge removal from output suppression or measurement artifacts.",
        "audit_check": "Final reviewer panel treats GT/UPQI as diagnostics until joined with official metrics and claim-gate evidence.",
        "stage": "claim",
    },
)

HARD_LOG_PATTERNS = (
    "Traceback",
    "zero answer",
    "source lock",
    "script SHA",
    "No checkpoint directory",
    "CUDA out",
    "OUT_OF_MEMORY",
    "validation failed",
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json_if_exists(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def current_lock_sha(source_lock: Path) -> str:
    obj = load_json(source_lock)
    return str(obj.get("source_lock_sha256", ""))


def manifest_status(manifest_path: Path, expected_n: int, expected_lock: str | None = None) -> Tuple[str, List[str]]:
    issues: List[str] = []
    if not manifest_path.exists():
        return "missing", ["manifest_missing"]
    try:
        obj = load_json(manifest_path)
    except Exception as exc:
        return "dirty", [f"bad_json:{exc}"]
    if not obj.get("manifest_sha256"):
        issues.append("manifest_sha_missing")
    samples = obj.get("samples")
    if not isinstance(samples, list):
        issues.append("samples_not_list")
        samples = []
    if len(samples) < expected_n:
        issues.append(f"sample_count_short:{len(samples)}<{expected_n}")
    if "forget" in manifest_path.name:
        bad_pert = 0
        missing_hash = 0
        for sample in samples[:expected_n]:
            perts = sample.get("perturbations") if isinstance(sample, Mapping) else None
            if not isinstance(perts, list) or not perts:
                bad_pert += 1
                continue
            for perturbation in perts:
                if not isinstance(perturbation, Mapping) or not str(perturbation.get("answer", "")).strip():
                    bad_pert += 1
                    break
                if not perturbation.get("perturbation_hash"):
                    missing_hash += 1
                    break
        if bad_pert:
            issues.append(f"truth_ratio_candidates_bad:{bad_pert}")
        if missing_hash:
            issues.append(f"perturbation_hash_missing:{missing_hash}")
    if expected_lock and obj.get("source_lock_sha256") not in (None, expected_lock):
        issues.append("manifest_source_lock_mismatch")
    return ("pass" if not issues else "dirty"), issues


def complete_lock_ok(obj: Mapping[str, Any], expected_lock: str) -> bool:
    return obj.get("provenance", {}).get("source_lock_sha256") == expected_lock


def partial_progress(path: Path, expected_lock: str) -> Tuple[str, str, int, int]:
    if not path.exists():
        return "missing", "missing", 0, 224
    try:
        obj = load_json(path)
    except Exception as exc:
        return "dirty_partial", f"bad_json:{exc}", 0, 224
    if not complete_lock_ok(obj, expected_lock):
        return "dirty_partial", "source_lock_mismatch", 0, int(obj.get("n_layers", 224) or 224)
    per = obj.get("per_layer")
    done = len(per) if isinstance(per, list) else int(obj.get("n_layers_completed", 0) or 0)
    total = int(obj.get("n_layers", 224) or 224)
    return "partial", f"{done}/{total}", done, max(0, total - done)


def stage_cell_row(
    *,
    cell: Mapping[str, str],
    results_root: Path,
    expected_lock: str,
    stage: str,
    validate_fn: Callable[[Mapping[str, Any], Path], Mapping[str, Any]],
    complete_name: str,
    partial_name: str,
) -> Dict[str, Any]:
    cell_dir = results_root / cell["cell_id"]
    complete = cell_dir / complete_name
    partial = cell_dir / partial_name
    issues: List[str] = []
    warnings: List[str] = []
    done_layers = 0
    remaining_layers = 224
    status = "missing"
    if complete.exists():
        try:
            obj = load_json(complete)
            report = validate_fn(obj, complete)
        except Exception as exc:
            obj = {}
            report = {"passed": False, "errors": [f"bad_json:{exc}"], "warnings": []}
        if not complete_lock_ok(obj, expected_lock):
            issues.append("source_lock_mismatch")
        issues.extend(str(x) for x in report.get("errors", []))
        warnings.extend(str(x) for x in report.get("warnings", []))
        if report.get("passed") and not issues:
            status = "clean_complete"
            done_layers = int(obj.get("n_layers", report.get("n_layers", 224)) or 224)
            remaining_layers = 0
        else:
            status = "dirty_complete"
            done_layers = int(obj.get("n_layers_completed", 0) or 0)
            remaining_layers = max(0, int(obj.get("n_layers", 224) or 224) - done_layers)
        if stage == "upqi":
            extra = extra_upqi_checks(obj)
            issues.extend(extra)
            if extra and status == "clean_complete":
                status = "dirty_complete"
        elif stage == "gt":
            extra = extra_gt_checks(obj)
            issues.extend(extra)
            if extra and status == "clean_complete":
                status = "dirty_complete"
    else:
        status, reason, done_layers, remaining_layers = partial_progress(partial, expected_lock)
        if status == "dirty_partial":
            issues.append(reason)
        elif status == "partial":
            warnings.append(reason)
    return {
        **cell,
        "stage": stage,
        "status": status,
        "passed": "YES" if status == "clean_complete" else "NO",
        "issues": ";".join(dict.fromkeys(issues)),
        "warnings": ";".join(dict.fromkeys(warnings)),
        "done_layers": str(done_layers),
        "remaining_layers": str(remaining_layers),
        "path": str(complete),
    }


def extra_upqi_checks(obj: Mapping[str, Any]) -> List[str]:
    issues: List[str] = []
    q = obj.get("quantizer")
    if q not in QUANTIZERS:
        issues.append(f"bad_quantizer:{q}")
    if int(obj.get("n_quadrature", -1)) != 8:
        issues.append(f"bad_n_quadrature:{obj.get('n_quadrature')}")
    per_layer = obj.get("per_layer")
    if isinstance(per_layer, list):
        seen: set[Any] = set()
        for idx, row in enumerate(per_layer):
            if not isinstance(row, Mapping):
                issues.append(f"bad_layer_entry:{idx}")
                continue
            layer_index = row.get("layer_index")
            if layer_index in seen:
                issues.append(f"duplicate_layer_index:{layer_index}")
            seen.add(layer_index)
            if int(row.get("n_quadrature", -1)) != 8:
                issues.append(f"bad_layer_n_quadrature:{idx}")
            if int(row.get("n_answer_tokens", 0)) <= 0:
                issues.append(f"zero_answer_tokens:{idx}")
    return issues


def extra_gt_checks(obj: Mapping[str, Any]) -> List[str]:
    issues: List[str] = []
    if obj.get("truth_ratio_policy") != "hard_required":
        issues.append("truth_ratio_not_hard_required")
    if "delta_log_AP" in obj.get("secondary_targets", []):
        issues.append("forbidden_delta_log_AP_secondary_target")
    evidence = obj.get("bf16_per_sample_truth_ratio_evidence")
    if not isinstance(evidence, list) or len(evidence) != int(obj.get("n_samples", -1)):
        issues.append("bf16_truth_ratio_evidence_not_full")
    layers = obj.get("layers")
    if isinstance(layers, Mapping):
        for layer, row in layers.items():
            if not isinstance(row, Mapping):
                continue
            if row.get("truth_ratio_reportable_bf16") is not True:
                issues.append(f"bf16_tr_not_reportable:{layer}")
            if row.get("truth_ratio_reportable_quantized") is not True:
                issues.append(f"quant_tr_not_reportable:{layer}")
            if "delta_AP" in row or "delta_log_AP" in row:
                issues.append(f"forbidden_ap_alias:{layer}")
    return issues


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_freeze(source_root: Path, manifests_root: Path, source_lock: Path) -> Dict[str, Any]:
    source_files = sorted(p for p in source_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    manifest_files = sorted(p for p in manifests_root.rglob("*.json") if p.is_file()) if manifests_root.exists() else []
    lock_obj = load_json(source_lock) if source_lock.exists() else {}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "manifest_root": str(manifests_root),
        "source_lock_path": str(source_lock),
        "source_lock_sha256": lock_obj.get("source_lock_sha256"),
        "source_files": [{"path": str(p), "sha256": sha256_file(p)} for p in source_files],
        "manifest_files": [{"path": str(p), "sha256": sha256_file(p)} for p in manifest_files],
        "freeze_sha256": sha256_text(canonical_json({
            "source": [(str(p), sha256_file(p)) for p in source_files],
            "manifests": [(str(p), sha256_file(p)) for p in manifest_files],
            "source_lock": lock_obj.get("source_lock_sha256"),
        })),
    }


def add_optional_file_to_freeze(freeze: MutableMapping[str, Any], label: str, path_text: str) -> None:
    if not path_text:
        return
    path = Path(path_text)
    entry: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        entry["sha256"] = sha256_file(path)
        entry["size"] = path.stat().st_size
    freeze.setdefault("external_snapshots", {})[label] = entry


def scan_logs(logs_root: Path, job_ids: Sequence[str]) -> List[Dict[str, str]]:
    if not logs_root.exists():
        return [{"path": str(logs_root), "pattern": "logs_root_missing", "line": ""}]
    includes = {".log", ".out", ".err"}
    paths = [p for p in logs_root.rglob("*") if p.is_file() and p.suffix in includes]
    if job_ids:
        paths = [p for p in paths if any(job_id in p.name for job_id in job_ids)]
    rows: List[Dict[str, str]] = []
    pattern = re.compile("|".join(re.escape(x) for x in HARD_LOG_PATTERNS), re.IGNORECASE)
    for path in sorted(paths):
        try:
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    rows.append({"path": str(path), "line_no": str(lineno), "line": line[:500]})
        except Exception as exc:
            rows.append({"path": str(path), "line_no": "0", "line": f"unreadable:{exc}"})
    return rows


def paper_standard_rows(
    gt_rows: Sequence[Mapping[str, Any]],
    upqi_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    log_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    counts = {
        "upqi_clean": sum(1 for r in upqi_rows if r["status"] == "clean_complete"),
        "gt_clean": sum(1 for r in gt_rows if r["status"] == "clean_complete"),
        "join_passed": sum(1 for r in claim_rows if r["join_status"] == "join_passed"),
        "paper_ready": sum(1 for r in claim_rows if r["paper_ready"] == "YES"),
        "manifest_pass": sum(1 for r in manifest_rows if r["status"] == "pass"),
        "log_hits": len(log_rows),
    }
    rows: List[Dict[str, str]] = []
    for standard in PAPER_STANDARDS:
        sid = standard["standard_id"]
        if sid == "tofu_truth_ratio":
            passed = counts["gt_clean"] == 144 and all(not r["issues"] for r in gt_rows)
        elif sid == "muse_benchmark_coverage":
            passed = counts["paper_ready"] == 144
        elif sid == "npo_unlearning_tradeoff":
            passed = counts["paper_ready"] == 144
        elif sid == "gptq_calibration":
            passed = not any("gptq" in r.get("issues", "") for r in list(gt_rows) + list(upqi_rows))
        elif sid == "nf4_baseline":
            passed = quantizer_balance_ok(gt_rows) and quantizer_balance_ok(upqi_rows)
        elif sid == "integrated_gradients_axioms":
            passed = counts["upqi_clean"] == 144 and not any(r["issues"] for r in upqi_rows)
        elif sid == "hazard_unlearning_skepticism":
            passed = counts["paper_ready"] == 144
        elif sid == "generative_ai_unlearning_limits":
            passed = counts["paper_ready"] == 144
        else:
            passed = False
        rows.append({
            **{k: str(v) for k, v in standard.items()},
            "status": "PASS" if passed else "BLOCKED",
            "evidence": f"upqi_clean={counts['upqi_clean']}/144 gt_clean={counts['gt_clean']}/144 join={counts['join_passed']}/144 paper_ready={counts['paper_ready']}/144 manifests_pass={counts['manifest_pass']} log_hits={counts['log_hits']}",
        })
    return rows


def quantizer_balance_ok(rows: Sequence[Mapping[str, Any]]) -> bool:
    clean = [r for r in rows if r["status"] == "clean_complete"]
    if len(clean) != 144:
        return False
    counts = Counter(r["quantizer"] for r in clean)
    return all(counts[q] == 48 for q in QUANTIZERS)


def reviewer_report(
    *,
    gt_rows: Sequence[Mapping[str, Any]],
    upqi_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
    standard_rows: Sequence[Mapping[str, str]],
    log_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[str, str]]:
    gt_clean = sum(1 for r in gt_rows if r["status"] == "clean_complete")
    upqi_clean = sum(1 for r in upqi_rows if r["status"] == "clean_complete")
    joins = sum(1 for r in claim_rows if r["join_status"] == "join_passed")
    paper_ready = sum(1 for r in claim_rows if r["paper_ready"] == "YES")
    standards_blocked = [r["standard_id"] for r in standard_rows if r["status"] != "PASS"]
    reports = {
        "Reviewer 1 - Methodology destroyer": {
            "verdict": "acceptable" if upqi_clean == 144 and "integrated_gradients_axioms" not in standards_blocked else "repair",
            "finding": "UPQI is methodologically usable only if every layer is finite, source-locked, and satisfies the path-integral signed/absolute contract; otherwise it remains diagnostic-only.",
        },
        "Reviewer 2 - Implementation destroyer": {
            "verdict": "acceptable" if not log_rows and gt_clean in (0, 144) and upqi_clean == 144 else "repair",
            "finding": "Implementation claims are blocked by stale logs, partials, source-lock mismatches, or any v2/v3 contamination. Incomplete GT is allowed only as not-yet-ready, not paper evidence.",
        },
        "Reviewer 3 - Experimental validity destroyer": {
            "verdict": "acceptable" if paper_ready == 144 else "repair",
            "finding": "The 144-cell core is balanced only when every model/algo/seed/quantizer cell is clean and official metrics exist; GT+UPQI alone cannot support final performance claims.",
        },
        "Reviewer 4 - Unlearning skepticism destroyer": {
            "verdict": "acceptable" if paper_ready == 144 else "repair",
            "finding": "No result should be phrased as true unlearning success until official TOFU/MUSE metrics and strict joins pass; UPQI/GT are robustness diagnostics, not proof of removal.",
        },
    }
    if log_rows:
        reports["Reviewer 2 - Implementation destroyer"]["verdict"] = "reject"
    if upqi_clean == 0 and gt_clean == 0:
        for block in reports.values():
            block["verdict"] = "reject"
    if gt_clean < 144 or upqi_clean < 144 or joins < 144:
        for block in reports.values():
            if block["verdict"] == "acceptable":
                block["verdict"] = "repair"
    return reports


def write_markdown_report(
    path: Path,
    *,
    title: str,
    expected_lock: str,
    gt_rows: Sequence[Mapping[str, Any]],
    upqi_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
    standard_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, Any]],
    log_rows: Sequence[Mapping[str, str]],
    reviewer_rows: Mapping[str, Mapping[str, str]],
    freeze: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_counts = Counter(r["status"] for r in gt_rows)
    upqi_counts = Counter(r["status"] for r in upqi_rows)
    claim_counts = Counter(r["claim_gate"] for r in claim_rows)
    ready_counts = Counter(r["paper_ready"] for r in claim_rows)
    standard_counts = Counter(r["status"] for r in standard_rows)
    verdict = "ACCEPTABLE_FOR_PAPER_READY_CLAIMS" if ready_counts.get("YES", 0) == 144 and not log_rows else "NOT_PAPER_READY"
    lines = [
        f"# {title}",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Expected source lock: `{expected_lock}`",
        f"Evidence freeze SHA: `{freeze.get('freeze_sha256')}`",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "This audit is intentionally destructive. Any failure is paper-blocking unless the affected claim is removed from the main paper or explicitly downgraded.",
        "",
        "## Counts",
        "",
        f"- UPQI: `{dict(upqi_counts)}`",
        f"- GT: `{dict(gt_counts)}`",
        f"- Claim gates: `{dict(claim_counts)}`",
        f"- Paper ready: `{dict(ready_counts)}`",
        f"- Paper standards: `{dict(standard_counts)}`",
        f"- Hard log hits: `{len(log_rows)}`",
        "",
        "## Paper-Standard Matrix",
        "",
        "| Standard | Anchor | Status | Audit check | Evidence |",
        "|---|---|---:|---|---|",
    ]
    for row in standard_rows:
        lines.append(f"| `{row['standard_id']}` | {row['anchor']} | **{row['status']}** | {row['audit_check']} | `{row['evidence']}` |")
    lines.extend(["", "## Reviewer Panel", ""])
    for reviewer, report in reviewer_rows.items():
        lines.append(f"### {reviewer}")
        lines.append("")
        lines.append(f"- Verdict: **{report['verdict']}**")
        lines.append(f"- Finding: {report['finding']}")
        lines.append("")
    lines.extend(["## Blocking Failures", ""])
    failures = [r for r in list(upqi_rows) + list(gt_rows) if r.get("issues")]
    if failures:
        for row in failures[:80]:
            lines.append(f"- `{row['stage']}:{row['cell_id']}` `{row['status']}`: {row['issues']}")
        if len(failures) > 80:
            lines.append(f"- ... {len(failures) - 80} more failures in TSV outputs.")
    else:
        lines.append("- No stage-level contract failures in completed clean outputs.")
    if log_rows:
        lines.extend(["", "## Hard Log Hits", ""])
        for row in log_rows[:80]:
            lines.append(f"- `{row['path']}:{row['line_no']}` {row['line']}")
        if len(log_rows) > 80:
            lines.append(f"- ... {len(log_rows) - 80} more log hits in log audit TSV.")
    lines.extend([
        "",
        "## Paper Claim Rules",
        "",
        "- UPQI-only outputs may support diagnostic quantization-sensitivity claims only.",
        "- GT-only outputs may support target-definition and truth-ratio/reportability checks only.",
        "- GT+UPQI joins may support layerwise robustness association claims only.",
        "- Official TOFU/MUSE metrics are required before any paper-ready unlearning-success claim.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICML-grade destructive Gate 2 v4 audit")
    parser.add_argument("--results_root", default="results/gate2_v4_standard")
    parser.add_argument("--metrics_root", default="results/gate2_v4_metrics")
    parser.add_argument("--logs_root", default="logs/gate2_v4")
    parser.add_argument("--manifests_root", default="manifests/gate2_v4")
    parser.add_argument("--source_root", default="gate2_v4_final")
    parser.add_argument("--source_lock", default="gate2_v4_final/V4_SOURCE_LOCK.json")
    parser.add_argument("--expected_source_lock_sha256", default="")
    parser.add_argument("--job_ids", default="", help="Comma-separated canonical job IDs; blank scans all v4 logs")
    parser.add_argument("--seeds", default=",".join(SEEDS))
    parser.add_argument("--output_prefix", default="")
    parser.add_argument("--queue_snapshot", default="", help="Optional squeue/sacct snapshot file to hash into the evidence freeze")
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = now_stamp()
    prefix = args.output_prefix or f"gate2_v4_icml_destructive_audit_{stamp}"
    results_root = Path(args.results_root)
    metrics_root = Path(args.metrics_root)
    logs_root = Path(args.logs_root)
    manifests_root = Path(args.manifests_root)
    source_root = Path(args.source_root)
    source_lock = Path(args.source_lock)
    for role, path in (
        ("results_root", results_root),
        ("logs_root", logs_root),
        ("manifests_root", manifests_root),
        ("source_root", source_root),
    ):
        if "gate2_v4" not in str(path).replace("\\", "/"):
            raise SystemExit(f"Refusing non-v4 {role}: {path}")
    expected_lock = args.expected_source_lock_sha256 or current_lock_sha(source_lock)
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    cells = list(iter_core_cells(seeds))
    job_ids = [x.strip() for x in args.job_ids.split(",") if x.strip()]

    upqi_rows = [
        stage_cell_row(
            cell=cell,
            results_root=results_root,
            expected_lock=expected_lock,
            stage="upqi",
            validate_fn=validate_upqi_obj,
            complete_name="upqi_v4.json",
            partial_name="upqi_v4.json.partial",
        )
        for cell in cells
    ]
    gt_rows = [
        stage_cell_row(
            cell=cell,
            results_root=results_root,
            expected_lock=expected_lock,
            stage="gt",
            validate_fn=validate_gt_obj,
            complete_name="ground_truth_v4.json",
            partial_name="ground_truth_v4.json.partial",
        )
        for cell in cells
    ]
    claim_rows = [claim_gate_row_for(results_root, metrics_root, cell) for cell in cells]

    manifest_rows: List[Dict[str, Any]] = []
    for path in sorted(manifests_root.glob("*_core128_seed*.json")) + sorted(manifests_root.glob("*_gptq_calib128_seed*.json")):
        status, issues = manifest_status(path, expected_n=128)
        manifest_rows.append({"path": str(path), "status": status, "issues": ";".join(issues)})
    log_rows = scan_logs(logs_root, job_ids)
    freeze = source_freeze(source_root, manifests_root, source_lock)
    add_optional_file_to_freeze(freeze, "queue_snapshot", args.queue_snapshot)
    standard_rows = paper_standard_rows(gt_rows, upqi_rows, claim_rows, manifest_rows, log_rows)
    reviewer_rows = reviewer_report(
        gt_rows=gt_rows,
        upqi_rows=upqi_rows,
        claim_rows=claim_rows,
        standard_rows=standard_rows,
        log_rows=log_rows,
    )

    if args.output_prefix:
        out_prefix = Path(prefix)
        upqi_tsv = out_prefix.with_name(out_prefix.name + "_upqi_v4_icml_brutal_cells.tsv")
        gt_tsv = out_prefix.with_name(out_prefix.name + "_gt_v4_icml_brutal_cells.tsv")
        claim_tsv = out_prefix.with_name(out_prefix.name + "_gate2_v4_claim_gate_matrix.tsv")
        standard_tsv = out_prefix.with_name(out_prefix.name + "_gate2_v4_paper_standard_matrix.tsv")
        log_tsv = out_prefix.with_name(out_prefix.name + "_gate2_v4_hard_log_audit.tsv")
        manifest_tsv = out_prefix.with_name(out_prefix.name + "_gate2_v4_manifest_audit.tsv")
        freeze_json = out_prefix.with_name(out_prefix.name + "_gate2_v4_evidence_freeze.json")
        upqi_md = out_prefix.with_name(out_prefix.name + "_UPQI_V4_ICML_BRUTAL_AUDIT.md")
        reviewer_md = out_prefix.with_name(out_prefix.name + "_GATE2_V4_FINAL_REVIEWER_PANEL_REPORT.md")
    else:
        upqi_tsv = Path(f"upqi_v4_icml_brutal_cells_{stamp}.tsv")
        gt_tsv = Path(f"gt_v4_icml_brutal_cells_{stamp}.tsv")
        claim_tsv = Path(f"gate2_v4_claim_gate_matrix_{stamp}.tsv")
        standard_tsv = Path(f"gate2_v4_paper_standard_matrix_{stamp}.tsv")
        log_tsv = Path(f"gate2_v4_hard_log_audit_{stamp}.tsv")
        manifest_tsv = Path(f"gate2_v4_manifest_audit_{stamp}.tsv")
        freeze_json = Path(f"gate2_v4_evidence_freeze_{stamp}.json")
        upqi_md = Path(f"UPQI_V4_ICML_BRUTAL_AUDIT_{stamp}.md")
        reviewer_md = Path(f"GATE2_V4_FINAL_REVIEWER_PANEL_REPORT_{stamp}.md")

    cell_fields = [
        "cell_id", "label", "model_tag", "dataset", "algo", "seed", "quantizer",
        "stage", "status", "passed", "issues", "warnings", "done_layers", "remaining_layers", "path",
    ]
    write_tsv(upqi_tsv, upqi_rows, cell_fields)
    write_tsv(gt_tsv, gt_rows, cell_fields)
    write_tsv(claim_tsv, claim_rows)
    write_tsv(standard_tsv, standard_rows)
    write_tsv(log_tsv, log_rows, ["path", "line_no", "line"])
    write_tsv(manifest_tsv, manifest_rows, ["path", "status", "issues"])
    freeze_json.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(
        upqi_md,
        title="UPQI v4 ICML Brutal Audit",
        expected_lock=expected_lock,
        gt_rows=gt_rows,
        upqi_rows=upqi_rows,
        claim_rows=claim_rows,
        standard_rows=standard_rows,
        manifest_rows=manifest_rows,
        log_rows=log_rows,
        reviewer_rows=reviewer_rows,
        freeze=freeze,
    )
    write_markdown_report(
        reviewer_md,
        title="Gate 2 v4 Final Reviewer Panel Report",
        expected_lock=expected_lock,
        gt_rows=gt_rows,
        upqi_rows=upqi_rows,
        claim_rows=claim_rows,
        standard_rows=standard_rows,
        manifest_rows=manifest_rows,
        log_rows=log_rows,
        reviewer_rows=reviewer_rows,
        freeze=freeze,
    )

    upqi_counts = Counter(r["status"] for r in upqi_rows)
    gt_counts = Counter(r["status"] for r in gt_rows)
    claim_counts = Counter(r["claim_gate"] for r in claim_rows)
    print(f"expected_lock {expected_lock}")
    print(f"UPQI {dict(upqi_counts)}")
    print(f"GT {dict(gt_counts)}")
    print(f"claim_gate {dict(claim_counts)}")
    print(f"hard_log_hits {len(log_rows)}")
    for path in (upqi_md, upqi_tsv, gt_tsv, claim_tsv, reviewer_md, standard_tsv, log_tsv, manifest_tsv, freeze_json):
        print(f"WROTE {path}")

    paper_ready = sum(1 for r in claim_rows if r["paper_ready"] == "YES")
    upqi_clean = upqi_counts.get("clean_complete", 0)
    gt_clean = gt_counts.get("clean_complete", 0)
    if not args.allow_incomplete and (upqi_clean != 144 or gt_clean != 144 or paper_ready != 144 or log_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
