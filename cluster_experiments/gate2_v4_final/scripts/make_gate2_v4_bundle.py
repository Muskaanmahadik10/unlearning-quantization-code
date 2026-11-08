#!/usr/bin/env python3
"""Create a Gate 2 v4 bundle only after strict artifact preflight passes."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import load_json, load_source_lock, sha256_file, source_lock_hash  # noqa: E402
from validate_gt_v4 import validate_obj as validate_gt_obj  # noqa: E402
from validate_upqi_v4 import validate_obj as validate_upqi_obj  # noqa: E402


FORBIDDEN_LOG_PATTERNS = (
    "Traceback",
    "CUDA out of memory",
    "OUT_OF_MEMORY",
    "ModuleNotFoundError",
    "SyntaxError",
)
LEGACY_RESULT_MARKERS = ("gate2_v2", "gate2_v3")


def add_tree(tar: tarfile.TarFile, root: Path, arc_prefix: str, *, only_files: Optional[Sequence[Path]] = None) -> None:
    if not root.exists():
        return
    allowed = {path.resolve() for path in only_files} if only_files is not None else None
    for path in sorted(root.rglob("*")):
        if path.is_file() and (allowed is None or path.resolve() in allowed):
            tar.add(path, arcname=str(Path(arc_prefix) / path.relative_to(root)))


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def matching_log_files(logs_root: Path, job_ids: Sequence[str]) -> List[Path]:
    if not logs_root.exists():
        return []
    paths = sorted(path for path in logs_root.rglob("*") if path.is_file())
    if not job_ids:
        return paths
    return [path for path in paths if any(job_id in path.name for job_id in job_ids)]


def audit_logs(logs_root: Path, job_ids: Sequence[str]) -> tuple[List[Dict[str, Any]], List[Path]]:
    rows: List[Dict[str, Any]] = []
    included = matching_log_files(logs_root, job_ids)
    if not included:
        rows.append({"path": str(logs_root), "status": "FAIL", "issue": "no_canonical_logs_found"})
        return rows, included
    for path in included:
        issue = ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            text = ""
            issue = f"unreadable:{exc}"
        if not issue:
            for pattern in FORBIDDEN_LOG_PATTERNS:
                if pattern in text:
                    issue = f"forbidden_pattern:{pattern}"
                    break
        rows.append({"path": str(path), "status": "FAIL" if issue else "PASS", "issue": issue})
    return rows, included


def current_source_lock_issue(obj: Mapping[str, Any], expected_source_lock_sha: str | None) -> str:
    if not expected_source_lock_sha:
        return ""
    observed = obj.get("provenance", {}).get("source_lock_sha256")
    return "" if observed == expected_source_lock_sha else "source_lock_not_current"


def audit_results(results_root: Path, expected_source_lock_sha: str | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not results_root.exists():
        return [{"path": str(results_root), "status": "FAIL", "stage": "", "issue": "missing_results_root"}]
    for marker in LEGACY_RESULT_MARKERS:
        if marker in str(results_root).replace("\\", "/"):
            rows.append({"path": str(results_root), "status": "FAIL", "stage": "", "issue": f"legacy_results_root:{marker}"})
    for partial in sorted(results_root.rglob("*.partial")):
        rows.append({"path": str(partial), "status": "FAIL", "stage": "partial", "issue": "partial_file_in_final_bundle"})
    for path in sorted(results_root.rglob("*.json")):
        rel_text = str(path).replace("\\", "/")
        if any(marker in rel_text for marker in LEGACY_RESULT_MARKERS):
            rows.append({"path": str(path), "status": "FAIL", "stage": "", "issue": "legacy_path_marker"})
            continue
        try:
            obj = load_json(path)
        except Exception as exc:
            rows.append({"path": str(path), "status": "FAIL", "stage": "", "issue": f"bad_json:{exc}"})
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in raw for marker in LEGACY_RESULT_MARKERS):
            rows.append({"path": str(path), "status": "FAIL", "stage": str(obj.get("stage", "")), "issue": "legacy_marker_inside_json"})
            continue
        stage = str(obj.get("stage", ""))
        if stage == "gt_v4":
            report = validate_gt_obj(obj, path)
            issue = "" if report["passed"] else ";".join(report["errors"][:12])
            issue = issue or current_source_lock_issue(obj, expected_source_lock_sha)
            rows.append({"path": str(path), "status": "PASS" if not issue else "FAIL", "stage": stage, "issue": issue})
        elif stage == "upqi_v4":
            report = validate_upqi_obj(obj, path)
            issue = "" if report["passed"] else ";".join(report["errors"][:12])
            issue = issue or current_source_lock_issue(obj, expected_source_lock_sha)
            rows.append({"path": str(path), "status": "PASS" if not issue else "FAIL", "stage": stage, "issue": issue})
    if not rows:
        rows.append({"path": str(results_root), "status": "FAIL", "stage": "", "issue": "no_json_results_found"})
    return rows


def git_dirty_rows(repo_root: Path, source_root: Path, manifests_root: Path) -> List[Dict[str, Any]]:
    paths = [source_root, manifests_root]
    try:
        cmd = ["git", "status", "--short", "--", *(str(path) for path in paths if path.exists())]
        output = subprocess.check_output(cmd, cwd=str(repo_root), stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return [{"path": str(repo_root), "status": "FAIL", "issue": f"git_status_unavailable:{exc}"}]
    rows: List[Dict[str, Any]] = []
    for line in output.splitlines():
        if line.strip():
            rows.append({"path": line[3:].strip(), "status": "FAIL", "issue": f"dirty_source:{line[:2].strip()}"})
    return rows


def audit_source(source_root: Path, manifests_root: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not source_root.exists():
        rows.append({"path": str(source_root), "status": "FAIL", "issue": "missing_source_root"})
        return rows
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.stat().st_size == 0:
            rows.append({"path": str(path), "status": "FAIL", "issue": "zero_byte_source_file"})
    lock_path = source_root / "V4_SOURCE_LOCK.json"
    if not lock_path.exists():
        rows.append({"path": str(lock_path), "status": "FAIL", "issue": "missing_source_lock"})
    else:
        try:
            lock = load_source_lock(lock_path)
            files = lock.get("files", {})
            for rel, expected_sha in sorted(files.items()):
                path = repo_root / rel
                if not path.exists():
                    rows.append({"path": str(path), "status": "FAIL", "issue": "source_lock_file_missing"})
                    continue
                actual = sha256_file(path)
                if actual != expected_sha:
                    rows.append({"path": str(path), "status": "FAIL", "issue": "source_lock_sha_mismatch"})
        except Exception as exc:
            rows.append({"path": str(lock_path), "status": "FAIL", "issue": f"bad_source_lock:{exc}"})
    rows.extend(git_dirty_rows(repo_root, source_root, manifests_root))
    if not rows:
        rows.append({"path": str(source_root), "status": "PASS", "issue": ""})
    return rows


def audit_tsv_contents(path: Path, label: str) -> List[str]:
    issues: List[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    except Exception as exc:
        return [f"unreadable_{label}:{exc}"]
    if not rows:
        return [f"empty_rows_{label}"]
    if "status" in rows[0]:
        bad = [row for row in rows if row.get("status") != "PASS"]
        if bad:
            issues.append(f"{label}_nonpass_rows:{len(bad)}")
    if "paper_ready" in rows[0]:
        bad = [row for row in rows if row.get("paper_ready") != "YES"]
        if bad:
            issues.append(f"{label}_not_paper_ready_rows:{len(bad)}")
    if "join_status" in rows[0]:
        bad = [row for row in rows if row.get("join_status") != "passed"]
        if bad:
            issues.append(f"{label}_join_not_passed_rows:{len(bad)}")
    if "gt_status" in rows[0]:
        bad = [row for row in rows if row.get("gt_status") != "clean_complete"]
        if bad:
            issues.append(f"{label}_gt_not_clean_rows:{len(bad)}")
    if "upqi_status" in rows[0]:
        bad = [row for row in rows if row.get("upqi_status") != "clean_complete"]
        if bad:
            issues.append(f"{label}_upqi_not_clean_rows:{len(bad)}")
    return issues


def require_audit_file(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "FAIL", "issue": f"missing_{label}"}
    if path.stat().st_size == 0:
        return {"path": str(path), "status": "FAIL", "issue": f"empty_{label}"}
    issues = audit_tsv_contents(path, label) if path.suffix.lower() == ".tsv" else []
    if issues:
        return {"path": str(path), "status": "FAIL", "issue": ";".join(issues)}
    return {"path": str(path), "status": "PASS", "issue": ""}


def parse_job_ids(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Gate 2 v4 final artifacts after strict preflight")
    parser.add_argument("--results_root", default="results/gate2_v4_standard")
    parser.add_argument("--logs_root", default="logs/gate2_v4")
    parser.add_argument("--manifests_root", default="manifests/gate2_v4")
    parser.add_argument("--source_root", default="gate2_v4_final")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output", default="gate2_v4_final_bundle.tar.gz")
    parser.add_argument("--job_ids", default="", help="Comma-separated canonical production job IDs; if omitted, all v4 logs are audited")
    parser.add_argument("--manifest_audit_tsv", default="manifest_truth_ratio_audit_v4.tsv")
    parser.add_argument("--claim_gate_tsv", default="claim_gate_matrix_v4.tsv")
    parser.add_argument("--join_audit_tsv", default="icml_core_144_v4_status.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    audit_prefix = output.with_suffix("").with_suffix("")
    log_audit_path = Path(f"{audit_prefix}_bundle_log_audit_v4.tsv")
    result_audit_path = Path(f"{audit_prefix}_bundle_result_audit_v4.tsv")
    source_audit_path = Path(f"{audit_prefix}_bundle_source_audit_v4.tsv")
    required_audit_path = Path(f"{audit_prefix}_bundle_required_audits_v4.tsv")

    job_ids = parse_job_ids(args.job_ids)
    expected_source_lock_sha = None
    lock_path = Path(args.source_root) / "V4_SOURCE_LOCK.json"
    if lock_path.exists():
        lock = load_source_lock(lock_path)
        expected_source_lock_sha = lock.get("source_lock_sha256", source_lock_hash(lock))
    log_rows, included_logs = audit_logs(Path(args.logs_root), job_ids)
    result_rows = audit_results(Path(args.results_root), expected_source_lock_sha)
    source_rows = audit_source(Path(args.source_root), Path(args.manifests_root), Path(args.repo_root))
    required_rows = [
        require_audit_file(Path(args.manifest_audit_tsv), "manifest_truth_ratio_audit"),
        require_audit_file(Path(args.claim_gate_tsv), "claim_gate_matrix"),
        require_audit_file(Path(args.join_audit_tsv), "strict_join_status_audit"),
    ]

    write_tsv(log_audit_path, log_rows, ["path", "status", "issue"])
    write_tsv(result_audit_path, result_rows, ["path", "status", "stage", "issue"])
    write_tsv(source_audit_path, source_rows, ["path", "status", "issue"])
    write_tsv(required_audit_path, required_rows, ["path", "status", "issue"])

    all_rows = [*log_rows, *result_rows, *source_rows, *required_rows]
    failures = [row for row in all_rows if row.get("status") != "PASS"]
    if failures:
        print(f"Bundle preflight failed with {len(failures)} issue(s)")
        for row in failures[:40]:
            print(f"{row.get('status')} {row.get('path')}: {row.get('issue')}")
        print(f"wrote: {log_audit_path}")
        print(f"wrote: {result_audit_path}")
        print(f"wrote: {source_audit_path}")
        print(f"wrote: {required_audit_path}")
        raise SystemExit(1)

    with tarfile.open(output, "w:gz") as tar:
        add_tree(tar, Path(args.results_root), "results/gate2_v4_standard")
        add_tree(tar, Path(args.logs_root), "logs/gate2_v4", only_files=included_logs)
        add_tree(tar, Path(args.manifests_root), "manifests/gate2_v4")
        add_tree(tar, Path(args.source_root), "gate2_v4_final")
        for audit_file in (log_audit_path, result_audit_path, source_audit_path, required_audit_path, Path(args.manifest_audit_tsv), Path(args.claim_gate_tsv), Path(args.join_audit_tsv)):
            tar.add(audit_file, arcname=str(Path("audits") / audit_file.name))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
