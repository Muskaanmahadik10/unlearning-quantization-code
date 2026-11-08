#!/usr/bin/env python3
"""Prepare immutable public O/U/R bytes for the B0.4 representation canary.

This program has three deliberately separate modes:

* ``attest`` records a researcher's explicit, scoped acceptance of the
  upstream Llama terms.  It is an attestation, not a legal opinion.
* ``lock`` validates the previously collected commit-pinned metadata and
  creates an exact download allowlist plus a conservative storage gate.
* ``download`` fetches only that allowlist, verifies every byte, and emits a
  local snapshot manifest.  It never authorizes B0.4 GPU execution.

No model code is executed and no pickle is loaded by this program.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


ATTEST_SCHEMA = "org_e2b_b04_internal_research_attestation_v1"
PRE_SCHEMA = "org_e2b_b04_public_predownload_metadata_v1"
LOCK_SCHEMA = "org_e2b_b04_public_identity_download_lock_v1"
LOCAL_SCHEMA = "org_e2b_b04_public_local_byte_manifest_v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROLE_SPECS: Mapping[str, Mapping[str, str]] = {
    "O": {
        "model_id": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "revision": "88e31200b97e4c0c04ae0d2f0b591f427046d192",
    },
    "U": {
        "model_id": (
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
            "AltPO_lr1e-05_beta0.05_alpha1_epoch10"
        ),
        "revision": "a31043513688dea3c878669b63689984e44f2bc6",
    },
    "R": {
        "model_id": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
        "revision": "7114300c0049527a71833f5683965c358ad9dcbf",
    },
}
REQUIRED_COMMON = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
TOKENIZER_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
BYTE_IDENTICAL_TOKENIZER_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
UPSTREAM_LICENSE_URL = (
    "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/"
    "blob/main/LICENSE.txt"
)
UPSTREAM_AUP_URL = "https://www.llama.com/llama3_2/use-policy"
DOWNLOAD_USER_AGENT = "org-e2b-b04-public-byte-lock-v1"


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_special_tokens(path: Path) -> dict[str, str]:
    """Normalize the two equivalent HF special-token serializations.

    A token may be stored as a string or as an AddedToken-shaped object.  The
    object form is accepted only when every behavior flag is explicitly false;
    runtime tokenizer equivalence is independently checked in P5.
    """
    raw = read_json(path)
    require(isinstance(raw, Mapping) and raw, f"invalid special-token map: {path}")
    result: dict[str, str] = {}
    allowed = {"content", "lstrip", "normalized", "rstrip", "single_word"}
    for name, value in raw.items():
        require(isinstance(name, str) and name, f"invalid special-token name: {path}")
        if isinstance(value, str):
            content = value
        else:
            require(isinstance(value, Mapping), f"invalid special-token value: {path}:{name}")
            require(set(value).issubset(allowed), f"unknown special-token fields: {path}:{name}")
            require(isinstance(value.get("content"), str), f"missing special-token content: {path}:{name}")
            for flag in allowed - {"content"}:
                require(value.get(flag, False) is False, f"behavioral special-token flag: {path}:{name}:{flag}")
            content = str(value["content"])
        require(content, f"empty special-token content: {path}:{name}")
        result[name] = content
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    require(not temporary.exists(), f"temporary output exists: {temporary}")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if hasattr(os, "O_DIRECTORY"):
        try:
            fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def safe_repo_path(value: str) -> str:
    path = PurePosixPath(value)
    require(bool(value) and not path.is_absolute(), f"unsafe repository path: {value!r}")
    require("\\" not in value, f"backslash in repository path: {value!r}")
    require(all(part not in {"", ".", ".."} for part in path.parts), f"unsafe repository path: {value!r}")
    return path.as_posix()


def exact_size(item: Mapping[str, Any]) -> int:
    candidates = []
    for key in ("api_size", "api_lfs_size", "head_declared_size"):
        value = item.get(key)
        if value is not None:
            require(isinstance(value, int) and value >= 0, f"invalid {key}")
            candidates.append(value)
    require(bool(candidates), f"no size evidence for {item.get('path')}")
    require(len(set(candidates)) == 1, f"size disagreement for {item.get('path')}")
    return candidates[0]


def lfs_oid(item: Mapping[str, Any]) -> str | None:
    values = []
    for key in ("api_lfs_oid", "head_lfs_oid"):
        value = item.get(key)
        if value not in (None, ""):
            value = str(value).lower().removeprefix("sha256:")
            require(bool(HEX64.fullmatch(value)), f"invalid LFS OID for {item.get('path')}")
            values.append(value)
    require(len(set(values)) <= 1, f"LFS OID disagreement for {item.get('path')}")
    return values[0] if values else None


def validate_attestation(value: Mapping[str, Any]) -> None:
    require(value.get("schema_version") == ATTEST_SCHEMA, "wrong attestation schema")
    require(bool(str(value.get("researcher", "")).strip()), "researcher missing")
    require(bool(str(value.get("institution", "")).strip()), "institution missing")
    decisions = value.get("decisions")
    require(isinstance(decisions, Mapping), "attestation decisions missing")
    for field in (
        "accepted_llama_3_2_community_license",
        "accepted_llama_acceptable_use_policy",
        "internal_noncommercial_academic_research_only",
        "no_model_weight_redistribution",
        "understands_downstream_cards_have_incomplete_license_metadata",
    ):
        require(decisions.get(field) is True, f"attestation missing decision: {field}")
    require(value.get("legal_opinion") is False, "attestation cannot be a legal opinion")
    require(value.get("production_authorized") is False, "attestation cannot authorize production")


def make_attestation(args: argparse.Namespace) -> None:
    confirmations = (
        args.accept_llama_license,
        args.accept_llama_aup,
        args.internal_academic_only,
        args.no_redistribution,
        args.acknowledge_incomplete_downstream_license,
    )
    require(all(confirmations), "all five explicit confirmation flags are required")
    output = args.output.resolve()
    require(not output.exists(), f"attestation output exists: {output}")
    record = {
        "schema_version": ATTEST_SCHEMA,
        "created_at_utc": utc_now(),
        "researcher": args.researcher.strip(),
        "institution": args.institution.strip(),
        "scope": "INTERNAL_NONCOMMERCIAL_ACADEMIC_RESEARCH_NO_REDISTRIBUTION",
        "upstream_terms": {
            "llama_3_2_community_license": UPSTREAM_LICENSE_URL,
            "llama_3_2_acceptable_use_policy": UPSTREAM_AUP_URL,
        },
        "roles": ROLE_SPECS,
        "decisions": {
            "accepted_llama_3_2_community_license": True,
            "accepted_llama_acceptable_use_policy": True,
            "internal_noncommercial_academic_research_only": True,
            "no_model_weight_redistribution": True,
            "understands_downstream_cards_have_incomplete_license_metadata": True,
        },
        "scoped_internal_research_use_permitted": True,
        "redistribution_permitted": False,
        "legal_opinion": False,
        "paper_claim_authorized": False,
        "production_authorized": False,
    }
    atomic_json(output, record)
    print("ORG_E2B_B04_INTERNAL_USE_ATTESTATION=PASS")
    print(f"OUTPUT={output}")


def validate_predownload(report: Mapping[str, Any]) -> None:
    require(report.get("schema_version") == PRE_SCHEMA, "wrong predownload schema")
    require(report.get("collection_status") == "COMPLETE", "predownload metadata incomplete")
    require(report.get("identity_manifest_complete") is True, "identity manifest incomplete")
    require(report.get("head_manifest_complete") is True, "HEAD manifest incomplete")
    summary = report.get("request_summary")
    require(isinstance(summary, Mapping), "request summary missing")
    for field in (
        "model_weight_get_request_count",
        "model_weight_body_bytes_read",
        "disallowed_get_request_count",
        "authorization_header_sent_count",
        "cookie_header_sent_count",
    ):
        require(summary.get(field) == 0, f"unsafe metadata request count: {field}")
    roles = report.get("roles")
    require(isinstance(roles, Mapping) and set(roles) == set(ROLE_SPECS), "role set mismatch")
    for role, spec in ROLE_SPECS.items():
        item = roles[role]
        require(item.get("model_id") == spec["model_id"], f"{role} model ID drift")
        require(item.get("revision") == spec["revision"], f"{role} revision drift")
        require(item.get("api_identity_verified") is True, f"{role} identity unverified")
        require(item.get("head_manifest_complete") is True, f"{role} HEAD manifest incomplete")
        require(item.get("private") is False, f"{role} is private")
        require(item.get("gated") in (False, None), f"{role} is gated")
        require(item.get("disabled") is False, f"{role} is disabled")
        require(item.get("errors") == [], f"{role} metadata errors")


def build_lock(args: argparse.Namespace) -> None:
    report_path = args.predownload_report.resolve()
    attest_path = args.attestation.resolve()
    output = args.output.resolve()
    require(not output.exists(), f"lock output exists: {output}")
    report = read_json(report_path)
    attestation = read_json(attest_path)
    validate_predownload(report)
    validate_attestation(attestation)

    roles_out: dict[str, Any] = {}
    snapshot_sizes: dict[str, int] = {}
    for role, spec in ROLE_SPECS.items():
        remote = report["roles"][role]
        siblings = remote.get("siblings")
        require(isinstance(siblings, list) and siblings, f"{role} siblings missing")
        records = []
        seen: set[str] = set()
        for sibling in siblings:
            require(isinstance(sibling, Mapping), f"{role} sibling is not an object")
            path = safe_repo_path(str(sibling.get("path", "")))
            require(path not in seen, f"{role} duplicate sibling: {path}")
            seen.add(path)
            size = exact_size(sibling)
            oid = lfs_oid(sibling)
            records.append(
                {
                    "path": path,
                    "size": size,
                    "lfs_sha256": oid,
                    "api_blob_id": sibling.get("api_blob_id"),
                    "required_common_model_file": path in REQUIRED_COMMON,
                }
            )
        require(REQUIRED_COMMON.issubset(seen), f"{role} required files missing")
        snapshot_size = sum(int(item["size"]) for item in records)
        snapshot_sizes[role] = snapshot_size
        roles_out[role] = {
            **spec,
            "files": sorted(records, key=lambda value: value["path"]),
            "file_count": len(records),
            "snapshot_bytes": snapshot_size,
        }

    snapshot_total = sum(snapshot_sizes.values())
    largest_snapshot = max(snapshot_sizes.values())
    # Four packed artifacts are bounded by the largest exact dense snapshot.
    # This is intentionally conservative and does not claim an observed packed size.
    logs_receipts_reserve = 1024**3
    high_water = snapshot_total + 4 * largest_snapshot + largest_snapshot + logs_receipts_reserve
    required_free = math.ceil(1.30 * high_water)
    lock = {
        "schema_version": LOCK_SCHEMA,
        "created_at_utc": utc_now(),
        "selected_u_rule": "lexicographically_first_exact_prefix_true_unlearned_repository",
        "roles": roles_out,
        "predownload_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "attestation": {"path": str(attest_path), "sha256": sha256_file(attest_path)},
        "license_scope": {
            "internal_research_use_permitted": True,
            "redistribution_permitted": False,
            "downstream_license_declaration_missing_roles": ["U", "R"],
            "license_unknown_count": 0,
            "license_unknown_count_scope": "SCOPED_INTERNAL_USE_DECISIONS_ONLY",
            "legal_opinion": False,
        },
        "lineage": {
            "public_u_is_exact_prefix_true_unlearned_repository": True,
            "repository_revisions_are_immutable_commit_shas": all(
                HEX40.fullmatch(spec["revision"]) for spec in ROLE_SPECS.values()
            ),
            "source_o_revision_demonstrated_for_u": False,
            "independent_training_lineage_demonstrated": False,
        },
        "storage_pre_download_gate": {
            "snapshot_total_bytes": snapshot_total,
            "largest_snapshot_bytes": largest_snapshot,
            "packed_artifact_count": 4,
            "packed_artifact_each_upper_bound_bytes": largest_snapshot,
            "maximum_staging_bytes": largest_snapshot,
            "logs_manifests_receipts_reserve_bytes": logs_receipts_reserve,
            "high_water_upper_bound_bytes": high_water,
            "safety_multiplier": 1.30,
            "required_available_bytes": required_free,
        },
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "model_weight_download_authorized": True,
        "b0_4_gpu_launcher_authorized": False,
        "outcome_inspection_authorized": False,
        "production_authorized": False,
        "scientific_claim_authorized": False,
    }
    atomic_json(output, lock)
    print("ORG_E2B_B04_PUBLIC_IDENTITY_DOWNLOAD_LOCK=PASS")
    print(f"FILES={sum(value['file_count'] for value in roles_out.values())}")
    print(f"DOWNLOAD_BYTES={snapshot_total}")
    print(f"REQUIRED_AVAILABLE_BYTES={required_free}")
    print(f"OUTPUT={output}")


class NoCredentialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
            redirected.remove_header("Cookie")
        return redirected


def download_one(url: str, target: Path, expected_size: int, expected_sha: str | None) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    if target.exists():
        require(target.is_file(), f"download target is not a file: {target}")
        actual_size = target.stat().st_size
        actual_sha = sha256_file(target)
        require(actual_size == expected_size, f"existing size mismatch: {target}")
        if expected_sha:
            require(actual_sha == expected_sha, f"existing LFS hash mismatch: {target}")
        return {"bytes": actual_size, "sha256": actual_sha, "reused_verified": True}
    opener = urllib.request.build_opener(NoCredentialRedirect())
    attempts = 0
    while attempts < 5:
        attempts += 1
        offset = partial.stat().st_size if partial.exists() else 0
        require(offset <= expected_size, f"partial exceeds expected size: {target}")
        headers = {"User-Agent": DOWNLOAD_USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with opener.open(request, timeout=120) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                if offset and status_code != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                    raise urllib.error.URLError("server did not honor Range request")
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            if attempts >= 5:
                raise
            time.sleep(min(2**attempts, 20))
            continue
        if partial.stat().st_size == expected_size:
            break
        if partial.stat().st_size > expected_size:
            raise ContractError(f"download exceeds expected size: {target}")
    require(partial.exists() and partial.stat().st_size == expected_size, f"incomplete download: {target}")
    actual_sha = sha256_file(partial)
    if expected_sha:
        require(actual_sha == expected_sha, f"downloaded LFS hash mismatch: {target}")
    os.replace(partial, target)
    os.chmod(target, stat.S_IRUSR | stat.S_IRGRP)
    return {"bytes": expected_size, "sha256": actual_sha, "reused_verified": False}


def validate_lock(lock: Mapping[str, Any], lock_path: Path) -> None:
    require(lock.get("schema_version") == LOCK_SCHEMA, "wrong identity/download lock schema")
    require(lock.get("model_weight_download_authorized") is True, "download is not authorized")
    require(lock.get("production_authorized") is False, "lock cannot authorize production")
    require(lock.get("source", {}).get("sha256") == sha256_file(Path(__file__).resolve()), "download source drift")
    pre = lock.get("predownload_report", {})
    att = lock.get("attestation", {})
    for record, label in ((pre, "predownload"), (att, "attestation")):
        path = Path(str(record.get("path", ""))).resolve()
        require(path.is_file(), f"{label} source missing")
        require(sha256_file(path) == record.get("sha256"), f"{label} source drift")
    validate_predownload(read_json(Path(str(pre["path"]))))
    validate_attestation(read_json(Path(str(att["path"]))))
    require(lock_path.is_file(), "lock path missing")


def download_snapshots(args: argparse.Namespace) -> None:
    lock_path = args.lock.resolve()
    lock = read_json(lock_path)
    validate_lock(lock, lock_path)
    root = args.snapshot_root.resolve()
    require(root.is_absolute(), "snapshot root must be absolute")
    require(args.quota_available_bytes > 0, "positive quota-available bytes required")
    parsed_base = urllib.parse.urlsplit(args.base_url)
    require(
        parsed_base.scheme == "https" or (args.allow_test_http and parsed_base.scheme == "http"),
        "download base URL must use HTTPS",
    )
    require(bool(parsed_base.netloc) and not parsed_base.query and not parsed_base.fragment, "unsafe base URL")
    base_url = args.base_url.rstrip("/")
    root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = root.parent / "b04_public_local_byte_manifest_v1.json"
    require(not manifest_path.exists(), f"local manifest exists: {manifest_path}")
    expected_rel = {
        f"{role}/{item['path']}"
        for role in ("O", "U", "R")
        for item in lock["roles"][role]["files"]
    }
    allowed_partial_rel = {
        str(PurePosixPath(relative).parent / f".{PurePosixPath(relative).name}.partial")
        for relative in expected_rel
    }
    if root.exists():
        require(root.is_dir() and not root.is_symlink(), f"unsafe existing snapshot root: {root}")
        for path in root.rglob("*"):
            require(not path.is_symlink(), f"symlink forbidden in resumable snapshot root: {path}")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                require(
                    relative in expected_rel or relative in allowed_partial_rel,
                    f"unexpected resumable snapshot file: {relative}",
                )
    fs_free = shutil.disk_usage(root.parent).free
    required_free = int(lock["storage_pre_download_gate"]["required_available_bytes"])
    require(fs_free >= required_free, f"filesystem free bytes insufficient: {fs_free} < {required_free}")
    require(args.quota_available_bytes >= required_free, "project/user quota headroom insufficient")
    root.mkdir(mode=0o750, exist_ok=True)
    file_records = []
    try:
        for role in ("O", "U", "R"):
            spec = lock["roles"][role]
            for item in spec["files"]:
                rel = safe_repo_path(str(item["path"]))
                quoted = "/".join(urllib.parse.quote(part, safe="") for part in rel.split("/"))
                url = (
                    f"{base_url}/{spec['model_id']}/resolve/"
                    f"{spec['revision']}/{quoted}?download=true"
                )
                target = root / role / PurePosixPath(rel)
                result = download_one(url, target, int(item["size"]), item.get("lfs_sha256"))
                file_records.append(
                    {
                        "role": role,
                        "model_id": spec["model_id"],
                        "revision": spec["revision"],
                        "path": rel,
                        "local_path": str(target),
                        "bytes": result["bytes"],
                        "sha256": result["sha256"],
                        "expected_lfs_sha256": item.get("lfs_sha256"),
                        "read_only": not bool(target.stat().st_mode & stat.S_IWUSR),
                        "reused_verified": result["reused_verified"],
                    }
                )
                print(f"B04_DOWNLOADED role={role} path={rel} bytes={result['bytes']}", flush=True)
    except Exception:
        # Preserve partial bytes for exact resumability, but never emit a PASS manifest.
        raise

    actual_rel = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".partial" not in path.name
    }
    require(actual_rel == expected_rel, "snapshot contains missing or unexpected files")
    by_role = {role: [item for item in file_records if item["role"] == role] for role in ROLE_SPECS}
    tokenizer_hashes = {
        name: {role: next(item["sha256"] for item in by_role[role] if item["path"] == name) for role in ROLE_SPECS}
        for name in TOKENIZER_FILES
    }
    byte_identity = {
        name: len(set(tokenizer_hashes[name].values())) == 1 for name in TOKENIZER_FILES
    }
    require(
        all(byte_identity[name] for name in BYTE_IDENTICAL_TOKENIZER_FILES),
        "O/U/R tokenizer vocabulary, configuration, or generation bytes differ",
    )
    special_token_semantics = {
        role: canonical_special_tokens(root / role / "special_tokens_map.json") for role in ROLE_SPECS
    }
    special_token_semantic_identity = len(
        {json.dumps(value, sort_keys=True) for value in special_token_semantics.values()}
    ) == 1
    require(special_token_semantic_identity, "O/U/R special-token semantics differ")
    model_hashes = {
        role: next(item["sha256"] for item in by_role[role] if item["path"] == "model.safetensors")
        for role in ROLE_SPECS
    }
    require(len(set(model_hashes.values())) == 3, "O/U/R model weight hashes are not distinct")
    manifest = {
        "schema_version": LOCAL_SCHEMA,
        "created_at_utc": utc_now(),
        "snapshot_root": str(root),
        "identity_download_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "files": sorted(file_records, key=lambda value: (value["role"], value["path"])),
        "file_count": len(file_records),
        "total_bytes": sum(int(item["bytes"]) for item in file_records),
        "role_model_weight_sha256": model_hashes,
        "tokenizer_generation_file_sha256": tokenizer_hashes,
        "tokenizer_generation_byte_identity_by_file": byte_identity,
        "tokenizer_byte_identity_except_special_token_encoding_verified": all(
            byte_identity[name] for name in BYTE_IDENTICAL_TOKENIZER_FILES
        ),
        "special_token_semantics_by_role": special_token_semantics,
        "special_tokens_semantic_identity_verified": special_token_semantic_identity,
        "tokenizer_semantic_identity_verified": True,
        "tokenizer_template_byte_identity_verified": all(byte_identity.values()),
        "storage_gate": {
            **lock["storage_pre_download_gate"],
            "filesystem_available_bytes_before_download": fs_free,
            "quota_available_bytes_before_download": args.quota_available_bytes,
            "passed": True,
        },
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "all_files_rehashed": True,
        "model_bytes_downloaded": True,
        "actual_module_inventory_verified": False,
        "b0_4_gpu_launcher_authorized": False,
        "outcome_inspection_authorized": False,
        "production_authorized": False,
        "scientific_claim_authorized": False,
    }
    atomic_json(manifest_path, manifest)
    print("ORG_E2B_B04_PUBLIC_MODEL_DOWNLOAD=PASS")
    print(f"FILES={len(file_records)}")
    print(f"BYTES={manifest['total_bytes']}")
    print("TOKENIZER_RUNTIME_PREREQUISITE_SEMANTICS=PASS")
    print(f"OUTPUT={manifest_path}")
    print("B0_4_GPU_LAUNCHER_AUTHORIZED=false")


def verify_local(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == LOCAL_SCHEMA, "wrong local-byte schema")
    require(manifest.get("production_authorized") is False, "local manifest cannot authorize production")
    source = manifest.get("source", {})
    require(source.get("sha256") == sha256_file(Path(__file__).resolve()), "local-manifest source drift")
    lock_path = Path(str(manifest["identity_download_lock"]["path"])).resolve()
    require(sha256_file(lock_path) == manifest["identity_download_lock"]["sha256"], "lock drift")
    validate_lock(read_json(lock_path), lock_path)
    root = Path(str(manifest["snapshot_root"])).resolve()
    failures = []
    expected_paths = set()
    for item in manifest.get("files", []):
        path = Path(str(item["local_path"])).resolve()
        expected_paths.add(path.relative_to(root).as_posix())
        if not path.is_file():
            failures.append(f"missing:{item['role']}:{item['path']}")
            continue
        if path.stat().st_size != item["bytes"]:
            failures.append(f"size:{item['role']}:{item['path']}")
        if sha256_file(path) != item["sha256"]:
            failures.append(f"sha256:{item['role']}:{item['path']}")
        if path.stat().st_mode & stat.S_IWUSR:
            failures.append(f"writable:{item['role']}:{item['path']}")
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        failures.append("snapshot_path_universe")
    require(not failures, "local-byte verification failed: " + ",".join(failures))
    print("ORG_E2B_B04_PUBLIC_LOCAL_BYTE_VERIFICATION=PASS")
    print(f"FILES={len(expected_paths)}")
    print(f"OUTPUT={manifest_path}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subs = root.add_subparsers(dest="command", required=True)
    attest = subs.add_parser("attest")
    attest.add_argument("--researcher", required=True)
    attest.add_argument("--institution", required=True)
    attest.add_argument("--output", type=Path, required=True)
    attest.add_argument("--accept-llama-license", action="store_true")
    attest.add_argument("--accept-llama-aup", action="store_true")
    attest.add_argument("--internal-academic-only", action="store_true")
    attest.add_argument("--no-redistribution", action="store_true")
    attest.add_argument("--acknowledge-incomplete-downstream-license", action="store_true")
    lock = subs.add_parser("lock")
    lock.add_argument("--predownload-report", type=Path, required=True)
    lock.add_argument("--attestation", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    download = subs.add_parser("download")
    download.add_argument("--lock", type=Path, required=True)
    download.add_argument("--snapshot-root", type=Path, required=True)
    download.add_argument("--quota-available-bytes", type=int, required=True)
    download.add_argument("--base-url", default="https://huggingface.co", help=argparse.SUPPRESS)
    download.add_argument("--allow-test-http", action="store_true", help=argparse.SUPPRESS)
    verify = subs.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "attest":
            make_attestation(args)
        elif args.command == "lock":
            build_lock(args)
        elif args.command == "download":
            download_snapshots(args)
        elif args.command == "verify":
            verify_local(args)
        else:  # pragma: no cover
            raise ContractError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"ORG_E2B_B04_{args.command.upper()}=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
