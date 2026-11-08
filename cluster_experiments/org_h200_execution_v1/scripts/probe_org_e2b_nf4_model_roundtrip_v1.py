#!/usr/bin/env python3
"""Fail-closed Transformers NF4 model roundtrip canary for ORG B0.3b.

This canary creates a deterministic tiny *local* Llama checkpoint, loads it
through the Transformers bitsandbytes quantizer with an exact mixed set of
protected module FQNs, serializes it with ``save_pretrained`` and safetensors,
terminates the producer, and reloads it in a fresh process using
``from_pretrained`` without supplying a quantization override.

It is deliberately not a scientific experiment.  A PASS certifies only this
small H200/Transformers/NF4 representation path and cannot authorize E2B/E3
production by itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import site
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

import probe_org_e2b_nf4_fresh_reload_v1 as nf4_engine_v1
import probe_org_e2b_nf4_fresh_reload_v2 as nf4_gate_v2


SCHEMA_VERSION = "org_e2b_nf4_transformers_model_roundtrip_evidence_v1"
REFERENCE_SCHEMA = "org_e2b_nf4_transformers_model_roundtrip_reference_v1"
LIVE_SCHEMA = "org_e2b_live_gpu_runtime_probe_v2"
LOW_LEVEL_SCHEMA = "org_e2b_nf4_fresh_reload_evidence_v2"
MODEL_CONTRACT = "tiny_local_llama_mixed_nf4_v1"
ENVIRONMENT_LOCK_SCHEMA = "org_e2b_packed_h200_environment_lock_v1"
SEED = 1729
REPEATABILITY_FLOOR = 1.0e-6
REPEATABILITY_MULTIPLIER = 5.0
REPEATABILITY_HARD_CEILING = 0.0
MAX_PREREQUISITE_AGE_HOURS = 72.0

MODEL_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1.0e-5,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "use_cache": False,
    "attention_implementation": "eager",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}

PROTECTED_FQNS = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.1.mlp.up_proj",
)


class CanaryError(RuntimeError):
    """A fail-closed model-roundtrip requirement was not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def environment_tree_content_sha256(value: Mapping[str, Any]) -> str:
    """Reproduce the exact B0.2 environment-inventory digest algorithm."""

    payload = {
        "files": value.get("files"),
        "symlinks": value.get("symlinks"),
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inventory_environment_tree(root: Path) -> dict[str, Any]:
    """Hash every file and internal symlink under an immutable environment."""

    root = root.resolve()
    _require(root.is_dir(), f"environment root unavailable: {root}")
    files: list[dict[str, Any]] = []
    symlinks: list[dict[str, str]] = []
    total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            resolved = path.resolve()
            _require(
                path_is_within(resolved, root),
                f"environment symlink escapes target root: {relative} -> {resolved}",
            )
            symlinks.append({"path": relative, "target": os.readlink(path)})
        elif path.is_file():
            size = path.stat().st_size
            total += size
            files.append(
                {"path": relative, "bytes": size, "sha256": sha256_file(path)}
            )
    result = {
        "root": str(root),
        "files": files,
        "symlinks": symlinks,
        "file_count": len(files),
        "symlink_count": len(symlinks),
        "total_bytes": total,
    }
    result["content_manifest_sha256"] = environment_tree_content_sha256(result)
    return result


def environment_tree_revalidation_failures(
    locked: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    expected_environment_root: Path,
) -> list[str]:
    """Compare the complete current tree to the complete frozen B0.2 tree."""

    failures: list[str] = []
    expected_root = str(expected_environment_root.resolve())
    if str(Path(str(locked.get("root", ""))).resolve()) != expected_root:
        failures.append("locked_environment_tree_root")
    if str(Path(str(current.get("root", ""))).resolve()) != expected_root:
        failures.append("current_environment_tree_root")
    for label, value in (("locked", locked), ("current", current)):
        files = value.get("files")
        symlinks = value.get("symlinks")
        if not isinstance(files, list):
            failures.append(f"{label}_environment_tree_files_type")
            files = []
        if not isinstance(symlinks, list):
            failures.append(f"{label}_environment_tree_symlinks_type")
            symlinks = []
        if value.get("file_count") != len(files):
            failures.append(f"{label}_environment_tree_file_count")
        if value.get("symlink_count") != len(symlinks):
            failures.append(f"{label}_environment_tree_symlink_count")
        if value.get("total_bytes") != sum(
            int(item.get("bytes", -1))
            for item in files
            if isinstance(item, dict)
        ):
            failures.append(f"{label}_environment_tree_total_bytes")
        if value.get("content_manifest_sha256") != environment_tree_content_sha256(
            {"files": files, "symlinks": symlinks}
        ):
            failures.append(f"{label}_environment_tree_content_digest")
    for field in (
        "files",
        "symlinks",
        "file_count",
        "symlink_count",
        "total_bytes",
        "content_manifest_sha256",
    ):
        if locked.get(field) != current.get(field):
            failures.append(f"environment_tree_drift:{field}")
    return failures


def executable_record(name: str) -> dict[str, Any]:
    """Reproduce the exact B0.2 external-tool identity record."""

    raw = shutil.which(name)
    if not raw:
        return {"available": False, "path": None, "sha256": None, "version": None}
    path = Path(raw).resolve()
    completed = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, check=False
    )
    return {
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "version": " ".join((completed.stdout + " " + completed.stderr).split())[
            :1000
        ],
    }


def executable_record_at_path(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        return {"available": False, "path": str(path), "sha256": None, "version": None}
    completed = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, check=False
    )
    return {
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "version": " ".join((completed.stdout + " " + completed.stderr).split())[
            :1000
        ],
    }


def nvidia_driver_version(executable: Path) -> str | None:
    completed = subprocess.run(
        [str(executable), "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    versions = sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )
    return versions[0] if completed.returncode == 0 and len(versions) == 1 else None


def frozen_absolute_tolerance(repeat_max_abs: float) -> float:
    """Freeze a fail-closed absolute tolerance from measured repeatability."""

    value = float(repeat_max_abs)
    _require(value >= 0.0 and value < float("inf"), "invalid repeatability metric")
    _require(
        value <= REPEATABILITY_HARD_CEILING,
        "same-process repeatability is not bitwise exact",
    )
    return max(REPEATABILITY_FLOOR, REPEATABILITY_MULTIPLIER * value)


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    _require(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
    _require(not path.exists(), f"refusing to overwrite evidence: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        _require(not path.exists(), f"refusing to overwrite evidence: {path}")
        os.replace(temporary, path)
        temporary = None
        _fsync_path(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _age_hours(value: str, now: datetime | None = None) -> float:
    collected = _parse_utc(value)
    current = (now or _utc_now()).astimezone(timezone.utc)
    return (current - collected).total_seconds() / 3600.0


def _boot_id() -> str | None:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="ascii").strip() if path.is_file() else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def expected_linear_fqns() -> tuple[str, ...]:
    names: list[str] = []
    for layer in range(int(MODEL_GEOMETRY["num_hidden_layers"])):
        prefix = f"model.layers.{layer}"
        names.extend(
            [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.down_proj",
            ]
        )
    names.append("lm_head")
    return tuple(names)


def target_fqns() -> tuple[str, ...]:
    return tuple(name for name in expected_linear_fqns() if name != "lm_head")


def expected_packed_fqns() -> tuple[str, ...]:
    protected = set(PROTECTED_FQNS)
    return tuple(name for name in target_fqns() if name not in protected)


def expected_dense_fqns() -> tuple[str, ...]:
    return (*PROTECTED_FQNS, "lm_head")


def protected_set_failures(
    actual_quantized: Sequence[str], actual_dense: Sequence[str]
) -> list[str]:
    failures: list[str] = []
    if tuple(sorted(actual_quantized)) != tuple(sorted(expected_packed_fqns())):
        failures.append("quantized_module_fqn_set_mismatch")
    if tuple(sorted(actual_dense)) != tuple(sorted(expected_dense_fqns())):
        failures.append("dense_module_fqn_set_mismatch")
    if set(actual_quantized) & set(actual_dense):
        failures.append("quantized_dense_module_overlap")
    if set(actual_quantized) | set(actual_dense) != set(expected_linear_fqns()):
        failures.append("linear_module_partition_incomplete")
    protected = set(PROTECTED_FQNS)
    if not any(".self_attn." in name for name in protected):
        failures.append("protected_set_lacks_attention_module")
    if not any(".mlp." in name for name in protected):
        failures.append("protected_set_lacks_mlp_module")
    layers = {
        int(match.group(1))
        for name in protected
        if (match := re.search(r"model\.layers\.(\d+)\.", name)) is not None
    }
    if layers != {0, 1}:
        failures.append("protected_set_does_not_span_both_layers")
    return failures


def exact_scalar_equal(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def quantization_metadata_failures(
    metadata: Mapping[str, Any], expected_skip_modules: Sequence[str]
) -> list[str]:
    failures: list[str] = []
    expected = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_use_double_quant": False,
        "llm_int8_enable_fp32_cpu_offload": False,
    }
    for key, value in expected.items():
        if not exact_scalar_equal(metadata.get(key), value):
            failures.append(f"quantization_metadata_{key}")
    actual_skip = metadata.get("llm_int8_skip_modules")
    if not isinstance(actual_skip, list):
        failures.append("quantization_metadata_skip_modules_type")
    elif tuple(actual_skip) != tuple(expected_skip_modules):
        failures.append("quantization_metadata_skip_modules_exact_order")
    if metadata.get("quant_method") != "bitsandbytes":
        failures.append("quantization_metadata_quant_method")
    return failures


def model_config_failures(config: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if config.get("model_type") != "llama":
        failures.append("config_model_type")
    architectures = config.get("architectures")
    if architectures != ["LlamaForCausalLM"]:
        failures.append("config_architectures")
    if "auto_map" in config:
        failures.append("config_auto_map_forbidden")
    for key, expected in MODEL_GEOMETRY.items():
        if not exact_scalar_equal(config.get(key), expected):
            failures.append(f"config_geometry_{key}")
    qmeta = config.get("quantization_config")
    if not isinstance(qmeta, dict):
        failures.append("config_quantization_metadata_missing")
    else:
        failures.extend(
            quantization_metadata_failures(qmeta, expected_dense_fqns())
        )
    return failures


def loading_info_failures(info: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key) != []:
            failures.append(f"loading_info_{key}")
    return failures


def parse_sha256_manifest(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", raw)
        _require(match is not None, f"invalid SHA256 manifest line {line_number}")
        digest, name = match.groups()
        normalized = name.replace("\\", "/")
        path = Path(normalized)
        _require(not path.is_absolute(), f"absolute manifest path: {name}")
        _require(".." not in path.parts, f"parent traversal in manifest: {name}")
        _require(normalized not in rows, f"duplicate manifest path: {name}")
        rows[normalized] = digest
    _require(bool(rows), "empty source manifest")
    return rows


def validate_source_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    required_paths: Sequence[Path],
) -> dict[str, Any]:
    rows = parse_sha256_manifest(manifest_path.read_text(encoding="utf-8"))
    required_names: list[str] = []
    for path in required_paths:
        resolved = path.resolve()
        _require(
            resolved.is_relative_to(repository_root.resolve()),
            f"source path outside repository: {resolved}",
        )
        name = resolved.relative_to(repository_root.resolve()).as_posix()
        required_names.append(name)
        _require(name in rows, f"source manifest lacks required path: {name}")
    failures: list[str] = []
    for name, expected in rows.items():
        path = repository_root / name
        if not path.is_file():
            failures.append(f"missing:{name}")
        elif sha256_file(path) != expected:
            failures.append(f"sha256:{name}")
    _require(not failures, "source manifest verification failed: " + ",".join(failures))
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "entries": len(rows),
        "required_paths": required_names,
        "all_entries_verified": True,
    }


def environment_lock_failures(
    receipt: Mapping[str, Any],
    *,
    expected_environment_root: str,
    expected_versions: Mapping[str, str],
    expected_cuda_version: str,
) -> list[str]:
    """Pure contract check for the immutable B0.2 environment receipt."""

    failures: list[str] = []
    checks = [
        (receipt.get("schema_version") == ENVIRONMENT_LOCK_SCHEMA, "environment_schema"),
        (receipt.get("overall_status") == "PASS", "environment_status"),
        (receipt.get("failures") == [], "environment_failures"),
        (str(Path(str(receipt.get("environment_root", ""))).resolve()) == str(Path(expected_environment_root).resolve()), "environment_root"),
        (receipt.get("interpreter_identity", {}).get("overall_status") == "PASS", "environment_interpreter_identity"),
        (receipt.get("environment_tree_manifest", {}).get("all_entries_verified") is True, "environment_tree_manifest_verification"),
        (re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("environment_tree_manifest", {}).get("content_manifest_sha256", ""))) is not None, "environment_tree_content_manifest_sha256"),
        (receipt.get("bitsandbytes_native", {}).get("compiled_with_cuda") is True, "environment_bnb_cuda"),
        (receipt.get("source_manifest", {}).get("all_entries_verified") is True, "environment_source_manifest_verification"),
        (receipt.get("scope_guards", {}).get("production_authorized") is False, "environment_scope_production"),
        (receipt.get("scope_guards", {}).get("scientific_claim_authorized") is False, "environment_scope_claim"),
        (receipt.get("scope_guards", {}).get("backend_representation_certified") is False, "environment_scope_backend"),
        (receipt.get("scope_guards", {}).get("b03_canary_authorized") is True, "environment_scope_b03"),
        (receipt.get("toolchain", {}).get("cuda_version") == expected_cuda_version, "environment_cuda_version"),
    ]
    for passed, name in checks:
        if not passed:
            failures.append(name)
    packages = receipt.get("packages", {})
    if not isinstance(packages, dict):
        failures.append("environment_packages_type")
        packages = {}
    root = Path(expected_environment_root).resolve()
    for name, version in expected_versions.items():
        item = packages.get(name, {})
        if item.get("version") != version:
            failures.append(f"environment_package_version:{name}")
        module_path = Path(str(item.get("module_path", ""))).resolve()
        if not str(item.get("module_path", "")) or not module_path.is_relative_to(root):
            failures.append(f"environment_package_path:{name}")
        if re.fullmatch(r"[0-9a-f]{64}", str(item.get("module_sha256", ""))) is None:
            failures.append(f"environment_package_sha256:{name}")
    native = receipt.get("bitsandbytes_native", {})
    native_path = Path(str(native.get("library_path", ""))).resolve()
    if not str(native.get("library_path", "")) or not native_path.is_relative_to(root):
        failures.append("environment_bnb_native_path")
    if re.fullmatch(r"[0-9a-f]{64}", str(native.get("library_sha256", ""))) is None:
        failures.append("environment_bnb_native_sha256")
    toolchain = receipt.get("toolchain", {})
    for name in ("driver_version", "compiler_fingerprint", "cuda_fingerprint"):
        if not str(toolchain.get(name, "")).strip():
            failures.append(f"environment_toolchain_{name}")
    return failures


def validate_environment_source_lock(
    source_lock_path: Path,
    *,
    expected_environment_root: Path,
    expected_content_manifest_sha256: str,
) -> dict[str, Any]:
    """Re-hash every artifact named by the B0.2 environment source lock."""

    source_lock_path = source_lock_path.resolve()
    source_lock = _read_json(source_lock_path)
    _require(
        source_lock.get("schema_version")
        == "org_e2b_b02_h200_environment_source_lock_v1",
        "environment source-lock schema mismatch",
    )
    _require(source_lock.get("status") == "PASS", "environment source lock did not PASS")
    _require(
        str(Path(str(source_lock.get("environment_root", ""))).resolve())
        == str(expected_environment_root.resolve()),
        "environment source-lock root mismatch",
    )
    _require(
        source_lock.get("environment_content_manifest_sha256")
        == expected_content_manifest_sha256,
        "environment source-lock content-manifest mismatch",
    )
    files = source_lock.get("files")
    _require(isinstance(files, dict) and bool(files), "environment source-lock files missing")
    failures: list[str] = []
    for raw_path, expected_sha256 in sorted(files.items()):
        path = Path(str(raw_path))
        if not path.is_absolute():
            failures.append(f"non_absolute:{raw_path}")
        elif re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) is None:
            failures.append(f"invalid_sha256:{raw_path}")
        elif not path.is_file():
            failures.append(f"missing:{raw_path}")
        elif sha256_file(path) != expected_sha256:
            failures.append(f"sha256:{raw_path}")
    _require(
        not failures,
        "environment source-lock entry verification failed: " + ",".join(failures),
    )
    return {
        "path": str(source_lock_path),
        "sha256": sha256_file(source_lock_path),
        "schema_version": source_lock["schema_version"],
        "entries": len(files),
        "all_entries_rehashed": True,
        "environment_content_manifest_sha256": expected_content_manifest_sha256,
    }


def validate_environment_toolchain(
    toolchain: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash external compiler/CUDA tools omitted from the environment tree."""

    expected_names = ("gcc", "g++", "ld", "nvcc", "ninja", "nvidia-smi")
    locked_records = toolchain.get("executables")
    _require(isinstance(locked_records, dict), "environment toolchain records missing")
    _require(
        set(locked_records) == set(expected_names),
        "environment toolchain executable set mismatch",
    )
    current_records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name in expected_names:
        locked = locked_records[name]
        if not isinstance(locked, dict) or locked.get("available") is not True:
            failures.append(f"locked_tool_unavailable:{name}")
            continue
        locked_path = Path(str(locked.get("path", "")))
        if not locked_path.is_absolute():
            failures.append(f"locked_tool_path_not_absolute:{name}")
            continue
        current = executable_record_at_path(locked_path)
        current_records[name] = current
        if current != locked:
            failures.append(f"tool_identity_drift:{name}")
    if set(current_records) != set(expected_names):
        failures.append("current_toolchain_record_set")
    compiler_payload = {
        name: current_records.get(name) for name in ("gcc", "g++", "ld", "ninja")
    }
    compiler_fingerprint = canonical_json_sha256(compiler_payload)
    if compiler_fingerprint != toolchain.get("compiler_fingerprint"):
        failures.append("compiler_fingerprint_drift")
    cuda_module = str(toolchain.get("cuda_module", ""))
    loaded_modules = [
        value for value in os.environ.get("LOADEDMODULES", "").split(":") if value
    ]
    if not cuda_module or cuda_module not in loaded_modules:
        failures.append("frozen_cuda_module_not_loaded")
    active_paths = {
        name: str(Path(raw).resolve()) if (raw := shutil.which(name)) else None
        for name in expected_names
    }
    for name in expected_names:
        if active_paths[name] != current_records.get(name, {}).get("path"):
            failures.append(f"active_tool_path_drift:{name}")
    _require(
        not failures,
        "environment toolchain revalidation failed: " + ",".join(failures),
    )
    return {
        "status": "PASS",
        "executables": current_records,
        "all_executables_rehashed": True,
        "all_versions_reexecuted": True,
        "active_paths": active_paths,
        "compiler_fingerprint": compiler_fingerprint,
        "cuda_module": cuda_module,
        "loaded_modules": loaded_modules,
        "locked_cuda_fingerprint": toolchain.get("cuda_fingerprint"),
        "locked_driver_version": toolchain.get("driver_version"),
    }


def validate_environment_lock(args: argparse.Namespace) -> dict[str, Any]:
    if args.execution_mode == "engineering_preflight":
        _require(
            args.environment_lock_receipt is None,
            "engineering preflight must not consume a strict environment receipt",
        )
        return {
            "status": "NOT_APPLICABLE_ENGINEERING_PREFLIGHT",
            "strict_environment_lock_linked": False,
            "environment_root": str(Path(args.expected_environment_root).resolve()),
        }
    _require(args.environment_lock_receipt is not None, "strict mode requires environment-lock receipt")
    _require(args.environment_lock_builder is not None, "strict mode requires environment-lock builder")
    _require(args.environment_lock_source_manifest is not None, "strict mode requires environment-lock source manifest")
    receipt_path = args.environment_lock_receipt.resolve()
    receipt = _read_json(receipt_path)
    expected_versions = {
        "torch": args.expected_torch_version,
        "bitsandbytes": args.expected_bnb_version,
        "transformers": args.expected_transformers_version,
        "accelerate": args.expected_accelerate_version,
        "safetensors": args.expected_safetensors_version,
    }
    failures = environment_lock_failures(
        receipt,
        expected_environment_root=args.expected_environment_root,
        expected_versions=expected_versions,
        expected_cuda_version=args.expected_cuda_version,
    )
    _require(not failures, "environment-lock contract failure: " + ",".join(failures))
    age = _age_hours(str(receipt.get("collected_utc", "")))
    _require(0.0 <= age <= MAX_PREREQUISITE_AGE_HOURS, f"environment lock age is {age:.3f} hours")
    tree_manifest_path = Path(
        str(receipt["environment_tree_manifest"]["path"])
    ).resolve()
    paths_and_hashes = [
        (
            tree_manifest_path,
            str(receipt["environment_tree_manifest"]["sha256"]),
            "environment tree manifest",
        ),
        (
            args.environment_lock_source_manifest.resolve(),
            str(receipt["source_manifest"]["sha256"]),
            "environment source manifest",
        ),
        (
            args.environment_lock_builder.resolve(),
            str(receipt["builder"]["sha256"]),
            "environment lock builder",
        ),
    ]
    for path, expected, label in paths_and_hashes:
        _require(path.is_file(), f"{label} unavailable: {path}")
        _require(sha256_file(path) == expected, f"{label} hash mismatch")
    _require(
        Path(str(receipt["source_manifest"]["path"])).resolve()
        == args.environment_lock_source_manifest.resolve(),
        "environment source-manifest path mismatch",
    )
    _require(
        Path(str(receipt["builder"]["path"])).resolve()
        == args.environment_lock_builder.resolve(),
        "environment builder path mismatch",
    )
    locked_tree = _read_json(tree_manifest_path)
    expected_root = Path(args.expected_environment_root).resolve()
    current_tree = inventory_environment_tree(expected_root)
    tree_failures = environment_tree_revalidation_failures(
        locked_tree,
        current_tree,
        expected_environment_root=expected_root,
    )
    locked_content_sha256 = str(
        receipt["environment_tree_manifest"]["content_manifest_sha256"]
    )
    if locked_tree.get("content_manifest_sha256") != locked_content_sha256:
        tree_failures.append("receipt_locked_tree_content_digest_mismatch")
    if current_tree.get("content_manifest_sha256") != locked_content_sha256:
        tree_failures.append("receipt_current_tree_content_digest_mismatch")
    _require(
        not tree_failures,
        "full environment-tree revalidation failed: " + ",".join(tree_failures),
    )
    source_lock_revalidation = validate_environment_source_lock(
        args.environment_lock_source_manifest,
        expected_environment_root=expected_root,
        expected_content_manifest_sha256=locked_content_sha256,
    )
    toolchain_revalidation = validate_environment_toolchain(receipt["toolchain"])
    return {
        "status": "PASS",
        "strict_environment_lock_linked": True,
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "age_hours": age,
        "environment_root": str(expected_root),
        "interpreter_identity": receipt["interpreter_identity"],
        "packages": receipt["packages"],
        "bitsandbytes_native": receipt["bitsandbytes_native"],
        "toolchain": receipt["toolchain"],
        "environment_tree_manifest": receipt["environment_tree_manifest"],
        "full_environment_tree_revalidation": {
            "status": "PASS",
            "algorithm": "org_e2b_b02_inventory_tree_exact_v1",
            "no_exclusions": True,
            "all_files_rehashed": True,
            "all_symlinks_revalidated": True,
            "file_count": current_tree["file_count"],
            "symlink_count": current_tree["symlink_count"],
            "total_bytes": current_tree["total_bytes"],
            "content_manifest_sha256": current_tree[
                "content_manifest_sha256"
            ],
        },
        "source_manifest": {
            **receipt["source_manifest"],
            "current_revalidation": source_lock_revalidation,
        },
        "toolchain_revalidation": toolchain_revalidation,
        "builder": receipt["builder"],
    }


def directory_file_manifest(directory: Path) -> dict[str, str]:
    _require(directory.is_dir(), f"directory unavailable: {directory}")
    rows: dict[str, str] = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        rows[path.relative_to(directory).as_posix()] = sha256_file(path)
    _require(bool(rows), f"directory has no files: {directory}")
    return rows


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_save_pretrained(model: Any, destination: Path) -> None:
    """Commit a save_pretrained directory with a same-filesystem rename."""

    destination = destination.resolve()
    parent = destination.parent
    _require(parent.is_dir(), f"checkpoint parent unavailable: {parent}")
    _require(not destination.exists(), f"refusing to overwrite checkpoint: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
    )
    committed = False
    try:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="1GB",
        )
        files = sorted(item for item in temporary.rglob("*") if item.is_file())
        _require(bool(files), "save_pretrained produced no files")
        for path in files:
            _fsync_path(path)
        directories = sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for directory in directories:
            _fsync_path(directory)
        _fsync_path(temporary)
        _require(not destination.exists(), f"refusing to overwrite checkpoint: {destination}")
        os.replace(temporary, destination)
        committed = True
        _fsync_path(parent)
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def initialize_model_by_fqn(torch: Any, model: Any) -> dict[str, str]:
    """Initialize every parameter deterministically from its full name."""

    hashes: dict[str, str] = {}
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            token = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
            index = torch.arange(
                parameter.numel(), dtype=torch.float32, device=parameter.device
            )
            phase = (token % 10007) / 10007.0
            if name.endswith("norm.weight"):
                values = 1.0 + 0.01 * torch.sin(index * 0.017 + phase)
            else:
                values = 0.035 * torch.sin(index * 0.013 + phase)
                values += 0.015 * torch.cos(index * 0.007 + phase * 0.5)
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))
            hashes[name] = nf4_engine_v1.tensor_sha256(parameter)
    _require(bool(hashes), "deterministic initializer found no parameters")
    return hashes


def validate_live_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Validate job 345096 as engineering provenance, never current strict runtime."""

    receipt_path = args.live_runtime_receipt.resolve()
    receipt = _read_json(receipt_path)
    actual_age = _age_hours(str(receipt.get("collected_utc", "")))
    _require(actual_age >= 0.0, f"live receipt age is {actual_age:.3f} hours")
    validation_now = (
        None
        if args.execution_mode == "engineering_preflight"
        else _parse_utc(str(receipt.get("collected_utc", "")))
    )
    # The V2 wrapper delegates to a frozen V1 validator whose schema constant
    # is normally configured by the wrapper CLI. B0.3b imports the wrapper as a
    # library, so scope the V2 schema to this call and restore the signed
    # historical module state afterward. This preserves the job-345122 source
    # chain byte-for-byte while removing import-order dependence here.
    previous_live_schema = nf4_gate_v2.engine_v1.LIVE_SCHEMA
    nf4_gate_v2.engine_v1.LIVE_SCHEMA = nf4_gate_v2.live_v2.SCHEMA_VERSION
    try:
        gate = nf4_gate_v2._validate_live_runtime_receipt_v2(
            receipt,
            expected_machine=args.expected_machine,
            expected_gpu_name_regex=args.expected_gpu_name_regex,
            expected_compute_capability=args.expected_compute_capability,
            expected_torch_version=args.provenance_torch_version,
            expected_bnb_version=args.provenance_bnb_version,
            live_probe_sha256=sha256_file(args.live_probe_script.resolve()),
            now=validation_now,
        )
    finally:
        nf4_gate_v2.engine_v1.LIVE_SCHEMA = previous_live_schema
    _require(
        str(gate.get("slurm_job_id")) == str(args.expected_live_job_id),
        "live receipt Slurm job ID mismatch",
    )
    return {
        **gate,
        "age_hours": actual_age,
        "role": (
            "historical_engineering_provenance_only"
            if args.execution_mode == "strict"
            else "engineering_runtime_prerequisite"
        ),
        "matched_to_current_runtime": args.execution_mode == "engineering_preflight",
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "schema_version": receipt.get("schema_version"),
    }


def low_level_receipt_failures(
    final: Mapping[str, Any],
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    *,
    expected_fresh_job_id: str,
    expected_live_receipt_sha256: str,
) -> list[str]:
    failures: list[str] = []
    checks = [
        (final.get("schema_version") == LOW_LEVEL_SCHEMA, "fresh_schema"),
        (final.get("phase") == "final", "fresh_phase"),
        (final.get("overall_status") == "PASS", "fresh_status"),
        (final.get("failures") == [], "fresh_failures"),
        (producer.get("phase") == "produce" and producer.get("status") == "PASS", "fresh_producer_status"),
        (consumer.get("phase") == "reload" and consumer.get("status") == "PASS", "fresh_consumer_status"),
        (consumer.get("producer_process_exited_before_reload") is True, "fresh_process_separation"),
        (producer.get("pid") != consumer.get("pid"), "fresh_process_pid_distinct"),
        (producer.get("runtime") == consumer.get("runtime"), "fresh_runtime_drift"),
        (str(producer.get("runtime", {}).get("slurm_job_id")) == str(expected_fresh_job_id), "fresh_job_id"),
        (str(consumer.get("runtime", {}).get("slurm_job_id")) == str(expected_fresh_job_id), "fresh_consumer_job_id"),
        (producer.get("live_runtime_receipt", {}).get("sha256") == expected_live_receipt_sha256, "fresh_live_receipt_link"),
        (consumer.get("live_runtime_receipt", {}).get("sha256") == expected_live_receipt_sha256, "fresh_consumer_live_receipt_link"),
    ]
    certification = final.get("certification", {})
    checks.extend(
        [
            (certification.get("synthetic_nf4_save_exit_fresh_reload") is True, "fresh_save_reload_certification"),
            (certification.get("plain_non_nested_quant_state") is True, "fresh_plain_nf4_certification"),
            (certification.get("exact_packed_state_hashes") is True, "fresh_hash_certification"),
            (certification.get("model_checkpoint_tested") is False, "fresh_scope_model"),
            (certification.get("mixed_protected_module_set_tested") is False, "fresh_scope_mixed"),
            (certification.get("b0_representation_certified") is False, "fresh_scope_b0"),
            (certification.get("production_authorized") is False, "fresh_scope_production"),
            (certification.get("scientific_claim_authorized") is False, "fresh_scope_claim"),
        ]
    )
    for passed, name in checks:
        if not passed:
            failures.append(name)
    return failures


def validate_low_level_receipt(
    args: argparse.Namespace, live_gate: Mapping[str, Any]
) -> dict[str, Any]:
    final_path = args.low_level_fresh_receipt.resolve()
    final = _read_json(final_path)
    producer_path = final_path.parent / "producer.json"
    consumer_path = final_path.parent / "consumer.json"
    artifact_path = final_path.parent / "packed_nf4.pt"
    reference_path = final_path.parent / "numerical_reference.pt"
    for path in (producer_path, consumer_path, artifact_path, reference_path):
        _require(path.is_file(), f"low-level evidence file unavailable: {path}")
    producer = _read_json(producer_path)
    consumer = _read_json(consumer_path)
    evidence = final.get("evidence", {})
    expected_hashes = {
        producer_path: evidence.get("producer_report_sha256"),
        consumer_path: evidence.get("consumer_report_sha256"),
        artifact_path: evidence.get("artifact_sha256"),
        reference_path: evidence.get("reference_sha256"),
    }
    for path, expected in expected_hashes.items():
        _require(expected == sha256_file(path), f"low-level evidence hash mismatch: {path.name}")
    failures = low_level_receipt_failures(
        final,
        producer,
        consumer,
        expected_fresh_job_id=args.expected_fresh_job_id,
        expected_live_receipt_sha256=str(live_gate["sha256"]),
    )
    _require(not failures, "low-level fresh receipt failed: " + ",".join(failures))
    age = _age_hours(str(final.get("collected_utc", "")))
    _require(age >= 0.0, f"low-level receipt age is {age:.3f} hours")
    if args.execution_mode == "engineering_preflight":
        _require(
            age <= MAX_PREREQUISITE_AGE_HOURS,
            f"low-level receipt age is {age:.3f} hours",
        )
    current_probe_sha = sha256_file(args.low_level_probe_script.resolve())
    current_engine_sha = sha256_file(args.low_level_engine_script.resolve())
    current_manifest_sha = sha256_file(args.low_level_source_manifest.resolve())
    current_launcher_sha = sha256_file(args.low_level_launcher.resolve())
    source = final.get("source", {})
    source_checks = [
        (source.get("v2_wrapper_sha256") == current_probe_sha, "low_level_probe_source_hash"),
        (source.get("v1_engine_sha256") == current_engine_sha, "low_level_engine_source_hash"),
        (source.get("source_manifest_sha256") == current_manifest_sha, "low_level_manifest_hash"),
        (source.get("launcher_sha256") == current_launcher_sha, "low_level_launcher_hash"),
    ]
    source_failures = [name for passed, name in source_checks if not passed]
    _require(
        not source_failures,
        "low-level source-chain failure: " + ",".join(source_failures),
    )
    return {
        "status": "PASS",
        "role": (
            "historical_engineering_provenance_only"
            if args.execution_mode == "strict"
            else "engineering_runtime_prerequisite"
        ),
        "matched_to_current_runtime": args.execution_mode == "engineering_preflight",
        "path": str(final_path),
        "sha256": sha256_file(final_path),
        "age_hours": age,
        "slurm_job_id": str(args.expected_fresh_job_id),
        "producer_report_sha256": sha256_file(producer_path),
        "consumer_report_sha256": sha256_file(consumer_path),
        "artifact_sha256": sha256_file(artifact_path),
        "reference_sha256": sha256_file(reference_path),
        "source_chain_verified": True,
        "linked_live_receipt_sha256": live_gate["sha256"],
    }


def prerequisite_gates(args: argparse.Namespace) -> dict[str, Any]:
    # Strictly validate the complete current runtime before consuming any
    # historical engineering prerequisites.
    environment = validate_environment_lock(args)
    live = validate_live_receipt(args)
    fresh = validate_low_level_receipt(args, live)
    return {
        "live_runtime_v2": live,
        "low_level_nf4_fresh_reload_v2": fresh,
        "packed_h200_environment_v1": environment,
    }


def stable_prerequisite_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop observation-time-only ages before prerequisite identity comparison."""

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: clean(child)
                for key, child in item.items()
                if key != "age_hours"
            }
        if isinstance(item, list):
            return [clean(child) for child in item]
        return item

    return clean(dict(value))


def stable_representation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare realized model facts while allowing only phase pre_quantized state."""

    cleaned = json.loads(json.dumps(value))
    quantizer = cleaned.get("hf_quantizer")
    if isinstance(quantizer, dict):
        quantizer.pop("pre_quantized", None)
    return cleaned


def _runtime(
    args: argparse.Namespace, prerequisites: Mapping[str, Any]
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    nf4_engine_v1._install_no_network_audit_guard()
    _require(platform.machine() == args.expected_machine, "host architecture mismatch")
    _require(os.environ.get("SLURM_JOB_ID"), "Slurm allocation is required")
    visible = [
        token
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    _require(len(visible) == 1, "exactly one CUDA_VISIBLE_DEVICES token is required")
    root = Path(args.expected_environment_root).resolve()
    executable = Path(sys.executable).resolve()
    environment_gate = prerequisites["packed_h200_environment_v1"]
    _require(executable.is_relative_to(root), "Python is outside expected environment")
    _require(Path(sys.prefix).resolve() == root, "sys.prefix differs from environment root")
    _require(Path(sys.exec_prefix).resolve() == root, "sys.exec_prefix differs from environment root")
    if args.execution_mode == "strict":
        locked_interpreter = environment_gate["interpreter_identity"]
        _require(
            executable == Path(str(locked_interpreter["executable"])).resolve(),
            "interpreter path differs from environment lock",
        )
        _require(
            sha256_file(executable) == locked_interpreter["executable_sha256"],
            "interpreter hash differs from environment lock",
        )
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE must equal 1")
    _require(not site.ENABLE_USER_SITE, "Python user site must be disabled")
    _require(not os.environ.get("PYTHONHOME"), "PYTHONHOME must be unset")
    torch = importlib.import_module("torch")
    bnb = importlib.import_module("bitsandbytes")
    torch_version = importlib.metadata.version("torch")
    bnb_version = importlib.metadata.version("bitsandbytes")
    _require(torch_version == args.expected_torch_version, f"Torch drift: {torch_version}")
    _require(bnb_version == args.expected_bnb_version, f"bitsandbytes drift: {bnb_version}")
    _require(str(getattr(torch.version, "cuda", None)) == args.expected_cuda_version, "Torch CUDA build drift")
    for package, module in (("torch", torch), ("bitsandbytes", bnb)):
        module_path = Path(module.__file__).resolve()
        _require(module_path.is_relative_to(root), f"{package} is outside expected environment")
    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "one live CUDA device is required",
    )
    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    capability = (int(properties.major), int(properties.minor))
    _require(
        re.search(args.expected_gpu_name_regex, gpu_name, re.I) is not None,
        f"GPU mismatch: {gpu_name}",
    )
    _require(
        capability == args.expected_compute_capability,
        f"compute capability mismatch: {capability}",
    )
    driver_version: str | None = None
    if args.execution_mode == "strict":
        nvidia_smi = Path(
            str(
                environment_gate["toolchain_revalidation"]["executables"]
                ["nvidia-smi"]["path"]
            )
        ).resolve()
        driver_version = nvidia_driver_version(nvidia_smi)
        _require(
            driver_version
            == environment_gate["toolchain_revalidation"]["locked_driver_version"],
            "NVIDIA driver version differs from environment lock",
        )
    cextension = importlib.import_module("bitsandbytes.cextension")
    native = getattr(cextension, "lib", None)
    _require(
        bool(getattr(native, "compiled_with_cuda", False)),
        "bitsandbytes is not CUDA-enabled",
    )
    native_candidates = (
        getattr(native, "_name", None),
        getattr(getattr(native, "_lib", None), "_name", None),
    )
    native_path = next(
        (
            Path(value).resolve()
            for value in native_candidates
            if value and Path(value).is_file()
        ),
        None,
    )
    _require(native_path is not None, "bitsandbytes native library path unavailable")
    _require(native_path.is_relative_to(root), "bitsandbytes native library outside environment")
    _require(hasattr(bnb, "matmul_4bit"), "bitsandbytes matmul_4bit missing")
    _require(hasattr(bnb.nn, "Linear4bit"), "bitsandbytes Linear4bit missing")
    runtime = {
        "machine": platform.machine(),
        "gpu_name": gpu_name,
        "compute_capability": list(capability),
        "python": sys.version,
        "executable": str(executable),
        "torch": torch_version,
        "bitsandbytes": bnb_version,
        "backend_contract": "transformers_bitsandbytes_plain_nf4_mixed_v1",
        "torch_compiled_cuda": getattr(torch.version, "cuda", None),
        "driver_version": driver_version,
        "bitsandbytes_native_library": str(native_path),
        "bitsandbytes_native_library_sha256": sha256_file(native_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8",
    )
    versions = {
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "safetensors": importlib.metadata.version("safetensors"),
    }
    expected_versions = {
        "transformers": args.expected_transformers_version,
        "accelerate": args.expected_accelerate_version,
        "safetensors": args.expected_safetensors_version,
    }
    for package, expected in expected_versions.items():
        _require(versions[package] == expected, f"{package} drift: {versions[package]}")
    transformers = importlib.import_module("transformers")
    accelerate = importlib.import_module("accelerate")
    safetensors = importlib.import_module("safetensors")
    safetensors_torch = importlib.import_module("safetensors.torch")
    package_paths = {
        "transformers": Path(transformers.__file__).resolve(),
        "accelerate": Path(accelerate.__file__).resolve(),
        "safetensors": Path(safetensors.__file__).resolve(),
    }
    for package, path in package_paths.items():
        _require(path.is_relative_to(root), f"{package} is outside expected environment")
    if args.execution_mode == "strict":
        locked_packages = environment_gate["packages"]
        for package, path in package_paths.items():
            locked = locked_packages[package]
            _require(path == Path(str(locked["module_path"])).resolve(), f"{package} module path differs from environment lock")
            _require(sha256_file(path) == locked["module_sha256"], f"{package} module hash differs from environment lock")
        for package, module in (("torch", torch), ("bitsandbytes", bnb)):
            path = Path(module.__file__).resolve()
            locked = locked_packages[package]
            _require(path == Path(str(locked["module_path"])).resolve(), f"{package} module path differs from environment lock")
            _require(sha256_file(path) == locked["module_sha256"], f"{package} module hash differs from environment lock")
        locked_native = environment_gate["bitsandbytes_native"]
        actual_native = Path(runtime["bitsandbytes_native_library"]).resolve()
        _require(actual_native == Path(str(locked_native["library_path"])).resolve(), "bitsandbytes native path differs from environment lock")
        _require(sha256_file(actual_native) == locked_native["library_sha256"], "bitsandbytes native hash differs from environment lock")
        current_gpu_identity = {
            "name": gpu_name,
            "compute_capability": list(capability),
            "total_memory": int(properties.total_memory),
            "torch_compiled_cuda": str(torch.version.cuda),
            "driver_version": driver_version,
        }
        toolchain_gate = environment_gate["toolchain_revalidation"]
        current_cuda_payload = {
            "module": toolchain_gate["cuda_module"],
            "nvcc": toolchain_gate["executables"]["nvcc"],
            "nvidia_smi": toolchain_gate["executables"]["nvidia-smi"],
            "gpu": current_gpu_identity,
        }
        current_cuda_fingerprint = canonical_json_sha256(current_cuda_payload)
        _require(
            current_cuda_fingerprint == toolchain_gate["locked_cuda_fingerprint"],
            "CUDA/GPU fingerprint differs from environment lock",
        )
        runtime["current_cuda_fingerprint"] = current_cuda_fingerprint
        runtime["compiler_fingerprint"] = toolchain_gate["compiler_fingerprint"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {
        **runtime,
        "transformers": versions["transformers"],
        "accelerate": versions["accelerate"],
        "safetensors": versions["safetensors"],
        "package_paths": {key: str(value) for key, value in package_paths.items()},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": SEED,
        "execution_mode": args.execution_mode,
        "strict_environment_lock_linked": args.execution_mode == "strict",
    }
    return torch, bnb, transformers, safetensors_torch, runtime


def _tiny_config(transformers: Any) -> Any:
    config = transformers.LlamaConfig(**MODEL_GEOMETRY)
    config.use_cache = False
    config.attention_implementation = "eager"
    config._attn_implementation = "eager"
    return config


def _quantization_config(torch: Any, transformers: Any) -> Any:
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_skip_modules=list(expected_dense_fqns()),
        llm_int8_enable_fp32_cpu_offload=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=torch.uint8,
    )


def _model_inputs(torch: Any, device: Any) -> dict[str, Any]:
    input_ids = torch.tensor(
        [
            [1, 7, 11, 19, 23, 31, 43, 2],
            [1, 5, 13, 17, 29, 37, 47, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _forward(torch: Any, model: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            use_cache=False,
            return_dict=True,
        )
    torch.cuda.synchronize()
    logits = outputs.logits.detach().contiguous().cpu()
    loss = outputs.loss.detach().float().reshape(1).cpu()
    _require(bool(torch.isfinite(logits).all().item()), "model logits are nonfinite")
    _require(bool(torch.isfinite(loss).all().item()), "model loss is nonfinite")
    return {"logits": logits, "loss": loss}


def matmul_attribution_failures(
    counts: Mapping[str, int], *, repeats: int
) -> list[str]:
    failures: list[str] = []
    expected = set(expected_packed_fqns())
    if set(counts) != expected:
        failures.append("bnb_matmul_fqn_universe")
    for name in sorted(expected):
        if counts.get(name) != repeats:
            failures.append(f"bnb_matmul_count:{name}")
    return failures


def _forward_with_execution_trace(
    torch: Any,
    bnb: Any,
    model: Any,
    inputs: Mapping[str, Any],
    *,
    repeats: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run forwards while proving every expected dense/packed module executes."""

    _require(repeats >= 2, "at least two traced forwards are required")
    named = dict(model.named_modules())
    expected = expected_linear_fqns()
    missing = [name for name in expected if name not in named]
    _require(not missing, "execution-trace modules missing: " + ",".join(missing))
    hook_counts = {name: 0 for name in expected}
    handles = []

    def hook_for(name: str) -> Callable[..., None]:
        def hook(*_unused: Any) -> None:
            hook_counts[name] += 1

        return hook

    for name in expected:
        handles.append(named[name].register_forward_hook(hook_for(name)))
    matmul_by_fqn = {name: 0 for name in expected_packed_fqns()}
    active_packed_fqns: list[str] = []
    for name in expected_packed_fqns():
        def enter(_module: Any, _inputs: Any, *, fqn: str = name) -> None:
            _require(not active_packed_fqns, "nested packed Linear4bit execution")
            active_packed_fqns.append(fqn)

        def leave(
            _module: Any, _inputs: Any, _output: Any, *, fqn: str = name
        ) -> None:
            _require(
                active_packed_fqns == [fqn],
                f"packed execution context mismatch: {fqn}",
            )
            active_packed_fqns.pop()

        handles.append(named[name].register_forward_pre_hook(enter))
        handles.append(named[name].register_forward_hook(leave))
    original_matmul = bnb.matmul_4bit

    def counted_matmul(*call_args: Any, **call_kwargs: Any) -> Any:
        _require(
            len(active_packed_fqns) == 1,
            "bnb.matmul_4bit call lacks an exact active module FQN",
        )
        matmul_by_fqn[active_packed_fqns[0]] += 1
        return original_matmul(*call_args, **call_kwargs)

    outputs: list[dict[str, Any]] = []
    bnb.matmul_4bit = counted_matmul
    try:
        for _ in range(repeats):
            outputs.append(_forward(torch, model, inputs))
    finally:
        bnb.matmul_4bit = original_matmul
        for handle in handles:
            handle.remove()
    expected_hook_counts = {name: repeats for name in expected}
    _require(hook_counts == expected_hook_counts, "model execution-hook count mismatch")
    attribution_failures = matmul_attribution_failures(
        matmul_by_fqn, repeats=repeats
    )
    _require(
        not attribution_failures,
        "bnb.matmul_4bit FQN attribution failure: "
        + ",".join(attribution_failures),
    )
    matmul_calls = sum(matmul_by_fqn.values())
    expected_matmul_calls = repeats * len(expected_packed_fqns())
    _require(
        matmul_calls == expected_matmul_calls,
        f"bnb.matmul_4bit count mismatch: {matmul_calls}!={expected_matmul_calls}",
    )
    return outputs, {
        "forwards": repeats,
        "hook_counts": hook_counts,
        "hook_counts_exact": True,
        "packed_fqns": list(expected_packed_fqns()),
        "dense_fqns": list(expected_dense_fqns()),
        "bnb_matmul_4bit_calls": matmul_calls,
        "expected_bnb_matmul_4bit_calls": expected_matmul_calls,
        "bnb_matmul_4bit_exact": True,
        "bnb_matmul_4bit_calls_by_fqn": matmul_by_fqn,
        "bnb_matmul_4bit_per_fqn_exact": True,
    }


def _tensor_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    delta = actual.float() - expected.float()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
    }


def _quant_state_tensor_manifest(
    tensor_hasher: Callable[[Any], str], state: Any
) -> dict[str, str]:
    packed = state.as_dict(packed=True)
    _require(bool(packed), "packed quantization state is empty")
    result: dict[str, str] = {}
    for key, value in sorted(packed.items()):
        _require(hasattr(value, "detach"), f"quantization state is not tensor: {key}")
        result[key] = tensor_hasher(value)
    return result


def nf4_realization_failures(facts: Mapping[str, Any]) -> list[str]:
    expected = {
        "module_class": "bitsandbytes.nn.modules.Linear4bit",
        "weight_class": "bitsandbytes.nn.modules.Params4bit",
        "quant_state_class": "bitsandbytes.functional.QuantState",
        "module_quant_state_same_object": True,
        "bnb_quantized": True,
        "module_compute_dtype": "torch.bfloat16",
        "weight_quant_storage": "torch.uint8",
        "weight_compress_statistics": False,
        "weight_quant_type": "nf4",
        "weight_blocksize": 64,
        "packed_dtype": "torch.uint8",
        "packed_device": "cuda:0",
        "quant_type": "nf4",
        "blocksize": 64,
        "quant_state_dtype": "torch.bfloat16",
        "nested": False,
        "state2_is_none": True,
        "offset_is_none": True,
        "bias_is_none": True,
        "absmax_class": "torch.Tensor",
        "absmax_dtype": "torch.float32",
        "absmax_device": "cuda:0",
        "absmax_finite": True,
        "code_class": "torch.Tensor",
        "code_dtype": "torch.float32",
        "code_device": "cuda:0",
        "code_shape": [16],
        "code_numel": 16,
        "code_finite": True,
    }
    failures: list[str] = []
    for key, value in expected.items():
        observed = facts.get(key)
        if isinstance(value, bool):
            matches = type(observed) is bool and observed is value
        else:
            matches = observed == value
        if not matches:
            failures.append(f"nf4_realization:{key}")
    logical_shape = facts.get("logical_shape")
    if facts.get("original_shape") != logical_shape:
        failures.append("nf4_realization:original_shape")
    if facts.get("packed_bytes") != facts.get("expected_packed_bytes"):
        failures.append("nf4_realization:packed_bytes")
    if not isinstance(logical_shape, list) or len(logical_shape) != 2:
        failures.append("nf4_realization:logical_shape")
    else:
        expected_absmax_numel = (
            int(logical_shape[0]) * int(logical_shape[1])
            + int(facts.get("blocksize", -1))
            - 1
        ) // int(facts.get("blocksize", -1)) if int(facts.get("blocksize", -1)) > 0 else -1
        if facts.get("absmax_shape") != [expected_absmax_numel]:
            failures.append("nf4_realization:absmax_shape")
        if facts.get("absmax_numel") != expected_absmax_numel:
            failures.append("nf4_realization:absmax_numel")
    for key in ("absmax_sha256", "code_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(facts.get(key, ""))) is None:
            failures.append(f"nf4_realization:{key}")
    if not isinstance(facts.get("quant_state_sha256"), dict) or not facts.get(
        "quant_state_sha256"
    ):
        failures.append("nf4_realization:quant_state_tensors")
    return failures


def exact_boolean_attribute(obj: Any, name: str, expected: bool) -> bool:
    marker = object()
    observed = getattr(obj, name, marker)
    _require(observed is not marker, f"missing exact boolean attribute: {name}")
    _require(type(observed) is bool, f"non-bool exact attribute: {name}")
    _require(observed is expected, f"exact boolean attribute mismatch: {name}")
    return observed


def live_parameter_manifest_failures(
    live: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> list[str]:
    """Tie every live named parameter to the frozen baseline tensor universe."""

    failures: list[str] = []
    baseline_tensors = baseline.get("tensor_manifest", {})
    baseline_facts = baseline.get("tensor_facts", {})
    expected_keys = set(baseline_tensors)
    actual_keys = set(live)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing:
            failures.append("live_parameter_universe_missing:" + "|".join(missing))
        if extra:
            failures.append("live_parameter_universe_extra:" + "|".join(extra))
    packed_keys = {f"{name}.weight" for name in expected_packed_fqns()}
    for key in sorted(expected_keys & actual_keys):
        facts = live[key]
        is_packed = key in packed_keys
        if type(facts.get("is_packed_nf4")) is not bool or facts.get(
            "is_packed_nf4"
        ) is not is_packed:
            failures.append(f"live_parameter_packed_role:{key}")
        if is_packed:
            expected = {
                "parameter_class": "bitsandbytes.nn.modules.Params4bit",
                "dtype": "torch.uint8",
                "device": "cuda:0",
            }
        else:
            expected = {
                "parameter_class": "torch.nn.parameter.Parameter",
                "dtype": "torch.bfloat16",
                "device": "cuda:0",
                "shape": baseline_facts.get(key, {}).get("shape"),
                "bytes": baseline_facts.get(key, {}).get("bytes"),
                "sha256": baseline_tensors.get(key),
            }
        for field, value in expected.items():
            if facts.get(field) != value:
                failures.append(f"live_parameter_{field}:{key}")
    return failures


def inspect_model(
    torch: Any,
    bnb: Any,
    model: Any,
    *,
    expected_pre_quantized: bool,
    baseline_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    quantized: list[str] = []
    dense: list[str] = []
    modules: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            quantized.append(name)
            parameter = module.weight
            _require(
                type(module) is bnb.nn.Linear4bit,
                f"Linear4bit subclass is forbidden: {name}",
            )
            _require(
                type(parameter) is bnb.nn.Params4bit,
                f"Params4bit subclass is forbidden: {name}",
            )
            state = getattr(parameter, "quant_state", None)
            _require(state is not None, f"missing quantization state: {name}")
            _require(
                type(state) is bnb.functional.QuantState,
                f"wrong QuantState class: {name}",
            )
            stats = state.as_dict(packed=True)
            _require(stats and all(torch.is_tensor(v) for v in stats.values()), f"non-tensor packed state: {name}")
            _require(
                getattr(module, "quant_state", None) is state,
                f"module and weight quantization-state identity differs: {name}",
            )
            absmax = getattr(state, "absmax", None)
            code = getattr(state, "code", None)
            _require(
                type(absmax) is torch.Tensor,
                f"QuantState.absmax is not an exact torch.Tensor: {name}",
            )
            _require(
                type(code) is torch.Tensor,
                f"QuantState.code is not an exact torch.Tensor: {name}",
            )
            original_shape = list(getattr(state, "shape", ()) or ())
            logical_shape = [int(module.out_features), int(module.in_features)]
            expected_bytes = (int(original_shape[0]) * int(original_shape[1]) + 1) // 2
            facts = {
                "module_class": f"{type(module).__module__}.{type(module).__name__}",
                "weight_class": f"{type(parameter).__module__}.{type(parameter).__name__}",
                "bnb_quantized": getattr(parameter, "bnb_quantized", None),
                "module_compute_dtype": str(getattr(module, "compute_dtype", None)),
                "weight_quant_storage": str(getattr(parameter, "quant_storage", None)),
                "weight_compress_statistics": getattr(
                    parameter, "compress_statistics", None
                ),
                "weight_quant_type": getattr(parameter, "quant_type", None),
                "weight_blocksize": int(getattr(parameter, "blocksize", -1)),
                "packed_dtype": str(parameter.data.dtype),
                "packed_device": str(parameter.data.device),
                "packed_shape": list(parameter.data.shape),
                "packed_bytes": int(parameter.data.numel() * parameter.data.element_size()),
                "expected_packed_bytes": expected_bytes,
                "quant_type": getattr(state, "quant_type", None),
                "blocksize": int(getattr(state, "blocksize", -1)),
                "quant_state_class": f"{type(state).__module__}.{type(state).__name__}",
                "module_quant_state_same_object": getattr(module, "quant_state", None)
                is state,
                "quant_state_dtype": str(getattr(state, "dtype", None)),
                "nested": getattr(state, "nested", None),
                "state2_is_none": getattr(state, "state2", None) is None,
                "offset_is_none": getattr(state, "offset", None) is None,
                "original_shape": original_shape,
                "logical_shape": logical_shape,
                "bias_is_none": module.bias is None,
                "absmax_class": f"{type(absmax).__module__}.{type(absmax).__name__}",
                "absmax_dtype": str(absmax.dtype),
                "absmax_device": str(absmax.device),
                "absmax_shape": list(absmax.shape),
                "absmax_numel": int(absmax.numel()),
                "absmax_finite": bool(torch.isfinite(absmax).all().item()),
                "absmax_sha256": nf4_engine_v1.tensor_sha256(absmax),
                "code_class": f"{type(code).__module__}.{type(code).__name__}",
                "code_dtype": str(code.dtype),
                "code_device": str(code.device),
                "code_shape": list(code.shape),
                "code_numel": int(code.numel()),
                "code_finite": bool(torch.isfinite(code).all().item()),
                "code_sha256": nf4_engine_v1.tensor_sha256(code),
                "packed_sha256": nf4_engine_v1.tensor_sha256(parameter.data),
                "quant_state_sha256": _quant_state_tensor_manifest(
                    nf4_engine_v1.tensor_sha256, state
                ),
            }
            realization_failures = nf4_realization_failures(facts)
            _require(
                not realization_failures,
                f"invalid NF4 representation: {name}:"
                + ",".join(realization_failures),
            )
            modules[name] = facts
        elif type(module) is torch.nn.Linear:
            dense.append(name)
            facts = {
                "module_class": f"{type(module).__module__}.{type(module).__name__}",
                "weight_dtype": str(module.weight.dtype),
                "weight_device": str(module.weight.device),
                "weight_shape": list(module.weight.shape),
                "weight_sha256": nf4_engine_v1.tensor_sha256(module.weight),
            }
            _require(facts["weight_dtype"] == "torch.bfloat16", f"dense dtype drift: {name}")
            _require(facts["weight_device"] == "cuda:0", f"dense device drift: {name}")
            modules[name] = facts
    failures = protected_set_failures(quantized, dense)
    _require(not failures, "model module identity failure: " + ",".join(failures))
    _require(getattr(model, "is_loaded_in_4bit", False) is True, "model lacks is_loaded_in_4bit")
    _require(getattr(model, "is_4bit_serializable", False) is True, "model is not 4-bit serializable")
    _require(getattr(model, "is_quantized", False) is True, "model lacks is_quantized")
    quantizer = getattr(model, "hf_quantizer", None)
    _require(quantizer is not None, "model lacks hf_quantizer")
    _require(type(quantizer).__name__ == "Bnb4BitHfQuantizer", "wrong hf_quantizer class")
    _require(
        type(quantizer).__module__ == "transformers.quantizers.quantizer_bnb_4bit",
        "wrong hf_quantizer module",
    )
    observed_pre_quantized = exact_boolean_attribute(
        quantizer, "pre_quantized", expected_pre_quantized
    )
    _require(getattr(model, "hf_device_map", None) == {"": 0}, "hf_device_map drift")
    _require(
        tuple(getattr(quantizer, "modules_to_not_convert", ()))
        == tuple(expected_dense_fqns()),
        "hf_quantizer modules_to_not_convert drift",
    )
    model_config = model.config
    _require(model_config.use_cache is False, "model config use_cache drift")
    _require(
        float(model_config.attention_dropout) == 0.0,
        "model config attention_dropout drift",
    )
    _require(
        getattr(model_config, "attention_implementation", None) == "eager",
        "model config public attention implementation drift",
    )
    _require(
        getattr(model_config, "_attn_implementation", None) == "eager",
        "model config realized attention implementation drift",
    )
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
    _require(parameter_devices == ["cuda:0"], "model parameter offload/device drift")
    _require(
        all(device == "cuda:0" for device in buffer_devices),
        "model buffer offload/device drift",
    )
    live_parameters = {
        name: {
            "parameter_class": f"{type(parameter).__module__}.{type(parameter).__name__}",
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "shape": list(parameter.shape),
            "bytes": int(parameter.numel() * parameter.element_size()),
            "sha256": nf4_engine_v1.tensor_sha256(parameter),
            "is_packed_nf4": type(parameter) is bnb.nn.Params4bit,
        }
        for name, parameter in sorted(model.named_parameters())
    }
    live_failures = live_parameter_manifest_failures(
        live_parameters, baseline_checkpoint
    )
    _require(
        not live_failures,
        "live parameter identity failure: " + ",".join(live_failures),
    )
    module_manifest = {
        name: {
            key: value
            for key, value in facts.items()
            if key.endswith("sha256") or key in {"quant_state_sha256"}
        }
        for name, facts in sorted(modules.items())
    }
    manifest = {
        "modules": module_manifest,
        "live_parameters": {
            name: facts["sha256"] for name, facts in sorted(live_parameters.items())
        },
    }
    return {
        "contract": MODEL_CONTRACT,
        "quantized_fqns": sorted(quantized),
        "dense_fqns": sorted(dense),
        "target_fqns": list(target_fqns()),
        "protected_dense_fqns": list(PROTECTED_FQNS),
        "expected_quantized_fqns": sorted(expected_packed_fqns()),
        "expected_dense_fqns": sorted(expected_dense_fqns()),
        "mixed_attention_and_mlp": True,
        "spans_layers": [0, 1],
        "modules": modules,
        "is_loaded_in_4bit": True,
        "is_4bit_serializable": True,
        "is_quantized": True,
        "hf_device_map": {"": 0},
        "parameter_devices": parameter_devices,
        "buffer_devices": buffer_devices,
        "cpu_or_disk_offload_detected": False,
        "live_parameter_manifest": live_parameters,
        "live_parameter_manifest_sha256": canonical_json_sha256(live_parameters),
        "live_parameter_universe_exact": True,
        "all_nonpacked_live_parameters_match_baseline": True,
        "hf_quantizer": {
            "class": f"{type(quantizer).__module__}.{type(quantizer).__name__}",
            "pre_quantized": observed_pre_quantized,
            "pre_quantized_attribute_present": True,
            "pre_quantized_exact_bool": True,
            "modules_to_not_convert": list(
                getattr(quantizer, "modules_to_not_convert", ())
            ),
        },
        "model_config_runtime": {
            "use_cache": model_config.use_cache,
            "attention_dropout": float(model_config.attention_dropout),
            "attention_implementation_public": getattr(
                model_config, "attention_implementation", None
            ),
            "attention_implementation_realized": getattr(
                model_config, "_attn_implementation", None
            ),
        },
    }, {
        "manifest": manifest,
        "sha256": canonical_json_sha256(manifest),
    }


def _checkpoint_tensor_manifest(
    torch: Any, safetensors_torch: Any, checkpoint_dir: Path
) -> dict[str, Any]:
    safe_files = sorted(checkpoint_dir.glob("*.safetensors"))
    _require(len(safe_files) == 1, "checkpoint must contain exactly one safetensors file")
    _require(not list(checkpoint_dir.glob("*.bin")), "pickle checkpoint file is forbidden")
    tensors = safetensors_torch.load_file(str(safe_files[0]), device="cpu")
    manifest = {
        key: nf4_engine_v1.tensor_sha256(value)
        for key, value in sorted(tensors.items())
    }
    facts = {
        key: {
            "sha256": manifest[key],
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": int(value.numel() * value.element_size()),
        }
        for key, value in sorted(tensors.items())
    }
    return {
        "file": safe_files[0].name,
        "file_sha256": sha256_file(safe_files[0]),
        "tensor_count": len(manifest),
        "tensor_manifest": manifest,
        "tensor_manifest_sha256": canonical_json_sha256(manifest),
        "tensor_facts": facts,
        "tensor_facts_sha256": canonical_json_sha256(facts),
    }


def checkpoint_representation_failures(
    representation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    tensors = checkpoint.get("tensor_manifest", {})
    tensor_facts = checkpoint.get("tensor_facts", {})
    baseline_tensors = baseline.get("tensor_manifest", {})
    baseline_facts = baseline.get("tensor_facts", {})
    modules = representation.get("modules", {})
    expected_tensor_keys = set(baseline_tensors)
    packed_weight_keys = {
        f"{name}.weight" for name in expected_packed_fqns()
    }
    for name in expected_packed_fqns():
        facts = modules.get(name, {})
        if tensors.get(f"{name}.weight") != facts.get("packed_sha256"):
            failures.append(f"checkpoint_packed_weight_hash:{name}")
        state_hashes = facts.get("quant_state_sha256", {})
        for key, digest in state_hashes.items():
            state_key = f"{name}.weight.{key}"
            expected_tensor_keys.add(state_key)
            if tensors.get(state_key) != digest:
                failures.append(f"checkpoint_quant_state_hash:{name}:{key}")
    for name in expected_dense_fqns():
        key = f"{name}.weight"
        facts = modules.get(name, {})
        if tensors.get(key) != facts.get("weight_sha256"):
            failures.append(f"checkpoint_dense_weight_hash:{name}")
        if tensors.get(key) != baseline_tensors.get(key):
            failures.append(f"checkpoint_dense_baseline_hash:{name}")
        if tensor_facts.get(key, {}).get("dtype") != "torch.bfloat16":
            failures.append(f"checkpoint_dense_dtype:{name}")
        if baseline_facts.get(key, {}).get("dtype") != "torch.bfloat16":
            failures.append(f"baseline_dense_dtype:{name}")
        quant_state_keys = sorted(
            item for item in tensors if item.startswith(key + ".")
        )
        if quant_state_keys:
            failures.append(f"checkpoint_dense_has_quant_state:{name}")
    actual_tensor_keys = set(tensors)
    if actual_tensor_keys != expected_tensor_keys:
        missing = sorted(expected_tensor_keys - actual_tensor_keys)
        extra = sorted(actual_tensor_keys - expected_tensor_keys)
        if missing:
            failures.append("checkpoint_tensor_universe_missing:" + "|".join(missing))
        if extra:
            failures.append("checkpoint_tensor_universe_extra:" + "|".join(extra))
    for key, baseline_digest in sorted(baseline_tensors.items()):
        if key not in packed_weight_keys and tensors.get(key) != baseline_digest:
            failures.append(f"checkpoint_nontarget_baseline_hash:{key}")
    return failures


def _save_reference(
    safetensors_torch: Any,
    path: Path,
    tensors: Mapping[str, Any],
) -> None:
    _require(path.parent.is_dir(), f"reference parent unavailable: {path.parent}")
    _require(not path.exists(), f"refusing to overwrite reference: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        safetensors_torch.save_file(
            {key: value.detach().contiguous().cpu() for key, value in tensors.items()},
            str(temporary),
            metadata={"schema_version": REFERENCE_SCHEMA},
        )
        _fsync_path(temporary)
        _require(not path.exists(), f"refusing to overwrite reference: {path}")
        os.replace(temporary, path)
        _fsync_path(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _source_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "launcher_path": str(args.launcher.resolve()),
        "launcher_sha256": sha256_file(args.launcher.resolve()),
        "source_manifest_path": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest.resolve()),
        "live_probe_sha256": sha256_file(args.live_probe_script.resolve()),
        "low_level_probe_sha256": sha256_file(args.low_level_probe_script.resolve()),
        "low_level_engine_sha256": sha256_file(args.low_level_engine_script.resolve()),
        "low_level_launcher_sha256": sha256_file(args.low_level_launcher.resolve()),
        "low_level_source_manifest_sha256": sha256_file(args.low_level_source_manifest.resolve()),
    }


def _manifest_gate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    return validate_source_manifest(
        args.source_manifest.resolve(),
        repository_root=root,
        required_paths=(Path(__file__), args.launcher),
    )


def _config_from_disk(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    _require(config_path.is_file(), "checkpoint config.json unavailable")
    config = _read_json(config_path)
    failures = model_config_failures(config)
    _require(not failures, "checkpoint config failure: " + ",".join(failures))
    return config


def produce(args: argparse.Namespace) -> dict[str, Any]:
    process_uuid = str(uuid.uuid4())
    manifest_gate = _manifest_gate(args)
    prerequisites = prerequisite_gates(args)
    torch, bnb, transformers, safetensors_torch, runtime = _runtime(
        args, prerequisites
    )
    _require(not args.baseline_dir.exists(), "baseline directory already exists")
    _require(not args.checkpoint_dir.exists(), "checkpoint directory already exists")
    config = _tiny_config(transformers)
    baseline = transformers.LlamaForCausalLM(config).to(dtype=torch.bfloat16)
    initialized_parameter_hashes = initialize_model_by_fqn(torch, baseline)
    baseline.eval()
    atomic_save_pretrained(baseline, args.baseline_dir)
    baseline_tensors = _checkpoint_tensor_manifest(
        torch, safetensors_torch, args.baseline_dir
    )
    baseline_manifest = directory_file_manifest(args.baseline_dir)
    _require("model.safetensors" in baseline_manifest, "baseline safetensors unavailable")
    _require(not any(name.endswith(".bin") for name in baseline_manifest), "baseline pickle file forbidden")
    for name, digest in initialized_parameter_hashes.items():
        _require(
            baseline_tensors["tensor_manifest"].get(name) == digest,
            f"baseline parameter persistence mismatch: {name}",
        )
    del baseline

    quantization_config = _quantization_config(torch, transformers)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.baseline_dir.resolve()),
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        attn_implementation="eager",
        output_loading_info=True,
    )
    _require(isinstance(loaded, tuple) and len(loaded) == 2, "producer loading_info unavailable")
    model, loading_info = loaded
    loading_failures = loading_info_failures(loading_info)
    _require(not loading_failures, "producer loading_info failure: " + ",".join(loading_failures))
    representation, representation_hashes = inspect_model(
        torch,
        bnb,
        model,
        expected_pre_quantized=False,
        baseline_checkpoint=baseline_tensors,
    )
    inputs = _model_inputs(torch, torch.device("cuda:0"))
    forward_outputs, execution_trace = _forward_with_execution_trace(
        torch, bnb, model, inputs, repeats=2
    )
    first, second = forward_outputs
    logits_repeat = _tensor_metrics(torch, second["logits"], first["logits"])
    loss_repeat = _tensor_metrics(torch, second["loss"], first["loss"])
    _require(
        logits_repeat["exact"] is True and logits_repeat["max_abs"] == 0.0,
        "producer same-process logits are not bitwise exact",
    )
    _require(
        loss_repeat["exact"] is True and loss_repeat["max_abs"] == 0.0,
        "producer same-process loss is not bitwise exact",
    )
    logits_atol = frozen_absolute_tolerance(logits_repeat["max_abs"])
    loss_atol = frozen_absolute_tolerance(loss_repeat["max_abs"])

    atomic_save_pretrained(model, args.checkpoint_dir)
    config_on_disk = _config_from_disk(args.checkpoint_dir)
    checkpoint_tensors = _checkpoint_tensor_manifest(
        torch, safetensors_torch, args.checkpoint_dir
    )
    checkpoint_failures = checkpoint_representation_failures(
        representation, checkpoint_tensors, baseline_tensors
    )
    _require(
        not checkpoint_failures,
        "saved checkpoint representation mismatch: " + ",".join(checkpoint_failures),
    )
    checkpoint_files = directory_file_manifest(args.checkpoint_dir)
    reference_tensors = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": inputs["labels"],
        "producer_logits": first["logits"],
        "producer_loss": first["loss"],
    }
    _save_reference(safetensors_torch, args.reference, reference_tensors)
    reference_loaded = safetensors_torch.load_file(str(args.reference), device="cpu")
    _require(
        set(reference_loaded) == set(reference_tensors),
        "reference safetensors key mismatch",
    )
    for key, expected in reference_tensors.items():
        _require(
            torch.equal(reference_loaded[key], expected.detach().cpu()),
            f"reference tensor persistence mismatch: {key}",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "produce",
        "status": "PASS",
        "collected_utc": _utc_now().isoformat(),
        "pid": os.getpid(),
        "process_uuid": process_uuid,
        "boot_id": _boot_id(),
        "runtime": runtime,
        "prerequisites": prerequisites,
        "source_manifest_gate": manifest_gate,
        "source": _source_record(args),
        "model_contract": {
            "name": MODEL_CONTRACT,
            "seed": SEED,
            "deterministic_initializer": "fqn_sha256_sinusoid_bfloat16_v1",
            "initialized_parameter_hashes": initialized_parameter_hashes,
            "initialized_parameter_manifest_sha256": canonical_json_sha256(
                initialized_parameter_hashes
            ),
            "geometry": MODEL_GEOMETRY,
            "baseline_model_class": "transformers.LlamaForCausalLM",
            "load_api": "transformers.AutoModelForCausalLM.from_pretrained",
            "attention_implementation": "eager",
            "trust_remote_code": False,
            "target_fqns": list(target_fqns()),
            "protected_fqns": list(PROTECTED_FQNS),
            "packed_fqns": list(expected_packed_fqns()),
            "skip_fqns": list(expected_dense_fqns()),
            "quantization_config": config_on_disk["quantization_config"],
        },
        "representation": representation,
        "representation_hashes": representation_hashes,
        "loading_info": loading_info,
        "loading_info_clean": True,
        "execution_trace": execution_trace,
        "baseline_checkpoint": {
            "path": str(args.baseline_dir.resolve()),
            "file_manifest": baseline_manifest,
            "file_manifest_sha256": canonical_json_sha256(baseline_manifest),
            "safetensors": baseline_tensors,
            "atomic_save_pretrained_commit": True,
        },
        "packed_checkpoint": {
            "path": str(args.checkpoint_dir.resolve()),
            "file_manifest": checkpoint_files,
            "file_manifest_sha256": canonical_json_sha256(checkpoint_files),
            "safetensors": checkpoint_tensors,
            "save_api": "transformers.PreTrainedModel.save_pretrained",
            "safe_serialization": True,
            "atomic_save_pretrained_commit": True,
        },
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
            "format": "safetensors",
            "schema_version": REFERENCE_SCHEMA,
            "tensor_sha256": {
                key: nf4_engine_v1.tensor_sha256(value)
                for key, value in sorted(reference_loaded.items())
            },
        },
        "numerics": {
            "same_process_logits_exact": logits_repeat["exact"],
            "same_process_loss_exact": loss_repeat["exact"],
            "same_process_logits_repeatability": logits_repeat,
            "same_process_loss_repeatability": loss_repeat,
            "producer_logits_sha256": nf4_engine_v1.tensor_sha256(first["logits"]),
            "producer_loss_sha256": nf4_engine_v1.tensor_sha256(first["loss"]),
            "frozen_tolerances": {
                "logits_atol": logits_atol,
                "loss_atol": loss_atol,
                "rtol": 0.0,
                "same_process_hard_ceiling": REPEATABILITY_HARD_CEILING,
                "rule": "bitwise_exact_same_process_then_max(1e-6,5x_repeatability)",
            },
            "same_process_logits_within_frozen_tolerance": True,
            "same_process_loss_within_frozen_tolerance": True,
        },
        "scope_guards": {
            "execution_mode": args.execution_mode,
            "strict_environment_lock_linked": args.execution_mode == "strict",
            "network_guard_installed": True,
            "local_files_only": True,
            "package_install_attempted": False,
            "model_download_attempted": False,
            "remote_code_allowed": False,
            "tiny_synthetic_model_only": True,
            "ouqoqu_roles_tested": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
    }


def reload(args: argparse.Namespace) -> dict[str, Any]:
    process_uuid = str(uuid.uuid4())
    producer = _read_json(args.producer_report)
    _require(producer.get("schema_version") == SCHEMA_VERSION, "producer schema mismatch")
    _require(producer.get("phase") == "produce" and producer.get("status") == "PASS", "producer did not PASS")
    producer_pid = int(producer.get("pid", -1))
    _require(producer_pid == args.expected_producer_pid, "producer PID argument mismatch")
    _require(os.getpid() != producer_pid, "reload is not a fresh process")
    _require(process_uuid != producer.get("process_uuid"), "process UUID collision")
    _require(not _pid_alive(producer_pid), "producer remains alive at reload")
    _require(producer.get("boot_id") == _boot_id(), "producer/reload boot mismatch")
    _require(
        directory_file_manifest(args.checkpoint_dir)
        == producer.get("packed_checkpoint", {}).get("file_manifest"),
        "packed checkpoint file manifest drift",
    )
    _require(
        directory_file_manifest(args.baseline_dir)
        == producer.get("baseline_checkpoint", {}).get("file_manifest"),
        "baseline checkpoint file manifest drift",
    )
    _require(
        sha256_file(args.reference) == producer.get("reference", {}).get("sha256"),
        "reference file hash drift",
    )
    manifest_gate = _manifest_gate(args)
    prerequisites = prerequisite_gates(args)
    torch, bnb, transformers, safetensors_torch, runtime = _runtime(
        args, prerequisites
    )
    config_on_disk = _config_from_disk(args.checkpoint_dir)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint_dir.resolve()),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        attn_implementation="eager",
        output_loading_info=True,
    )
    _require(isinstance(loaded, tuple) and len(loaded) == 2, "consumer loading_info unavailable")
    model, loading_info = loaded
    loading_failures = loading_info_failures(loading_info)
    _require(not loading_failures, "consumer loading_info failure: " + ",".join(loading_failures))
    baseline_tensors = _checkpoint_tensor_manifest(
        torch, safetensors_torch, args.baseline_dir
    )
    representation, representation_hashes = inspect_model(
        torch,
        bnb,
        model,
        expected_pre_quantized=True,
        baseline_checkpoint=baseline_tensors,
    )
    _require(
        stable_representation_identity(representation)
        == stable_representation_identity(producer.get("representation", {})),
        "realized model representation facts drift after from_pretrained",
    )
    _require(
        representation_hashes.get("sha256")
        == producer.get("representation_hashes", {}).get("sha256"),
        "packed/dense representation hash drift after from_pretrained",
    )
    checkpoint_tensors = _checkpoint_tensor_manifest(
        torch, safetensors_torch, args.checkpoint_dir
    )
    _require(
        baseline_tensors == producer.get("baseline_checkpoint", {}).get("safetensors"),
        "baseline safetensors manifest drift",
    )
    _require(
        checkpoint_tensors
        == producer.get("packed_checkpoint", {}).get("safetensors"),
        "checkpoint safetensors manifest drift",
    )
    checkpoint_failures = checkpoint_representation_failures(
        representation, checkpoint_tensors, baseline_tensors
    )
    _require(
        not checkpoint_failures,
        "reloaded checkpoint representation mismatch: " + ",".join(checkpoint_failures),
    )
    reference = safetensors_torch.load_file(str(args.reference), device="cpu")
    _require(
        nf4_engine_v1.tensor_sha256(reference["producer_logits"])
        == producer.get("numerics", {}).get("producer_logits_sha256"),
        "producer logits reference hash mismatch",
    )
    _require(
        nf4_engine_v1.tensor_sha256(reference["producer_loss"])
        == producer.get("numerics", {}).get("producer_loss_sha256"),
        "producer loss reference hash mismatch",
    )
    device = torch.device("cuda:0")
    inputs = {
        "input_ids": reference["input_ids"].to(device),
        "attention_mask": reference["attention_mask"].to(device),
        "labels": reference["labels"].to(device),
    }
    forward_outputs, execution_trace = _forward_with_execution_trace(
        torch, bnb, model, inputs, repeats=2
    )
    first, second = forward_outputs
    logits_repeat = _tensor_metrics(torch, second["logits"], first["logits"])
    loss_repeat = _tensor_metrics(torch, second["loss"], first["loss"])
    _require(
        logits_repeat["exact"] is True and logits_repeat["max_abs"] == 0.0,
        "consumer same-process logits are not bitwise exact",
    )
    _require(
        loss_repeat["exact"] is True and loss_repeat["max_abs"] == 0.0,
        "consumer same-process loss is not bitwise exact",
    )
    logits_metrics = _tensor_metrics(torch, first["logits"], reference["producer_logits"])
    loss_metrics = _tensor_metrics(torch, first["loss"], reference["producer_loss"])
    frozen = producer.get("numerics", {}).get("frozen_tolerances", {})
    producer_logits_repeat = producer.get("numerics", {}).get(
        "same_process_logits_repeatability", {}
    )
    producer_loss_repeat = producer.get("numerics", {}).get(
        "same_process_loss_repeatability", {}
    )
    logits_atol = float(frozen.get("logits_atol", -1.0))
    loss_atol = float(frozen.get("loss_atol", -1.0))
    _require(frozen.get("rtol") == 0.0, "frozen relative tolerance must be zero")
    _require(
        frozen.get("rule")
        == "bitwise_exact_same_process_then_max(1e-6,5x_repeatability)",
        "frozen tolerance rule drift",
    )
    _require(
        frozen.get("same_process_hard_ceiling")
        == REPEATABILITY_HARD_CEILING,
        "same-process hard ceiling drift",
    )
    _require(
        logits_atol
        == frozen_absolute_tolerance(float(producer_logits_repeat.get("max_abs", -1.0))),
        "frozen logits tolerance derivation mismatch",
    )
    _require(
        loss_atol
        == frozen_absolute_tolerance(float(producer_loss_repeat.get("max_abs", -1.0))),
        "frozen loss tolerance derivation mismatch",
    )
    _require(
        logits_repeat["max_abs"] <= logits_atol,
        "consumer same-artifact logits exceed frozen tolerance",
    )
    _require(
        loss_repeat["max_abs"] <= loss_atol,
        "consumer same-artifact loss exceeds frozen tolerance",
    )
    _require(
        logits_metrics["max_abs"] <= logits_atol,
        "cross-process logits exceed frozen tolerance",
    )
    _require(
        loss_metrics["max_abs"] <= loss_atol,
        "cross-process loss exceeds frozen tolerance",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "reload",
        "status": "PASS",
        "collected_utc": _utc_now().isoformat(),
        "pid": os.getpid(),
        "process_uuid": process_uuid,
        "boot_id": _boot_id(),
        "producer_pid": producer_pid,
        "producer_process_uuid": producer.get("process_uuid"),
        "producer_alive_at_reload": False,
        "producer_process_exited_before_reload": True,
        "producer_report": {
            "path": str(args.producer_report.resolve()),
            "sha256": sha256_file(args.producer_report),
        },
        "runtime": runtime,
        "prerequisites": prerequisites,
        "source_manifest_gate": manifest_gate,
        "source": _source_record(args),
        "model_contract": producer.get("model_contract"),
        "config_quantization_metadata": config_on_disk.get("quantization_config"),
        "representation": representation,
        "representation_hashes": representation_hashes,
        "loading_info": loading_info,
        "loading_info_clean": True,
        "execution_trace": execution_trace,
        "baseline_checkpoint": {
            "path": str(args.baseline_dir.resolve()),
            "file_manifest": directory_file_manifest(args.baseline_dir),
            "file_manifest_sha256": canonical_json_sha256(
                directory_file_manifest(args.baseline_dir)
            ),
            "safetensors": baseline_tensors,
        },
        "packed_checkpoint": {
            "path": str(args.checkpoint_dir.resolve()),
            "file_manifest": directory_file_manifest(args.checkpoint_dir),
            "file_manifest_sha256": canonical_json_sha256(
                directory_file_manifest(args.checkpoint_dir)
            ),
            "safetensors": checkpoint_tensors,
            "load_api": "transformers.AutoModelForCausalLM.from_pretrained",
            "quantization_config_override_supplied": False,
            "local_files_only": True,
        },
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
        },
        "numerics": {
            "same_process_logits_exact": logits_repeat["exact"],
            "same_process_loss_exact": loss_repeat["exact"],
            "same_process_logits_repeatability": logits_repeat,
            "same_process_loss_repeatability": loss_repeat,
            "same_process_logits_within_frozen_tolerance": True,
            "same_process_loss_within_frozen_tolerance": True,
            "cross_process_logits": logits_metrics,
            "cross_process_loss": loss_metrics,
            "cross_process_logits_within_frozen_tolerance": True,
            "cross_process_loss_within_frozen_tolerance": True,
            "frozen_tolerances": producer.get("numerics", {}).get("frozen_tolerances"),
        },
        "scope_guards": {
            "execution_mode": args.execution_mode,
            "strict_environment_lock_linked": args.execution_mode == "strict",
            "network_guard_installed": True,
            "local_files_only": True,
            "package_install_attempted": False,
            "model_download_attempted": False,
            "remote_code_allowed": False,
            "tiny_synthetic_model_only": True,
            "ouqoqu_roles_tested": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
    }


def evidence_chain_failures(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    *,
    baseline_file_manifest: Mapping[str, str],
    packed_file_manifest: Mapping[str, str],
    reference_sha256: str,
) -> list[str]:
    failures: list[str] = []
    checks = [
        (producer.get("schema_version") == SCHEMA_VERSION, "producer_schema"),
        (consumer.get("schema_version") == SCHEMA_VERSION, "consumer_schema"),
        (producer.get("phase") == "produce" and producer.get("status") == "PASS", "producer_status"),
        (consumer.get("phase") == "reload" and consumer.get("status") == "PASS", "consumer_status"),
        (consumer.get("producer_process_exited_before_reload") is True, "process_separation"),
        (consumer.get("producer_alive_at_reload") is False, "producer_alive"),
        (producer.get("pid") == consumer.get("producer_pid"), "producer_pid_link"),
        (producer.get("pid") != consumer.get("pid"), "process_pid_distinct"),
        (bool(producer.get("process_uuid")) and producer.get("process_uuid") == consumer.get("producer_process_uuid"), "producer_uuid_link"),
        (producer.get("process_uuid") != consumer.get("process_uuid"), "process_uuid_distinct"),
        (producer.get("boot_id") == consumer.get("boot_id"), "boot_identity"),
        (producer.get("runtime") == consumer.get("runtime"), "runtime_drift"),
        (producer.get("model_contract") == consumer.get("model_contract"), "model_contract_drift"),
        (producer.get("representation_hashes", {}).get("sha256") == consumer.get("representation_hashes", {}).get("sha256"), "representation_hash_drift"),
        (stable_representation_identity(producer.get("representation", {})) == stable_representation_identity(consumer.get("representation", {})), "realized_representation_drift"),
        (producer.get("loading_info_clean") is True, "producer_loading_info"),
        (consumer.get("loading_info_clean") is True, "consumer_loading_info"),
        (producer.get("representation", {}).get("cpu_or_disk_offload_detected") is False, "producer_offload"),
        (consumer.get("representation", {}).get("cpu_or_disk_offload_detected") is False, "consumer_offload"),
        (producer.get("execution_trace") == consumer.get("execution_trace"), "execution_trace_drift"),
        (producer.get("execution_trace", {}).get("bnb_matmul_4bit_per_fqn_exact") is True, "producer_matmul_fqn_attribution"),
        (consumer.get("execution_trace", {}).get("bnb_matmul_4bit_per_fqn_exact") is True, "consumer_matmul_fqn_attribution"),
        (producer.get("representation", {}).get("hf_quantizer", {}).get("pre_quantized") is False, "producer_quantizer_pre_quantized"),
        (consumer.get("representation", {}).get("hf_quantizer", {}).get("pre_quantized") is True, "consumer_quantizer_pre_quantized"),
        (producer.get("representation", {}).get("hf_quantizer", {}).get("pre_quantized_attribute_present") is True, "producer_quantizer_attribute_presence"),
        (consumer.get("representation", {}).get("hf_quantizer", {}).get("pre_quantized_attribute_present") is True, "consumer_quantizer_attribute_presence"),
        (producer.get("representation", {}).get("hf_quantizer", {}).get("pre_quantized_exact_bool") is True, "producer_quantizer_exact_bool"),
        (consumer.get("representation", {}).get("hf_quantizer", {}).get("pre_quantized_exact_bool") is True, "consumer_quantizer_exact_bool"),
        (producer.get("baseline_checkpoint", {}).get("file_manifest") == baseline_file_manifest, "baseline_file_manifest"),
        (consumer.get("baseline_checkpoint", {}).get("file_manifest") == baseline_file_manifest, "consumer_baseline_file_manifest"),
        (producer.get("packed_checkpoint", {}).get("file_manifest") == packed_file_manifest, "packed_file_manifest"),
        (consumer.get("packed_checkpoint", {}).get("file_manifest") == packed_file_manifest, "consumer_packed_file_manifest"),
        (producer.get("reference", {}).get("sha256") == reference_sha256, "reference_hash"),
        (consumer.get("reference", {}).get("sha256") == reference_sha256, "consumer_reference_hash"),
        (producer.get("numerics", {}).get("same_process_logits_within_frozen_tolerance") is True, "producer_logits_repeat"),
        (producer.get("numerics", {}).get("same_process_loss_within_frozen_tolerance") is True, "producer_loss_repeat"),
        (consumer.get("numerics", {}).get("same_process_logits_within_frozen_tolerance") is True, "consumer_logits_repeat"),
        (consumer.get("numerics", {}).get("same_process_loss_within_frozen_tolerance") is True, "consumer_loss_repeat"),
        (producer.get("numerics", {}).get("same_process_logits_exact") is True, "producer_logits_not_exact"),
        (producer.get("numerics", {}).get("same_process_loss_exact") is True, "producer_loss_not_exact"),
        (consumer.get("numerics", {}).get("same_process_logits_exact") is True, "consumer_logits_not_exact"),
        (consumer.get("numerics", {}).get("same_process_loss_exact") is True, "consumer_loss_not_exact"),
        (consumer.get("numerics", {}).get("cross_process_logits_within_frozen_tolerance") is True, "cross_process_logits"),
        (consumer.get("numerics", {}).get("cross_process_loss_within_frozen_tolerance") is True, "cross_process_loss"),
        (producer.get("scope_guards", {}).get("production_authorized") is False, "producer_scope_production"),
        (consumer.get("scope_guards", {}).get("production_authorized") is False, "consumer_scope_production"),
        (producer.get("scope_guards", {}).get("scientific_claim_authorized") is False, "producer_scope_claim"),
        (consumer.get("scope_guards", {}).get("scientific_claim_authorized") is False, "consumer_scope_claim"),
    ]
    for passed, name in checks:
        if not passed:
            failures.append(name)
    return failures


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_gate = _manifest_gate(args)
    prerequisites = prerequisite_gates(args)
    producer = _read_json(args.producer_report)
    consumer = _read_json(args.consumer_report)
    current_files = directory_file_manifest(args.checkpoint_dir)
    current_baseline_files = directory_file_manifest(args.baseline_dir)
    failures = evidence_chain_failures(
        producer,
        consumer,
        baseline_file_manifest=current_baseline_files,
        packed_file_manifest=current_files,
        reference_sha256=sha256_file(args.reference),
    )
    if consumer.get("producer_report", {}).get("sha256") != sha256_file(args.producer_report):
        failures.append("producer_report_hash_chain")
    current_source = _source_record(args)
    if producer.get("source") != current_source:
        failures.append("producer_source_hash_chain")
    if consumer.get("source") != current_source:
        failures.append("consumer_source_hash_chain")
    producer_prerequisites = stable_prerequisite_identity(
        producer.get("prerequisites", {})
    )
    consumer_prerequisites = stable_prerequisite_identity(
        consumer.get("prerequisites", {})
    )
    current_prerequisites = stable_prerequisite_identity(prerequisites)
    if producer_prerequisites != consumer_prerequisites:
        failures.append("producer_consumer_prerequisite_drift")
    if producer_prerequisites != current_prerequisites:
        failures.append("current_prerequisite_drift")
    if producer.get("source_manifest_gate") != manifest_gate:
        failures.append("producer_source_manifest_gate_drift")
    if consumer.get("source_manifest_gate") != manifest_gate:
        failures.append("consumer_source_manifest_gate_drift")
    failures = list(dict.fromkeys(failures))
    status = "PASS" if not failures else "FAIL"
    strict_certified = (
        status == "PASS"
        and producer.get("scope_guards", {}).get("execution_mode") == "strict"
        and consumer.get("scope_guards", {}).get("execution_mode") == "strict"
        and prerequisites["packed_h200_environment_v1"].get(
            "strict_environment_lock_linked"
        )
        is True
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "final",
        "overall_status": status,
        "collected_utc": _utc_now().isoformat(),
        "failures": failures,
        "qualification": (
            "STRICT_B0_3B_CANARY_PASS"
            if strict_certified
            else "ENGINEERING_PREFLIGHT_PASS"
            if status == "PASS"
            else "FAIL"
        ),
        "source": current_source,
        "source_manifest_gate": manifest_gate,
        "prerequisites": prerequisites,
        "stable_prerequisite_identity_sha256": canonical_json_sha256(
            current_prerequisites
        ),
        "evidence": {
            "producer_report_sha256": sha256_file(args.producer_report),
            "consumer_report_sha256": sha256_file(args.consumer_report),
            "reference_sha256": sha256_file(args.reference),
            "baseline_checkpoint_file_manifest": current_baseline_files,
            "baseline_checkpoint_file_manifest_sha256": canonical_json_sha256(
                current_baseline_files
            ),
            "packed_checkpoint_file_manifest": current_files,
            "packed_checkpoint_file_manifest_sha256": canonical_json_sha256(current_files),
        },
        "certification": {
            "strict_b0_3b_model_roundtrip_certified": strict_certified,
            "eligible_as_strict_b0_3b_prerequisite_for_b0_4": strict_certified,
            "tiny_local_llama_model_checkpoint_tested": status == "PASS",
            "transformers_save_pretrained_safetensors": status == "PASS",
            "fresh_from_pretrained_without_quantization_override": status == "PASS",
            "exact_mixed_protected_module_fqns": status == "PASS",
            "exact_packed_parameter_and_state_hashes": status == "PASS",
            "dense_bfloat16_modules_preserved": status == "PASS",
            "dense_protected_weights_match_baseline_safetensors": status == "PASS",
            "protected_modules_have_no_quant_state_tensors": status == "PASS",
            "bnb_matmul_4bit_execution_path_observed": status == "PASS",
            "quantizer_pre_quantized_false_then_true": status == "PASS",
            "hf_device_map_exact_cuda0": status == "PASS",
            "loading_info_clean": status == "PASS",
            "cpu_or_disk_offload_absent": status == "PASS",
            "logits_and_loss_repeatability": status == "PASS",
            "low_level_nf4_prerequisite_linked": status == "PASS",
            "model_checkpoint_tested": status == "PASS",
            "mixed_protected_module_set_tested": status == "PASS",
            "ouqoqu_roles_tested": False,
            "b0_representation_certified": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
        "scope_guards": {
            "execution_mode": producer.get("scope_guards", {}).get(
                "execution_mode"
            ),
            "strict_environment_lock_linked": strict_certified,
            "strict_environment_lock_sha256": (
                prerequisites["packed_h200_environment_v1"].get("sha256")
                if strict_certified
                else None
            ),
            "tiny_synthetic_model_only": True,
            "network_or_download_allowed": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
        "limitations": [
            "This canary uses a deterministic tiny local two-layer Llama model, not O, U, R, QO, or QU.",
            (
                "It certifies one H200 plain-NF4 mixed-module roundtrip under "
                f"Transformers {producer.get('runtime', {}).get('transformers')} and "
                f"bitsandbytes {producer.get('runtime', {}).get('bitsandbytes')} only."
            ),
            "It does not certify GPTQ INT4, RTN INT8, model-scale memory behavior, evaluator validity, or any scientific outcome.",
            "A PASS is one prerequisite for later bounded canaries and cannot independently authorize E2B/E3 production.",
        ],
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execution-mode",
        choices=("strict", "engineering_preflight"),
        required=True,
    )
    parser.add_argument("--environment-lock-receipt", type=Path)
    parser.add_argument("--environment-lock-builder", type=Path)
    parser.add_argument("--environment-lock-source-manifest", type=Path)
    parser.add_argument("--live-runtime-receipt", type=Path, required=True)
    parser.add_argument("--low-level-fresh-receipt", type=Path, required=True)
    parser.add_argument("--live-probe-script", type=Path, required=True)
    parser.add_argument("--low-level-probe-script", type=Path, required=True)
    parser.add_argument("--low-level-engine-script", type=Path, required=True)
    parser.add_argument("--low-level-launcher", type=Path, required=True)
    parser.add_argument("--low-level-source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--expected-live-job-id", required=True)
    parser.add_argument("--expected-fresh-job-id", required=True)
    parser.add_argument("--provenance-torch-version", required=True)
    parser.add_argument("--provenance-bnb-version", required=True)
    parser.add_argument("--expected-machine", required=True)
    parser.add_argument("--expected-gpu-name-regex", required=True)
    parser.add_argument(
        "--expected-compute-capability",
        required=True,
        type=lambda value: tuple(int(item) for item in value.split(".")),
    )
    parser.add_argument("--expected-environment-root", required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-bnb-version", required=True)
    parser.add_argument("--expected-transformers-version", required=True)
    parser.add_argument("--expected-accelerate-version", required=True)
    parser.add_argument("--expected-safetensors-version", required=True)
    parser.add_argument("--expected-cuda-version", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    producer = sub.add_parser("produce")
    _common(producer)
    producer.add_argument("--baseline-dir", type=Path, required=True)
    producer.add_argument("--checkpoint-dir", type=Path, required=True)
    producer.add_argument("--reference", type=Path, required=True)
    producer.add_argument("--report", type=Path, required=True)
    consumer = sub.add_parser("reload")
    _common(consumer)
    consumer.add_argument("--baseline-dir", type=Path, required=True)
    consumer.add_argument("--checkpoint-dir", type=Path, required=True)
    consumer.add_argument("--reference", type=Path, required=True)
    consumer.add_argument("--producer-report", type=Path, required=True)
    consumer.add_argument("--expected-producer-pid", type=int, required=True)
    consumer.add_argument("--report", type=Path, required=True)
    finalizer = sub.add_parser("finalize")
    _common(finalizer)
    finalizer.add_argument("--baseline-dir", type=Path, required=True)
    finalizer.add_argument("--checkpoint-dir", type=Path, required=True)
    finalizer.add_argument("--reference", type=Path, required=True)
    finalizer.add_argument("--producer-report", type=Path, required=True)
    finalizer.add_argument("--consumer-report", type=Path, required=True)
    finalizer.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.phase == "produce":
            report = produce(args)
        elif args.phase == "reload":
            report = reload(args)
        else:
            report = finalize(args)
        write_json_atomic(report, args.report)
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "phase": args.phase,
            "status": "FAIL" if args.phase != "finalize" else None,
            "overall_status": "FAIL" if args.phase == "finalize" else None,
            "collected_utc": _utc_now().isoformat(),
            "failures": [
                f"{type(exc).__name__}: {' '.join(str(exc).split())[:1600]}"
            ],
            "scope_guards": {
                "production_authorized": False,
                "scientific_claim_authorized": False,
            },
        }
        try:
            write_json_atomic(failure, args.report)
        except Exception:
            pass
        print(
            f"ORG_E2B_NF4_MODEL_ROUNDTRIP_V1_{args.phase.upper()}=FAIL {exc}",
            file=sys.stderr,
        )
        return 1
    status = report.get("overall_status", report.get("status"))
    print(f"ORG_E2B_NF4_MODEL_ROUNDTRIP_V1_{args.phase.upper()}={status}")
    print(f"OUTPUT={args.report.resolve()}")
    print("PRODUCTION_AUTHORIZED=false")
    print("SCIENTIFIC_CLAIM_AUTHORIZED=false")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
