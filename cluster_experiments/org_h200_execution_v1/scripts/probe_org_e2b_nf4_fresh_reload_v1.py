#!/usr/bin/env python3
"""Fail-closed synthetic NF4 save/exit/fresh-reload canary for ORG B0.

The three phases are separate CLI invocations. ``produce`` creates a tiny
packed NF4 artifact and exits. ``reload`` must run in a different, later
process and reconstructs the packed tensors with
``bitsandbytes.nn.Params4bit.from_prequantized``. ``finalize`` validates the
evidence chain without importing CUDA libraries.

This is deliberately a synthetic representation canary.  It never downloads
or loads a model and cannot authorize E2B/E3 production or a scientific claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import socket
import sys
import tempfile
from typing import Any, Callable, Mapping
import uuid


SCHEMA_VERSION = "org_e2b_nf4_fresh_reload_evidence_v1"
ARTIFACT_SCHEMA = "org_e2b_synthetic_nf4_packed_artifact_v1"
REFERENCE_SCHEMA = "org_e2b_synthetic_nf4_numerical_reference_v1"
LIVE_SCHEMA = "org_e2b_live_gpu_runtime_probe_v1"
MAX_LIVE_RECEIPT_AGE_HOURS = 72.0
SUPPORTED_BNB_CONTRACTS = {
    "0.45.0": "bnb_quantstate_packed_params4bit_v1",
    "0.50.2": "bnb_quantstate_packed_params4bit_v1",
}
PROFILES = {
    "nf4_plain_stats": False,
}


class CanaryError(RuntimeError):
    """A fail-closed canary requirement was not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    _require(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
    _require(not path.exists(), f"refusing to overwrite evidence: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        _require(not path.exists(), f"refusing to overwrite evidence: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_torch_atomic(torch: Any, value: Any, path: Path) -> None:
    path = path.resolve()
    _require(path.parent.is_dir(), f"artifact parent does not exist: {path.parent}")
    _require(not path.exists(), f"refusing to overwrite artifact: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _require(not path.exists(), f"refusing to overwrite artifact: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_torch_weights_only(torch: Any, path: Path) -> Any:
    signature = inspect.signature(torch.load)
    _require("weights_only" in signature.parameters, "torch.load lacks weights_only")
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    raw = value.view(-1).view(getattr(importlib.import_module("torch"), "uint8"))
    data = raw.numpy().tobytes(order="C")
    header = f"{value.dtype}|{tuple(value.shape)}|".encode("ascii")
    return _sha256_bytes(header + data)


def tensor_tree_manifest(
    value: Any,
    *,
    is_tensor: Callable[[Any], bool],
    hasher: Callable[[Any], str],
    prefix: str = "root",
) -> dict[str, str]:
    """Hash all tensor leaves; dependencies are injectable for CPU-only tests."""

    result: dict[str, str] = {}
    if is_tensor(value):
        result[prefix] = hasher(value)
    elif isinstance(value, dict):
        for key in sorted(value):
            result.update(
                tensor_tree_manifest(
                    value[key], is_tensor=is_tensor, hasher=hasher,
                    prefix=f"{prefix}.{key}",
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(
                tensor_tree_manifest(
                    item, is_tensor=is_tensor, hasher=hasher,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def validate_live_runtime_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_machine: str,
    expected_gpu_name_regex: str,
    expected_compute_capability: tuple[int, int],
    expected_torch_version: str,
    expected_bnb_version: str,
    live_probe_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate the prerequisite live probe without importing Torch."""

    _require(receipt.get("schema_version") == LIVE_SCHEMA, "wrong live receipt schema")
    _require(receipt.get("overall_status") == "PASS", "live runtime receipt did not PASS")
    _require(receipt.get("failures") == [], "live runtime receipt has failures")
    _require(receipt.get("probe_scope") == "live_single_gpu_nf4_kernel_smoke_only", "wrong live probe scope")
    guards = receipt.get("scope_guards", {})
    _require(guards.get("save_reload_certified") is False, "live receipt has impossible save/reload claim")
    _require(guards.get("scientific_outcome_produced") is False, "live receipt claims a scientific outcome")
    _require(receipt.get("host", {}).get("machine") == expected_machine, "live receipt machine mismatch")
    cuda = receipt.get("cuda", {})
    _require(cuda.get("device_count") == 1, "live receipt did not isolate one CUDA device")
    _require(re.search(expected_gpu_name_regex, str(cuda.get("name", "")), re.I) is not None, "live receipt GPU mismatch")
    _require(tuple(cuda.get("capability", [])) == expected_compute_capability, "live receipt compute capability mismatch")
    packages = receipt.get("packages", {})
    _require(packages.get("torch") == expected_torch_version, "live receipt Torch version mismatch")
    _require(packages.get("bitsandbytes") == expected_bnb_version, "live receipt bitsandbytes version mismatch")
    _require(receipt.get("bitsandbytes_native", {}).get("compiled_with_cuda") is True, "bitsandbytes CUDA backend was not live")
    _require(receipt.get("nf4_functional", {}).get("status") == "PASS", "live NF4 functional smoke failed")
    _require(receipt.get("linear4bit_forward", {}).get("status") == "PASS", "live Linear4bit smoke failed")
    checks = receipt.get("checks", [])
    _require(isinstance(checks, list) and len(checks) >= 8, "live receipt check set is incomplete")
    _require(all(row.get("status") == "PASS" for row in checks), "live receipt contains a non-PASS check")
    _require(bool(receipt.get("slurm", {}).get("SLURM_JOB_ID")), "live receipt lacks a Slurm job ID")
    _require(receipt.get("probe_script", {}).get("sha256") == live_probe_sha256, "live probe source hash mismatch")
    collected = _parse_utc(str(receipt.get("collected_utc", "")))
    current = (now or _utc_now()).astimezone(timezone.utc)
    age_hours = (current - collected).total_seconds() / 3600.0
    _require(0.0 <= age_hours <= MAX_LIVE_RECEIPT_AGE_HOURS, f"live receipt age is {age_hours:.3f} hours")
    return {
        "status": "PASS",
        "age_hours": age_hours,
        "slurm_job_id": receipt.get("slurm", {}).get("SLURM_JOB_ID"),
        "gpu_name": cuda.get("name"),
        "torch": packages.get("torch"),
        "bitsandbytes": packages.get("bitsandbytes"),
    }


def backend_contract_for(version: str) -> str:
    _require(version in SUPPORTED_BNB_CONTRACTS, f"unsupported bitsandbytes version: {version}")
    return SUPPORTED_BNB_CONTRACTS[version]


def representation_fact_failures(facts: Mapping[str, Any]) -> list[str]:
    """Pure representation validator used by live code and CPU-only tests."""

    expected = {
        "linear4bit_instance": True,
        "params4bit_instance": True,
        "bnb_quantized": True,
        "packed_dtype": "torch.uint8",
        "packed_device": "cuda:0",
        "quant_type": "nf4",
        "blocksize": 64,
        "nested": False,
        "state2_is_none": True,
        "offset_is_none": True,
        "original_shape": [64, 128],
        "packed_bytes": 4096,
        "state_tensors_cuda0": True,
    }
    return [f"representation_{key}" for key, value in expected.items() if facts.get(key) != value]


def artifact_identity_failures(
    artifact: Mapping[str, Any], producer: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[str]:
    """Pure identity/linkage checks for a loaded packed artifact."""

    checks = [
        (artifact.get("schema_version") == ARTIFACT_SCHEMA, "artifact_schema_mismatch"),
        (artifact.get("backend_contract") == runtime.get("backend_contract"), "artifact_backend_contract_drift"),
        (artifact.get("producer_pid") == producer.get("pid"), "artifact_producer_pid_mismatch"),
        (artifact.get("producer_process_uuid") == producer.get("process_uuid"), "artifact_producer_uuid_mismatch"),
        (artifact.get("producer_boot_id") == producer.get("boot_id"), "artifact_producer_boot_mismatch"),
    ]
    return [name for passed, name in checks if not passed]


def _install_no_network_audit_guard() -> None:
    blocked = {"socket.connect", "socket.connect_ex", "socket.bind", "socket.sendto"}

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event in blocked:
            raise CanaryError(f"network operation blocked by audit guard: {event}")

    sys.addaudithook(hook)


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


def _runtime(
    *, expected_machine: str, expected_gpu_name_regex: str,
    expected_compute_capability: tuple[int, int], expected_environment_root: str,
    expected_torch_version: str, expected_bnb_version: str,
) -> tuple[Any, Any, dict[str, Any]]:
    _install_no_network_audit_guard()
    _require(platform.machine() == expected_machine, "host architecture mismatch")
    _require(os.environ.get("SLURM_JOB_ID"), "Slurm allocation is required")
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    _require(len(visible) == 1, "exactly one CUDA_VISIBLE_DEVICES token is required")
    torch = importlib.import_module("torch")
    bnb = importlib.import_module("bitsandbytes")
    torch_version = importlib.metadata.version("torch")
    bnb_version = importlib.metadata.version("bitsandbytes")
    _require(torch_version == expected_torch_version, f"Torch drift: {torch_version}")
    _require(bnb_version == expected_bnb_version, f"bitsandbytes drift: {bnb_version}")
    contract = backend_contract_for(bnb_version)
    root = Path(expected_environment_root).resolve()
    _require(Path(sys.executable).resolve().is_relative_to(root), "Python is outside expected environment")
    _require(Path(torch.__file__).resolve().is_relative_to(root), "Torch is outside expected environment")
    _require(Path(bnb.__file__).resolve().is_relative_to(root), "bitsandbytes is outside expected environment")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one live CUDA device is required")
    props = torch.cuda.get_device_properties(0)
    name = str(props.name)
    capability = (int(props.major), int(props.minor))
    _require(re.search(expected_gpu_name_regex, name, re.I) is not None, f"GPU mismatch: {name}")
    _require(capability == expected_compute_capability, f"compute capability mismatch: {capability}")
    cext = importlib.import_module("bitsandbytes.cextension")
    native = getattr(cext, "lib", None)
    _require(bool(getattr(native, "compiled_with_cuda", False)), "bitsandbytes is not CUDA-enabled")
    native_candidates = [
        getattr(native, "_name", None),
        getattr(getattr(native, "_lib", None), "_name", None),
    ]
    native_path = next(
        (Path(value).resolve() for value in native_candidates if value and Path(value).is_file()),
        None,
    )
    _require(native_path is not None, "bitsandbytes native library path is unavailable")
    _require(native_path.is_relative_to(root), "bitsandbytes native library is outside expected environment")
    _require(hasattr(bnb.functional.QuantState, "from_dict"), "QuantState.from_dict missing")
    _require(hasattr(bnb.nn.Params4bit, "from_prequantized"), "Params4bit.from_prequantized missing")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, bnb, {
        "machine": platform.machine(), "gpu_name": name,
        "compute_capability": list(capability), "python": sys.version,
        "executable": sys.executable, "torch": torch_version,
        "bitsandbytes": bnb_version, "backend_contract": contract,
        "torch_compiled_cuda": getattr(torch.version, "cuda", None),
        "bitsandbytes_native_library": str(native_path),
        "bitsandbytes_native_library_sha256": sha256_file(native_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _clone_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in stats.items()}


def _make_linear(torch: Any, bnb: Any, packed: Any, stats: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    linear = bnb.nn.Linear4bit(
        128, 64, bias=False, compute_dtype=torch.float32,
        compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8,
    )
    parameter = bnb.nn.Params4bit.from_prequantized(
        data=packed, quantized_stats=_clone_stats(stats), requires_grad=False,
        device=torch.device("cuda:0"), module=linear,
    )
    linear.weight = parameter
    linear.quant_state = parameter.quant_state
    state = parameter.quant_state
    state_tensors = [getattr(state, "absmax", None), getattr(state, "code", None)]
    facts = {
        "linear4bit_instance": isinstance(linear, bnb.nn.Linear4bit),
        "params4bit_instance": isinstance(parameter, bnb.nn.Params4bit),
        "bnb_quantized": bool(getattr(parameter, "bnb_quantized", False)),
        "packed_dtype": str(parameter.data.dtype),
        "packed_device": str(parameter.data.device),
        "quant_type": getattr(state, "quant_type", None),
        "blocksize": int(getattr(state, "blocksize", -1)),
        "nested": bool(getattr(state, "nested", False)),
        "state2_is_none": getattr(state, "state2", None) is None,
        "offset_is_none": getattr(state, "offset", None) is None,
        "original_shape": list(getattr(state, "shape", ()) or ()),
        "packed_bytes": int(parameter.data.numel() * parameter.data.element_size()),
        "state_tensors_cuda0": all(
            tensor is not None and str(tensor.device) == "cuda:0"
            for tensor in state_tensors
        ),
    }
    failures = representation_fact_failures(facts)
    _require(not failures, ",".join(failures))
    return linear, facts


def _output_metrics(torch: Any, actual: Any, reference: Any) -> dict[str, Any]:
    delta = actual.float() - reference.float()
    return {
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
    }


def produce(args: argparse.Namespace) -> dict[str, Any]:
    process_uuid = str(uuid.uuid4())
    live_path = args.live_runtime_receipt.resolve()
    live_probe_path = args.live_probe_script.resolve()
    live = _read_json(live_path)
    live_gate = validate_live_runtime_receipt(
        live, expected_machine=args.expected_machine,
        expected_gpu_name_regex=args.expected_gpu_name_regex,
        expected_compute_capability=args.expected_compute_capability,
        expected_torch_version=args.expected_torch_version,
        expected_bnb_version=args.expected_bnb_version,
        live_probe_sha256=sha256_file(live_probe_path),
    )
    torch, bnb, runtime = _runtime(
        expected_machine=args.expected_machine,
        expected_gpu_name_regex=args.expected_gpu_name_regex,
        expected_compute_capability=args.expected_compute_capability,
        expected_environment_root=args.expected_environment_root,
        expected_torch_version=args.expected_torch_version,
        expected_bnb_version=args.expected_bnb_version,
    )
    device = torch.device("cuda:0")
    weight = torch.linspace(-1.25, 1.25, steps=64 * 128, dtype=torch.float32, device=device).reshape(64, 128)
    input_single = torch.linspace(-0.75, 0.75, steps=128, dtype=torch.float32, device=device).reshape(1, 128)
    input_batch = torch.sin(torch.linspace(-2.0, 2.0, steps=4 * 128, dtype=torch.float32, device=device)).reshape(4, 128)
    artifact: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA,
        "backend_contract": runtime["backend_contract"],
        "producer_pid": os.getpid(), "producer_process_uuid": process_uuid,
        "producer_boot_id": _boot_id(),
        "profiles": {},
    }
    reference: dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA,
        "input_single": input_single.cpu(), "input_batch": input_batch.cpu(),
        "profiles": {},
    }
    profile_reports: dict[str, Any] = {}
    for profile, nested in PROFILES.items():
        packed, state = bnb.functional.quantize_4bit(
            weight, blocksize=64, compress_statistics=nested,
            quant_type="nf4", quant_storage=torch.uint8,
        )
        stats = state.as_dict(packed=True)
        _require(stats and all(torch.is_tensor(v) for v in stats.values()), "QuantState packed form must contain tensors only")
        dequantized = bnb.functional.dequantize_4bit(packed, quant_state=state).float()
        linear, representation = _make_linear(torch, bnb, packed, stats)
        with torch.inference_mode():
            out_single_a = linear(input_single)
            out_single_b = linear(input_single)
            out_batch_a = linear(input_batch)
            out_batch_b = linear(input_batch)
            ref_single = torch.nn.functional.linear(input_single, dequantized, None)
            ref_batch = torch.nn.functional.linear(input_batch, dequantized, None)
        torch.cuda.synchronize()
        _require(torch.equal(out_single_a, out_single_b), f"{profile} producer single repeat drift")
        _require(torch.equal(out_batch_a, out_batch_b), f"{profile} producer batch repeat drift")
        _require(bool(torch.isfinite(out_single_a).all().item()), f"{profile} producer single output is nonfinite")
        _require(bool(torch.isfinite(out_batch_a).all().item()), f"{profile} producer batch output is nonfinite")
        _require(torch.allclose(out_single_a.float(), ref_single.float(), atol=0.025, rtol=0.05), f"{profile} producer single reference mismatch")
        _require(torch.allclose(out_batch_a.float(), ref_batch.float(), atol=0.025, rtol=0.05), f"{profile} producer batch reference mismatch")
        artifact["profiles"][profile] = {
            "compress_statistics": nested,
            "packed": packed.detach().cpu(),
            "quantized_stats": {k: v.detach().cpu() for k, v in stats.items()},
        }
        reference["profiles"][profile] = {
            "dequantized_sha256": tensor_sha256(dequantized),
            "representation": representation,
            "producer_output_single": out_single_a.detach().cpu(),
            "producer_output_batch": out_batch_a.detach().cpu(),
            "dequantized_reference_single": ref_single.detach().cpu(),
            "dequantized_reference_batch": ref_batch.detach().cpu(),
        }
        profile_reports[profile] = {
            "packed_sha256": tensor_sha256(packed),
            "packed_state_sha256": {
                key: tensor_sha256(value) for key, value in sorted(stats.items())
            },
            "dequantized_sha256": tensor_sha256(dequantized),
            "single_reference_error": _output_metrics(torch, out_single_a, ref_single),
            "batch_reference_error": _output_metrics(torch, out_batch_a, ref_batch),
            "repeat_exact": True,
        }
    artifact_manifest = tensor_tree_manifest(artifact, is_tensor=torch.is_tensor, hasher=tensor_sha256)
    reference_manifest = tensor_tree_manifest(reference, is_tensor=torch.is_tensor, hasher=tensor_sha256)
    save_torch_atomic(torch, artifact, args.artifact)
    save_torch_atomic(torch, reference, args.reference)
    persisted_artifact = load_torch_weights_only(torch, args.artifact)
    persisted_manifest = tensor_tree_manifest(
        persisted_artifact, is_tensor=torch.is_tensor, hasher=tensor_sha256
    )
    _require(persisted_manifest == artifact_manifest, "packed tensors changed during save")
    return {
        "schema_version": SCHEMA_VERSION, "phase": "produce", "status": "PASS",
        "collected_utc": _utc_now().isoformat(), "pid": os.getpid(),
        "process_uuid": process_uuid,
        "boot_id": _boot_id(), "runtime": runtime,
        "live_runtime_gate": live_gate,
        "live_runtime_receipt": {"path": str(live_path), "sha256": sha256_file(live_path)},
        "source": {"script_sha256": sha256_file(Path(__file__).resolve()), "live_probe_sha256": sha256_file(live_probe_path), "source_manifest_sha256": sha256_file(args.source_manifest.resolve())},
        "artifact": {"path": str(args.artifact.resolve()), "sha256": sha256_file(args.artifact), "tensor_manifest": artifact_manifest, "tensor_manifest_sha256": _canonical_json_sha256(artifact_manifest), "producer_post_save_manifest_exact": True},
        "reference": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference), "tensor_manifest": reference_manifest, "tensor_manifest_sha256": _canonical_json_sha256(reference_manifest)},
        "profiles": profile_reports,
        "scope_guards": {
            "network_guard_installed": True, "package_install_attempted": False,
            "model_download_attempted": False, "model_checkpoint_loaded": False,
            "synthetic_artifact_only": True, "producer_process_exited_before_reload": False,
            "production_authorized": False, "scientific_claim_authorized": False,
        },
    }


def reload(args: argparse.Namespace) -> dict[str, Any]:
    process_uuid = str(uuid.uuid4())
    producer = _read_json(args.producer_report)
    _require(producer.get("status") == "PASS" and producer.get("phase") == "produce", "producer did not PASS")
    producer_pid = int(producer.get("pid", -1))
    _require(producer_pid == args.expected_producer_pid, "producer PID argument mismatch")
    _require(os.getpid() != producer_pid, "reload is not a fresh process")
    _require(process_uuid != producer.get("process_uuid"), "reload process UUID collision")
    _require(not _pid_alive(producer_pid), "producer process is still alive")
    _require(producer.get("boot_id") == _boot_id(), "producer and reload are from different boots")
    _require(sha256_file(args.artifact) == producer["artifact"]["sha256"], "artifact file hash mismatch")
    _require(sha256_file(args.reference) == producer["reference"]["sha256"], "reference file hash mismatch")
    live_path = args.live_runtime_receipt.resolve()
    live_probe_path = args.live_probe_script.resolve()
    live_gate = validate_live_runtime_receipt(
        _read_json(live_path), expected_machine=args.expected_machine,
        expected_gpu_name_regex=args.expected_gpu_name_regex,
        expected_compute_capability=args.expected_compute_capability,
        expected_torch_version=args.expected_torch_version,
        expected_bnb_version=args.expected_bnb_version,
        live_probe_sha256=sha256_file(live_probe_path),
    )
    torch, bnb, runtime = _runtime(
        expected_machine=args.expected_machine,
        expected_gpu_name_regex=args.expected_gpu_name_regex,
        expected_compute_capability=args.expected_compute_capability,
        expected_environment_root=args.expected_environment_root,
        expected_torch_version=args.expected_torch_version,
        expected_bnb_version=args.expected_bnb_version,
    )
    artifact = load_torch_weights_only(torch, args.artifact)
    reference = load_torch_weights_only(torch, args.reference)
    identity_failures = artifact_identity_failures(artifact, producer, runtime)
    _require(not identity_failures, ",".join(identity_failures))
    _require(reference.get("schema_version") == REFERENCE_SCHEMA, "reference schema mismatch")
    artifact_manifest = tensor_tree_manifest(artifact, is_tensor=torch.is_tensor, hasher=tensor_sha256)
    reference_manifest = tensor_tree_manifest(reference, is_tensor=torch.is_tensor, hasher=tensor_sha256)
    _require(artifact_manifest == producer["artifact"]["tensor_manifest"], "artifact tensor manifest mismatch")
    _require(reference_manifest == producer["reference"]["tensor_manifest"], "reference tensor manifest mismatch")
    device = torch.device("cuda:0")
    input_single = reference["input_single"].to(device)
    input_batch = reference["input_batch"].to(device)
    profile_reports: dict[str, Any] = {}
    for profile, nested in PROFILES.items():
        item = artifact["profiles"][profile]
        _require(item["compress_statistics"] is nested, f"{profile} profile flag mismatch")
        packed = item["packed"]
        stats = item["quantized_stats"]
        _require(tensor_sha256(packed) == producer["profiles"][profile]["packed_sha256"], f"{profile} packed bytes changed on disk")
        _require({k: tensor_sha256(v) for k, v in sorted(stats.items())} == producer["profiles"][profile]["packed_state_sha256"], f"{profile} packed state changed on disk")
        linear, representation = _make_linear(torch, bnb, packed, stats)
        rehydrated_stats = linear.weight.quant_state.as_dict(packed=True)
        _require(tensor_sha256(linear.weight.data) == producer["profiles"][profile]["packed_sha256"], f"{profile} packed weight changed during reconstruction")
        _require({k: tensor_sha256(v) for k, v in sorted(rehydrated_stats.items())} == producer["profiles"][profile]["packed_state_sha256"], f"{profile} state changed during reconstruction")
        dequantized = bnb.functional.dequantize_4bit(linear.weight.data, quant_state=linear.weight.quant_state).float()
        _require(tensor_sha256(dequantized) == producer["profiles"][profile]["dequantized_sha256"], f"{profile} dequantized tensor drift")
        with torch.inference_mode():
            single_a, single_b = linear(input_single), linear(input_single)
            batch_a, batch_b = linear(input_batch), linear(input_batch)
        torch.cuda.synchronize()
        expected = reference["profiles"][profile]
        _require(torch.equal(single_a, single_b), f"{profile} child single repeat drift")
        _require(torch.equal(batch_a, batch_b), f"{profile} child batch repeat drift")
        _require(bool(torch.isfinite(single_a).all().item()), f"{profile} child single output is nonfinite")
        _require(bool(torch.isfinite(batch_a).all().item()), f"{profile} child batch output is nonfinite")
        _require(torch.equal(single_a.cpu(), expected["producer_output_single"]), f"{profile} cross-process single output drift")
        _require(torch.equal(batch_a.cpu(), expected["producer_output_batch"]), f"{profile} cross-process batch output drift")
        _require(torch.allclose(single_a.cpu().float(), expected["dequantized_reference_single"].float(), atol=0.025, rtol=0.05), f"{profile} child single reference mismatch")
        _require(torch.allclose(batch_a.cpu().float(), expected["dequantized_reference_batch"].float(), atol=0.025, rtol=0.05), f"{profile} child batch reference mismatch")
        profile_reports[profile] = {
            "representation": representation,
            "packed_and_state_hashes_exact": True,
            "dequantized_hash_exact": True,
            "same_process_repeat_exact": True,
            "cross_process_output_exact": True,
            "dequantized_reference_agreement": True,
        }
    return {
        "schema_version": SCHEMA_VERSION, "phase": "reload", "status": "PASS",
        "collected_utc": _utc_now().isoformat(), "pid": os.getpid(),
        "process_uuid": process_uuid, "producer_process_uuid": producer.get("process_uuid"),
        "producer_pid": producer_pid, "producer_alive_at_reload": False,
        "producer_process_exited_before_reload": True, "boot_id": _boot_id(),
        "runtime": runtime, "live_runtime_gate": live_gate,
        "live_runtime_receipt": {"path": str(live_path), "sha256": sha256_file(live_path)},
        "source": {"script_sha256": sha256_file(Path(__file__).resolve()), "live_probe_sha256": sha256_file(live_probe_path), "source_manifest_sha256": sha256_file(args.source_manifest.resolve())},
        "producer_report": {"path": str(args.producer_report.resolve()), "sha256": sha256_file(args.producer_report)},
        "artifact_sha256": sha256_file(args.artifact), "reference_sha256": sha256_file(args.reference),
        "profiles": profile_reports,
        "scope_guards": {
            "network_guard_installed": True, "package_install_attempted": False,
            "model_download_attempted": False, "model_checkpoint_loaded": False,
            "synthetic_artifact_only": True, "production_authorized": False,
            "scientific_claim_authorized": False,
        },
    }


def validate_evidence_chain(
    producer: Mapping[str, Any], consumer: Mapping[str, Any],
    *, artifact_sha256: str, reference_sha256: str,
) -> list[str]:
    """Pure validation core, intentionally injectable/testable without CUDA."""

    failures: list[str] = []
    checks = [
        (producer.get("status") == "PASS" and producer.get("phase") == "produce", "producer_not_pass"),
        (consumer.get("status") == "PASS" and consumer.get("phase") == "reload", "consumer_not_pass"),
        (consumer.get("producer_process_exited_before_reload") is True, "process_separation_not_proven"),
        (consumer.get("producer_alive_at_reload") is False, "producer_alive_at_reload"),
        (producer.get("pid") == consumer.get("producer_pid"), "producer_pid_link_mismatch"),
        (producer.get("pid") != consumer.get("pid"), "same_process_used"),
        (bool(producer.get("process_uuid")) and producer.get("process_uuid") == consumer.get("producer_process_uuid"), "producer_process_uuid_link_mismatch"),
        (producer.get("process_uuid") != consumer.get("process_uuid"), "same_process_uuid_used"),
        (producer.get("artifact", {}).get("sha256") == artifact_sha256 == consumer.get("artifact_sha256"), "artifact_hash_chain_mismatch"),
        (producer.get("reference", {}).get("sha256") == reference_sha256 == consumer.get("reference_sha256"), "reference_hash_chain_mismatch"),
        (set(producer.get("profiles", {})) == set(PROFILES), "producer_profile_set_mismatch"),
        (set(consumer.get("profiles", {})) == set(PROFILES), "consumer_profile_set_mismatch"),
        (all(v.get("packed_and_state_hashes_exact") is True for v in consumer.get("profiles", {}).values()), "packed_state_hash_failure"),
        (all(v.get("cross_process_output_exact") is True for v in consumer.get("profiles", {}).values()), "cross_process_output_failure"),
        (producer.get("artifact", {}).get("producer_post_save_manifest_exact") is True, "producer_save_mutated_packed_state"),
        (producer.get("scope_guards", {}).get("production_authorized") is False, "producer_authorized_production"),
        (consumer.get("scope_guards", {}).get("production_authorized") is False, "consumer_authorized_production"),
        (producer.get("scope_guards", {}).get("scientific_claim_authorized") is False, "producer_authorized_claim"),
        (consumer.get("scope_guards", {}).get("scientific_claim_authorized") is False, "consumer_authorized_claim"),
    ]
    for passed, name in checks:
        if not passed:
            failures.append(name)
    return failures


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    producer = _read_json(args.producer_report)
    consumer = _read_json(args.consumer_report)
    failures = validate_evidence_chain(
        producer, consumer, artifact_sha256=sha256_file(args.artifact),
        reference_sha256=sha256_file(args.reference),
    )
    if consumer.get("producer_report", {}).get("sha256") != sha256_file(args.producer_report):
        failures.append("producer_report_hash_chain_mismatch")
    current_script_sha256 = sha256_file(Path(__file__).resolve())
    if producer.get("source", {}).get("script_sha256") != current_script_sha256:
        failures.append("producer_source_hash_drift")
    if consumer.get("source", {}).get("script_sha256") != current_script_sha256:
        failures.append("consumer_source_hash_drift")
    if producer.get("runtime") != consumer.get("runtime"):
        failures.append("producer_consumer_runtime_drift")
    if producer.get("live_runtime_receipt", {}).get("sha256") != consumer.get("live_runtime_receipt", {}).get("sha256"):
        failures.append("live_runtime_receipt_drift")
    if producer.get("source", {}).get("source_manifest_sha256") != consumer.get("source", {}).get("source_manifest_sha256"):
        failures.append("source_manifest_drift")
    source_manifest_sha256 = sha256_file(args.source_manifest.resolve())
    if producer.get("source", {}).get("source_manifest_sha256") != source_manifest_sha256:
        failures.append("source_manifest_hash_chain_mismatch")
    failures = list(dict.fromkeys(failures))
    status = "PASS" if not failures else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION, "phase": "final", "overall_status": status,
        "collected_utc": _utc_now().isoformat(), "failures": failures,
        "source": {
            "script_sha256": current_script_sha256,
            "launcher_path": str(args.launcher.resolve()),
            "launcher_sha256": sha256_file(args.launcher),
            "source_manifest_path": str(args.source_manifest.resolve()),
            "source_manifest_sha256": source_manifest_sha256,
        },
        "evidence": {
            "producer_report_sha256": sha256_file(args.producer_report),
            "consumer_report_sha256": sha256_file(args.consumer_report),
            "artifact_sha256": sha256_file(args.artifact),
            "reference_sha256": sha256_file(args.reference),
        },
        "certification": {
            "synthetic_nf4_save_exit_fresh_reload": status == "PASS",
            "plain_non_nested_quant_state": status == "PASS",
            "exact_packed_state_hashes": status == "PASS",
            "model_checkpoint_tested": False,
            "ouqoqu_roles_tested": False,
            "mixed_protected_module_set_tested": False,
            "b0_representation_certified": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
        "limitations": [
            "This is a tiny synthetic bitsandbytes NF4 representation canary only.",
            "compress_statistics=False is frozen to match the historical NF4 functional call; nested-state NF4 is a different intervention and is not tested.",
            "No O, U, R, QO, or QU checkpoint and no protected-module mixture was tested.",
            "A PASS cannot authorize E2B/E3 production or support a scientific result.",
            "GPTQ INT4 and frozen-algorithm RTN INT8 remain untested and blocked.",
        ],
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live-runtime-receipt", type=Path, required=True)
    parser.add_argument("--live-probe-script", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-machine", required=True)
    parser.add_argument("--expected-gpu-name-regex", required=True)
    parser.add_argument("--expected-compute-capability", required=True, type=lambda x: tuple(int(v) for v in x.split(".")))
    parser.add_argument("--expected-environment-root", required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-bnb-version", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("produce")
    _common(p)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    r = sub.add_parser("reload")
    _common(r)
    r.add_argument("--artifact", type=Path, required=True)
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--producer-report", type=Path, required=True)
    r.add_argument("--expected-producer-pid", type=int, required=True)
    r.add_argument("--report", type=Path, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--artifact", type=Path, required=True)
    f.add_argument("--reference", type=Path, required=True)
    f.add_argument("--producer-report", type=Path, required=True)
    f.add_argument("--consumer-report", type=Path, required=True)
    f.add_argument("--launcher", type=Path, required=True)
    f.add_argument("--source-manifest", type=Path, required=True)
    f.add_argument("--report", type=Path, required=True)
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
            "schema_version": SCHEMA_VERSION, "phase": args.phase,
            "status": "FAIL" if args.phase != "finalize" else None,
            "overall_status": "FAIL" if args.phase == "finalize" else None,
            "collected_utc": _utc_now().isoformat(),
            "failures": [f"{type(exc).__name__}: {' '.join(str(exc).split())[:1200]}"],
            "scope_guards": {"production_authorized": False, "scientific_claim_authorized": False},
        }
        try:
            write_json_atomic(failure, args.report)
        except Exception:
            pass
        print(f"ORG_E2B_NF4_FRESH_RELOAD_{args.phase.upper()}=FAIL {exc}", file=sys.stderr)
        return 1
    status = report.get("overall_status", report.get("status"))
    print(f"ORG_E2B_NF4_FRESH_RELOAD_{args.phase.upper()}={status}")
    print(f"OUTPUT={args.report.resolve()}")
    print("PRODUCTION_AUTHORIZED=false")
    print("SCIENTIFIC_CLAIM_AUTHORIZED=false")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
