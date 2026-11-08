#!/usr/bin/env python3
"""Live one-GPU kernel smoke for the ORG E2B packed-runtime gate.

This script must run inside an already granted one-GPU allocation.  It checks
CUDA visibility/isolation, the loaded bitsandbytes native library, deterministic
NF4 functional quantization/dequantization, and a tiny ``Linear4bit`` forward.

It deliberately does not load a model, download anything, install anything, or
exercise serialization/reload.  A PASS is only a live runtime/kernel smoke; it
is never packed-artifact or B0 representation certification.
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
import socket
import sys
import tempfile
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "org_e2b_live_gpu_runtime_probe_v1"


class RuntimeProbeError(RuntimeError):
    """A fail-closed live runtime check failed."""


def _short_error(exc: BaseException, limit: int = 1200) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": " ".join(str(exc).split())[:limit],
    }


def _distribution_version(
    name: str, version_provider: Callable[[str], str]
) -> str | None:
    try:
        return version_provider(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeProbeError(message)


def _path_is_within(path: str | os.PathLike[str] | None, root: str | os.PathLike[str]) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _require_cuda_tensor(tensor: Any, label: str) -> None:
    device = getattr(tensor, "device", None)
    _require(
        getattr(device, "type", None) == "cuda" and int(getattr(device, "index", 0) or 0) == 0,
        f"{label} is not on cuda:0 (device={device})",
    )


def parse_compute_capability(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("compute capability must have MAJOR.MINOR form")
    return int(match.group(1)), int(match.group(2))


def run_nf4_functional_smoke(torch_module: Any, bnb_module: Any) -> dict[str, Any]:
    """Exercise deterministic CUDA NF4 packing and dequantization."""

    torch = torch_module
    bnb = bnb_module
    device = torch.device("cuda:0")
    source = torch.linspace(
        -2.0,
        2.0,
        steps=64 * 64,
        dtype=torch.float32,
        device=device,
    ).reshape(64, 64)

    packed_a, state_a = bnb.functional.quantize_4bit(
        source,
        blocksize=64,
        compress_statistics=False,
        quant_type="nf4",
        quant_storage=torch.uint8,
    )
    packed_b, state_b = bnb.functional.quantize_4bit(
        source,
        blocksize=64,
        compress_statistics=False,
        quant_type="nf4",
        quant_storage=torch.uint8,
    )
    torch.cuda.synchronize()

    _require_cuda_tensor(source, "NF4 source")
    _require_cuda_tensor(packed_a, "NF4 packed tensor")
    _require_cuda_tensor(state_a.absmax, "NF4 absmax")
    _require_cuda_tensor(state_a.code, "NF4 quantization map")
    _require(packed_a.dtype == torch.uint8, f"NF4 storage is {packed_a.dtype}, not uint8")
    expected_packed_bytes = (source.numel() + 1) // 2
    actual_packed_bytes = packed_a.numel() * packed_a.element_size()
    _require(
        actual_packed_bytes == expected_packed_bytes,
        f"NF4 packed bytes {actual_packed_bytes} != expected {expected_packed_bytes}",
    )
    _require(torch.equal(packed_a, packed_b), "repeated NF4 packed tensors differ")
    _require(
        torch.equal(state_a.absmax, state_b.absmax),
        "repeated NF4 absmax tensors differ",
    )
    _require(getattr(state_a, "quant_type", None) == "nf4", "quant_state is not NF4")
    _require(int(getattr(state_a, "blocksize", -1)) == 64, "NF4 blocksize is not 64")
    _require(tuple(state_a.shape) == tuple(state_b.shape), "repeated NF4 state shapes differ")
    _require(state_a.dtype == state_b.dtype, "repeated NF4 state dtypes differ")
    _require(state_a.quant_type == state_b.quant_type, "repeated NF4 state types differ")
    _require(state_a.blocksize == state_b.blocksize, "repeated NF4 state blocks differ")
    _require(torch.equal(state_a.code, state_b.code), "repeated NF4 quantization maps differ")
    _require(state_a.state2 is None and state_b.state2 is None, "unexpected nested NF4 state")
    _require(state_a.offset is None and state_b.offset is None, "unexpected NF4 offset")
    packed_unique_values = int(torch.unique(packed_a).numel())
    _require(packed_unique_values > 1, "NF4 packed output is degenerate")

    restored_a = bnb.functional.dequantize_4bit(packed_a, quant_state=state_a)
    restored_b = bnb.functional.dequantize_4bit(packed_b, quant_state=state_b)
    torch.cuda.synchronize()
    _require_cuda_tensor(restored_a, "NF4 dequantized tensor")
    _require(tuple(restored_a.shape) == tuple(source.shape), "dequantized shape changed")
    _require(bool(torch.isfinite(restored_a).all().item()), "NF4 dequantization produced NaN/Inf")
    _require(torch.equal(restored_a, restored_b), "repeated NF4 dequantized tensors differ")

    error = restored_a.float() - source.float()
    max_abs_error = float(torch.max(torch.abs(error)).item())
    rmse = float(torch.sqrt(torch.mean(error.square())).item())
    restored_max_abs = float(torch.max(torch.abs(restored_a.float())).item())
    _require(restored_max_abs > 0.5, "NF4 dequantized output is degenerate")
    _require(1.0e-8 < rmse < 0.5, f"NF4 RMSE outside smoke bound: {rmse}")
    _require(max_abs_error < 0.75, f"NF4 max error outside smoke bound: {max_abs_error}")
    return {
        "status": "PASS",
        "input_shape": list(source.shape),
        "input_dtype": str(source.dtype),
        "quant_type": str(state_a.quant_type),
        "blocksize": int(state_a.blocksize),
        "packed_dtype": str(packed_a.dtype),
        "packed_shape": list(packed_a.shape),
        "packed_bytes": int(actual_packed_bytes),
        "expected_packed_bytes": int(expected_packed_bytes),
        "packed_sha256": _tensor_sha256(packed_a),
        "absmax_sha256": _tensor_sha256(state_a.absmax),
        "quant_map_sha256": _tensor_sha256(state_a.code),
        "packed_unique_values": packed_unique_values,
        "cuda_placement_verified": True,
        "full_non_nested_quant_state_repeat_verified": True,
        "repeat_packed_exact": True,
        "repeat_dequantized_exact": True,
        "max_abs_quantization_error": max_abs_error,
        "rmse_quantization_error": rmse,
    }


def run_linear4bit_forward_smoke(torch_module: Any, bnb_module: Any) -> dict[str, Any]:
    """Exercise a one-row NF4 Linear4bit forward without claiming dispatch tracing."""

    torch = torch_module
    bnb = bnb_module
    linear = bnb.nn.Linear4bit(
        64,
        32,
        bias=False,
        compute_dtype=torch.float32,
        compress_statistics=False,
        quant_type="nf4",
        quant_storage=torch.uint8,
    )
    reference_weight = torch.linspace(
        -1.0,
        1.0,
        steps=32 * 64,
        dtype=torch.float32,
    ).reshape(32, 64)
    with torch.no_grad():
        linear.weight.copy_(reference_weight)
    linear = linear.to(torch.device("cuda:0"))

    _require(isinstance(linear.weight, bnb.nn.Params4bit), "Linear4bit weight is not Params4bit")
    _require(bool(getattr(linear.weight, "bnb_quantized", False)), "Params4bit is not marked quantized")
    _require(linear.weight.data.dtype == torch.uint8, "Linear4bit weight storage is not uint8")
    quant_state = getattr(linear.weight, "quant_state", None)
    _require(quant_state is not None, "Linear4bit weight has no quant_state")
    _require(getattr(quant_state, "quant_type", None) == "nf4", "Linear4bit is not NF4")
    _require_cuda_tensor(linear.weight.data, "Linear4bit packed weight")
    _require_cuda_tensor(quant_state.absmax, "Linear4bit absmax")
    _require_cuda_tensor(quant_state.code, "Linear4bit quantization map")

    expected_packed_bytes = (reference_weight.numel() + 1) // 2
    actual_packed_bytes = linear.weight.data.numel() * linear.weight.data.element_size()
    _require(
        actual_packed_bytes == expected_packed_bytes,
        f"Linear4bit packed bytes {actual_packed_bytes} != expected {expected_packed_bytes}",
    )

    inputs = torch.linspace(
        -0.5,
        0.5,
        steps=64,
        dtype=torch.float32,
        device=torch.device("cuda:0"),
    ).reshape(1, 64)
    _require_cuda_tensor(inputs, "Linear4bit input")
    single_token_dispatch_preconditions = (
        inputs.numel() == inputs.shape[-1] and not bool(inputs.requires_grad)
    )
    _require(single_token_dispatch_preconditions, "single-token packed-forward preconditions not met")
    with torch.inference_mode():
        output_a = linear(inputs)
        output_b = linear(inputs)
        dequantized_weight = bnb.functional.dequantize_4bit(
            linear.weight.data,
            quant_state=quant_state,
        )
        reference_output = torch.nn.functional.linear(inputs, dequantized_weight, None)
    torch.cuda.synchronize()
    _require_cuda_tensor(output_a, "Linear4bit output")
    _require_cuda_tensor(dequantized_weight, "Linear4bit dequantized reference weight")
    _require_cuda_tensor(reference_output, "Linear4bit reference output")
    _require(bool(torch.isfinite(output_a).all().item()), "Linear4bit forward produced NaN/Inf")
    _require(torch.equal(output_a, output_b), "repeated Linear4bit outputs differ")
    _require(float(torch.max(torch.abs(output_a.float())).item()) > 1.0e-6, "Linear4bit output is degenerate")
    reference_diff = output_a.float() - reference_output.float()
    reference_max_abs = float(torch.max(torch.abs(reference_diff)).item())
    reference_rmse = float(torch.sqrt(torch.mean(reference_diff.square())).item())
    _require(
        bool(torch.allclose(output_a.float(), reference_output.float(), atol=0.025, rtol=0.05)),
        f"Linear4bit forward disagrees with dequantized-weight reference (max={reference_max_abs})",
    )

    return {
        "status": "PASS",
        "module_class": f"{type(linear).__module__}.{type(linear).__name__}",
        "weight_class": f"{type(linear.weight).__module__}.{type(linear.weight).__name__}",
        "weight_data_dtype": str(linear.weight.data.dtype),
        "weight_data_shape": list(linear.weight.data.shape),
        "weight_packed_bytes": int(actual_packed_bytes),
        "expected_packed_bytes": int(expected_packed_bytes),
        "weight_packed_sha256": _tensor_sha256(linear.weight.data),
        "bnb_quantized": True,
        "quant_type": str(quant_state.quant_type),
        "blocksize": int(quant_state.blocksize),
        "input_shape": list(inputs.shape),
        "output_shape": list(output_a.shape),
        "output_dtype": str(output_a.dtype),
        "cuda_placement_verified": True,
        "single_token_dispatch_preconditions_met": True,
        "native_kernel_dispatch_instrumented": False,
        "reference_atol": 0.025,
        "reference_rtol": 0.05,
        "reference_max_abs_difference": reference_max_abs,
        "reference_rmse_difference": reference_rmse,
        "dequantized_reference_agreement": True,
        "repeat_output_exact": True,
        "repeat_output_max_abs_drift": 0.0,
    }


def _add_check(
    report: dict[str, Any], name: str, passed: bool, detail: Any
) -> None:
    report["checks"].append(
        {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}
    )
    if not passed:
        report["failures"].append(name)


def collect_live_gpu_report(
    *,
    expected_machine: str,
    expected_gpu_name_regex: str,
    expected_compute_capability: tuple[int, int],
    expected_environment_root: str,
    require_slurm: bool,
    importer: Callable[[str], Any] = importlib.import_module,
    version_provider: Callable[[str], str] = importlib.metadata.version,
    environ: Mapping[str, str] | None = None,
    machine_provider: Callable[[], str] = platform.machine,
    hostname_provider: Callable[[], str] = socket.gethostname,
    executable_provider: Callable[[], str] = lambda: sys.executable,
    now: Callable[[], datetime] | None = None,
    functional_runner: Callable[[Any, Any], dict[str, Any]] = run_nf4_functional_smoke,
    linear_runner: Callable[[Any, Any], dict[str, Any]] = run_linear4bit_forward_smoke,
) -> dict[str, Any]:
    """Run fail-closed checks and return a report even when a check fails."""

    env = dict(os.environ if environ is None else environ)
    timestamp = (now or (lambda: datetime.now(timezone.utc)))()
    machine = machine_provider()
    executable = executable_provider()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collected_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "overall_status": "FAIL",
        "probe_scope": "live_single_gpu_nf4_kernel_smoke_only",
        "scope_guards": {
            "package_install_attempted": False,
            "network_access_attempted": False,
            "model_download_attempted": False,
            "model_weight_loaded": False,
            "model_artifact_serialization_attempted": False,
            "fresh_process_reload_attempted": False,
            "save_reload_certified": False,
            "mixed_protected_module_identity_tested": False,
            "native_linear4bit_dispatch_instrumented": False,
            "b0_representation_certified": False,
            "scientific_outcome_produced": False,
        },
        "host": {
            "hostname": hostname_provider(),
            "machine": machine,
            "platform": platform.platform(),
            "python": sys.version,
            "executable": executable,
        },
        "slurm": {
            key: env.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_NODELIST",
                "SLURM_GPUS_ON_NODE",
                "SLURM_STEP_GPUS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "expected": {
            "machine": expected_machine,
            "gpu_name_regex": expected_gpu_name_regex,
            "compute_capability": list(expected_compute_capability),
            "environment_root": expected_environment_root,
            "visible_cuda_devices": 1,
            "slurm_required": bool(require_slurm),
        },
        "packages": {
            name: _distribution_version(name, version_provider)
            for name in ("torch", "bitsandbytes")
        },
        "checks": [],
        "failures": [],
        "cuda": {},
        "bitsandbytes_native": {},
        "nf4_functional": None,
        "linear4bit_forward": None,
        "limitations": [
            "PASS means only that tiny NF4 CUDA kernels executed in this allocation.",
            "No model or quantized artifact was serialized and no fresh-process reload occurred.",
            "No arbitrary protected-module set or model checkpoint was tested.",
            "The Linear4bit call uses a one-row single-token shape, but native dispatch was not instrumented and is version-specific.",
            "GPTQ INT4 and frozen-algorithm RTN INT8 were not tested.",
            "Unit tests use dependency injection and cover orchestration only; CUDA kernels require this live job.",
            "This report cannot authorize B0, E2B, or E3 production.",
        ],
    }

    _add_check(report, "host_machine", machine == expected_machine, machine)
    _add_check(
        report,
        "python_executable_environment_root",
        _path_is_within(executable, expected_environment_root),
        {"executable": executable, "expected_root": expected_environment_root},
    )
    if require_slurm:
        _add_check(
            report,
            "slurm_allocation",
            bool(env.get("SLURM_JOB_ID")),
            env.get("SLURM_JOB_ID"),
        )

    try:
        torch = importer("torch")
        _add_check(report, "torch_import", True, getattr(torch, "__file__", None))
        _add_check(
            report,
            "torch_module_environment_root",
            _path_is_within(getattr(torch, "__file__", None), expected_environment_root),
            getattr(torch, "__file__", None),
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        _add_check(report, "torch_import", False, _short_error(exc))
        torch = None

    try:
        bnb = importer("bitsandbytes")
        _add_check(report, "bitsandbytes_import", True, getattr(bnb, "__file__", None))
        _add_check(
            report,
            "bitsandbytes_module_environment_root",
            _path_is_within(getattr(bnb, "__file__", None), expected_environment_root),
            getattr(bnb, "__file__", None),
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        _add_check(report, "bitsandbytes_import", False, _short_error(exc))
        bnb = None

    cextension = None
    if bnb is not None:
        try:
            cextension = importer("bitsandbytes.cextension")
            lib = getattr(cextension, "lib", None)
            compiled_with_cuda = bool(
                lib is not None and getattr(lib, "compiled_with_cuda", False)
            )
            report["bitsandbytes_native"] = {
                "compiled_with_cuda": compiled_with_cuda,
                "library_class": None if lib is None else type(lib).__name__,
                "library_path": None if lib is None else getattr(lib, "_name", None),
            }
            _add_check(
                report,
                "bitsandbytes_compiled_with_cuda",
                compiled_with_cuda,
                report["bitsandbytes_native"],
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            _add_check(report, "bitsandbytes_compiled_with_cuda", False, _short_error(exc))

    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count())
            report["cuda"].update(
                {
                    "available": cuda_available,
                    "device_count": device_count,
                    "torch_compiled_cuda": getattr(torch.version, "cuda", None),
                }
            )
            _add_check(report, "torch_cuda_available", cuda_available, cuda_available)
            _add_check(report, "single_visible_cuda_device", device_count == 1, device_count)

            visible_raw = env.get("CUDA_VISIBLE_DEVICES")
            visible_tokens = (
                [token.strip() for token in visible_raw.split(",") if token.strip()]
                if visible_raw
                else []
            )
            _add_check(
                report,
                "cuda_visible_devices_single_token",
                len(visible_tokens) == 1,
                visible_raw,
            )

            if cuda_available and device_count == 1:
                current_device = int(torch.cuda.current_device())
                properties = torch.cuda.get_device_properties(0)
                gpu_name = str(getattr(properties, "name", torch.cuda.get_device_name(0)))
                report["cuda"].update(
                    {
                        "current_device": current_device,
                        "name": gpu_name,
                        "capability": [
                            int(getattr(properties, "major")),
                            int(getattr(properties, "minor")),
                        ],
                        "total_memory_bytes": int(getattr(properties, "total_memory")),
                        "multi_processor_count": int(
                            getattr(properties, "multi_processor_count", -1)
                        ),
                        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                    }
                )
                actual_capability = (
                    int(getattr(properties, "major")),
                    int(getattr(properties, "minor")),
                )
                _add_check(report, "cuda_current_device_zero", current_device == 0, current_device)
                _add_check(
                    report,
                    "expected_gpu_name",
                    re.search(expected_gpu_name_regex, gpu_name, flags=re.IGNORECASE) is not None,
                    gpu_name,
                )
                _add_check(
                    report,
                    "expected_compute_capability",
                    actual_capability == expected_compute_capability,
                    list(actual_capability),
                )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            _add_check(report, "cuda_runtime_introspection", False, _short_error(exc))

    prerequisites_passed = not report["failures"]
    if prerequisites_passed and torch is not None and bnb is not None:
        try:
            torch.cuda.reset_peak_memory_stats(0)
            result = functional_runner(torch, bnb)
            report["nf4_functional"] = result
            _add_check(report, "nf4_functional_kernel", result.get("status") == "PASS", result)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            report["nf4_functional"] = {"status": "FAIL", "error": _short_error(exc)}
            _add_check(report, "nf4_functional_kernel", False, report["nf4_functional"])

        try:
            result = linear_runner(torch, bnb)
            report["linear4bit_forward"] = result
            _add_check(report, "linear4bit_forward_kernel", result.get("status") == "PASS", result)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            report["linear4bit_forward"] = {"status": "FAIL", "error": _short_error(exc)}
            _add_check(report, "linear4bit_forward_kernel", False, report["linear4bit_forward"])

        try:
            torch.cuda.synchronize()
            report["cuda"]["peak_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(0)
            )
            report["cuda"]["peak_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(0)
            )
        except Exception as exc:
            _add_check(report, "cuda_peak_memory_accounting", False, _short_error(exc))

    report["overall_status"] = "PASS" if not report["failures"] else "FAIL"
    return report


def write_json_atomic(report: Mapping[str, Any], output: Path, *, overwrite: bool) -> None:
    output = output.resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists (use --overwrite): {output}")
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-machine", required=True)
    parser.add_argument("--expected-gpu-name-regex", required=True)
    parser.add_argument("--expected-compute-capability", required=True, type=parse_compute_capability)
    parser.add_argument("--expected-environment-root", required=True)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect_live_gpu_report(
        expected_machine=args.expected_machine,
        expected_gpu_name_regex=args.expected_gpu_name_regex,
        expected_compute_capability=args.expected_compute_capability,
        expected_environment_root=args.expected_environment_root,
        require_slurm=args.require_slurm,
    )
    script_path = Path(__file__).resolve()
    report["probe_script"] = {
        "path": str(script_path),
        "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }
    try:
        write_json_atomic(report, args.output, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"ORG_E2B_LIVE_GPU_RUNTIME_PROBE={report['overall_status']} "
        f"output={args.output.resolve()} save_reload_certified=false"
    )
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
