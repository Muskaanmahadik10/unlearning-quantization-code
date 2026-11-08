#!/usr/bin/env python3
"""Hardened V2 wrapper for the synthetic NF4 fresh-process reload canary.

The V1 artifact, process-separation, numerical, and evidence-chain logic is
preserved as an immutable engine.  V2 replaces its interpreter check with the
same symlink-aware, fail-closed environment identity used by live-probe V2 and
accepts only a live-probe V2 receipt.  V1 files and V1 evidence remain intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import site
import sys
from typing import Any, Callable, Mapping

import probe_org_e2b_live_gpu_runtime_v2 as live_v2
import probe_org_e2b_nf4_fresh_reload_v1 as engine_v1


SCHEMA_VERSION = "org_e2b_nf4_fresh_reload_evidence_v2"
LIVE_SOURCE_MANIFEST = Path(
    "org_h200_execution_v1/ORG_E2B_B0_LIVE_GPU_PROBES_TRANSFER_SHA256_V2.txt"
)
_VALIDATE_LIVE_RUNTIME_RECEIPT_V1 = engine_v1.validate_live_runtime_receipt


def _runtime_v2(
    *,
    expected_machine: str,
    expected_gpu_name_regex: str,
    expected_compute_capability: tuple[int, int],
    expected_environment_root: str,
    expected_torch_version: str,
    expected_bnb_version: str,
    importer: Callable[[str], Any] = importlib.import_module,
    version_provider: Callable[[str], str] = importlib.metadata.version,
    environ: Mapping[str, str] | None = None,
    machine_provider: Callable[[], str] = platform.machine,
    executable_provider: Callable[[], str] = lambda: sys.executable,
    prefix_provider: Callable[[], str] = lambda: sys.prefix,
    exec_prefix_provider: Callable[[], str] = lambda: sys.exec_prefix,
    base_prefix_provider: Callable[[], str] = lambda: sys.base_prefix,
    base_executable_provider: Callable[[], str | None] = lambda: getattr(
        sys, "_base_executable", None
    ),
    proc_self_exe_provider: Callable[[], str | None] = live_v2._default_proc_self_exe_provider,
    sys_path_provider: Callable[[], list[str]] = lambda: list(sys.path),
    enable_user_site_provider: Callable[[], bool | None] = lambda: getattr(
        site, "ENABLE_USER_SITE", None
    ),
    user_site_provider: Callable[[], str | list[str] | None] = site.getusersitepackages,
    user_base_provider: Callable[[], str | None] = site.getuserbase,
    path_resolver: Callable[[str], str] = live_v2._default_path_resolver,
    pyvenv_reader: Callable[[Path], str] = live_v2._default_pyvenv_reader,
    install_network_guard: Callable[[], None] = engine_v1._install_no_network_audit_guard,
) -> tuple[Any, Any, dict[str, Any]]:
    """Run the fresh-reload runtime gate with complete environment identity."""

    env = os.environ if environ is None else environ
    install_network_guard()
    machine = machine_provider()
    engine_v1._require(machine == expected_machine, "host architecture mismatch")
    engine_v1._require(env.get("SLURM_JOB_ID"), "Slurm allocation is required")
    visible = [
        value
        for value in env.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    engine_v1._require(
        len(visible) == 1, "exactly one CUDA_VISIBLE_DEVICES token is required"
    )

    torch = importer("torch")
    bnb = importer("bitsandbytes")
    torch_version = version_provider("torch")
    bnb_version = version_provider("bitsandbytes")
    engine_v1._require(
        torch_version == expected_torch_version, f"Torch drift: {torch_version}"
    )
    engine_v1._require(
        bnb_version == expected_bnb_version, f"bitsandbytes drift: {bnb_version}"
    )
    contract = engine_v1.backend_contract_for(bnb_version)

    identity = live_v2.collect_interpreter_identity(
        expected_environment_root=expected_environment_root,
        executable=executable_provider(),
        prefix=prefix_provider(),
        exec_prefix=exec_prefix_provider(),
        base_prefix=base_prefix_provider(),
        base_executable=base_executable_provider(),
        proc_self_exe=proc_self_exe_provider(),
        path_resolver=path_resolver,
        pyvenv_reader=pyvenv_reader,
    )
    identity_failures = [
        row["name"] for row in identity["checks"] if row["status"] != "PASS"
    ]
    engine_v1._require(
        not identity_failures,
        "interpreter identity failure: " + ",".join(identity_failures),
    )
    path_hygiene = live_v2.collect_python_path_hygiene(
        environ=env,
        expected_environment_root=expected_environment_root,
        sys_path=sys_path_provider(),
        enable_user_site=enable_user_site_provider(),
        user_site_paths=user_site_provider(),
        user_base=user_base_provider(),
        path_resolver=path_resolver,
    )
    path_failures = [
        row["name"]
        for row in path_hygiene["checks"]
        if row["status"] != "PASS"
    ]
    engine_v1._require(
        not path_failures,
        "Python path hygiene failure: " + ",".join(path_failures),
    )

    root = Path(expected_environment_root).resolve()
    torch_path = Path(torch.__file__).resolve()
    bnb_path = Path(bnb.__file__).resolve()
    engine_v1._require(
        torch_path.is_relative_to(root), "Torch is outside expected environment"
    )
    engine_v1._require(
        bnb_path.is_relative_to(root),
        "bitsandbytes is outside expected environment",
    )
    engine_v1._require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "one live CUDA device is required",
    )
    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    capability = (int(properties.major), int(properties.minor))
    engine_v1._require(
        re.search(expected_gpu_name_regex, gpu_name, re.I) is not None,
        f"GPU mismatch: {gpu_name}",
    )
    engine_v1._require(
        capability == expected_compute_capability,
        f"compute capability mismatch: {capability}",
    )

    cextension = importer("bitsandbytes.cextension")
    native = getattr(cextension, "lib", None)
    engine_v1._require(
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
    engine_v1._require(
        native_path is not None,
        "bitsandbytes native library path is unavailable",
    )
    engine_v1._require(
        native_path.is_relative_to(root),
        "bitsandbytes native library is outside expected environment",
    )
    engine_v1._require(
        hasattr(bnb.functional.QuantState, "from_dict"),
        "QuantState.from_dict missing",
    )
    engine_v1._require(
        hasattr(bnb.nn.Params4bit, "from_prequantized"),
        "Params4bit.from_prequantized missing",
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, bnb, {
        "machine": machine,
        "gpu_name": gpu_name,
        "compute_capability": list(capability),
        "python": sys.version,
        "executable": executable_provider(),
        "interpreter_identity": identity,
        "python_path_hygiene": path_hygiene,
        "torch": torch_version,
        "bitsandbytes": bnb_version,
        "backend_contract": contract,
        "torch_module_path": str(torch_path),
        "bitsandbytes_module_path": str(bnb_path),
        "torch_compiled_cuda": getattr(torch.version, "cuda", None),
        "bitsandbytes_native_library": str(native_path),
        "bitsandbytes_native_library_sha256": engine_v1.sha256_file(native_path),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
    }


def _augment_source(report: dict[str, Any]) -> dict[str, Any]:
    source = report.setdefault("source", {})
    wrapper_path = Path(__file__).resolve()
    source["v2_wrapper_path"] = str(wrapper_path)
    source["v2_wrapper_sha256"] = engine_v1.sha256_file(wrapper_path)
    source["v1_engine_path"] = str(Path(engine_v1.__file__).resolve())
    source["v1_engine_sha256"] = engine_v1.sha256_file(
        Path(engine_v1.__file__).resolve()
    )
    source["live_probe_schema"] = live_v2.SCHEMA_VERSION
    return report


def _validate_live_runtime_receipt_v2(
    receipt: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    result = _VALIDATE_LIVE_RUNTIME_RECEIPT_V1(receipt, **kwargs)
    live_engine_path = Path(live_v2.probe_v1.__file__).resolve()
    live_engine_sha256 = engine_v1.sha256_file(live_engine_path)
    engine_v1._require(
        receipt.get("probe_engine_v1", {}).get("sha256")
        == live_engine_sha256,
        "live probe V1 engine source hash mismatch",
    )
    manifest_path = LIVE_SOURCE_MANIFEST.resolve()
    engine_v1._require(
        manifest_path.is_file(), "live V2 source manifest is unavailable"
    )
    manifest_sha256 = engine_v1.sha256_file(manifest_path)
    engine_v1._require(
        receipt.get("source_manifest", {}).get("sha256") == manifest_sha256,
        "live V2 source manifest hash mismatch",
    )
    result.update(
        {
            "live_probe_v1_engine_sha256": live_engine_sha256,
            "live_source_manifest_sha256": manifest_sha256,
        }
    )
    return result


def v2_wrapper_hash_chain_failures(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    *,
    current_wrapper_sha256: str,
) -> list[str]:
    producer_hash = producer.get("source", {}).get("v2_wrapper_sha256")
    consumer_hash = consumer.get("source", {}).get("v2_wrapper_sha256")
    failures: list[str] = []
    if producer_hash != current_wrapper_sha256:
        failures.append("producer_v2_wrapper_source_hash_drift")
    if consumer_hash != current_wrapper_sha256:
        failures.append("consumer_v2_wrapper_source_hash_drift")
    if producer_hash != consumer_hash:
        failures.append("producer_consumer_v2_wrapper_source_drift")
    return failures


def finalize_v2(args: argparse.Namespace) -> dict[str, Any]:
    report = engine_v1.finalize(args)
    producer = engine_v1._read_json(args.producer_report)
    consumer = engine_v1._read_json(args.consumer_report)
    wrapper_path = Path(__file__).resolve()
    wrapper_sha256 = engine_v1.sha256_file(wrapper_path)
    report["failures"].extend(
        v2_wrapper_hash_chain_failures(
            producer,
            consumer,
            current_wrapper_sha256=wrapper_sha256,
        )
    )
    report["failures"] = list(dict.fromkeys(report["failures"]))
    report["overall_status"] = "PASS" if not report["failures"] else "FAIL"
    for key in (
        "synthetic_nf4_save_exit_fresh_reload",
        "plain_non_nested_quant_state",
        "exact_packed_state_hashes",
    ):
        report["certification"][key] = report["overall_status"] == "PASS"
    return _augment_source(report)


def _configure_engine() -> None:
    engine_v1.SCHEMA_VERSION = SCHEMA_VERSION
    engine_v1.LIVE_SCHEMA = live_v2.SCHEMA_VERSION
    engine_v1._runtime = _runtime_v2
    engine_v1.validate_live_runtime_receipt = _validate_live_runtime_receipt_v2


def main(argv: list[str] | None = None) -> int:
    _configure_engine()
    args = engine_v1.parse_args(argv)
    try:
        if args.phase == "produce":
            report = _augment_source(engine_v1.produce(args))
        elif args.phase == "reload":
            report = _augment_source(engine_v1.reload(args))
        else:
            report = finalize_v2(args)
        engine_v1.write_json_atomic(report, args.report)
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "phase": args.phase,
            "status": "FAIL" if args.phase != "finalize" else None,
            "overall_status": "FAIL" if args.phase == "finalize" else None,
            "collected_utc": datetime.now(timezone.utc).isoformat(),
            "failures": [
                f"{type(exc).__name__}: {' '.join(str(exc).split())[:1200]}"
            ],
            "scope_guards": {
                "production_authorized": False,
                "scientific_claim_authorized": False,
            },
        }
        try:
            engine_v1.write_json_atomic(failure, args.report)
        except Exception:
            pass
        print(
            f"ORG_E2B_NF4_FRESH_RELOAD_V2_{args.phase.upper()}=FAIL {exc}",
            file=sys.stderr,
        )
        return 1
    status = report.get("overall_status", report.get("status"))
    print(f"ORG_E2B_NF4_FRESH_RELOAD_V2_{args.phase.upper()}={status}")
    print(f"OUTPUT={args.report.resolve()}")
    print("PRODUCTION_AUTHORIZED=false")
    print("SCIENTIFIC_CLAIM_AUTHORIZED=false")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
