#!/usr/bin/env python3
"""Build the fail-closed evidence/source lock for repaired H200 adversarial canary v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CORE_MANIFEST_SHA256 = "0e0637e324be42d7be1c36b10735dc6f2e2728cad4d5f09560da1e18c66cd8d1"
EXPECTED_CANARY_MANIFEST_SHA256 = "21bf949bda8e5e56709f052481a6289596cc2c5b0d3fcaf17b0dd4fb929ef646"
GB200_RESULTS_ROOT = Path(
    "results/org_gb200_execution_v1/gate2_v4_bridge_standard_bundled_canary2_v2"
)
GB200_VALIDATION = Path(
    "results/org_gb200_execution_v1/parity/bundled_canary_validation_v2.json"
)
GB200_PARITY128 = Path(
    "results/org_gb200_execution_v1/parity/gptq_production128_parity_v5_340393.json"
)
GB200_BOUNDARY = Path(
    "results/org_gb200_execution_v1/parity/gptq_cpu_boundary_parity_v4_340302.json"
)
H200_PREFLIGHT_ROOT = Path("results/org_gb200_execution_v1/h200_preflight_340413")

STATIC_FILES = (
    "org_h200_execution_v1/PREREGISTRATION_AMENDMENT_H200_PRIMARY_V1.md",
    "org_h200_execution_v1/H200_CERTIFICATION_AND_EXECUTION_RUNBOOK_V1.md",
    "org_h200_execution_v1/FAILURE_340480_TIME_WRAPPER.md",
    "org_h200_execution_v1/H200_CANARY_V2_REPAIR_RUNBOOK.md",
    "org_h200_execution_v1/locks/ORG_H200_ADVERSARIAL_CANARY8_LOCK_V1.json",
    "logs/org_h200/org_h200_canary8_340480.out",
    "logs/org_h200/org_h200_canary8_340480.err",
    "results/org_h200_execution_v1/failures/job340480_sacct.txt",
    "results/org_h200_execution_v1/failures/job340480_evidence.sha256",
    "org_h200_execution_v1/scripts/build_org_h200_adversarial_lock_v2.py",
    "org_h200_execution_v1/scripts/validate_org_h200_adversarial_canary_v1.py",
    "org_h200_execution_v1/slurm/org_h200_adversarial_canary8_v2.slurm",
    "org_gb200_execution_v1/scripts/run_org_bridge_bundle_v2.py",
    "org_gb200_execution_v1/scripts/org_gptq_shared_cpu_finalize_collector_v3.py",
    "org_gb200_execution_v1/scripts/test_org_gptq_production128_parity_v5.py",
    "org_gb200_execution_v1/scripts/test_org_gptq_cpu_boundary_parity_v4.py",
    "org_gb200_execution_v1/scripts/validate_org_bundled_canary_v1.py",
    "org_gb200_execution_v1/scripts/validate_org_core72_v1.py",
    "gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py",
    "gate2_v4_final/scripts/gate2_v4_backend.py",
    "gate2_v4_final/scripts/gate2_v4_contract.py",
    "gate2_v4_final/scripts/gate2_v4_dataset.py",
    "gate2_v4_final/scripts/run_gate2_ground_truth_v4.py",
    "gate2_v4_final/scripts/validate_bridge_v4.py",
    "manifests/gate2_v4/tofu_forget_core128_seed20260625.json",
    "manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json",
    "manifests/gate2_v4/muse_news_forget_core128_seed20260625.json",
    "manifests/gate2_v4/muse_news_retain_gptq_calib128_seed20260625.json",
    "org_gb200_execution_v1/preregistration/core72_v1/org_core72_manifest_v1.json",
    "org_gb200_execution_v1/preregistration/core72_v1/org_canary2_manifest_v1.json",
    "org_gb200_execution_v1/preregistration/core72_v1/org_core72_cells.tsv",
    "org_gb200_execution_v1/preregistration/core72_v1/org_canary2_cells.tsv",
    "org_gb200_execution_v1/preregistration/core72_v1/ORG_CORE72_PREREGISTRATION.md",
    "org_gb200_execution_v1/preregistration/core72_v1/org_core72_rankings_v1.jsonl",
    "org_gb200_execution_v1/preregistration/core72_v1/output_sha256.json",
    "org_gb200_execution_v1/locks/ORG_BUNDLED_CANARY_GB200_LOCK_V2.json",
    "results/org_gb200_execution_v1/parity/gptq_cpu_boundary_parity_v4_340302.json",
    "results/org_gb200_execution_v1/parity/gptq_cpu_boundary_parity_v4_340302.json.sha256",
    "results/org_gb200_execution_v1/parity/gptq_cpu_boundary_parity_v4_340302.json.script.sha256",
    "results/org_gb200_execution_v1/parity/gptq_production128_parity_v5_340393.json",
    "results/org_gb200_execution_v1/parity/gptq_production128_parity_v5_340393.json.sha256",
    "results/org_gb200_execution_v1/parity/bundled_canary_validation_v2.json",
    "results/org_gb200_execution_v1/parity/bundled_canary_validation_v2.json.sha256",
    "results/org_gb200_execution_v1/h200_preflight_340413/gpu.txt",
    "results/org_gb200_execution_v1/h200_preflight_340413/runtime.json",
    "results/org_gb200_execution_v1/h200_preflight_340413/status.txt",
    "results/org_gb200_execution_v1/h200_preflight_340413/uname.txt",
    "results/org_gb200_execution_v1/h200_preflight_340413/evidence_sha256.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def require_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise SystemExit(f"Missing SHA sidecar: {sidecar}")
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if recorded != actual:
        raise SystemExit(f"SHA sidecar mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gb200_results = sorted(GB200_RESULTS_ROOT.rglob("bridge_v4.json"))
    if len(gb200_results) != 12:
        raise SystemExit(f"Expected 12 corrected GB200 canary results, found {len(gb200_results)}")
    files = [Path(name) for name in STATIC_FILES] + gb200_results
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit("Missing locked files:\n" + "\n".join(missing))

    core = load_json(
        Path("org_gb200_execution_v1/preregistration/core72_v1/org_core72_manifest_v1.json")
    )
    canary = load_json(
        Path("org_gb200_execution_v1/preregistration/core72_v1/org_canary2_manifest_v1.json")
    )
    if core.get("manifest_sha256") != EXPECTED_CORE_MANIFEST_SHA256:
        raise SystemExit("Core72 manifest identity mismatch")
    if canary.get("manifest_sha256") != EXPECTED_CANARY_MANIFEST_SHA256:
        raise SystemExit("Canary2 manifest identity mismatch")

    preflight = load_json(H200_PREFLIGHT_ROOT / "runtime.json")
    if preflight.get("status") != "PASS" or preflight.get("gpu") != "NVIDIA H200":
        raise SystemExit("H200 runtime preflight did not pass on NVIDIA H200")
    if preflight.get("transformers") != "4.46.3":
        raise SystemExit("Unexpected H200 Transformers version")
    if preflight.get("torch_compile") != "PASS":
        raise SystemExit("H200 TorchInductor/Triton preflight did not pass")
    if (H200_PREFLIGHT_ROOT / "status.txt").read_text(encoding="utf-8").strip() != "H200_RUNTIME_PREFLIGHT=PASS":
        raise SystemExit("H200 preflight status marker mismatch")

    boundary = load_json(GB200_BOUNDARY)
    if boundary.get("status") != "PASS_FAST":
        raise SystemExit("GB200 all-layer boundary parity status is not PASS_FAST")
    if boundary.get("fast", {}).get("hessian", {}).get("exact_count") != 224:
        raise SystemExit("GB200 fast Hessian parity is not 224/224 exact")
    if boundary.get("fast", {}).get("quantized", {}).get("exact_count") != 21:
        raise SystemExit("GB200 fast quantized parity is not 21/21 exact")
    require_sidecar(GB200_BOUNDARY)

    parity = load_json(GB200_PARITY128)
    if parity.get("status") != "PASS" or parity.get("n_samples") != 128:
        raise SystemExit("GB200 production-size GPTQ parity did not pass at n=128")
    if parity.get("hessian_exact_count") != 21 or parity.get("quantized_exact_count") != 21:
        raise SystemExit("GB200 production-size GPTQ parity is not 21/21 exact")
    parity_script = Path("org_gb200_execution_v1/scripts/test_org_gptq_production128_parity_v5.py")
    collector_script = Path("org_gb200_execution_v1/scripts/org_gptq_shared_cpu_finalize_collector_v3.py")
    if parity.get("script_sha256") != sha256_file(parity_script):
        raise SystemExit("GB200 production-size parity script identity mismatch")
    if parity.get("collector_sha256") != sha256_file(collector_script):
        raise SystemExit("GB200 production-size collector identity mismatch")
    require_sidecar(GB200_PARITY128)

    validation = load_json(GB200_VALIDATION)
    if validation.get("status") != "PASS" or validation.get("new_result_count") != 12:
        raise SystemExit("Corrected GB200 bundled validation did not pass with 12 results")
    if validation.get("failures"):
        raise SystemExit("Corrected GB200 bundled validation contains failures")
    require_sidecar(GB200_VALIDATION)

    locked = {path.as_posix(): sha256_file(path) for path in files}
    output = Path(args.output)
    if output.exists():
        existing = load_json(output)
        if existing.get("files") != locked:
            raise SystemExit("Existing H200 lock differs; preserve it and create a new version")
        print("ORG_H200_ADVERSARIAL_LOCK_V2=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={existing['source_lock_sha256']}")
        return

    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "ORG H200 E1-E3 adversarial canary v2; time-wrapper-only repair after quarantined job 340480",
        "hardware_policy": {
            "primary": "NVIDIA H200",
            "gb200_role": "engineering and hardware-sensitivity evidence only",
            "cross_hardware_endpoint_pooling": False,
        },
        "files": locked,
    }
    payload = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    lock["source_lock_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_ADVERSARIAL_LOCK_V2=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print(f"BUNDLE_SCRIPT_SHA256={locked['org_gb200_execution_v1/scripts/run_org_bridge_bundle_v2.py']}")


if __name__ == "__main__":
    main()


