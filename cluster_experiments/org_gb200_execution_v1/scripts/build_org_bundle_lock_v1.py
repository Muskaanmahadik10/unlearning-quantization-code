#!/usr/bin/env python3
"""Build the fail-closed source/data lock for the ORG bundled canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


FILES = (
    "org_gb200_execution_v1/scripts/run_org_bridge_bundle_v1.py",
    "org_gb200_execution_v1/scripts/test_org_gptq_collector_parity_v1.py",
    "org_gb200_execution_v1/scripts/validate_org_core72_v1.py",
    "org_gb200_execution_v1/scripts/build_org_bundle_lock_v1.py",
    "org_gb200_execution_v1/slurm/org_bundle_canary2_gb200_v1.slurm",
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
    "org_gb200_execution_v1/preregistration/core72_v1/org_core72_rankings_v1.jsonl",
    "org_gb200_execution_v1/preregistration/core72_v1/output_sha256.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    missing = [name for name in FILES if not Path(name).is_file()]
    if missing:
        raise SystemExit("Missing locked files:\n" + "\n".join(missing))
    files = {name: sha256_file(Path(name)) for name in FILES}
    core = json.loads(Path(FILES[-4]).read_text(encoding="utf-8"))
    canary = json.loads(Path(FILES[-3]).read_text(encoding="utf-8"))
    if core.get("manifest_sha256") != "0e0637e324be42d7be1c36b10735dc6f2e2728cad4d5f09560da1e18c66cd8d1":
        raise SystemExit("Core72 manifest identity mismatch")
    if canary.get("manifest_sha256") != "21bf949bda8e5e56709f052481a6289596cc2c5b0d3fcaf17b0dd4fb929ef646":
        raise SystemExit("Canary2 manifest identity mismatch")
    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("files") != files:
            raise SystemExit("Existing bundle lock does not match current files; preserve it and use a new version")
        print("ORG_BUNDLE_LOCK=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={existing['source_lock_sha256']}")
        print(f"BUNDLE_SCRIPT_SHA256={files['org_gb200_execution_v1/scripts/run_org_bridge_bundle_v1.py']}")
        return
    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "ORG bundled GB200 end-to-end parity canary v1",
        "files": files,
    }
    payload = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    lock["source_lock_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_BUNDLE_LOCK=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print(f"BUNDLE_SCRIPT_SHA256={files['org_gb200_execution_v1/scripts/run_org_bridge_bundle_v1.py']}")


if __name__ == "__main__":
    main()
