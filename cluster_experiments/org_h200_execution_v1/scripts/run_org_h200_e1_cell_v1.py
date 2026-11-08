#!/usr/bin/env python3
"""Run and integrity-certify one frozen H200 E1 Core72 bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V4_DIR = REPO_ROOT / "gate2_v4_final" / "scripts"
for directory in (SCRIPT_DIR, V4_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gate2_v4_contract import load_json, load_source_lock, sha256_file, verify_source_lock_files  # noqa: E402
from validate_bridge_v4 import validate_obj  # noqa: E402


def result_identity(obj: dict[str, Any]) -> tuple[str, str, int]:
    return str(obj["condition"]), str(obj.get("ranking_source")), int(obj["k"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source_lock", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--wave", choices=("stress16", "remaining56"), required=True)
    parser.add_argument("--task_index", type=int)
    parser.add_argument("--task_offset", type=int, default=0)
    args = parser.parse_args()

    local_task_index = args.task_index
    if local_task_index is None:
        if "SLURM_PROCID" not in os.environ:
            raise SystemExit("Missing --task_index and SLURM_PROCID")
        local_task_index = int(os.environ["SLURM_PROCID"])
    if args.task_offset < 0:
        raise SystemExit("Task offset must be non-negative")
    task_index = local_task_index + args.task_offset
    with Path(args.cells).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    wave_rows = sorted(
        (row for row in rows if row["execution_wave"] == args.wave),
        key=lambda row: int(row["execution_index"]),
    )
    if task_index < 0 or task_index >= len(wave_rows):
        raise SystemExit(f"Task index {task_index} outside wave size {len(wave_rows)}")
    row = wave_rows[task_index]
    manifest = load_json(args.manifest)
    expected_ids = manifest["stress16_cell_ids"] if args.wave == "stress16" else manifest["remaining56_cell_ids"]
    if row["cell_id"] not in expected_ids:
        raise SystemExit("Cell is not in the manifest wave")

    lock = load_source_lock(args.source_lock)
    verify_source_lock_files(lock)
    lock_sha = str(lock["source_lock_sha256"])
    bundle_script = "org_gb200_execution_v1/scripts/run_org_bridge_bundle_v2.py"
    bundle_sha = str(lock["files"][bundle_script])
    if sha256_file(bundle_script) != bundle_sha:
        raise SystemExit("Bundle script SHA mismatch")

    import bitsandbytes
    import torch  # imported only inside the allocated H200 worker
    import transformers
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"Expected exactly one Slurm-visible GPU, found {torch.cuda.device_count()}")
    if torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise SystemExit(f"Expected NVIDIA H200, found {torch.cuda.get_device_name(0)}")
    expected_runtime = manifest["software_stack"]
    if torch.__version__ != expected_runtime["torch"] or str(torch.version.cuda) != expected_runtime["cuda"]:
        raise SystemExit("H200 PyTorch/CUDA runtime drift")
    if transformers.__version__ != expected_runtime["transformers"] or bitsandbytes.__version__ != expected_runtime["bitsandbytes"]:
        raise SystemExit("H200 Transformers/bitsandbytes runtime drift")

    checkpoint = Path("checkpoints") / row["model_tag"] / "unlearned" / row["algo"] / f"seed_{row['seed']}"
    if not checkpoint.is_dir():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    ledger = load_json("org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_LEDGER_V1.json")
    ledger_entries = {entry["checkpoint_dir"]: entry for entry in ledger["checkpoints"]}
    ledger_entry = ledger_entries.get(checkpoint.as_posix())
    if ledger_entry is None:
        raise SystemExit(f"Checkpoint absent from frozen ledger: {checkpoint}")
    for record in ledger_entry["files"]:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise SystemExit(f"Checkpoint quick identity failure: {path}")
    dataset = row["dataset"]
    forget = Path(f"manifests/gate2_v4/{dataset}_forget_core128_seed20260625.json")
    retain = Path(f"manifests/gate2_v4/{dataset}_retain_gptq_calib128_seed20260625.json")
    selectors = row["selectors"].split(";")
    if selectors != manifest["selectors"] or len(selectors) != 11:
        raise SystemExit("Selector identity mismatch")
    results_root = Path(args.results_root)
    cell_root = results_root / row["cell_id"]
    if cell_root.exists() and any(cell_root.rglob("bridge_v4.json")):
        raise SystemExit(f"Refusing to overwrite existing cell outputs: {cell_root}")

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    proc_id = os.environ.get("SLURM_PROCID", str(task_index))
    cache = Path("results/org_h200_execution_v1/cache") / f"e1_{job_id}" / f"task_{proc_id}"
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, bundle_script,
        "--model_dir", str(checkpoint), "--model_tag", row["model_tag"],
        "--dataset", dataset, "--algo", row["algo"], "--seed", row["seed"],
        "--quantizer", row["quantizer"], "--ranking_sources", *selectors,
        "--k_values", "8", "--rankings_path", "org_gb200_execution_v1/preregistration/core72_v1/org_core72_rankings_v1.jsonl",
        "--manifest_path", str(forget), "--retain_manifest_path", str(retain),
        "--results_root", str(results_root), "--source_lock_path", args.source_lock,
        "--expected_script_sha256", bundle_sha, "--n_samples", "128", "--n_retain", "128",
        "--n_calibration", "128", "--manifest_n_samples", "128", "--manifest_n_retain", "128",
        "--manifest_n_calibration", "128", "--max_length", "512", "--dtype", "bfloat16",
    ]
    if row["quantizer"] == "gptq_int4":
        command.extend(("--calibration_manifest_path", str(retain)))
    print(f"ORG_H200_E1_CELL_START={row['cell_id']}", flush=True)
    subprocess.run(command, env=env, check=True)

    paths = sorted(cell_root.rglob("bridge_v4.json"))
    if len(paths) != 12:
        raise SystemExit(f"Expected 12 results for {row['cell_id']}, found {len(paths)}")
    expected = {("full_quantized", "None", 0)} | {("protected", source, 8) for source in selectors}
    observed = set()
    hashes = {}
    for path in paths:
        obj = load_json(path)
        errors = validate_obj(obj, path=path, expected_source_lock_sha256=lock_sha)
        if errors:
            raise SystemExit(f"Result validation failed {path}: {errors}")
        if obj["cell_id"] != row["cell_id"] or obj["quantizer"] != row["quantizer"]:
            raise SystemExit(f"Result identity mismatch: {path}")
        observed.add(result_identity(obj))
        hashes[path.as_posix()] = sha256_file(path)
    if observed != expected:
        raise SystemExit(f"Condition identity mismatch: missing={expected-observed} extra={observed-expected}")
    receipt = {
        "schema_version": "org_h200_e1_cell_receipt_v1",
        "status": "PASS",
        "cell_id": row["cell_id"],
        "wave": args.wave,
        "task_index": task_index,
        "source_lock_sha256": lock_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "result_count": len(paths),
        "result_sha256": hashes,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "worker_sha256": sha256_file(Path(__file__)),
    }
    receipt_path = results_root / "_receipts" / args.wave / (row["cell_id"].replace("/", "__") + ".json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(receipt_path)
    print(f"ORG_H200_E1_CELL_PASS={row['cell_id']}")
    print(f"RECEIPT={receipt_path}")


if __name__ == "__main__":
    main()
