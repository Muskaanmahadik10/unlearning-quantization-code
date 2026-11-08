#!/usr/bin/env python3
"""Certify the actual loaded O/U/R module universe for B0.4 on one H200.

This is a representation-only inventory.  It runs no TOFU/MUSE prompt and
computes no forget, retain, recovery, utility, privacy, GQDP, or UPQI outcome.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence
import uuid

import prepare_org_e2b_b04_public_models_v1 as prep


SCHEMA_VERSION = "org_e2b_b04_actual_our_module_inventory_v1"
B03_SCHEMA = "org_e2b_nf4_transformers_model_roundtrip_evidence_v1"
ENV_SCHEMA = "org_e2b_packed_h200_environment_lock_v1"
EXPECTED_TARGET_PARAMETERS = 973_078_528
TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
FORBIDDEN_OUTCOME_TOKENS = (
    "forget_ce",
    "retain_ce",
    "recovery",
    "resurgence",
    "truth_ratio",
    "membership",
    "privacy",
    "gqdp",
    "upqi",
)


class InventoryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def expected_target_fqns(layers: int = 16) -> tuple[str, ...]:
    suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    return tuple(f"model.layers.{layer}.{suffix}" for layer in range(layers) for suffix in suffixes)


def tensor_sha256(torch: Any, tensor: Any) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def validate_receipts(
    local_manifest_path: Path,
    identity_lock_path: Path,
    b03_path: Path,
    environment_lock_path: Path,
) -> dict[str, Any]:
    local = read_json(local_manifest_path)
    identity = read_json(identity_lock_path)
    b03 = read_json(b03_path)
    environment = read_json(environment_lock_path)
    require(local.get("schema_version") == prep.LOCAL_SCHEMA, "wrong local-byte manifest schema")
    require(local.get("identity_download_lock", {}).get("sha256") == sha256_file(identity_lock_path), "identity lock link mismatch")
    require(identity.get("schema_version") == prep.LOCK_SCHEMA, "wrong identity lock schema")
    require(identity.get("model_weight_download_authorized") is True, "model download was not authorized")
    require(identity.get("production_authorized") is False, "identity lock cannot authorize production")
    require(local.get("all_files_rehashed") is True, "local files were not rehashed")
    require(local.get("tokenizer_semantic_identity_verified") is True, "tokenizer semantic identity failed")
    require(local.get("special_tokens_semantic_identity_verified") is True, "special-token identity failed")
    require(local.get("storage_gate", {}).get("passed") is True, "storage gate failed")
    require(local.get("production_authorized") is False, "local manifest cannot authorize production")
    require(b03.get("schema_version") == B03_SCHEMA, "wrong B0.3b schema")
    require(b03.get("overall_status") == "PASS", "B0.3b did not pass")
    require(b03.get("qualification") == "STRICT_B0_3B_CANARY_PASS", "B0.3b qualification mismatch")
    certification = b03.get("certification", {})
    require(certification.get("strict_b0_3b_model_roundtrip_certified") is True, "B0.3b certification absent")
    require(certification.get("eligible_as_strict_b0_3b_prerequisite_for_b0_4") is True, "B0.3b not eligible for B0.4")
    require(certification.get("production_authorized") is False, "B0.3b cannot authorize production")
    require(environment.get("schema_version") == ENV_SCHEMA, "wrong environment-lock schema")
    require(environment.get("overall_status") == "PASS", "environment lock did not pass")
    environment_sha = sha256_file(environment_lock_path)
    require(b03.get("scope_guards", {}).get("strict_environment_lock_sha256") == environment_sha, "B0.3b/environment link mismatch")
    prep.verify_local(argparse.Namespace(manifest=local_manifest_path))
    return {
        "local_byte_manifest": {"path": str(local_manifest_path), "sha256": sha256_file(local_manifest_path)},
        "identity_download_lock": {"path": str(identity_lock_path), "sha256": sha256_file(identity_lock_path)},
        "b03b_final": {"path": str(b03_path), "sha256": sha256_file(b03_path)},
        "environment_lock": {"path": str(environment_lock_path), "sha256": environment_sha},
    }


def parameter_record(torch: Any, name: str, parameter: Any) -> dict[str, Any]:
    return {
        "fqn": name,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "numel": int(parameter.numel()),
        "requires_grad": bool(parameter.requires_grad),
        "sha256": tensor_sha256(torch, parameter),
        "data_ptr": int(parameter.data_ptr()),
    }


def named_parameters_with_aliases(model: Any) -> list[tuple[str, Any]]:
    """Return the complete FQN universe, including tied parameter aliases."""
    return list(model.named_parameters(remove_duplicate=False))


def inventory_model(torch: Any, transformers: Any, role: str, snapshot: Path) -> dict[str, Any]:
    before = torch.cuda.memory_allocated(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    tokenizer_fingerprint = {
        "class": tokenizer.__class__.__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "length": int(len(tokenizer)),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "all_special_tokens": list(tokenizer.all_special_tokens),
        "all_special_ids": list(tokenizer.all_special_ids),
        "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest(),
        "backend_tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "probe_input_ids": [
            tokenizer.encode(text, add_special_tokens=add)
            for text, add in (("A reproducible evaluation probe.", False), ("A reproducible evaluation probe.", True))
        ],
    }
    model, loading = transformers.AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    require(isinstance(loading, Mapping), f"{role} loading info missing")
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        require(loading.get(field, []) == [], f"{role} loading info is not clean: {field}")
    require(model.__class__.__name__ == "LlamaForCausalLM", f"{role} model class mismatch")
    require(model.config.model_type == "llama", f"{role} model type mismatch")
    require(model.config.architectures == ["LlamaForCausalLM"], f"{role} architecture mismatch")
    require(model.config.num_hidden_layers == 16, f"{role} layer count mismatch")
    model.eval().to("cuda:0")
    torch.cuda.synchronize()

    parameters = [
        parameter_record(torch, name, parameter)
        for name, parameter in named_parameters_with_aliases(model)
    ]
    parameter_by_name = {item["fqn"]: item for item in parameters}
    modules = []
    target_modules = []
    for name, module in model.named_modules():
        direct_parameters = []
        for local_name, parameter in module.named_parameters(recurse=False):
            fqn = f"{name}.{local_name}" if name else local_name
            record = parameter_by_name.get(fqn)
            require(record is not None, f"{role} parameter missing from universe: {fqn}")
            direct_parameters.append(
                {
                    "name": local_name,
                    "fqn": fqn,
                    "shape": record["shape"],
                    "dtype": record["dtype"],
                    "device": record["device"],
                    "numel": record["numel"],
                    "sha256": record["sha256"],
                }
            )
        is_target = bool(TARGET_RE.fullmatch(name))
        record = {
            "fqn": name,
            "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            "target_linear_eligible": is_target,
            "direct_parameters": direct_parameters,
        }
        modules.append(record)
        if is_target:
            require(isinstance(module, torch.nn.Linear), f"{role} target is not torch.nn.Linear: {name}")
            require(len(direct_parameters) == 1 and direct_parameters[0]["name"] == "weight", f"{role} target parameter contract mismatch: {name}")
            target_modules.append(
                {
                    "fqn": name,
                    "class": record["class"],
                    "weight_shape": direct_parameters[0]["shape"],
                    "weight_dtype": direct_parameters[0]["dtype"],
                    "weight_numel": direct_parameters[0]["numel"],
                    "weight_sha256": direct_parameters[0]["sha256"],
                }
            )

    embedding = model.get_input_embeddings().weight
    output = model.get_output_embeddings().weight
    tied = embedding is output and embedding.data_ptr() == output.data_ptr()
    peak = torch.cuda.max_memory_allocated(0)
    result = {
        "role": role,
        "snapshot": str(snapshot.resolve()),
        "model_class": model.__class__.__name__,
        "config": model.config.to_dict(),
        "tokenizer_runtime_fingerprint": tokenizer_fingerprint,
        "module_count": len(modules),
        "modules": modules,
        "parameter_count": len(parameters),
        "unique_parameter_storage_count": len({item["data_ptr"] for item in parameters}),
        "parameters": parameters,
        "target_linear_count": len(target_modules),
        "target_linear_parameters": sum(item["weight_numel"] for item in target_modules),
        "target_linears": target_modules,
        "target_fqn_shape_universe": {item["fqn"]: item["weight_shape"] for item in target_modules},
        "embedding_lm_head_tied": tied,
        "embedding_weight_sha256": tensor_sha256(torch, embedding),
        "lm_head_weight_sha256": tensor_sha256(torch, output),
        "all_parameters_cuda0": all(item["device"] == "cuda:0" for item in parameters),
        "cuda_memory_allocated_before": before,
        "cuda_peak_memory_allocated": peak,
    }
    del model, loading, embedding, output, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    return result


def acceptance_failures(roles: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures = []
    expected = set(expected_target_fqns())
    universes = []
    tokenizer_fingerprints = []
    for role in ("O", "U", "R"):
        item = roles.get(role, {})
        if item.get("model_class") != "LlamaForCausalLM":
            failures.append(f"model_class:{role}")
        if item.get("target_linear_count") != 112:
            failures.append(f"target_count:{role}")
        if item.get("target_linear_parameters") != EXPECTED_TARGET_PARAMETERS:
            failures.append(f"target_parameters:{role}")
        universe = item.get("target_fqn_shape_universe", {})
        if set(universe) != expected:
            failures.append(f"target_universe:{role}")
        universes.append(universe)
        tokenizer_fingerprints.append(item.get("tokenizer_runtime_fingerprint"))
        if item.get("embedding_lm_head_tied") is not True:
            failures.append(f"embedding_tie:{role}")
        if item.get("all_parameters_cuda0") is not True:
            failures.append(f"parameter_device:{role}")
    if not all(universe == universes[0] for universe in universes[1:]):
        failures.append("target_fqn_shape_universe_equal")
    if not all(value == tokenizer_fingerprints[0] for value in tokenizer_fingerprints[1:]):
        failures.append("tokenizer_runtime_fingerprint_equal")
    return failures


def runtime_record(torch: Any, transformers: Any) -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import safetensors

    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.device_count() == 1, "exactly one visible GPU is required")
    name = torch.cuda.get_device_name(0)
    require("H200" in name, f"GPU is not H200: {name}")
    require(tuple(torch.cuda.get_device_capability(0)) == (9, 0), "compute capability mismatch")
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "safetensors": safetensors.__version__,
        "cuda_build": torch.version.cuda,
        "gpu_name": name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "visible_gpu_count": torch.cuda.device_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    require(not output.exists(), f"output exists: {output}")
    links = validate_receipts(
        args.local_byte_manifest.resolve(),
        args.identity_download_lock.resolve(),
        args.b03b_final.resolve(),
        args.environment_lock.resolve(),
    )
    import torch
    import transformers

    runtime = runtime_record(torch, transformers)
    local = read_json(args.local_byte_manifest.resolve())
    snapshots = Path(str(local["snapshot_root"])).resolve()
    roles = {}
    for role in ("O", "U", "R"):
        print(f"B04_INVENTORY_START role={role}", flush=True)
        roles[role] = inventory_model(torch, transformers, role, snapshots / role)
        print(
            f"B04_INVENTORY_PASS role={role} targets={roles[role]['target_linear_count']} "
            f"parameters={roles[role]['target_linear_parameters']}",
            flush=True,
        )
    failures = acceptance_failures(roles)
    status = "PASS" if not failures else "FAIL"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_uuid": str(uuid.uuid4()),
        "status": status,
        "failures": failures,
        "runtime": runtime,
        "prerequisites": links,
        "roles": roles,
        "acceptance": {
            "architectures_exactly_llama_for_causal_lm": all(item["model_class"] == "LlamaForCausalLM" for item in roles.values()),
            "target_linear_count_O": roles["O"]["target_linear_count"],
            "target_linear_count_U": roles["U"]["target_linear_count"],
            "target_linear_count_R": roles["R"]["target_linear_count"],
            "target_linear_fqn_shape_universe_equal": roles["O"]["target_fqn_shape_universe"] == roles["U"]["target_fqn_shape_universe"] == roles["R"]["target_fqn_shape_universe"],
            "target_linear_parameter_count_each": [roles[role]["target_linear_parameters"] for role in ("O", "U", "R")],
            "embedding_lm_head_tie_preserved": all(item["embedding_lm_head_tied"] for item in roles.values()),
            "tokenizer_runtime_fingerprint_equal": all(
                roles[role]["tokenizer_runtime_fingerprint"] == roles["O"]["tokenizer_runtime_fingerprint"]
                for role in ("U", "R")
            ),
            "unexpected_module_count": 0 if not any(failure.startswith("target_universe:") for failure in failures) else -1,
        },
        "scope_guards": {
            "outcome_inspected": False,
            "outcome_metric_count": 0,
            "network_allowed": False,
            "remote_code_used": False,
            "pickle_loaded": False,
            "b0_4_representation_canary_passed": False,
            "b0_4_gpu_launcher_authorized": status == "PASS",
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    serialized = json.dumps(report, sort_keys=True).lower()
    # Names inside the explicit scope guard are allowed; outcome-bearing result
    # keys are forbidden elsewhere by construction and checked in unit tests.
    for token in FORBIDDEN_OUTCOME_TOKENS:
        require(token not in serialized, f"forbidden outcome field leaked into inventory: {token}")
    atomic_json(output, report)
    require(status == "PASS", "actual O/U/R module inventory failed: " + ",".join(failures))
    print("ORG_E2B_B04_ACTUAL_OUR_MODULE_INVENTORY=PASS")
    print("TARGET_LINEAR_COUNT=112")
    print(f"OUTPUT={output}")
    print("B0_4_GPU_LAUNCHER_AUTHORIZED=true")
    print("PRODUCTION_AUTHORIZED=false")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-byte-manifest", type=Path, required=True)
    parser.add_argument("--identity-download-lock", type=Path, required=True)
    parser.add_argument("--b03b-final", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except Exception as exc:
        print(f"ORG_E2B_B04_ACTUAL_OUR_MODULE_INVENTORY=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
