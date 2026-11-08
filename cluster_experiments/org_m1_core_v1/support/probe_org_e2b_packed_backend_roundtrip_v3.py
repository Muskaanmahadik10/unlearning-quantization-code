#!/usr/bin/env python3
"""B0.3c/d packed GPTQ and packed-storage RTN model roundtrip canaries, v3.

The GPTQ arm uses GPTQModel 4.2.5 with its TorchQuantLinear packed INT4
backend.  The RTN arm is deliberately and permanently named
``org_packed_storage_rtn_int8_dequant_matmul_v1``: INT8 weights are stored at
rest and reloaded exactly, while inference explicitly decodes to the activation
dtype before torch linear.  It is not represented as a fused INT8 kernel.

Each arm creates a deterministic local tiny Llama checkpoint, protects two
mixed module FQNs, writes a packed checkpoint atomically, exits the producer,
reloads in a fresh process, and validates representation plus outputs.  These
are engineering canaries only, not scientific outcomes or Atlas authorization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid


SCHEMA = "org_e2b_packed_backend_model_roundtrip_v3"
ENV_LOCK_SCHEMA = "org_e2b_packed_h200_environment_lock_v1"
EXPECTED_ENV_LOCK_SHA256 = "9ac5f111ba26aaefbdd104f6e3464b4252aee16daf9c09e3427bf4de56d898f7"
EXPECTED_GPTQ_VERSION = "4.2.5"
EXPECTED_GPTQ_SOURCE_SHA256 = {
    "gptqmodel.models.auto": "11c11845fd450bfb4605a9a360c1deb5803db75ef3c8477c1121b58cdc4a001d",
    "gptqmodel.models.base": "87753b431362eb520351f4b6846623048758b7a5b8f1a0e2e82bee322774c98f",
    "gptqmodel.quantization.config": "beedf360d6e9daebdba93f79fb883a6ef64617a6d17a15bda4ae9e7633bf02a7",
    "gptqmodel.nn_modules.qlinear.torch": "72611e301b599d1d4cbeaca051cccdd18a75e832783c488f3f0bc81480edaf52",
}
RTN_INTERVENTION = "org_packed_storage_rtn_int8_dequant_matmul_v1"
SEED = 1907
ATOL = 1.0e-5

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
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}
PROTECTED_FQNS = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.1.mlp.up_proj",
)


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    require(path.parent.is_dir(), f"output parent missing: {path.parent}")
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
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
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    }


def parse_source_manifest(path: Path, repository_root: Path) -> dict[str, Any]:
    rows: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", raw)
        require(match is not None, "invalid source manifest row")
        digest, name = match.groups()
        require(name not in rows, f"duplicate source manifest row: {name}")
        target = (repository_root / name).resolve()
        require(target.is_relative_to(repository_root.resolve()), "source path escapes repository")
        require(target.is_file(), f"source file missing: {name}")
        require(sha256_file(target) == digest, f"source hash mismatch: {name}")
        rows[name] = digest
    require(bool(rows), "empty source manifest")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "entries": rows}


def validate_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    environment_root = args.environment_root.resolve()
    executable = Path(sys.executable).resolve()
    require(executable.is_relative_to(environment_root), "interpreter outside frozen environment")
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.device_count() == 1, "exactly one visible GPU required")
    require(torch.cuda.get_device_name(0) == "NVIDIA H200", "non-H200 GPU")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE not set")
    require(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE not set")
    require(os.environ.get("SLURM_JOB_ID"), "missing SLURM_JOB_ID")
    lock_path = args.environment_lock.resolve()
    require(sha256_file(lock_path) == EXPECTED_ENV_LOCK_SHA256, "environment lock hash drift")
    lock = read_json(lock_path)
    require(lock.get("schema_version") == ENV_LOCK_SCHEMA, "environment lock schema")
    require(lock.get("overall_status") == "PASS", "environment lock status")
    require(lock.get("failures") == [], "environment lock failures")
    require(
        Path(str(lock.get("environment_root"))).resolve() == environment_root,
        "environment root mismatch",
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "safetensors", "gptqmodel")
    }
    require(versions["gptqmodel"] == EXPECTED_GPTQ_VERSION, "GPTQModel version drift")
    source_gate = parse_source_manifest(args.source_manifest, args.repository_root)
    return {
        "job_id": os.environ["SLURM_JOB_ID"],
        "pid": os.getpid(),
        "process_uuid": str(uuid.uuid4()),
        "python": sys.version,
        "python_executable": str(executable),
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "environment_lock": {"path": str(lock_path), "sha256": EXPECTED_ENV_LOCK_SHA256},
        "source_manifest": source_gate,
    }


def expected_linear_fqns() -> tuple[str, ...]:
    names: list[str] = []
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        names.extend(
            f"{prefix}.{suffix}"
            for suffix in (
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            )
        )
    names.append("lm_head")
    return tuple(names)


def target_fqns() -> tuple[str, ...]:
    return tuple(name for name in expected_linear_fqns() if name != "lm_head")


def packed_fqns() -> tuple[str, ...]:
    return tuple(name for name in target_fqns() if name not in PROTECTED_FQNS)


def get_module(model: Any, fqn: str) -> Any:
    current = model
    for part in fqn.split("."):
        current = getattr(current, part)
    return current


def set_module(model: Any, fqn: str, module: Any) -> None:
    parent, leaf = fqn.rsplit(".", 1)
    setattr(get_module(model, parent), leaf, module)


def initialize_model(torch: Any, model: Any) -> None:
    with torch.no_grad():
        for index, (name, parameter) in enumerate(sorted(model.named_parameters())):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            values = torch.sin(values * (0.003 + index * 0.00001)) * 0.035
            parameter.copy_(values.to(parameter.dtype))


def make_tiny_baseline(torch: Any, transformers: Any, tokenizers: Any, destination: Path) -> None:
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    require(not destination.exists(), f"baseline already exists: {destination}")
    tmp = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    require(not tmp.exists(), f"baseline staging exists: {tmp}")
    tmp.mkdir(parents=True)
    try:
        config = transformers.LlamaConfig(**MODEL_GEOMETRY)
        model = transformers.LlamaForCausalLM(config).to(dtype=torch.float16)
        initialize_model(torch, model)
        model.save_pretrained(tmp, safe_serialization=True)
        vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        vocab.update({f"t{i}": i for i in range(4, 128)})
        raw = tokenizers.Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=raw,
            bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
        )
        tokenizer.save_pretrained(tmp)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def inputs(torch: Any, device: str = "cuda:0") -> dict[str, Any]:
    rows = []
    for row in range(3):
        rows.append([4 + ((row * 19 + col * 7) % 120) for col in range(24)])
    ids = torch.tensor(rows, dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def calibration(torch: Any) -> list[dict[str, Any]]:
    rows = []
    for row in range(16):
        ids = [4 + ((row * 13 + col * 11) % 120) for col in range(32)]
        rows.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(32, dtype=torch.long),
        })
    return rows


def forward_record(torch: Any, model: Any) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        first = model(**inputs(torch)).logits.detach().cpu().contiguous()
        second = model(**inputs(torch)).logits.detach().cpu().contiguous()
    require(torch.isfinite(first).all().item(), "nonfinite logits")
    require(torch.equal(first, second), "same-process output not bitwise repeatable")
    return {
        "tensor": first,
        "shape": list(first.shape),
        "dtype": str(first.dtype),
        "sha256": tensor_sha256(first),
        "min": float(first.float().min()),
        "max": float(first.float().max()),
    }


def tensor_state_manifest(torch: Any, model: Any, module_names: Sequence[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    total = 0
    for name in module_names:
        module = get_module(model, name)
        tensors: dict[str, Any] = {}
        for key, value in sorted(module.state_dict().items()):
            size = int(value.numel() * value.element_size())
            total += size
            tensors[key] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "bytes": size, "sha256": tensor_sha256(value),
            }
        require(bool(tensors), f"module state is empty: {name}")
        rows[name] = {"class": f"{type(module).__module__}.{type(module).__qualname__}", "tensors": tensors}
    return {"modules": rows, "total_bytes": total, "sha256": canonical_sha256(rows)}


def dense_weight_manifest(model: Any, names: Sequence[str]) -> dict[str, str]:
    rows = {}
    for name in names:
        module = get_module(model, name)
        require(hasattr(module, "weight"), f"dense weight missing: {name}")
        rows[name] = tensor_sha256(module.weight)
    return rows


def build_rtn_linear_class(torch: Any):
    class PackedRTNLinear(torch.nn.Module):
        intervention_name = RTN_INTERVENTION

        def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float32))
            if bias:
                self.register_buffer("bias", torch.empty(out_features, dtype=torch.float16))
            else:
                self.bias = None

        @classmethod
        def from_dense(cls, module: Any):
            result = cls(module.in_features, module.out_features, module.bias is not None)
            weight = module.weight.detach().float().cpu()
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-12) / 127.0
            result.qweight.copy_(torch.round(weight / scale).clamp(-127, 127).to(torch.int8))
            result.scale.copy_(scale)
            if module.bias is not None:
                result.bias.copy_(module.bias.detach().to(torch.float16).cpu())
            return result

        def forward(self, x: Any) -> Any:
            weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
            bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
            return torch.nn.functional.linear(x, weight, bias)

    return PackedRTNLinear


def install_rtn(torch: Any, model: Any, *, from_dense: bool) -> None:
    cls = build_rtn_linear_class(torch)
    for name in packed_fqns():
        old = get_module(model, name)
        replacement = cls.from_dense(old) if from_dense else cls(
            old.in_features, old.out_features, old.bias is not None
        )
        set_module(model, name, replacement)


def save_tensor_reference(torch: Any, tensor: Any, path: Path) -> None:
    from safetensors.torch import save_file

    require(not path.exists(), f"reference exists: {path}")
    save_file({"logits": tensor.detach().cpu().contiguous()}, str(path))


def load_tensor_reference(path: Path) -> Any:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")["logits"]


def atomic_directory(destination: Path, producer: Any) -> None:
    require(not destination.exists(), f"destination exists: {destination}")
    tmp = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    require(not tmp.exists(), f"staging exists: {tmp}")
    tmp.mkdir(parents=True)
    try:
        producer(tmp)
        require(any(tmp.iterdir()), "empty checkpoint staging directory")
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def inspect_gptq_sources() -> dict[str, str]:
    import importlib

    rows = {}
    for name, expected in EXPECTED_GPTQ_SOURCE_SHA256.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        actual = sha256_file(path)
        require(actual == expected, f"GPTQ source hash drift: {name}")
        rows[name] = actual
    return rows


def produce_gptq(args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import tokenizers
    import transformers
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    make_tiny_baseline(torch, transformers, tokenizers, args.baseline_dir)
    baseline = transformers.LlamaForCausalLM.from_pretrained(
        args.baseline_dir, torch_dtype=torch.float16, local_files_only=True
    )
    protected_before = dense_weight_manifest(baseline, (*PROTECTED_FQNS, "lm_head"))
    dense_bytes = sum(
        int(get_module(baseline, name).weight.numel() * 2) for name in packed_fqns()
    )
    del baseline

    dynamic = {
        r"-:^model\.layers\.0\.self_attn\.q_proj$": {},
        r"-:^model\.layers\.1\.mlp\.up_proj$": {},
    }
    qcfg = QuantizeConfig(
        bits=4, group_size=128, desc_act=False, sym=True, dynamic=dynamic,
        device="cuda:0", parallel_packing=True,
        meta={"quantizer": [f"gptqmodel:{EXPECTED_GPTQ_VERSION}"], "org_arm": "b03c"},
    )
    require(qcfg.dynamic_get(PROTECTED_FQNS[0]) is False, "first dynamic skip does not match")
    require(qcfg.dynamic_get(PROTECTED_FQNS[1]) is False, "second dynamic skip does not match")
    require(qcfg.dynamic_get(packed_fqns()[0]) is None, "dynamic skip overmatches")
    wrapper = GPTQModel.from_pretrained(
        str(args.baseline_dir), qcfg, torch_dtype=torch.float16,
        trust_remote_code=False, local_files_only=True,
    )
    wrapper.quantize(
        calibration(torch), backend=BACKEND.TORCH, batch_size=1,
        calibration_enable_gpu_cache=True, auto_gc=True, fail_safe=False,
    )
    wrapper.to("cuda:0")
    packed_modules = sorted(
        (
            (name, module) for name, module in wrapper.model.named_modules()
            if isinstance(module, TorchQuantLinear)
        ),
        key=lambda item: item[0],
    )
    actual_packed = [name for name, _ in packed_modules]
    require(actual_packed == sorted(packed_fqns()), "GPTQ packed module set mismatch")
    protected_after = dense_weight_manifest(wrapper.model, (*PROTECTED_FQNS, "lm_head"))
    require(protected_after == protected_before, "GPTQ protected dense weights changed")
    state = tensor_state_manifest(torch, wrapper.model, packed_fqns())
    require(state["total_bytes"] < dense_bytes, "GPTQ packed state is not smaller than BF16 weights")
    for item in state["modules"].values():
        require("qweight" in item["tensors"], "GPTQ qweight missing")
        require(item["tensors"]["qweight"]["dtype"] == "torch.int32", "GPTQ qweight dtype")
    def save(tmp: Path) -> None:
        wrapper.save(str(tmp))
        for source in args.baseline_dir.iterdir():
            if source.name.startswith("tokenizer") or source.name in {
                "special_tokens_map.json", "generation_config.json",
            }:
                target = tmp / source.name
                if not target.exists():
                    shutil.copy2(source, target)

    atomic_directory(args.checkpoint_dir, save)
    return {
        "backend": "gptq_int4_torchquantlinear_v1",
        "intervention_identity": "gptqmodel_4.2.5_gptq_int4_torch_backend",
        "gptq_sources": inspect_gptq_sources(),
        "producer_inference_executed": False,
        "reference_origin": "independent_fresh_reload_a",
        "dynamic": dynamic,
        "actual_packed_fqns": actual_packed,
        "protected_weight_sha256": protected_after,
        "pre_save_in_memory_packed_state": state,
        "dense_reference_bytes": dense_bytes,
        "packed_checkpoint": file_manifest(args.checkpoint_dir),
    }


def create_rtn_model(torch: Any, transformers: Any, baseline_dir: Path, *, packed: bool) -> Any:
    model = transformers.LlamaForCausalLM.from_pretrained(
        baseline_dir, torch_dtype=torch.float16, local_files_only=True
    )
    if packed:
        install_rtn(torch, model, from_dense=True)
    return model


def produce_rtn(args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import tokenizers
    import transformers
    from safetensors.torch import save_file

    make_tiny_baseline(torch, transformers, tokenizers, args.baseline_dir)
    baseline = create_rtn_model(torch, transformers, args.baseline_dir, packed=False)
    protected_before = dense_weight_manifest(baseline, (*PROTECTED_FQNS, "lm_head"))
    dense_bytes = sum(int(get_module(baseline, name).weight.numel() * 2) for name in packed_fqns())
    model = create_rtn_model(torch, transformers, args.baseline_dir, packed=True)
    require(
        dense_weight_manifest(model, (*PROTECTED_FQNS, "lm_head")) == protected_before,
        "RTN protected dense weights changed",
    )
    model.to("cuda:0")
    state = tensor_state_manifest(torch, model, packed_fqns())
    require(state["total_bytes"] < dense_bytes, "RTN packed state not smaller than BF16")
    for item in state["modules"].values():
        require(item["class"].endswith("PackedRTNLinear"), "RTN module class mismatch")
        require(item["tensors"]["qweight"]["dtype"] == "torch.int8", "RTN qweight dtype")
        require(item["tensors"]["scale"]["dtype"] == "torch.float32", "RTN scale dtype")
    output = forward_record(torch, model)

    def save(tmp: Path) -> None:
        baseline_config = read_json(args.baseline_dir / "config.json")
        baseline_config["org_quantization"] = {
            "intervention": RTN_INTERVENTION,
            "packed_fqns": list(packed_fqns()),
            "protected_fqns": list(PROTECTED_FQNS),
            "decode_compute_dtype": "activation_dtype",
            "fused_integer_kernel": False,
        }
        write_json_atomic(baseline_config, tmp / "config.json")
        cpu_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        save_file(cpu_state, str(tmp / "model.safetensors"))
        for source in args.baseline_dir.iterdir():
            if source.name.startswith("tokenizer") or source.name in {
                "special_tokens_map.json", "generation_config.json",
            }:
                shutil.copy2(source, tmp / source.name)

    atomic_directory(args.checkpoint_dir, save)
    save_tensor_reference(torch, output["tensor"], args.reference)
    del output["tensor"]
    return {
        "backend": "rtn_int8_packed_storage_control_v1",
        "intervention_identity": RTN_INTERVENTION,
        "fused_integer_kernel": False,
        "actual_packed_fqns": sorted(packed_fqns()),
        "protected_weight_sha256": protected_before,
        "packed_state": state,
        "dense_reference_bytes": dense_bytes,
        "packed_checkpoint": file_manifest(args.checkpoint_dir),
        "output": output,
    }


def produce(args: argparse.Namespace) -> None:
    runtime = validate_runtime(args)
    require(not args.run_dir.exists(), f"run directory exists: {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    result = produce_gptq(args, runtime) if args.backend == "gptq" else produce_rtn(args, runtime)
    report = {
        "schema_version": SCHEMA, "phase": "produce", "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend_selector": args.backend, "runtime": runtime,
        "pid": os.getpid(), "process_uuid": runtime["process_uuid"],
        "baseline_checkpoint": file_manifest(args.baseline_dir),
        **result,
        "scope_guards": {
            "engineering_canary_only": True, "public_model_tested": False,
            "b05_authorized": False, "atlas_production_authorized": False,
            "scientific_claim_authorized": False,
        },
    }
    write_json_atomic(report, args.producer_report)
    print(f"ORG_E2B_{args.backend.upper()}_PRODUCE=PASS")
    print(f"PRODUCER_PID={os.getpid()}")


def load_rtn(torch: Any, transformers: Any, checkpoint_dir: Path) -> Any:
    from safetensors.torch import load_file

    config_payload = read_json(checkpoint_dir / "config.json")
    qmeta = config_payload.pop("org_quantization", None)
    require(isinstance(qmeta, dict), "RTN quantization metadata missing")
    require(qmeta.get("intervention") == RTN_INTERVENTION, "RTN identity drift")
    config = transformers.LlamaConfig.from_dict(config_payload)
    model = transformers.LlamaForCausalLM(config).to(dtype=torch.float16)
    install_rtn(torch, model, from_dense=False)
    state = load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    loaded = model.load_state_dict(state, strict=True)
    require(list(loaded.missing_keys) == [], "RTN missing state keys")
    require(list(loaded.unexpected_keys) == [], "RTN unexpected state keys")
    return model


def load_gptq_runtime(torch: Any, checkpoint_dir: Path) -> tuple[Any, Any, list[str]]:
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    model = GPTQModel.from_quantized(
        str(checkpoint_dir), device="cuda:0", backend=BACKEND.TORCH,
        torch_dtype=torch.float16, trust_remote_code=False, local_files_only=True,
    )
    underlying = model.model
    actual = sorted(
        name for name, module in underlying.named_modules()
        if isinstance(module, TorchQuantLinear)
    )
    require(actual == sorted(packed_fqns()), "reloaded GPTQ module set mismatch")
    inspect_gptq_sources()
    return model, underlying, actual


def validate_reload_entry(args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    producer = read_json(args.producer_report)
    require(producer.get("phase") == "produce" and producer.get("status") == "PASS", "producer not PASS")
    require(int(producer.get("pid", -1)) == args.expected_producer_pid, "producer PID mismatch")
    require(os.getpid() != args.expected_producer_pid, "reload PID equals producer PID")
    require(producer.get("runtime", {}).get("job_id") == runtime["job_id"], "job identity drift")
    require(file_manifest(args.checkpoint_dir) == producer["packed_checkpoint"], "checkpoint byte drift")
    return producer


def reload_reference(args: argparse.Namespace) -> None:
    import torch

    require(args.backend == "gptq", "reload-reference is GPTQ-only")
    runtime = validate_runtime(args)
    producer = validate_reload_entry(args, runtime)
    require(not args.reference.exists(), "reference already exists")
    model, underlying, actual = load_gptq_runtime(torch, args.checkpoint_dir)
    state = tensor_state_manifest(torch, underlying, packed_fqns())
    protected = dense_weight_manifest(underlying, (*PROTECTED_FQNS, "lm_head"))
    require(protected == producer["protected_weight_sha256"], "protected weight identity changed")
    output = forward_record(torch, model)
    save_tensor_reference(torch, output["tensor"], args.reference)
    del output["tensor"]
    report = {
        "schema_version": SCHEMA, "phase": "reload_reference", "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend_selector": args.backend, "runtime": runtime,
        "pid": os.getpid(), "process_uuid": runtime["process_uuid"],
        "producer_pid": args.expected_producer_pid,
        "producer_exited_before_reload": True,
        "actual_packed_fqns": actual,
        "packed_state": state,
        "protected_weight_sha256": protected,
        "output": output,
        "reference": {"path": str(args.reference), "sha256": sha256_file(args.reference)},
        "scope_guards": producer["scope_guards"],
    }
    write_json_atomic(report, args.reference_consumer_report)
    print("ORG_E2B_GPTQ_FRESH_RELOAD_A=PASS")


def reload(args: argparse.Namespace) -> None:
    import torch
    import transformers

    runtime = validate_runtime(args)
    producer = validate_reload_entry(args, runtime)

    if args.backend == "gptq":
        reference_consumer = read_json(args.reference_consumer_report)
        require(reference_consumer.get("phase") == "reload_reference", "reference reload phase")
        require(reference_consumer.get("status") == "PASS", "reference reload not PASS")
        require(reference_consumer.get("runtime", {}).get("job_id") == runtime["job_id"], "reference job drift")
        require(reference_consumer.get("pid") != os.getpid(), "fresh reload PID collision")
        require(reference_consumer.get("pid") != producer.get("pid"), "reference/producer PID collision")
        require(sha256_file(args.reference) == reference_consumer["reference"]["sha256"], "reference byte drift")
        model, underlying, actual = load_gptq_runtime(torch, args.checkpoint_dir)
    else:
        underlying = load_rtn(torch, transformers, args.checkpoint_dir).to("cuda:0")
        actual = sorted(
            name for name, module in underlying.named_modules()
            if type(module).__qualname__.endswith("PackedRTNLinear")
        )
        require(actual == sorted(packed_fqns()), "reloaded RTN module set mismatch")
        model = underlying

    state = tensor_state_manifest(torch, underlying, packed_fqns())
    if args.backend == "gptq":
        require(state == reference_consumer["packed_state"], "packed state differs across fresh reloads")
    else:
        require(state == producer["packed_state"], "packed tensor state changed across reload")
    require(
        dense_weight_manifest(underlying, (*PROTECTED_FQNS, "lm_head"))
        == producer["protected_weight_sha256"],
        "protected weight identity changed across reload",
    )
    output = forward_record(torch, model)
    reference = load_tensor_reference(args.reference)
    difference = (output["tensor"].float() - reference.float()).abs()
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    require(max_abs <= ATOL, f"fresh reload output mismatch: {max_abs}")
    del output["tensor"]
    report = {
        "schema_version": SCHEMA, "phase": "reload", "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend_selector": args.backend, "runtime": runtime,
        "pid": os.getpid(), "process_uuid": runtime["process_uuid"],
        "producer_pid": args.expected_producer_pid,
        "reference_reload_pid": reference_consumer["pid"] if args.backend == "gptq" else None,
        "producer_exited_before_reload": True,
        "actual_packed_fqns": actual,
        "packed_state": state,
        "protected_weight_sha256": producer["protected_weight_sha256"],
        "output": output,
        "fresh_reload_parity": {"atol": ATOL, "max_abs": max_abs, "mean_abs": mean_abs},
        "scope_guards": producer["scope_guards"],
    }
    write_json_atomic(report, args.consumer_report)
    print(f"ORG_E2B_{args.backend.upper()}_FRESH_RELOAD=PASS")


def validate_gptq_reload_pair(
    producer: Mapping[str, Any],
    reference_consumer: Mapping[str, Any],
    consumer: Mapping[str, Any],
) -> dict[str, Any]:
    require(reference_consumer.get("phase") == "reload_reference", "reference phase")
    require(consumer.get("phase") == "reload", "consumer phase")
    require(reference_consumer.get("status") == consumer.get("status") == "PASS", "reload phase failure")
    pids = {producer.get("pid"), reference_consumer.get("pid"), consumer.get("pid")}
    require(None not in pids and len(pids) == 3, "producer/reload process independence")
    jobs = {
        producer.get("runtime", {}).get("job_id"),
        reference_consumer.get("runtime", {}).get("job_id"),
        consumer.get("runtime", {}).get("job_id"),
    }
    require(None not in jobs and len(jobs) == 1, "job identity drift")
    require(reference_consumer.get("packed_state") == consumer.get("packed_state"), "fresh reload packed-state parity")
    require(
        producer.get("actual_packed_fqns")
        == reference_consumer.get("actual_packed_fqns")
        == consumer.get("actual_packed_fqns"),
        "fresh reload module-set parity",
    )
    require(
        producer.get("protected_weight_sha256")
        == reference_consumer.get("protected_weight_sha256")
        == consumer.get("protected_weight_sha256"),
        "protected-weight parity",
    )
    require(reference_consumer.get("output", {}).get("sha256") == consumer.get("output", {}).get("sha256"), "output hash parity")
    require(consumer.get("fresh_reload_parity", {}).get("max_abs", math.inf) <= ATOL, "output tolerance parity")
    return {
        "independent_processes": True,
        "exact_runtime_packed_state": True,
        "exact_output_hash": True,
        "producer_to_reload_runtime_normalization_observed": (
            producer.get("pre_save_in_memory_packed_state") != reference_consumer.get("packed_state")
        ),
    }


def finalize(args: argparse.Namespace) -> None:
    producer = read_json(args.producer_report)
    consumer = read_json(args.consumer_report)
    require(producer.get("schema_version") == SCHEMA, "producer schema")
    require(consumer.get("schema_version") == SCHEMA, "consumer schema")
    require(producer.get("status") == consumer.get("status") == "PASS", "phase failure")
    require(producer.get("backend_selector") == consumer.get("backend_selector") == args.backend, "backend drift")
    if args.backend == "gptq":
        reference_consumer = read_json(args.reference_consumer_report)
        lifecycle = validate_gptq_reload_pair(producer, reference_consumer, consumer)
    else:
        require(producer.get("pid") != consumer.get("pid"), "producer/reloader PID collision")
        require(consumer.get("producer_exited_before_reload") is True, "fresh process not proven")
        require(producer.get("runtime", {}).get("job_id") == consumer.get("runtime", {}).get("job_id"), "job drift")
        require(producer.get("packed_state") == consumer.get("packed_state"), "packed state parity")
        require(producer.get("actual_packed_fqns") == consumer.get("actual_packed_fqns"), "module set parity")
        require(consumer.get("fresh_reload_parity", {}).get("max_abs", math.inf) <= ATOL, "output parity")
        reference_consumer = None
        lifecycle = {"independent_processes": True, "exact_runtime_packed_state": True}
    final = {
        "schema_version": SCHEMA, "phase": "final", "overall_status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend_selector": args.backend,
        "qualification": f"STRICT_B0_3_{'C' if args.backend == 'gptq' else 'D'}_CANARY_PASS",
        "producer_report": {"path": str(args.producer_report), "sha256": sha256_file(args.producer_report)},
        "reference_consumer_report": (
            {"path": str(args.reference_consumer_report), "sha256": sha256_file(args.reference_consumer_report)}
            if reference_consumer is not None else None
        ),
        "consumer_report": {"path": str(args.consumer_report), "sha256": sha256_file(args.consumer_report)},
        "checkpoint_manifest_sha256": canonical_sha256(producer["packed_checkpoint"]),
        "certification": {
            "model_level_mixed_packed_roundtrip": True,
            "atomic_save_producer_exit_fresh_reload": True,
            "two_independent_fresh_reloads": args.backend == "gptq",
            "exact_packed_state_hashes_across_fresh_reloads": True,
            "exact_protected_weight_identity": True,
            "repeatable_finite_inference": True,
            "backend_public_representation_certified": False,
            "b05_authorized": False,
            "atlas_production_authorized": False,
            "scientific_claim_authorized": False,
        },
        "intervention_identity": producer["intervention_identity"],
        "lifecycle": lifecycle,
        "fused_integer_kernel": producer.get("fused_integer_kernel"),
        "failures": [],
    }
    write_json_atomic(final, args.final_report)
    print(f"ORG_E2B_{args.backend.upper()}_MODEL_ROUNDTRIP=PASS")
    print(f"FINAL={args.final_report}")
    print("PUBLIC_REPRESENTATION_CERTIFIED=false")
    print("B05_AUTHORIZED=false")
    print("ATLAS_PRODUCTION_AUTHORIZED=false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("produce", "reload-reference", "reload", "finalize"))
    parser.add_argument("--backend", choices=("gptq", "rtn"), required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--producer-report", type=Path, required=True)
    parser.add_argument("--reference-consumer-report", type=Path, required=True)
    parser.add_argument("--consumer-report", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--expected-producer-pid", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "repository_root", "environment_root", "environment_lock", "source_manifest",
        "run_dir", "baseline_dir", "checkpoint_dir", "reference",
        "producer_report", "reference_consumer_report", "consumer_report", "final_report",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.phase == "produce":
        produce(args)
    elif args.phase == "reload-reference":
        reload_reference(args)
    elif args.phase == "reload":
        reload(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
