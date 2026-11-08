#!/usr/bin/env python3
"""Project Full144 metric cost from pilot timings and an authoritative wallet delta."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_root", required=True)
    parser.add_argument("--pilot_gpus", type=int, default=16)
    parser.add_argument("--pilot_elapsed_seconds", type=float, required=True)
    parser.add_argument("--wallet_before", type=float, required=True)
    parser.add_argument("--wallet_after", type=float, required=True)
    parser.add_argument("--target_gpus", type=int, default=64)
    parser.add_argument("--target_fisher_forget", type=int, default=128)
    parser.add_argument("--target_fisher_retain", type=int, default=128)
    parser.add_argument("--budget_cap_ntd", type=float, default=5100.0)
    parser.add_argument("--safety_factor", type=float, default=1.20)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def schedule_makespan(durations: list[float], workers: int) -> float:
    heap = [0.0] * workers
    heapq.heapify(heap)
    for duration in sorted(durations, reverse=True):
        current = heapq.heappop(heap)
        heapq.heappush(heap, current + duration)
    return max(heap)


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.pilot_root).rglob("metrics_v42.json"))
    if len(paths) != 12:
        raise SystemExit(f"Expected 12 pilot metric results, found {len(paths)}")
    templates: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        obj = load(path)
        templates[(str(obj["model_tag"]), str(obj["quantizer"]))].append(obj)
    if len(templates) != 12:
        raise SystemExit(f"Expected 12 model/quantizer timing templates, found {len(templates)}")
    durations: list[float] = []
    template_rows: list[dict[str, Any]] = []
    for (model, quantizer), objects in sorted(templates.items()):
        obj = objects[0]
        timing = obj["timing"]
        pilot_forget = max(int(obj.get("fisher_forget_samples", 0)), 1)
        pilot_retain = max(int(obj.get("fisher_retain_samples", 0)), 1)
        forget_time = float(timing.get("fisher_forget_seconds", 0.0))
        retain_time = float(timing.get("fisher_retain_seconds", 0.0))
        base = max(0.0, float(timing["total_seconds"]) - forget_time - retain_time)
        projected = (
            base
            + forget_time * args.target_fisher_forget / pilot_forget
            + retain_time * args.target_fisher_retain / pilot_retain
        )
        durations.extend([projected] * 12)
        template_rows.append(
            {
                "model_tag": model,
                "quantizer": quantizer,
                "pilot_total_seconds": float(timing["total_seconds"]),
                "projected_cell_seconds": projected,
            }
        )
    makespan = schedule_makespan(durations, args.target_gpus)
    projected_gpu_hours = args.target_gpus * makespan / 3600.0
    pilot_gpu_hours = args.pilot_gpus * args.pilot_elapsed_seconds / 3600.0
    wallet_decrease = args.wallet_before - args.wallet_after
    errors: list[str] = []
    if wallet_decrease <= 0:
        errors.append("wallet_delta_not_positive_or_not_yet_posted")
    if pilot_gpu_hours <= 0:
        errors.append("pilot_gpu_hours_not_positive")
    rate = wallet_decrease / pilot_gpu_hours if not errors else None
    projected_ntd = projected_gpu_hours * rate if rate is not None else None
    guarded_ntd = projected_ntd * args.safety_factor if projected_ntd is not None else None
    if errors:
        decision = "WAIT_AUTHORITATIVE_WALLET_DELTA"
    elif guarded_ntd <= args.budget_cap_ntd:
        decision = "FULL144_METRICS_WITHIN_BUDGET"
    else:
        decision = "FULL144_FISHER128_OVER_BUDGET_REDUCE_SCOPE"
    result = {
        "decision": decision,
        "pilot_result_count": len(paths),
        "pilot_gpus": args.pilot_gpus,
        "pilot_elapsed_seconds": args.pilot_elapsed_seconds,
        "pilot_allocated_gpu_hours": pilot_gpu_hours,
        "wallet_before": args.wallet_before,
        "wallet_after": args.wallet_after,
        "wallet_decrease": wallet_decrease,
        "observed_ntd_per_allocated_gpu_hour": rate,
        "target_gpus": args.target_gpus,
        "target_fisher_forget": args.target_fisher_forget,
        "target_fisher_retain": args.target_fisher_retain,
        "projected_makespan_seconds": makespan,
        "projected_allocated_gpu_hours": projected_gpu_hours,
        "projected_ntd": projected_ntd,
        "safety_factor": args.safety_factor,
        "guarded_projected_ntd": guarded_ntd,
        "budget_cap_ntd": args.budget_cap_ntd,
        "errors": errors,
        "timing_templates": template_rows,
        "warning": "Wallet decrease can include other project activity; isolate the pilot interval before treating this as a billing rate.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"METRIC_FULL_COST_DECISION={decision}")
    print(f"OBSERVED_RATE_NTD_PER_ALLOCATED_GPU_HOUR={rate}")
    print(f"GUARDED_PROJECTED_NTD={guarded_ntd}")
    print(f"WROTE={out}")
    raise SystemExit(0 if decision == "FULL144_METRICS_WITHIN_BUDGET" else 2)


if __name__ == "__main__":
    main()
