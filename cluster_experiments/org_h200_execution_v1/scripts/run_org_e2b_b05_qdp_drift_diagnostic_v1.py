#!/usr/bin/env python3
"""Run the B0.5 feature computation as a non-authorizing drift diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "org_e2b_b05_qdp_drift_diagnostic_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source = Path(__file__).with_name("run_org_e2b_b05_feature_cell_v1.py")
    spec = importlib.util.spec_from_file_location("org_e2b_b05_feature_v1_diagnostic", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen B0.5 feature runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = list(sys.argv[1:])
    try:
        sha_index = argv.index("--expected-script-sha256") + 1
    except ValueError as exc:
        raise RuntimeError("missing expected-script-sha256 argument") from exc
    argv[sha_index] = sha256_file(source)

    original_atomic = module.atomic_json

    def diagnostic_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        value = dict(payload)
        value.update(
            {
                "schema_version": SCHEMA,
                "status": "DIAGNOSTIC_PASS",
                "production_eligible": False,
                "diagnostic_only": True,
                "historical_reference_enforced": False,
                "b05_behavior_authorized": False,
                "atlas_production_authorized": False,
                "frozen_feature_runner": {
                    "path": str(source.resolve()),
                    "sha256": sha256_file(source),
                },
                "diagnostic_wrapper": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__)),
                },
            }
        )
        original_atomic(path, value)

    module.SCHEMA = SCHEMA
    module.REFERENCE_ATOL = float("inf")
    module.atomic_json = diagnostic_atomic
    result = int(module.main(argv))
    print("ORG_E2B_B05_QDP_DRIFT_DIAGNOSTIC=PASS")
    print("PRODUCTION_ELIGIBLE=false")
    print("B05_BEHAVIOR_AUTHORIZED=false")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
