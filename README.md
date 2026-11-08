# Code for "Evaluating Machine Unlearning under Quantization: Layerwise Diagnostics, Execution Comparisons and Manufacturing Knowledge Revision"

This repository contains the source code, job scripts, source locks and provenance records behind the reported results. Personal account identifiers (cluster user name, scheduler project accounts and an e-mail address) have been removed; see [ANONYMIZATION.md](ANONYMIZATION.md). The result data are supplied in the `Data/` folder of the supplementary package. Model weights are not included (see [PROVENANCE.md](PROVENANCE.md)).

## Contents

| Folder | What it holds |
|---|---|
| `cluster_experiments/` | Code, Slurm scripts and source locks for the cluster experiments, in the original project layout. `historical_versions/` holds earlier versions of seven files that specific source locks require. |
| `manufacturing_experiment/` | Dataset builder, training and evaluation runner, GQDP diagnostic probe, the three notebooks used to run them, and the analysis script. |
| `local_analysis/` | Scripts that produced the retrospective tables and the analytical atlas from exported results. |
| `provenance/` | How each file was selected, the 2026-11-08 comparison with the cluster copies, and the scheduler job history. |
| `tools/verify_manifest.py` | Checks every file against `FILE_MANIFEST.csv`. |

## Where each part of the paper comes from

| Paper section | Code |
|---|---|
| 3.1–3.2, 4.1.1, 4.1.2, 4.1.4: layerwise diagnostics over 144 checkpoint–quantizer combinations, GQDP/UPQI scores, ranking transport | `cluster_experiments/gate2_v4_final/`: `run_gate2_ground_truth_v4.py` (isolated matrix responses), `run_gate2_metric_suite_v42.py`, `run_gate2_upqi_star_v42.py`, `gate2_metric_v42_math.py`; locks in `V4_SOURCE_LOCK.json` and `metric_locks/` |
| 4.1.3: whole-model recovery | `gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py`, `slurm/gate2_v4/bridge_v4_parallel_matrix.slurm` and `bridge_v4_slotfill.slurm`, `bridge_locks/` (pilot 18, diversity 12, remainder 120), `analyze_bridge_v41_full144.py` |
| 4.2.1: exploratory protection | `gate2_v4_final/scripts/*mitigation_pilot24_v42.py`, `*mitigation_bridge48*_v42.py`; `metric_locks/V4_MITIGATION_V42_SOURCE_LOCK_NANO4_20260901_180006.json` |
| 3.3, 4.2.2: matched protection (fixed budget of eight matrices) | `cluster_experiments/org_h200_execution_v1/`, files named `*e1*` (contract `ORG_H200_E1_SCIENTIFIC_CONTRACT_V1.md`, analyzer `scripts/analyze_org_h200_e1_gate_v1.py`) |
| 4.2.3: dose response and observed frontier | `org_h200_execution_v1/`, files named `*e2a*` (contract `ORG_H200_E2A_ADAPTIVE_TRADEOFF_SCIENTIFIC_CONTRACT_V1.md`, analyzer `scripts/analyze_org_h200_e2a_v1.py`) |
| 3.4.1, 4.3: prediction, dense emulation and packed execution | `org_h200_execution_v1/`, files named `*b02*`–`*b05*` (environment, packed representation and behaviour stages); cross-hardware parity in `cluster_experiments/org_gb200_execution_v1/` |
| 3.4.2, 4.4: fixed-artifact switching and restoration | `cluster_experiments/org_m1_core_v1/` |
| 3.5.1, 4.5.1: controlled benchmark | `cluster_experiments/org_m2_model_pilot_v2/`, `org_m2_training_release_v1/`, `org_m2_u_training_v1/`, `org_m2_eval_qualification_v2/`, `org_m2_metric_integration_v1/`, `org_m2_development_reuse_v3/` |
| 3.5.2, 4.5.2, 4.5.3: manufacturing knowledge revision and GQDP extension | `manufacturing_experiment/` |
| 4.3.2 and Supplement S14: retrospective contrasts | `local_analysis/retrospective_b05/` |
| Supplementary analytical atlas | `local_analysis/atlas/` |

## Checking the files

```bash
python tools/verify_manifest.py
```

On Windows, some archived paths are long; clone with `git -c core.longpaths=true clone <repository URL>`.

This is read-only and uses only the Python standard library. It confirms every file's size and SHA-256 against `FILE_MANIFEST.csv`.

## Re-running the analyses that need no GPU

Both commands use only the standard library (Python 3.11 or later) and the supplementary data package. Write to a new, empty output folder.

```bash
python manufacturing_experiment/analysis/analyze_m4.py --package "<package>/Data/Manufacturing" --output out_m4
python local_analysis/retrospective_b05/analyze_b05.py --source "<package>/Data/Supplementary_tables/Figure_data/B05Contrasts.csv" --output out_b05
```

- The first recreates the four CSV files in `Data/Manufacturing/Analysis` byte for byte.
- The second recreates the four CSV files in `Data/Retrospective_analysis` byte for byte.

The atlas scripts depend on the original local folder layout and are provided as a record rather than as a runnable pipeline.

## GPU experiments

The cluster and notebook code is archived exactly as it ran. It contains the original cluster paths and environment references, so it is not a portable pipeline. The phase-specific locks state which files, data manifests and checkpoints each result used. Do not run the archived Slurm scripts as a single workflow.

- **Cluster experiments:** Slurm on H200 nodes, with GB200 nodes for the parity checks.
- **Manufacturing experiment:** `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`, with Transformers, PEFT and bitsandbytes.
  - Training ran on Colab; the final evaluations and the GQDP probes ran on two A100 GPUs.
  - Seeds 20261101, 20261102 and 20261103.

## Licence

No licence has been granted yet; one will be added on publication. Until then all rights are reserved, and the code is provided so that it can be read and checked. Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
