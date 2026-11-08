# Provenance

## How the cluster files were selected

The files in `cluster_experiments/` are the ones the reported results point to:
- a result file records the file's exact SHA-256, or
- a source lock that a result was produced under lists it.

The selection records are in `provenance/source_selection/`. In those records, paths under `code/` correspond to `cluster_experiments/` here.

Two additions came later:
- `org_h200_execution_v1/scripts/analyze_org_h200_e1_gate_v1.py`, which is not listed by any lock (see below).
- `historical_versions/`. Each of its subfolders is named by the SHA-256 of the file inside. Seven files are earlier versions required by specific historical locks, because the file at the main path was edited after those runs.

## Comparison with the cluster (2026-11-08)

A read-only inventory of the cluster project directory was taken on 2026-11-08. It covered 53,678 files and fingerprinted 3,959 scripts and lock files. `provenance/cluster_recheck_20261108.csv` gives the result for every file in `cluster_experiments/`:

- **Identical by SHA-256 at the same path:** 391 files, including the E1 analyzer.
- **Same size at the same path:** 17 documentation and configuration files. The scan did not fingerprint these file types.
- **Historical versions:** each of the seven was found on the cluster with an identical SHA-256.
  - Six were already preserved locally.
  - One was recovered on 2026-11-08: the `bridge_v4_parallel_matrix.slurm` version required by `V4_BRIDGE_SOURCE_LOCK_NANO4_V41_VERIFIED_20260831_011901.json`.

`provenance/slurm_job_history.csv` is the scheduler accounting record for the project account since 2026-04-01. It includes every job, including jobs unrelated to this study, and gives each job's submission command and working directory.

## Manufacturing experiment

- **Runner:** `m4_runner.py` is the copy stored in the lifecycle results archive. Its SHA-256 (`75a00be3…e2b5`) equals `runner_sha256` in `Data/Manufacturing/Provenance/FINAL_TEST_FREEZE.json`.
- **Diagnostic scripts:** `m4_fix2_gqdp.py` and `m4_fix2_scheduler.py` are the copies stored in the diagnostic archive.
- **Archives:** the SHA-256 values of both archives equal those recorded in `Data/Manufacturing/Analysis/Analysis_receipt.json`.
- **Evaluation data:** the 36 evaluation files and `M4_FIX2_all192.csv` in the supplementary package are byte-identical to the archived copies.

## Re-execution checks

These checks were run on 2026-11-08.

| Script | Result |
|---|---|
| `manufacturing_experiment/analysis/analyze_m4.py --package` | Recreates the four analysis CSVs in `Data/Manufacturing/Analysis` byte for byte. |
| `manufacturing_experiment/analysis/analyze_m4.py --archives` | Recreates the same four CSVs and `Analysis_receipt.json` byte for byte. |
| `manufacturing_experiment/analysis/original/analyze.py` (as originally run) | Same five files byte for byte. |
| `local_analysis/retrospective_b05/analyze_b05.py` | Recreates the four CSVs in `Data/Retrospective_analysis` byte for byte. |
| `local_analysis/atlas/` (historical run of 2026-09-10) | 63 of the 65 atlas tables in the package are byte-identical to that run's output. `11_checkpoint_identities.csv` differs only where private paths were withheld. `12_nested_panel_differences.csv` comes from a later thesis-stage table. |

The tables in `Data/Supplementary_tables/Research/source`, `Research/full144` and `Figure_data` were exported from earlier analysis stages. Byte identity with those exports was confirmed, but their producing scripts were not re-run for this repository.

## Known limitations

1. **The diversity-12 bridge launcher (job of 2026-08-31 02:29) cannot be recovered byte for byte.**
   - **What is missing:** its source lock records `bridge_v4_parallel_matrix.slurm` with SHA-256 `78ede04a…af925`. The file was edited at 03:20 the same night, after the job finished, and the scheduler does not store submitted job scripts.
   - **Why the effect is bounded:** the file only selects which tasks to launch.
     - The run log reports scope `diversity12` with 12 tasks.
     - The 12 result files on the cluster correspond exactly to the 12 diversity-12 tasks listed in the later version, and all were written between 02:29 and 02:58.
     - The task runner `bridge_v4_slotfill.slurm` and the Python code are identical across all bridge locks.
2. **The E1 analyzer is linked by timestamps, not by a lock.** On 2026-09-05:
   - 08:33:26: method amendment written.
   - 08:35:19: analyzer last modified; it has not changed since.
   - 08:48:20: its final input written.
   - 08:48:28: report `e1_scientific_gate_v1.json` written.

   The report therefore came from the archived bytes. It records the SHA-256 of the method amendment and of the source lock it used.
3. **The GPU code is not portable.** It embeds the original cluster paths and environments. Exact re-execution needs those environments and the original checkpoints.
4. **Model weights are not distributed.** The core checkpoints total about 2.5 TB, and the base models' licences restrict redistribution.
5. **Removing account identifiers changed some files.** See [ANONYMIZATION.md](ANONYMIZATION.md). `FILE_MANIFEST.csv` gives each file's SHA-256 both as published and as originally run. Source-lock checks inside the archived code compare against the original values, so they report a mismatch for the changed files.
