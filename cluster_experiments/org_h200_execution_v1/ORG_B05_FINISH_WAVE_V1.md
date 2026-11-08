# B05 completion wave — 2026-09-07

The user requested completing B05 in one parallel wave after GPTQ repair job
349981 completed. Scope is the unchanged six historical checkpoint/quantizer
cells and six frozen conditions per cell. This is implementation/behavior
transport evidence, not six independent checkpoint replications or efficacy.

## Execution and budget amendment

Reuse ordinary RTN's five validated non-alias conditions, ordinary NF4 pair,
and ordinary GPTQ pair from the repaired serialized lifecycle. Keep their roots
unchanged and freeze/verify their evidence hashes. Do not copy or requantize them.
The remaining at most 23 unique conditions run independently on one H200 each.
Safe abstention and full abstention remain aliases; no duplicate work for aliases.

At most nine GPTQ tasks receive 60 minutes each; fourteen other tasks receive
30 minutes each. All tasks may run concurrently subject to Slurm limits. A
15-minute finalizer runs after both arrays terminate, even if a task failed.
Its GPU allocation is budgeted although its aggregation work is CPU-only.
No worker retries automatically and no failed condition is silently omitted.
Both input checkpoints are rehashed against the feature-linked certified ledger
before launch and after the wave; workers check frozen file size/mtime in between.

Additional hard allocation ceiling: 16.25 H200 GPU-hours, 487.50 NTD at the
user-provided 30 NTD/GPU-hour rate. This explicitly supersedes the earlier
7.5-hour cumulative B05 cap for this finishing wave only. Previous actual
accounting is retained separately. The last quoted wallet is not a live balance;
user must have at least the reported maximum available at launch. Filesystem
headroom must be at least 250 GiB; this does not establish personal quota.

## Unchanged validation

Use the repaired GPTQ serialization-before-inference lifecycle, all 128
calibration samples, both 128-sample evaluation manifests, repeated evaluation,
two independent fresh reloads, source-exact BF16 protection, no offload, and
strict state/buffer/logit/per-sample NLL checks. Dense-emulated comparison runs
for each non-alias condition. No tolerance, selector, sample, or sign changes.

Each worker validates all four phases before a complete receipt. The finalizer
requires terminal successful accounting, 36 logical conditions across six cells,
reused and new JSON integrity, and exact saved-artifact file hashes. Reports
include per-cell contrasts and the existing 90% sign-agreement rule with 1e-6
negligibility. Scientific failure does not become an execution failure or vanish.

## Final outputs / next phase

Root: results/org_h200_execution_v1/b05_finish_wave_v1

- final.json and final.json.sha256: integrity, six transport reports, lineage;
- condition_results.tsv: all 36 conditions with packed and emulated role NLL;
- wave_accounting.psv: actual worker allocation accounting;
- submissions.json and finalizer_job.json: submitted job identities.

If all six transport gates pass, preparation of the outcome-blind Atlas lock is
allowed. Atlas GPU production remains false: its lineage/splits/manifests,
controller/feature gates, smoke validation, live wallet, and cost projection
must still be verified. E3 is not automatically launched. A failed or inconclusive
transport gate stops advancement; thresholds are not relaxed to meet a deadline.

CPU tests validate orchestration and failure handling, not unseen GPU kernels.
The ordinary backend pair validations are the representative GPU acceptance
evidence. The adverse-cell conditions remain genuine validation work in this wave.
