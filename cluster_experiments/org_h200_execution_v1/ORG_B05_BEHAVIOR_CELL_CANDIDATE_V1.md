# B05 full-condition runner candidate

Implemented 2026-09-07. GPU execution NOT YET VALIDATED; no authorization supplied.

## Scope

One independent cell/quantizer per GPU. Six frozen conditions: BF16, full packing,
safe singleton, risky singleton, pair, abstention. Abstention refers to the exact
full-packed metrics/artifact; safe abstention does likewise. No selector changes.
Both checkpoint cells and all three quantizer code paths are implemented.

Each non-alias condition runs producer, reload A, reload B, and legacy dense-emulated
evaluation in separate processes. Every evaluation includes all 128 forget and
128 retain samples twice. Packed paper-facing metrics come from reload B.
Producer metrics only test transport. BF16 is serialized and reloaded too.
Persistent tensors, nonpersistent buffers, synthetic logits and per-sample NLL
must pass strict checks; errors retain evidence and prevent the final receipt.

GPTQ packed calibration uses all 128 actual manifest samples, answer-preserving
length-512 encoding. Dense GPTQ uses the historical backend with its Hessian
collector, deliberately NOT relabeled as GPTQModel. Cross-implementation transport
can fail. These are historical NLL diagnostics, not official benchmark efficacy.

## Prospective analysis definition awaiting approval

For each safe/risky/pair condition and forget/retain role, contrast is token-weighted
NLL minus full-packing NLL. Non-negligible means either absolute contrast exceeds
1e-6; agreement requires both exceed 1e-6 with the same sign. Zero eligible contrasts
are INCONCLUSIVE, never PASS. The threshold is proposed before these outcomes;
it is not licensed by unit tests. Per-cell reports retain counts for later pooled
descriptive transport checks; they are not independent scientific replications.
No automatically selected thresholds or changed signs after inspecting outcomes.

## Budget and next gate

B05 accounted usage through job 349553: 2.8258333333 GPU-hours, under amended 3.2.
This package cannot use the old single-pair authorization. A two-hour, one-GPU
full-condition validation allocation would need a further prospective amendment
(maximum 60 NTD at the stated rate). Six such jobs have a 12 GPU-hour reservation
ceiling, NOT a measured forecast; do not launch all before representative validation.
No fresh budget approval or submission occurs in this package.

## Verification boundary

Local pure-Python tests cover metrics, identities, six-condition aliases, transport
signs/zero cases, all child phases, and child failure propagation. PyTorch CPU loss
tests are provided for the cluster. GPU kernels, complete mixed models, wall time,
memory and numerical behavior remain to be validated for this expanded runner.
The earlier ordinary RTN pair canary does not validate these additional paths.

Run only the CPU test helper first. It neither loads models nor submits jobs.
