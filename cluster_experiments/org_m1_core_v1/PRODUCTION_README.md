# M1 core production

Six cells x ten arms x two historical data roles x 128 samples. Each arm is
evaluated in producer AND fresh reload processes. The successful RTN first cell
is production data, not discarded smoke. Reuse all passed frozen smoke evidence.
Phase change from the accepted runner: full behavior for every arm rather than
N0/Z_ab only. Decoders/loaders/state assertions, fixed 1e-6 reload threshold,
exact endpoint copies and unchanged midpoint remain in place. No requantization.

Scheduling retains the six-cell bundling protocol: one GPU per cell, 40min cap.
First run index5 (ordinary RTN; slowest measured backend). After COMPLETED 0:0,
the remaining-stage command independently validates all ten producer/reload
receipts and bytes before submitting indices0..4 concurrently. No duplicate
first-cell job. Total new cap4GPUh/NTD120, including prior smoke5.35223GPUh,
below M1 cap7GPUh. Forecast uses observed phase times, an extra eight times
the slowest measured full-arm timing per phase, then20% headroom. Queue and
filesystem variation can still cause timeout; no calendar guarantee.

Commands from cluster project root:
python3 org_m1_core_v1/m1_core_submit_v1.py --stage canary --submit
After successful canary:
python3 org_m1_core_v1/m1_core_submit_v1.py --stage remaining --submit

Do not rerun a submit command after an ambiguous response: inspect intent and
outcome receipts. No source/model deletion or overwrite. Cluster-only retention.
Full result reconciliation/analysis and M2/M3 remain separate, unreleased stages.
M1 is outcome-exposed development, not clean unlearning-efficacy confirmation.
