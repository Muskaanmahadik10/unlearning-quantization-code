# ORG E2B B0.5 Current-Runtime Reference Amendment V1

## Decision

The frozen cross-runtime analysis returned `HISTORICAL_TRANSPORT_FAIL`. The failure is isolated to GPTQ: both GPTQ cells failed the frozen sign, rank, and/or normalized-error gates and reproduced only 3/224 historical displacement hashes. NF4 and RTN passed with 224/224 displacement equality and near-perfect rank transport.

No protected-condition outcomes have been inspected. Therefore, before any behavior run, the six completed diagnostic feature computations are frozen as the authoritative isolated-packed-H200 current-runtime references. This is a runtime-boundary correction, not threshold tuning and not a favorable-outcome selection.

## Frozen rules

1. Preserve the historical transport failure and its thresholds unchanged.
2. Freeze all six current-runtime diagnostic receipts, including NF4 and RTN; do not selectively replace only unfavorable cells.
3. Independently recompute every feature cell at 128 forget, retain, and calibration samples.
4. Require exact 224/224 current-reference displacement hashes and `1e-10` absolute agreement for every forget-QDP scalar.
5. Keep the historical metric and legacy oracle as forensic evidence only; neither is an acceptance oracle for V2.
6. Keep protected outcomes uninspected throughout feature generation.
7. Feature success may authorize the separately locked B0.5 behavior canary, but never Atlas production or a paper claim directly.

Any failed V2 cell stops behavior authorization. Thresholds and selectors may not be changed after protected outcomes are inspected.
