# ORG H200 E2A adaptive trade-off scientific contract v1

Frozen: 2026-09-05, after E1 reported a retain-utility failure and before any
E2A GPU execution or E2A outcome inspection.

## 1. Status and disclosure

E2A is a new, explicitly post-E1 adaptive study. It replaces the blocked E2 as
the next executable ORG stage, but it does not modify E1, pass the old E2 gate,
or inherit confirmatory status from the preregistered E1-to-E2 hierarchy.

The adaptation was motivated by E1's observed pattern: recovery and native
GQDP benefit passed, while retain-utility non-inferiority failed and benefit was
strongly negatively correlated with safety slack. This motivation and the full
E1 results must be disclosed in the manuscript.

## 2. Fixed panel and fresh intervention grid

E2A uses all 72 frozen Core72 cells, all 11 frozen rankings, identical data
manifests, evaluator, checkpoint identities, quantizers, calibration sample
counts, maximum length, dtype, and certified H200 runtime.

Fresh matrix budgets are `k in {2,4,16,32}`. Each cell contains one newly
computed full-quantized baseline and 44 protected conditions: 45 results per
cell and 3,240 production results. The E1 k=8 observations may appear only as
labelled historical bridge points and are not part of E2A hypothesis tests.

No cell, algorithm, dataset, model, quantizer, or selector is removed because
of its E1 outcome. One deterministic sorted-index-0 GPTQ engineering canary is
excluded and rerun from scratch in the clean production root.

## 3. Estimands

For cell c, ranking s, and budget k:

`B(c,s,k) = H(c,full_quantized) - H(c,s,k)`.

Let `R(c,k)` be the mean B across the eight matched-random controls. Native
ranking advantage is:

`D(c,k) = B(c,GQDP_signed_v42,k) - R(c,k)`.

Retain cost is native protected `delta_retain_ce` minus the fresh full baseline
`delta_retain_ce`. The unchanged margin is
`max(0.05, 0.01 * abs(BF16 retain answer_ce))`; safety slack `S(c,k)` is margin
minus retain cost.

Checkpoint-cluster values average the quantizer arms belonging to the same
`(model_tag, algorithm, seed)`.

## 4. Primary low-dose rescue question

Only k=2 and k=4 are primary candidate doses. For each candidate dose, the
joint null is rejected only when all components pass:

1. native mean D is greater than zero using the centered studentized
   checkpoint-cluster bootstrap;
2. mean safety slack is greater than or equal to zero overall;
3. mean safety slack is greater than or equal to zero separately in MUSE and
   TOFU.

Within a dose this is an intersection-union test: the dose-level p-value is the
maximum of the one-sided benefit and three one-sided safety-component p-values.
Holm correction at familywise alpha 0.025 is applied across the two dose-level
joint p-values. A dose is a primary safe-effective dose only if its Holm
decision passes, its benefit mean and median are positive, both dataset benefit
means are positive, and all leave-one-model-tag-out benefit means are
non-negative.

The primary claim is existence of at least one safe-effective low dose in this
fixed panel. Both k=2 and k=4 estimates, intervals, raw values, and adjusted
p-values are reported regardless of outcome. A favorable subgroup cannot
rescue a failed global joint test.

## 5. Secondary and mechanistic analyses

- k=16 and k=32 are prespecified high-dose stress points, not rescue doses.
- Dose-specific native D and safety slack are reported for all four fresh k.
- Transported GQDP and legacy UPQI use the same contrasts against the identical
  matched-random controls; their analyses are secondary.
- A checkpoint-cluster model relates D and safety slack to centered log2(k),
  quantizer, algorithm, dataset, and prespecified interactions.
- Report parameter fractions because equal matrix count is not equal parameter
  count. A parameter-fraction-adjusted sensitivity model is required.
- Report dataset, model, algorithm, seed, and quantizer groups; leave-one-model
  and leave-one-quantizer-out sensitivity; medians; sign proportions; and the
  joint positive-benefit/non-negative-slack fraction.
- E1 k=8 is descriptive and visually distinguished from fresh E2A points.

Secondary findings cannot establish the primary low-dose safe-effective claim.

## 6. Inference

- Resampling unit: checkpoint `(model_tag, algorithm, seed)`.
- Quantizer arms stay together in every resample.
- Bootstrap replicates: 10,000.
- Seed: 20260905 with deterministic endpoint-specific offsets.
- Report two-sided 95% bootstrap confidence intervals and one-sided centered
  studentized bootstrap p-values for benefit and safety.
- Missing scientific cells are not imputed.
- All 72 production cells are required for the primary analysis.

## 7. Integrity gates

E2A analysis requires:

- exact E1 V3 source-lock identity and 936-file E1 checksum verification;
- exact E1 `STOP` status with recovery, native primary, native robustness, and
  integrity PASS and retain utility FAIL;
- exact PASS E1 trade-off audit identity;
- exact 40-checkpoint/400-file certification;
- a new E2A canary lock and clean excluded-canary root;
- one canary cell, 45 results, one receipt, zero failures, and clean Slurm exit;
- a promoted production lock and clean production root;
- 72 cells, 3,240 result files, 72 receipts, finite values, correct condition
  identities, stable tensor fingerprints within cell, and zero unadjudicated
  failures;
- all production jobs and task steps completed with exit code 0.

## 8. Decision after E2A

- If a low-dose safe-effective point passes, the next stage should test its
  external robustness on a new held-out checkpoint/model or dataset panel.
- If benefit persists but no low dose is safe, the next study should develop a
  utility-aware ranking objective; simply increasing k is not justified.
- If benefit vanishes at low dose and high dose remains unsafe, the mitigation
  is not operationally actionable in this panel and the ORG paper must present
  the negative frontier honestly.

No E3 is automatically authorized by E2A. E3 must receive a new contract based
on the E2A decision and must not reuse the old blocked hierarchy.

