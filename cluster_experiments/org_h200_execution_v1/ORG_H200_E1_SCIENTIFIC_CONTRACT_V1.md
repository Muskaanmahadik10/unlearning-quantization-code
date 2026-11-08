# ORG H200 E1 Scientific Contract v1

Frozen: 2026-09-05, before inspection of H200 canary scientific endpoint values
or any H200 E1 outcome.

## Claim boundary

E1 tests whether preregistered matrix rankings identify a fixed budget of eight
weight matrices whose preservation reduces post-unlearning quantization-induced
harmful recovery. It is a quantization-durability, ranking-transport, and
post-hoc precision-allocation study. It is not evidence of certified deletion,
retraining-equivalent forgetting, or standard TOFU/MUSE unlearning efficacy.

The original Core72 cell selection, rankings, data manifests, primary k=8, and
eight matched random controls remain unchanged. The H200 amendment supersedes
only the former GB200 execution/claim language. Canary outputs are engineering
certification only and are excluded from E1 analysis.

## Experimental unit and conditions

The analysis cluster is the unlearned checkpoint identified by model tag,
algorithm, and seed. Quantizer-specific cells within the same checkpoint remain
paired. Random-control seeds are repeated interventions, not independent cells.

Each cell has one full-quantized condition and eleven protected conditions:
native GQDP, cyclic cross-quantizer transported GQDP, legacy signed UPQI, and
eight frozen projection-type/depth-matched random controls. Every protected
condition uses k=8 matrices. Protected parameter fraction is reported for every
arm and is never silently treated as identical merely because k is identical.

## Frozen primary estimand

For cell c and condition s, let H(c,s) be the stored `harmful_recovery`, equal
to `max(0, -delta_forget_ce)` relative to the same cell's embedded BF16
reference. Define mitigation benefit

`B(c,s) = H(c,full_quantized) - H(c,s)`.

Positive B means protection reduced harmful recovery. Let R(c) be the mean of
the eight matched-random B values. The primary cell contrast is

`D(c) = B(c,GQDP_signed_v42) - R(c)`.

The primary population estimand is the checkpoint-cluster-weighted mean D,
with quantizer arms retained as paired observations. Transported GQDP and
legacy UPQI are ordered secondary contrasts against the identical R(c).

Raw signed `-delta_forget_ce`, answer log-probability, and truth-ratio movement
are sensitivity endpoints. Truth ratio is never substituted post hoc as the
primary endpoint.

## Utility and budget checks

For each condition, retain cost is its `delta_retain_ce` minus the corresponding
full-quantized `delta_retain_ce`. The primary non-inferiority margin is
`max(0.05, 0.01 * abs(BF16 retain answer_ce))` within cell. Any primary benefit
that fails this retain-utility condition is reported as an unsafe tradeoff.

The main analysis is k-matched. A sensitivity model adjusts for protected
parameter fraction. No selector may be rerun with a different k after outcomes
are seen.

## Inference

- Resampling unit: checkpoint `(model_tag, algo, seed)`; all its quantizer arms
  and random controls move together.
- Bootstrap: 10,000 checkpoint-cluster resamples, seed 20260905.
- Primary test: one-sided 0.025 test that mean D is greater than zero, with a
  two-sided 95% confidence interval also reported.
- Ordered secondary tests: transported GQDP then legacy UPQI; Holm adjustment
  across these two tests. They cannot rescue a failed primary claim.
- Always report means, medians, confidence intervals, sign proportions, raw
  cell values, and standardized effects.
- Required subgroup reporting: dataset, model tag, algorithm, seed, and
  quantizer; plus leave-one-model-tag-out and leave-one-quantizer-out analyses.
- Missing or failed cells are not imputed. Confirmatory inference requires all
  72 cells unless a failure is proven infrastructure-only and rerun under the
  same immutable lock.

## E1 continuation and futility gates

E2 is authorized only if all conditions below hold:

1. The unrectified recovery signal `-delta_forget_ce` for full quantization has
   positive checkpoint-cluster mean with a two-sided 95% lower bound above zero.
2. Native GQDP has primary mean D greater than zero with one-sided p below
   0.025 and a positive two-sided 95% lower confidence bound.
3. Native GQDP's median D is positive, its sign is positive in both datasets,
   and leave-one-model-tag-out estimates are all non-negative.
4. The retain-utility non-inferiority condition passes at the aggregate level
   and in both datasets.
5. No source, checkpoint, manifest, count, finite-value, budget, or execution
   integrity gate fails.

If the recovery signal is absent, E1 is a valid negative result and E2/E3 stop.
If recovery exists but native GQDP does not beat matched random, the mitigation
claim fails and E2/E3 stop. Favorable secondary metrics cannot override either
futility decision.

## Outcome-blind technical stress

Sixteen cells, selected deterministically as one cell from every model-tag by
algorithm stratum, form a two-node launcher stress wave. Only schemas, hashes,
counts, identities, finite values, and resource behavior may be inspected after
this wave. Scientific metric values remain sealed. Passing stress outputs are
valid E1 outputs and are reused in the final 72-cell analysis.

