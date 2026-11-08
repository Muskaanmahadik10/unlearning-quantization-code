# ORG Payload Audit: Full144 Metrics, GQDP, and Selected24 UPQI-star

Date: 2026-09-04  
Audit mode: read-only reconstruction plus independent numerical recomputation  
Scope: the historical ORG measurement paper only

## Executive verdict

The transferred evidence is real, complete at the file/join level, and useful for an ORG measurement paper. It is **not valid for the interpretation currently implied by several PASS labels**.

The defensible result is:

> Across 144 algorithm-labelled historical checkpoints and 32,256 one-matrix-at-a-time quantization interventions, locally recomputed gradient-displacement scores have moderate-to-strong **within-cell association** with the stored forget-set cross-entropy response, but their rankings mostly fail to transfer when the quantizer, algorithm, architecture, or dataset changes. Seed transfer is better but still materially below native scoring.

The payload does **not** establish that:

- GQDP is an exact first-order attribution of the stored target;
- GQDP prospectively predicts unseen examples or unseen conditions;
- UPQI-star K8/N64 is numerically converged;
- UPQI-star is superior to GQDP, legacy UPQI, or a standard comparator;
- the response is selective to forget knowledge rather than a shared likelihood shift;
- any checkpoint satisfies official TOFU/MUSE unlearning efficacy;
- a fall in forget-set CE is necessarily causal recovery of knowledge.

For ORG, the Full144 evidence can enter the main manuscript after a new dependency-aware retrospective analysis and strict relabelling. Selected24 UPQI-star should be secondary/supplementary evidence only.

## Evidence roots and audit reproducibility

Primary transferred payload root:

```text
nano4_audit_verified/org_plan_priority_20260904/research_priority_payloads_20260904T140140Z/payload
```

Independent recomputation script created for this audit:

```text
LATEST_RESEARCH_EXAMINATION_20260904/ORG_PLAN_PAYLOAD_AUDIT/drafts/recompute_payload_metrics.py
```

The script imports none of the producer or analysis code. It independently reads the frozen JSON files, reconstructs the joins, recomputes tied-rank Spearman coefficients, top-k overlap, sign agreement, target concentration, and directed cross-condition transfer. It uses Windows extended-length paths and does not modify source evidence.

Additional exact historical material already present in the workspace:

```text
GATE2_V42_METRIC_LAYERWISE_ANALYSIS_PREREGISTRATION_20260901.md
gate2_v42_metrics_mitigation_setup_20260901.tar.gz
gate2_v42_metrics_mitigation_setup_20260901.tar.gz.sha256
july/gate2_v4_metric_script_audit_20260731_062129/extracted_source/manifests/gate2_v4/
```

The setup archive SHA-256 matches its sidecar:

```text
6c5b1771f6004bed4fbd245cf7324ddd361a949062c51baaa8bb2d4aba606c44
```

It contains the exact UPQI-star producer and validator versions recorded in the source lock. They were extracted, without altering the archive, under:

```text
LATEST_RESEARCH_EXAMINATION_20260904/ORG_PLAN_PAYLOAD_AUDIT/tmp_metric_setup/gate2_v4_final/scripts/
```

## 1. What is actually complete

### 1.1 Factorial inventory

The raw metric evidence reconstructs exactly:

| Factor | Levels |
| --- | --- |
| Architecture | Llama 2, Llama 3 |
| Dataset-labelled branch | TOFU, MUSE-News |
| Algorithm label | GA, GDR, KLR, NPO |
| Seed | 42, 2024, 5000 |
| Quantizer | GPTQ-style int4, NF4, RTN-int8 |
| Cells | `2 × 2 × 4 × 3 × 3 = 144` |
| Target matrices per cell | 224: seven projection matrices in each of 32 transformer blocks |
| One-matrix interventions | `144 × 224 = 32,256` |

All four dataset/architecture branches contain 36 cells. Each quantizer contains 48 cells.

### 1.2 Integrity and join checks

Independent checks found:

| Check | Result |
| --- | ---: |
| Full144 `metrics_v42.json` files | 144/144 |
| Full144 result-list SHA-256 matches | 144/144 |
| Referenced GT SHA-256 matches | 144/144 |
| Referenced legacy-UPQI SHA-256 matches | 144/144 |
| Metric/GT/legacy cell identity matches | 144/144 |
| Exact 224-layer name-set joins | 144/144 |
| Selected24 K8/N64 files | 24/24 |
| Selected24 result-list SHA-256 matches | 24/24 |
| Duplicate metric cells | 0 |
| Missing metric/GT/legacy cells | 0 |

The canonical raw roots are:

```text
payload/results/gate2_v4_standard_metrics_v42_full144/
payload/results/gate2_v4_standard/
payload/results/gate2_v4_standard_upqi_star_v42_selected24/K8_N64/
```

This is enough to say that the historical response grid exists. It is not enough to adopt the original scientific interpretation.

## 2. Exact metric and target definitions

### 2.1 Stored one-matrix target

For each cell and matrix, the GT producer quantizes only that matrix, restores all others, and stores:

```math
\Delta CE_{sample} = \frac{1}{N}\sum_{i=1}^{N} CE_i(Q_l(W))
                   - \frac{1}{N}\sum_{i=1}^{N} CE_i(W),
```

where each `CE_i` is already the mean over the answer tokens of sample `i`.

The implementation computes per-sample mean-token CE in `run_gate2_ground_truth_v4.py:120-140`, appends one CE per sample in lines 145-215, averages those values at line 215, and stores `CE_quantized - CE_bf16` at line 529.

The frozen analysis then defines:

```math
H_l = \max(0,-\Delta CE_l)
```

as `harmful_recovery`, and `|ΔCE_l|` as `absolute_impact`. See `analyze_gate2_metric_suite_v42.py:319-320`.

Scientifically, `H_l` is a **clipped negative forget-set CE shift**. Calling it recovered knowledge requires evidence that the checkpoint previously knew the content, the suppression was valid, and the CE change is content-specific. This payload does not supply those conditions.

### 2.2 GQDP v4.2

The endpoint metric producer computes a gradient on the algorithm-labelled checkpoint and the actual matrix-specific quantization displacement:

```math
g_l = \nabla_{W_l}\left[
  \frac{1}{\sum_i T_i}\sum_i\sum_{t=1}^{T_i}\ell_{it}
\right],
\qquad
\epsilon_l = Q_l(W_l)-W_l.
```

It then records:

```math
GQDP_{signed,l}=g_l^\top\epsilon_l,
```

```math
GQDP_{abs,l}=|g_l^\top\epsilon_l|,
```

```math
GQDP_{cos,l}=\frac{g_l^\top\epsilon_l}{\|g_l\|_2\|\epsilon_l\|_2},
```

```math
GQDP_{per\_param,l}=\frac{g_l^\top\epsilon_l}{p_l}.
```

The formulas are in `gate2_metric_v42_math.py:72-105`. The gradient accumulator uses summed token loss divided by total answer tokens in `run_gate2_metric_suite_v42.py:104-151`; the result explicitly labels this `answer_token_weighted_mean_gradient` at lines 396-397.

For harmful-direction ranking, the analysis negates signed GQDP and cosine. `GQDP_abs` is the absolute value of the signed dot product; it is **not** the separate comparator `sum(abs(g) * abs(epsilon))`.

### 2.3 The decisive objective mismatch

The predictor and target do not optimize the same scalar:

| Quantity | Aggregation |
| --- | --- |
| GT `delta_forget_ce` | equal weight per sample after taking each sample's mean-token CE |
| GQDP gradient | equal weight per answer token across all samples |
| Legacy UPQI gradient | equal weight per answer token |
| UPQI-star gradient | equal weight per answer token |

Therefore GQDP is **not the exact endpoint derivative of the stored GT target**. It is a local derivative of a differently weighted CE objective associated with that target.

This is material because answer lengths are heterogeneous:

| Dataset panel | N | Min tokens | Median | Mean | Max | CV | Max/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TOFU | 128 | 9 | 43 | 44.65 | 85 | 0.33 | 9.44 |
| MUSE-News | 128 | 10 | 414 | 347.84 | 510 | 0.39 | 51.0 |

For representative checkpoints, merely changing baseline aggregation from sample-weighted to token-weighted changes CE by about `+0.04`, `+0.16`, `-1.15`, and `-9.41` across the four model/dataset branches. The exact one-layer delta under the token-weighted objective cannot be reconstructed from the present per-layer GT JSON because quantized per-sample responses were not stored.

This defect does not erase the observed association. It changes the permissible claim from **exact first-order attribution** to **association between a token-weighted local diagnostic and a sample-weighted perturbation response**.

### 2.4 Sample counts and denominator ledger

| Evidence | Samples | Other denominator |
| --- | ---: | --- |
| Full144 GT target | 128 | equal sample weighting |
| Full144 GQDP | 128 | total answer-token weighting |
| Legacy UPQI | 128 | K=8 path samples; total answer-token weighting |
| Forget Fisher / FQP | 128 | average of per-sample gradient-square summaries |
| Retain Fisher | 128 | average of per-sample gradient-square summaries |
| Selected24 UPQI-star | first 64 of locked 128 | K=8 Gauss-Legendre; total answer-token weighting |
| GPTQ calibration | 128 retain samples | quantizer-specific Hessian context |

The raw dot product is also size-dependent. `GQDP_per_param` was computed but omitted from the frozen analysis even though the registry required it.

## 3. Data-panel audit

The exact manifest files matching the source-lock hashes are recoverable under:

```text
july/gate2_v4_metric_script_audit_20260731_062129/extracted_source/manifests/gate2_v4/
```

Their contracts are:

| Panel | Eligible pool | Selected | Selection |
| --- | ---: | ---: | --- |
| TOFU forget | 400 | 128 | deterministic hash without replacement |
| MUSE-News forget | 3,550 | 128 | deterministic hash without replacement |

The same 128 examples are used to compute the score and stored target. There is no example-level holdout, so the native correlation is in-sample.

Selected24 UPQI-star uses `samples[:64]` from that same locked 128 while the referenced GT delta uses all 128. The 64 are an overlapping subset, not a held-out evaluation set.

The MUSE-News branch is a custom answer-only CE stress panel. The dataset code preserves the answer by trimming the prompt when the sequence exceeds 512 tokens (`gate2_v4_dataset.py:35-80`). With answers reaching 510 tokens, some prompts are reduced to almost only the `Answer:` boundary. This is valid as the declared operational response, but it must not be presented as official MUSE knowledge, privacy, verbatim-memorization, or utility evaluation.

## 4. Independent reconstruction of native associations

The following values were independently recomputed from raw metric, GT, legacy-UPQI, and Selected24 JSON—not copied from `metric_summary.tsv`.

| Score and target | Cells | Median Spearman | Mean | Positive-cell fraction | Mean top-4 overlap | Mean top-8 overlap | Mean sign accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GQDP signed vs clipped `H` | 144 | **0.5327** | 0.5251 | 95.1% | 65.8% | 59.5% | 70.4% |
| GQDP signed vs raw signed `-ΔCE` | 144 | **0.5801** | — | — | — | — | — |
| Legacy UPQI signed vs clipped `H` | 144 | **0.5451** | 0.5400 | 95.1% | 69.4% | 62.9% | 71.0% |
| Legacy UPQI signed vs raw signed `-ΔCE` | 144 | **0.5936** | — | — | — | — | — |
| UPQI-star signed vs clipped `H` | 24 selected | **0.5857** | 0.5803 | 100% | 71.9% | 70.3% | 71.8% |
| UPQI-star signed vs raw signed `-ΔCE` | 24 selected | **0.6341** | — | — | — | — | — |
| GQDP per parameter vs clipped `H` | 144 | **0.4894** | 0.4859 | 94.4% | 45.1% | 43.1% | — |
| GQDP cosine vs clipped `H` | 144 | **0.4855** | 0.4453 | 92.4% | 23.4% | 27.3% | — |
| Gradient L2 vs absolute impact | 144 | **0.2934** | 0.3225 | 87.5% | 44.6% | 36.9% | — |
| Parameter count vs absolute impact | 144 | **0.0513** | 0.0775 | 67.4% | 24.1% | 18.2% | — |

### 4.1 What the native result does establish

- The association is not only raw parameter count: per-parameter GQDP and cosine remain moderately associated.
- Raw signed direction matters. Signed metrics outperform their absolute counterparts for the clipped negative-CE target.
- The result is highly heterogeneous rather than uniformly strong.

Important subgroup medians for GQDP signed are:

| Group | Median native Spearman |
| --- | ---: |
| MUSE-News | 0.7415 |
| TOFU | 0.3122 |
| NF4 | 0.8084 |
| GPTQ-int4 | 0.3748 |
| RTN-int8 | 0.3400 |
| NPO | 0.7515 |
| GDR | 0.5907 |
| KLR | 0.4156 |
| GA | 0.3776 |

The 5th-to-95th percentile interval across cells is about `[0.027, 0.980]`; at least seven cells have non-positive correlation. The headline must show this distribution, not only the pooled median.

### 4.2 Clipping changes the scientific result

The clipped target sets every over-forgetting/positive-CE response to zero. Across cells, the median positive-layer fraction is only 57.8%; thus roughly 42% of layers in a typical cell share the same zero target rank. This reduces the median correlations:

- GQDP: `0.5801` raw signed to `0.5327` clipped;
- legacy UPQI: `0.5936` to `0.5451`;
- Selected24 UPQI-star: `0.6341` to `0.5857`.

Only one cell has fewer than eight positive layers, so top-8 zero-tie selection is directly arbitrary in one cell; five cells have fewer than 16 positive layers. Spearman is nevertheless broadly affected by the large tied-zero block in every sparse cell.

The paper should make raw signed `-ΔCE` the primary mathematical response and use clipped harmful-only response as a policy-oriented sensitivity analysis.

## 5. The main ORG result: native attribution does not transport

The frozen analysis did **not** calculate transport. Its factor contrasts compare native within-cell accuracy at different factor levels. They do not apply a source ranking to a target condition.

I independently performed the actual directed transfer calculation. For every pair differing in exactly one factor, the source-cell GQDP ranking was evaluated against the target cell's clipped response. These values are descriptive because directed pairs reuse cells and are not independent.

| Changed factor | Directed pairs | Median source score vs target response | Median native target association | Median native-minus-transfer gap | Median source/target score rank agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quantizer | 288 | **0.0364** | 0.5327 | 0.4326 | 0.0267 |
| Unlearning-algorithm label | 432 | **0.0605** | 0.5327 | 0.4343 | 0.0673 |
| Architecture | 144 | **0.0235** | 0.5327 | 0.4455 | 0.0563 |
| Dataset panel | 144 | **0.0254** | 0.5327 | 0.4759 | 0.0586 |
| Seed | 288 | **0.2256** | 0.5327 | 0.1261 | 0.5934 |

Legacy UPQI gives the same qualitative result: median transfer is approximately `0.043`, `0.062`, `0.040`, `0.034`, and `0.242` for quantizer, algorithm, architecture, dataset, and seed respectively.

This is the loud, defensible ORG story:

1. A score computed on the exact checkpoint/objective/displacement path can rank that same path moderately well.
2. The layer ordering is largely recomputed by quantizer, algorithm, architecture, and dataset rather than inherited across them.
3. Seed changes preserve considerably more ranking structure, but still lose meaningful target association.

An empirical layer-identity prior trained on the other three model tags obtains median Spearman `0.1379`. It is weaker than native GQDP but stronger than GQDP transferred across quantizer, algorithm, architecture, or dataset in this preliminary calculation. This is a useful structural control, not yet a final benchmark.

Before publication, transport intervals must resample independent checkpoint blocks—not the 144/288/432 directed pairs—and report source/target direction separately. The present figures are retrospective discovery results.

## 6. Fragility concentration and specificity

### 6.1 Response concentration

The negative-CE response is concentrated, but not uniformly so:

| Oracle-ranked matrices | Median fraction of total clipped response mass | Mean |
| --- | ---: | ---: |
| Top 4 | 29.5% | 38.8% |
| Top 8 | 40.9% | 46.7% |
| Top 16 | 55.0% | 57.1% |
| Top 32 | 69.6% | 70.4% |

Top-4 share ranges from about 5.1% to 99.5% across cells. The manuscript may claim heterogeneous concentration; it should not claim a universal small vulnerable set.

### 6.2 Matrix-level selectivity is absent

The 32,256 one-matrix GT outputs evaluate forget examples only. Retain Fisher is a sensitivity proxy, not an observed one-matrix retain-CE response. Therefore the payload cannot determine whether a matrix has a high forget effect and low retain cost.

At the **whole-model full-quantization cell level only**, the 144 bridge outcomes show:

- Spearman between clipped forget response and `-delta_retain_ce`: **0.8824**;
- retain CE decreases in 77.8% of cells and increases in 22.2%.

This strong shared-direction coupling suggests that much of the apparent forget recovery may reflect a broad likelihood/perplexity shift rather than target-specific recovered knowledge. It is not a substitute for per-matrix selectivity or official retained-utility evaluation.

## 7. Selected24 UPQI-star is not a superiority or convergence result

### 7.1 Outcome-dependent cell selection

Selected24 is not a random or representative subset. `build_mitigation_pilot24_v42.py:155-195` deliberately chooses:

- 12 high-risk cells: the maximum whole-model harmful response in each model-by-quantizer stratum;
- 12 balanced cells: near the stratum median, with algorithm-by-seed coverage.

The pooled UPQI-star median `0.5857` mixes sharply different panels:

| Panel | Cells | UPQI-star median | GQDP median | Legacy-UPQI median |
| --- | ---: | ---: | ---: | ---: |
| High-risk | 12 | 0.8270 | 0.84 | 0.84 |
| Balanced | 12 | 0.3547 | 0.35 | 0.35 |

The high pooled value is largely panel composition, not evidence that UPQI-star improves the metric.

### 7.2 Direct comparison finds no superiority

On the same 24 cells:

- mean `(UPQI-star rho - GQDP rho)` = `-0.00578`;
- median difference = `-0.00097`;
- UPQI-star is better in only 11/24 cells (45.8%);
- median rank agreement between UPQI-star and GQDP is `0.9341`;
- median rank agreement between UPQI-star and legacy UPQI is `0.9725`.

Across all 144 cells, legacy UPQI exceeds GQDP by only `0.00142` in median paired difference, is higher in 67.4% of cells, and has a large mean difference (`0.0149`) driven by a small number of cells. A paired, cluster-aware comparison—not separate positive-signal tests—is required for any ordering claim.

The frozen claim gate calls `UPQI_star_superiority` PASS merely when UPQI-star has a positive directional signal. The code at `analyze_gate2_metric_suite_v42.py:500` performs no UPQI-star-versus-GQDP, legacy-UPQI, or random superiority test. The PASS label is false as a superiority claim.

### 7.3 K8/N64 was not demonstrated converged

The convergence payload contains only six files: K4/N32 and K8/N64 on three cells. In `decision_fixed/upqi_star_convergence_cells.tsv`, each K8/N64 row is compared to itself and necessarily obtains Spearman/top-8 overlap equal to 1. There are no K16/N128 result files.

The generated `UPQI_STAR_CONVERGENCE.md` says the comparison is against K16/N128, but that sentence is hard-coded at `analyze_upqi_star_v42_convergence.py:220-221`. The fixed invocation changed the reference to K8/N64; the table itself records `reference_k=8`, `reference_n=64`.

K4/N32 signed-rank agreement with K8/N64 on the three cells is:

```text
0.9955, 0.9180, 0.9951
```

The middle cell fails the 0.95 threshold. The decision therefore correctly says `UPQI_STAR_REFERENCE_ONLY_PASS_RUN_FULLGRID48`: K8/N64 was the maximum tested operational reference, not a higher-order-validated converged estimator.

The later `reference_acceptance.json` transparently amends the path and chooses K8/N64 without FullGrid48. That may be an operational decision, but it cannot be reported as preregistered convergence.

### 7.4 Completeness residual is not a completeness test

The exact locked producer computes:

```text
completeness_residual = UPQI_star_signed - GT_delta_forget_ce
```

at `run_gate2_upqi_star_v42.py:147-162`. This is the opposite sign of the registry wording, which says observed endpoint change minus path attribution. Absolute residual summaries are unaffected, but signed nonlinearity plots would be reversed.

More importantly:

1. UPQI-star integrates gradients on the first 64 samples;
2. its GT reference is the 128-sample endpoint delta;
3. UPQI-star is token-weighted while GT is sample-weighted.

Thus the residual combines sample-subset change, aggregation mismatch, and quadrature error. It cannot isolate path-integration completeness.

Across 5,376 Selected24 matrix rows:

| Quantity | Median | Mean | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Absolute recorded residual | 0.001210 | 0.004167 | 0.010153 | 2.33139 |
| Absolute GT effect | 0.001776 | 0.021631 | 0.080282 | 14.78874 |

Six of 24 cells contain an absolute residual above 0.1. The validator only emits `large_completeness_residual_review_kn_convergence` when the maximum exceeds 0.1 and still returns PASS; see `validate_upqi_star_v42.py:134-136`.

UPQI-star may enter the supplement as a maximum-tested K8/N64 path score with this limitation. It must not be called a converged or complete reference.

## 8. Preregistration and analysis-contract audit

The file named `metric_v42_full144_prereg_freeze_20260901_052937` is an after-run evidence freeze, not a prospective registration of Full144 outcomes. The separate registry explicitly states at lines 9-13 that Full144 endpoint metrics were already 144/144 certified. Its local timestamp is later than the Full144 analysis/freeze names. It is prospective mainly for downstream UPQI and mitigation work.

The registry required substantially more than the frozen analysis delivered:

| Registered item | Delivered in frozen Full144 analysis? |
| --- | --- |
| Spearman | Yes |
| Kendall | No |
| Top-k precision/recall/overlap | Partial: one overlap-equivalent recall value |
| NDCG and enrichment over random | No |
| Calibration and out-of-cell calibration error | No |
| Raw signed near-zero analysis | No; clipped target used |
| `GQDP_per_param_v42` | Computed in JSON, omitted from analysis |
| Nonlinearity/completeness map | No |
| Gini/participation ratio/concentration | No |
| Anatomical heatmaps with block uncertainty | No |
| Actual cross-quantizer transfer | No |
| Cross-condition rank stability/entropy/influence | No |
| Observed forget-retain matrix geometry | Not available in raw GT |
| Multivariable mechanism decomposition | No |
| Leave-one-block sensitivity | No |

The registry also says at line 162 that if K4/N32 fails, the pipeline must stop and must not manually choose K8 or launch Pilot24. K4 failed one cell; K8 was later accepted by amendment. The amendment is usable as disclosed exploratory governance, not unchanged preregistration.

### Statistical-contract issues

- Native summary bootstrap treats 144 cells as exchangeable even though three quantizers share each algorithm-labelled checkpoint. Intervals should cluster at the 48 checkpoint blocks at minimum.
- The sign-flip p-value tests the mean cell correlation while the gate combines it with a bootstrap interval for the median. This hybrid is not a single coherent estimand.
- BH correction is applied across all predictor summaries together, mixing clipped-harmful and absolute-impact families, rather than an explicitly justified family structure.
- `predictor_gate` asks only whether GQDP or UPQI-star has positive signal above the random median; it does not test incremental value over the strongest non-random comparator.
- Factor contrasts are matched, but the reported blocks are still reused across several contrasts and predictors. Manuscript inference needs a declared cluster/bootstrap hierarchy and influence analysis.

Existing q-values may be reported only as historical pipeline outputs, not as the final confirmatory inference.

## 9. Claim-decision table for the ORG manuscript

| Proposed claim | Decision | Exact safe wording/boundary |
| --- | --- | --- |
| Full144 exists | **Accept** | 144 complete cells and 32,256 one-matrix interventions were reconstructed with exact joins and hashes. |
| GQDP has native signal | **Accept with relabelling** | Token-weighted local GQDP is associated with the sample-weighted negative forget-CE response within the same cell. |
| GQDP is exact | **Reject** | Predictor and target have different sample/token weighting. |
| GQDP predicts unseen examples | **Reject** | Score and target use overlapping fixed samples; no example holdout. |
| GQDP transports | **Reject; invert as finding** | Rankings largely collapse across quantizer, algorithm, architecture, and dataset; seed transfer is partial. |
| GQDP adds value beyond structure/magnitude | **Not established** | Requires paired incremental, permutation, and anatomy controls. |
| Fragility is concentrated | **Accept descriptively** | Concentration is strong in some cells and weak in others; report full distribution. |
| Matrix-level forget/retain selectivity | **Reject** | Per-matrix retain outcomes are absent. |
| Whole-model changes are forget-specific | **Reject** | Forget and retain CE shifts are strongly coupled across cells. |
| UPQI-star is superior | **Reject** | No superiority test; direct Selected24 comparison gives no improvement. |
| K8/N64 UPQI-star is converged | **Reject** | It is the maximum tested operational reference; no K16/N128 evidence. |
| UPQI-star residual proves completeness | **Reject** | N=64 vs 128 and token-vs-sample objective mismatch. |
| Official TOFU/MUSE success | **Reject** | This is a custom answer-only CE perturbation panel, not official evaluation. |
| Quantization restored knowledge | **Reject as causal wording** | Use negative forget-set CE shift or CE-decreasing response. |
| Historical checkpoints are valid unlearned states | **Not established here** | Call them algorithm-labelled historical checkpoints unless clean state lineage is proven elsewhere. |

## 10. How the ORG paper should use this payload

### Main-text evidence

1. The exact 144-cell/32,256-intervention design and response distribution.
2. Native raw-signed and clipped-target associations, with all cell distributions.
3. Normalized controls: per-parameter GQDP, cosine, gradient, displacement, parameter count, anatomy.
4. True directed transport across quantizer, seed, algorithm, architecture, and dataset.
5. Heterogeneous response concentration.
6. Whole-model forget/retain coupling as a specificity warning.

### Supplement only

- Selected24 UPQI-star K8/N64;
- K4-versus-K8 three-cell numerical sensitivity;
- completeness residual, explicitly labelled as a mixed discrepancy rather than pure quadrature error;
- historical PASS/decision files and amendment chronology.

### Prohibited terminology

Do not write:

- “exact GQDP attribution”;
- “prospective predictor”;
- “converged UPQI-star”;
- “UPQI-star superiority”;
- “recovered forgotten knowledge”;
- “official TOFU/MUSE performance”;
- “selective protection” from Full144 alone;
- “preregistered Full144 confirmation.”

Prefer:

- “same-cell local diagnostic association”;
- “one-matrix quantization response”;
- “negative forget-panel CE shift”;
- “retrospective cross-condition transfer test”;
- “maximum-tested K8/N64 path score”;
- “algorithm-labelled historical checkpoint.”

## 11. Work required before manuscript figures are final

### Required CPU/statistical reanalysis

1. Freeze a corrected claim and estimand ledger before generating new paper tables.
2. Make raw signed `-ΔCE` primary and clipped harmful-only response secondary.
3. Recompute native statistics with checkpoint-block bootstrap intervals and leave-one-block influence.
4. Run directed transport with cluster-aware intervals, separately by source/target direction.
5. Compare GQDP, per-parameter GQDP, cosine, legacy UPQI, gradient, displacement, anatomy, and random using paired differences rather than separate significance tests.
6. Add block/submodule permutation controls and a leakage-safe anatomical prior.
7. Add rank entropy, Gini, participation ratio, and concentration curves.
8. Define near-zero thresholds before sign/top-k reporting and run tie-robust sensitivity analyses.
9. Separate TOFU and MUSE-News throughout; never let the easier MUSE association dominate one pooled headline.
10. Audit all tables against the raw JSON with a generated claim-to-artifact manifest.

### Evidence that must be bundled immutably

The final ORG bundle should add:

```text
GATE2_V42_METRIC_LAYERWISE_ANALYSIS_PREREGISTRATION_20260901.md
gate2_v42_metrics_mitigation_setup_20260901.tar.gz
gate2_v42_metrics_mitigation_setup_20260901.tar.gz.sha256
the four exact source-locked manifest JSON files
the new analysis source lock and result hashes
```

The priority payload's current `gate2_v4_final/scripts/run_gate2_upqi_star_v42.py` and validator are later versions and do not match the producing hashes. The exact producing versions do exist in the verified setup archive:

```text
run_gate2_upqi_star_v42.py
32851b296b72a26881adde995e045ad2699ee08475a0935325d45207b8a8dfc5

validate_upqi_star_v42.py
4f81f906ea32ce406e5d5460bd6a7363adbd40defc876823da8736c550ed29a1
```

The later producer differs only by adding explicit subset-layer/resume-contract support; the core formula is unchanged. Still, the producing source—not only its hash—must be in the final bundle.

### When new GPU work is and is not needed

No new GPU work is needed for the core **transport-stress-test** paper if the claims remain descriptive and bounded as above.

New evaluation is required if the paper insists on any of the following:

- exact endpoint/path completeness under one identical scalar objective;
- example-held-out prospective prediction;
- observed per-matrix retain cost/selectivity;
- official TOFU/MUSE unlearning efficacy;
- clean causal claims about knowledge suppression and recovery.

The cheapest scientifically sound path is therefore to publish ORG as a retrospective measurement and transport audit, not to spend GPU time trying to force these artifacts into an official unlearning-efficacy claim.

## Final decision

**Use Full144 in the ORG main paper after corrected reanalysis. Use Selected24 UPQI-star only as an exploratory supplement.**

The strongest paper is not “our predictor works everywhere.” It is:

> A locally tailored attribution score can look convincing on its own quantization path while carrying little ranking information into another operational condition.

That conclusion is supported by the raw grid, survives normalization checks, is more defensible than the historical PASS labels, and gives the ORG paper a clear scientific purpose without pretending that the old lineage proves standard unlearning efficacy.
