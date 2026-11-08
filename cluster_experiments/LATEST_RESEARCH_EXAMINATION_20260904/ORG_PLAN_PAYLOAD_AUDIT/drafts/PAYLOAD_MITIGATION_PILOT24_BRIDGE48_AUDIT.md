# ORG Payload Audit: Pilot24 and Bridge48 GQDP-K8 Mitigation Evidence

**Audit date:** 2026-09-04  
**Audited payload:** `nano4_audit_verified/org_plan_priority_20260904/research_priority_payloads_20260904T140140Z`  
**Source-evidence policy:** read-only; no source result, manifest, lock, certification, or archive was modified.  
**Independent recomputation:** `LATEST_RESEARCH_EXAMINATION_20260904/ORG_PLAN_PAYLOAD_AUDIT/drafts/recompute_mitigation_payload.py`  
**Machine-readable recomputation:** `LATEST_RESEARCH_EXAMINATION_20260904/ORG_PLAN_PAYLOAD_AUDIT/drafts/PAYLOAD_MITIGATION_RECOMPUTATION.json`

## Executive decision

The payload is authentic, complete for the experiments it actually contains, and arithmetically reproducible. Pilot24 has 192/192 final protected results; Bridge48 has 96/96; every result hash in the two result ledgers matches; all 288 producer validator rows pass; both certified archive sidecars match; and the Pilot24 and Bridge48 manifests are identical across their preregistration, certification, and freeze copies.

The scientific claim must nevertheless be narrower than the producer label.

**Manuscript-usable:** a secondary, retrospective actionability probe showing that, across 48 disjoint cells, protecting eight matrices selected by a GQDP score recomputed inside each target cell produced a smaller zero-clipped forget-CE decrease than one deterministic, layer-count-matched random selection in 37 cells, a larger decrease in 3, and a tie in 8. The mean advantage was `0.385949` CE and the median was `0.035132` CE.

**Not manuscript-usable:** “confirmed unlearning mitigation,” “retained-utility-preserving defense,” “cross-condition GQDP transport,” “official TOFU/MUSE improvement,” or “parameter-budget-controlled superiority.” Those stronger interpretations are blocked by the invalid historical state lineage, absence of an original-knowledge/retraining denominator, same-objective/same-sample score construction and evaluation, a single random draw per cell, unmatched protected parameter budgets and anatomy, repeated checkpoints, severe BF16 retain-reference failure, and strong target/retain response coupling.

The correct ORG role is therefore **supporting actionability evidence with explicit limitations**, not a headline confirmatory mitigation result.

## 1. Evidence map and directory-name issue

The collector reports this target as absent:

`results/gate2_v4_bridge_standard_v42_mitigation_bridge48_confirmation`

That exact directory is absent, but the complete experiment exists under the name actually used by the locked runbook and launcher:

`payload/results/gate2_v4_bridge_standard_v42_mitigation_bridge48_gqdp_k8`

The latter contains 96 unique `bridge_v4.json` files: 48 GQDP-signed `k=8` results and 48 deterministic-random `k=8` results. It is also embedded in the certified freeze archive. The collector's `ABSENT` line is thus a target-name mismatch, not missing scientific output.

Primary evidence locations:

- Pilot preregistration: `payload/mitigation_pilot24_preregistration_20260901/`
- Pilot analysis: `payload/mitigation_pilot24_v42_analysis_20260901_190312/`
- Pilot certification: `payload/mitigation_pilot24_v42_certification_20260901_190312/`
- Pilot freeze: `payload/mitigation_pilot24_v42_certified_freeze_20260901_191041/`
- Pilot raw results: `payload/results/gate2_v4_bridge_standard_v42_mitigation_pilot24/`
- Bridge preregistration: `payload/mitigation_bridge48_v42_preregistration_20260901_195003/`
- Bridge analysis: `payload/mitigation_bridge48_v42_analysis_20260901_234113/`
- Bridge certification: `payload/mitigation_bridge48_v42_certification_20260901_234113/`
- Bridge freeze: `payload/mitigation_bridge48_v42_certified_freeze_20260901_234356/`
- Bridge raw results: `payload/results/gate2_v4_bridge_standard_v42_mitigation_bridge48_gqdp_k8/`
- Locked implementation: `payload/gate2_v4_final/scripts/` and `payload/gate2_v4_final/slurm/gate2_v4/`

## 2. Provenance and completeness audit

| Check | Independent result | Decision |
|---|---:|---|
| Pilot manifest cells | 24 unique | Pass |
| Pilot protected results | 192 paths / 192 unique keys | Pass |
| Pilot guided-vs-random pairs | 144 | Pass |
| Pilot validator rows | 192/192 pass, empty hard-error file | Pass as structural validation |
| Pilot result hashes | 192/192 match | Pass |
| Pilot archive sidecar | `47e4b5e6...11c1b3`, exact match | Pass |
| Bridge manifest cells | 48 unique | Pass |
| Pilot/Bridge cell overlap | 0 | Pass |
| Bridge protected results | 96 paths / 96 unique keys | Pass |
| Bridge GQDP/random pairs | 48/48 | Pass |
| Bridge validator rows | 96/96 pass, empty hard-error file | Pass as structural validation |
| Bridge result hashes | 96/96 match | Pass |
| Bridge archive sidecar | `324abced...383534`, exact match | Pass |

The Pilot manifest hash is `1f6a7303cd4666d3b7bc8302b70744d8a3754834d24fa8f77e208dce7026d24a` in the standalone preregistration, Selected24 composite freeze, and Pilot certified freeze. The Bridge manifest hash is `6b3c5c8a17448bbaa81b370890edf93ca60d959c1c7bc8c71fb18883287762f1` in preregistration, certification, and freeze copies.

The Bridge preregistration input ledger additionally freezes:

- Bridge manifest: `6b3c5c8a...7762f1`
- Bridge cell table: `4f9ba026...c30c9`
- ranking file: `8e2390e5...4c1bd`
- Pilot manifest: `1f6a7303...6d24a`

Slurm accounting records `COMPLETED 0:0` for Pilot job `323684` and Bridge job `324163`. All Bridge raw results use script SHA-256 `5bcc291d...a959fb` and source-lock SHA-256 `b905dee4...5ad330`; all Pilot raw results use the same runner script SHA and Pilot lock `498f66c9...faf61`.

These checks establish byte identity, completeness, and execution-contract consistency. They do not establish valid checkpoint semantics, useful retained performance, or a publishable causal interpretation.

## 3. Reconstructed design

### 3.1 Pilot24 is discovery, not confirmation

`build_mitigation_pilot24_v42.py` selected:

- 12 **high-risk** cells: the largest already observed Full144 harmful-recovery value in each model-by-quantizer stratum;
- 12 **balanced** cells: an outcome-dependent assignment nearest each stratum's median harmful recovery, covering each algorithm-by-seed combination once.

The Pilot therefore deliberately enriches and balances using historical outcome values. That is reasonable for discovery and method selection, but Pilot effect sizes cannot be represented as prevalence estimates for Full144.

Each Pilot cell evaluated four selectors at `k=4` and `k=8`: GQDP signed, corrected UPQI-star signed, displacement L2, and one deterministic-random ranking. The analyzer created 144 guided-versus-random pairs. Its expansion rule passed if any selector/k configuration had a positive cell-bootstrap interval, BH-adjusted `p<0.05`, and relative retain noninferiority in at least 80% of cells.

### 3.2 Bridge48 is a disjoint intervention replication

Bridge48 selected 48 cells with zero Pilot overlap, four cells in each of 12 model-by-quantizer strata, and four appearances of every algorithm-by-seed combination. It then compared GQDP signed and deterministic random at `k=8`, producing 96 results.

The Bridge manifest was frozen before the Bridge job. This is materially better than the Pilot and legitimately tests whether the **intervention contrast** repeats in unseen Pilot cells.

It does **not** test cross-condition score transport. For every Bridge target cell, GQDP is recomputed from that cell's own checkpoint, quantizer displacement, locked 128-example forget manifest, and forget-loss gradients. The protected outcome is then evaluated on the same forget manifest and objective. Raw `ranking_evidence.selected_manifest_sha256s` equals the result's `forget_manifest.sha256`. Thus the confirmation question is:

> Does an in-cell score-guided protection intervention repeat across new cells?

It is not:

> Does a score learned on one seed, model, dataset, algorithm, quantizer, or sample split predict or protect another condition?

The actual Bridge builder also hashes the Pilot manifest, not the Pilot decision artifact. The certified Pilot decision independently shows that GQDP-k8 passed, but the Bridge preregistration chain does not cryptographically bind the expansion to that decision. This is an auditability weakness, not evidence that the 96 results are missing or altered.

## 4. Independent numerical recomputation

### 4.1 Pilot24

All producer means and medians were exactly reproduced from raw JSON. Independent 200,000-resample intervals differ only by Monte Carlo seed/resolution.

| Selector | k | Mean advantage | Median | Independent 95% bootstrap CI | Wins/losses | Retain-noninferior |
|---|---:|---:|---:|---:|---:|---:|
| GQDP signed | 4 | 1.868168 | 0.327418 | [0.539823, 3.979940] | 23/1 | 22/24 |
| GQDP signed | 8 | 2.247106 | 0.356251 | [0.702094, 4.686827] | 23/1 | 22/24 |
| UPQI-star signed | 4 | 1.844909 | 0.456688 | [0.579194, 3.847427] | 24/0 | 23/24 |
| UPQI-star signed | 8 | 2.253237 | 0.493121 | [0.725890, 4.668965] | 24/0 | 23/24 |
| Displacement L2 | 4 | 0.122803 | -0.001859 | [-0.011662, 0.294228] | 9/15 | 15/24 |
| Displacement L2 | 8 | 0.133081 | -0.000562 | [-0.019502, 0.318203] | 10/14 | 17/24 |

The Pilot result is real as a discovery result. However, the four passing configurations are not four independent confirmations. GQDP and UPQI-star protected exactly the same layer set in 15/24 cells at `k=4` and 8/24 at `k=8`; their mean set Jaccard similarities were `0.8500` and `0.7879`. Their near-identical effects therefore mostly reflect rank coupling.

### 4.2 Bridge48

The producer summary is arithmetically correct:

- matched pairs: 48/48;
- mean harmful-recovery advantage: `0.3859493567` CE;
- median: `0.0351318717` CE;
- producer bootstrap interval: `[0.202358, 0.609481]`;
- independent 200,000-bootstrap interval: `[0.200445, 0.607816]`;
- wins/losses/ties: `37/3/8`;
- producer random-sign p-value: `0.00009999` at 10,000 draws;
- independent 200,000-draw sign-flip p-value: `0.000005`;
- exact two-sided sign-test p-value, excluding ties: `1.95e-8`;
- retain-noninferior fraction under the producer rule: `42/48 = 0.875`.

Using raw, untruncated `delta_forget_ce` rather than `max(0, -delta_forget_ce)` gives a similar direction: mean `0.429098`, median `0.087389`, and 45 positive versus 3 negative contrasts. The numeric direction is therefore not an artifact of zero clipping alone.

The magnitude is heterogeneous and right-skewed. The largest cell contributes 18.6% of the positive sum; the five largest contribute 58.6%. Aggregate harmful recovery is `100.9674` for GQDP protection versus `119.4929` for random protection, a 15.5% aggregate reduction, but raw CE totals pool conditions with very different scales and are not a normalized recovery fraction.

Treating repeated quantizers on a checkpoint as one cluster leaves 32 checkpoint clusters. Averaging inside checkpoint before inference gives mean `0.448958`, median `0.064126`, 27 positive, 1 negative, and 4 tied clusters, with an independent bootstrap interval `[0.208021, 0.748481]`. This supports a directional within-panel signal, while not solving shared-sample or four-model-tag generalization.

## 5. Compression-budget and structural-control audit

Both interventions protect eight matrices, but matrix count is not parameter count, byte budget, effective average precision, layer type, or depth. LLaMA attention and MLP matrices have different shapes, and LLaMA-3 GQA makes some key/value projections especially different in size.

For Bridge48, guided minus random protection had:

- mean protected-parameter-fraction difference: `+0.0031165` (`+0.312` percentage points);
- median: `+0.0018029`;
- range: `-0.0270433` to `+0.0312500`;
- GQDP protected more parameters in 24/48 cells, fewer in 14/48, and the same fraction in 10/48;
- mean effective-average-bit difference: `+0.032905` bits;
- range: `-0.324519` to `+0.375000` bits;
- Pearson correlation of harmful advantage with budget difference: `0.0103`.

The near-zero aggregate correlation argues against a simple “more protected parameters explains the entire mean” account. It does not make the experiment budget-controlled. Pairwise budget differences are large and bidirectional, and the selector changes matrix anatomy as well as score.

Only 10/48 pairs were within `0.0025` absolute protected-fraction difference. Their mean advantage was `0.365322`, but the median was only `0.003842`, and one cell supplied 94.5% of the positive sum. This sensitivity subset cannot substitute for a preregistered parameter-matched control.

GQDP and random selected markedly different matrix types across 384 protected positions:

| Matrix type | GQDP count | Random count |
|---|---:|---:|
| down projection | 109 | 66 |
| value projection | 107 | 54 |
| output projection | 60 | 57 |
| up projection | 43 | 48 |
| gate projection | 40 | 53 |
| query projection | 17 | 50 |
| key projection | 8 | 56 |

This difference can be the desired mechanism, a structural prior, or a size/anatomy confound. The current design cannot separate them. A valid control needs equal protected parameters or bytes and, ideally, matched matrix type and depth. The random baseline is also one SHA256 ranking per cell; there are no within-cell repeated random sets from which to estimate random-selection variance.

Pilot budgets do not explain its guided result in the naive direction: GQDP-k8 protected `0.002648` less target-parameter fraction than random on average, and UPQI-star-k8 protected `0.001169` less. Nevertheless, those cells were outcome-selected discovery cells and GQDP/UPQI rankings were highly coupled, so this does not certify either selector independently.

## 6. Retained utility and selectivity fail the manuscript gate

The producer's retain rule is relative:

`abs(guided delta_retain_ce) <= abs(random delta_retain_ce) + 0.01`

It does not require the BF16 unlearned checkpoint to have acceptable retain utility, and it does not impose an absolute post-intervention utility floor.

In Bridge48, the BF16 reference retain CE has:

- median `53.8002`;
- mean `70.2465`;
- range `1.1003` to `255.2895`;
- 26/48 (`54.17%`) above 10;
- 24/48 (`50%`) above 50;
- 15/48 (`31.25%`) above 100.

The runner computes mean answer-token CE, so these are not harmless sequence-length sums. More than half of the references are pathological on the locked retain evaluator before the protected intervention. Relative noninferiority to a random intervention cannot transform those checkpoints into retained-utility-qualified models.

GQDP-protected absolute retain shift has median `0.04043`, mean `0.86916`, and maximum `8.97609`. Only 19/48 are within `0.01`, 28/48 within `0.1`, and 39/48 within `1.0`. Sixteen of the 48 cells satisfy the producer's relative noninferiority label while the guided absolute retain shift still exceeds `0.1`.

The target and retain movements are also strongly coupled:

- Spearman(harmful advantage, retain-shift advantage) = `0.832882`;
- Spearman(random harmful recovery, random absolute retain shift) = `0.803778`;
- Spearman(guided harmful recovery, guided absolute retain shift) = `0.504682`.

This pattern is more consistent with protection reducing a general quantization-sensitivity response than with demonstrated selective protection of forgotten knowledge. It can be scientifically useful for ORG, but it should not be described as target-selective unlearning mitigation without a specificity contrast.

Restricting to the 16 cells with BF16 retain CE at most 5 leaves a positive but much smaller signal: mean `0.08073`, median `0.00785`, 12 wins, 0 losses, and 4 ties. The five largest effects account for 96.5% of the positive sum. This is supportive sensitivity evidence, not a robustness certificate.

## 7. Heterogeneity and denominators

Bridge48's overall effect is not uniform:

| Factor | Level | n | Mean advantage | Median | Wins/losses/ties | Retain-noninferior |
|---|---|---:|---:|---:|---:|---:|
| Quantizer | NF4 | 16 | 0.819556 | 0.206683 | 16/0/0 | 15/16 |
| Quantizer | GPTQ int4 | 16 | 0.300031 | 0.065421 | 12/1/3 | 15/16 |
| Quantizer | RTN int8 | 16 | 0.038261 | 0.001106 | 9/2/5 | 12/16 |
| Algorithm | NPO | 12 | 0.801731 | 0.496780 | 12/0/0 | 11/12 |
| Algorithm | GDR | 12 | 0.529602 | 0.072991 | 8/0/4 | 9/12 |
| Algorithm | GA | 12 | 0.206808 | 0.035132 | 8/3/1 | 10/12 |
| Algorithm | KLR | 12 | 0.005657 | 0.007557 | 9/0/3 | 12/12 |

The actual analyzer correctly labels subgroups descriptive. RTN does not meet the global 80% retain-noninferiority threshold within its subgroup, and KLR's magnitude is near zero. The main average should not be narrated as equal protection across operators or algorithms.

The raw result contains only the BF16 unlearned state and its protected quantized condition. It contains no valid original target-knowing checkpoint `O`, no independently retrained retain-only reference `R`, and no baseline gap against which to normalize “recovery.” Its primary outcome is:

`harmful_recovery = max(0, -(CE_condition - CE_bf16_unlearned))`

That is an operational negative forget-CE shift, not by itself proof that original knowledge returned. Pooling absolute CE differences across TOFU and MUSE, model families, algorithms, and checkpoint-quality regimes gives a direction test but not a portable effect magnitude.

Finally, the 48 cells are not 48 independent trained checkpoints: they reduce to 32 unique `(model_tag, algorithm, seed)` checkpoints, with 19 appearing under one quantizer, 10 under two, and 3 under all three. All cells also reuse one of two locked 128-example forget/retain manifests. Cell-level bootstrap intervals are therefore too optimistic as population-generalization intervals unless dependence is modeled.

## 8. Claim-decision matrix

| Proposed claim | Decision | Exact reason |
|---|---|---|
| Pilot24 and Bridge48 files are complete and frozen | **Accept** | Counts, hashes, validators, manifests, archives, and Slurm records agree. |
| Pilot guided selectors beat one deterministic random ranking in the selected discovery panel | **Accept, discovery only** | Raw effects reproduce, but cells were selected from Full144 outcomes. |
| Corrected UPQI-star has independent mitigation validation | **Reject** | Only Pilot evidence; high GQDP/UPQI layer-set coupling; no disjoint UPQI confirmation. |
| Bridge48 repeats an in-cell GQDP-guided protection contrast | **Accept narrowly** | 48 disjoint Pilot cells and 96 complete outcomes; direction survives independent recomputation. |
| GQDP score transports across conditions | **Reject** | Score is recomputed in every target cell from the same forget objective and samples used for evaluation. |
| GQDP is parameter-budget controlled against random | **Reject / unresolved** | `k` is matched, but parameters, bytes, effective bits, type, and depth are not; only one random set per cell. |
| The effect is target-specific | **Reject** | Target/retain advantages have Spearman `0.833`; general sensitivity is a live explanation. |
| Retained utility is preserved | **Reject** | Relative rule only; BF16 retain CE median `53.8`, with 54.17% above 10. |
| The effect is homogeneous across quantizers and algorithms | **Reject** | NF4/NPO dominate; RTN and KLR magnitudes are small; subgroups are descriptive. |
| Quantization-induced knowledge recovery was prevented | **Reject as causal language** | No valid `O/U/R/QO/QU` denominator; harmful recovery is a signed CE-shift label. |
| Official TOFU/MUSE efficacy improved | **Reject** | The preregistration and decision files explicitly keep official claims closed. |
| The evidence may appear in ORG | **Accept with wording below** | It is valuable as a bounded actionability/general-sensitivity probe. |

## 9. Manuscript-safe wording

Recommended result sentence:

> In a preregistered intervention replication on 48 cells disjoint from the Pilot24 discovery panel, preserving eight matrices selected by an in-cell GQDP ranking reduced the zero-clipped forget-CE decrease relative to one deterministic, layer-count-matched random ranking in 37 cells, increased it in 3, and tied in 8 (mean paired advantage 0.386 CE; median 0.035). This result demonstrates repeatable within-cell actionability on the historical evaluator, not cross-condition score transport or standard unlearning efficacy.

Required immediate caveat:

> The comparison did not match protected parameter bytes or matrix anatomy; ranking and evaluation used the same locked forget objective and samples; target and retain responses were strongly coupled; and many BF16 references failed an absolute retained-utility plausibility check. We therefore treat the experiment as a retrospective precision-allocation stress test rather than a validated unlearning defense.

Do not write:

- “GQDP prevents quantization-induced unlearning reversal.”
- “Bridge48 confirms transport across models, datasets, methods, and quantizers.”
- “Retain utility was preserved in 87.5% of cases.”
- “The control was compute- or parameter-matched.”
- “UPQI-star and GQDP independently confirm the mitigation.”
- “The result meets TOFU/MUSE standards.”

## 10. Repairs required for a stronger mitigation claim

1. Start from utility-qualified, semantically distinct `O/U/R/QO/QU` states and require target knowledge in `O`, suppression in `U`, and an absolute retain-utility floor before mitigation analysis.
2. Split the forget objective used to compute GQDP from an untouched evaluation set. For a transport claim, freeze a ranking or selection rule on discovery checkpoints and apply it without recomputing scores on target-cell evaluation data.
3. Replace the single deterministic random ranking with repeated parameter- or byte-matched random subsets per cell. Add matrix-type/depth-matched random and simple structural-prior controls.
4. Match effective quantization budget, not only `k`; report protected parameters, serialized bytes, and effective average bits for every pair.
5. Add a target-specificity estimand that contrasts forget benefit against retain/general-domain movement. The current positive target/retain coupling cannot support selectivity.
6. Normalize the recovery outcome using a valid original-to-unlearned gap or use a preregistered scale-free paired outcome. Do not pool raw CE magnitudes as though they share one physical scale.
7. Use checkpoint-clustered or hierarchical inference with crossed quantizer effects and predeclared interaction tests. Treat the four model/dataset tags as the external-generalization limit.
8. Run official maintained evaluators only if the paper makes an efficacy claim. The current ORG paper can instead keep this section explicitly secondary and diagnostic.

## Final audit verdict

**Engineering/provenance:** pass.  
**Producer arithmetic:** pass.  
**Discovery evidence:** real but outcome-selected.  
**Disjoint intervention replication:** real.  
**Parameter-budget control:** absent.  
**Retained-utility qualification:** fail.  
**Target specificity:** fail/unresolved.  
**Cross-condition score transport:** not tested.  
**Standard unlearning mitigation claim:** reject.  
**ORG secondary actionability/general-sensitivity result:** accept with strict wording and limitations.
