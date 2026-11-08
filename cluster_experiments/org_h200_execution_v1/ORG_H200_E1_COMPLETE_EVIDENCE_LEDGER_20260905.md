# ORG H200 E1 complete evidence ledger

Status date: 2026-09-05 (Asia/Taipei)

This is the authoritative human-readable index for the completed ORG H200 E1
campaign. It records what was executed, where the evidence lives on Nano4, what
was verified, what the scientific results say, and what they do not establish.
Machine-readable locks, receipts, checksum manifests, validators, and Slurm
accounting remain the primary evidence when this document and an artifact ever
disagree.

## 1. Study identity and claim boundary

E1 studies whether a frozen matrix ranking identifies eight matrices whose
preservation reduces quantization-induced harmful recovery after an existing
targeted-knowledge-suppression intervention. It is a quantization-durability,
ranking-actionability, and precision-allocation experiment.

E1 is not a valid standalone demonstration of certified machine unlearning,
retraining-equivalent forgetting, or official TOFU/MUSE efficacy. The older
v4.2 lineage conflated checkpoint roles and remains quarantined as exploratory.
The ORG paper must not imply that E1 repairs that causal defect.

Frozen E1 estimands and inference are in:

- `org_h200_execution_v1/ORG_H200_E1_SCIENTIFIC_CONTRACT_V1.md`
- `org_h200_execution_v1/ORG_H200_E1_ANALYSIS_METHOD_AMENDMENT_V1.md`
- `org_h200_execution_v1/PREREGISTRATION_AMENDMENT_H200_PRIMARY_V1.md`

## 2. Frozen design

- Hardware: NVIDIA H200.
- Experimental cells: 72.
- Checkpoint clusters: 40 distinct available checkpoint identities.
- Datasets/model families: TOFU-oriented and MUSE-oriented Llama-2/Llama-3
  checkpoint families as frozen in the Core72 manifest.
- Algorithms: GA, GDR, KLR, and NPO where checkpoints existed in the frozen
  table.
- Quantizers: GPTQ int4, NF4, and RTN int8.
- E1 budget: `k=8` protected matrices.
- Conditions per cell: one fully quantized baseline plus 11 protected arms.
- Protected selectors: native signed GQDP v4.2, cyclic cross-quantizer
  transported GQDP, legacy signed UPQI v4, and eight type/depth-matched random
  controls.
- Results expected and obtained: 72 x 12 = 864 `bridge_v4.json` files.
- Cell receipts expected and obtained: 72.

The experimental unit for population inference is the checkpoint
`(model_tag, algorithm, seed)`. Quantizer arms are paired observations within a
checkpoint. The eight random selectors are repeated controls, not independent
experimental units.

## 3. Exact source and checkpoint lineage

### Core72 execution source

- E1 V3 source lock:
  `org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V3.json`
- Parent V2 source-lock SHA-256:
  `dd1528d1d5378fb9a85353ae0960685862b15623291f41c4991fb3be949712e0`
- E1 V3 result root:
  `results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v3`
- Frozen cell table:
  `org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_cells_v2.tsv`
- Frozen manifest:
  `org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_manifest_v2.json`
- Frozen ranking file:
  `org_gb200_execution_v1/preregistration/core72_v1/org_core72_rankings_v1.jsonl`

### Checkpoint byte identity

- Ledger:
  `org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_LEDGER_V1.json`
- Certification receipt:
  `org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_CERTIFICATION_V1.json`
- Ledger SHA-256:
  `609f21baa23d08ec278a5e9deb5c921e21b5e0395ccc0a05ccc485db0efbc5c0`
- Verified checkpoint directories: 40/40.
- Verified files: 400/400.
- Verified bytes: 588,580,226,924.

This establishes byte-level identity of the checkpoint material used by the
campaign. It does not establish that those checkpoints have the causal roles
required for a standard unlearning-efficacy paper.

## 4. Hardware and numerical certification inherited by E1

The H200 runtime used Python 3.11.15, PyTorch 2.5.1+cu124, CUDA 12.4,
Transformers 4.46.3, and bitsandbytes 0.45.0. The adversarial eight-GPU canary
job 340505 completed with exit code 0 and certified:

- bundled results: 18;
- legacy comparisons: 12;
- repeat results: 6;
- failures: 0;
- production-128 GPTQ parity: PASS;
- 224-layer boundary parity: PASS;
- semantic certification: PASS.

Evidence paths:

- `results/org_h200_execution_v1/parity/h200_adversarial_canary_validation_v2_340505.json`
- `results/org_h200_execution_v1/parity/gptq_production128_parity_h200_v2_340505.json`
- `results/org_h200_execution_v1/parity/gptq_boundary224_h200_v2_340505.json`
- `results/org_h200_execution_v1/certification/job340505_sacct.txt`
- `logs/org_h200/org_h200_canary8_v2_340505.out`
- `logs/org_h200/org_h200_canary8_v2_340505.err`

The earlier shared-GPU optimization was accepted only after exact parity at the
CPU transfer boundary: 224/224 Hessians and 21/21 sampled GPTQ tensors matched
the legacy path exactly. Earlier GPU-resident candidate paths that produced
quantized-tensor differences were rejected and are not scientific authority.

## 5. E1 execution accounting

### Accepted stress wave

- Jobs: 340622 and 340623.
- State: both `COMPLETED`, exit code `0:0`.
- Tasks: eight per job, 16 cells total.
- Result files: 192/192.
- Receipts: 16/16.
- Validation:
  `results/org_h200_execution_v1/parity/e1_stress16_validation_v3_jobs340622_340623.json`

### Accepted remaining wave

- Jobs: 340655, 340656, 340657, 340658, 340659, 340660, 340661.
- State: all `COMPLETED`, exit code `0:0`.
- Tasks: eight per job, 56 cells total.
- Result files: 672/672.
- Receipts: 56/56.
- Validation:
  `results/org_h200_execution_v1/parity/e1_remaining56_validation_v3_jobs340655_340661.json`

### Complete accepted inventory

- Cells: 72/72.
- Results: 864/864.
- Receipts: 72/72.
- Frozen files: 936 unique files.
- Output checksum manifest:
  `results/org_h200_execution_v1/certification/e1_core72_v3_outputs.sha256`
- Slurm accounting:
  `results/org_h200_execution_v1/certification/e1_core72_v3_jobs_sacct.txt`

The nine accepted V3 jobs accumulated 4,764 seconds of one-node wall time. At
eight allocated H200 GPUs per node, that is approximately 10.5867 allocated
GPU-hours for the accepted E1 production waves. This is reconstructed from the
reported elapsed times; the cluster's billing ledger remains authoritative.

Jobs 340568, 340595, and 340596 failed with exit 137 because Slurm's default
step wait behavior terminated valid longer-running peers after early tasks
finished. Their partial V1/V2 roots are quarantined. A scheduler canary proved
`srun --wait=0`, and only the clean V3 root above is accepted.

## 6. E1 estimands

For cell `c` and selector `s`, harmful recovery is `H(c,s)`. Mitigation benefit
is `B(c,s) = H(c,full_quantized) - H(c,s)`. Let `R(c)` be the mean benefit of
the eight matched-random selectors. The native-ranking contrast is
`D(c) = B(c,GQDP_signed_v42) - R(c)`.

Retain cost is the protected arm's `delta_retain_ce` minus the same cell's full
quantization `delta_retain_ce`. The non-inferiority margin is
`max(0.05, 0.01 * abs(BF16 retain answer_ce))`. Safety slack is margin minus
retain cost; non-negative values are safe by this frozen rule.

## 7. Scientific gate result

Machine-readable output:
`results/org_h200_execution_v1/analysis/e1_scientific_gate_v1.json`

Final status: `STOP`; `e2_authorized=false` under the original preregistered
E1-to-E2 hierarchy.

| Gate | Result | Key evidence |
|---|---:|---|
| Recovery signal | PASS | mean 2.71755; 95% CI [0.86130, 5.16871]; one-sided p=0.00059994; 95% checkpoint-positive |
| Native GQDP primary | PASS | mean D=0.728936; 95% CI [0.209745, 1.520522]; one-sided p=0.00209979 |
| Native GQDP robustness | PASS | median D=0.067377; 92.5% positive; both datasets positive; all leave-one-model-tag-out means non-negative |
| Retain utility | FAIL | mean slack=-0.354146; 95% CI [-1.24243, 0.319168]; 62.5% checkpoint-positive |
| Integrity | PASS | all required source, count, finite-value, receipt, and execution checks passed |

Dataset utility summaries:

- MUSE: mean slack -0.8891; 95% CI [-2.5825, 0.3483]; median 0.0442;
  65% checkpoint-positive.
- TOFU: mean slack 0.18082; 95% CI [-0.24526, 0.62792]; median 0.0493;
  60% checkpoint-positive.

Leave-one-model-tag-out native D means were all positive:

- Llama-2 7B omitted: 0.9034.
- Llama-2 MUSE omitted: 0.2502.
- Llama-3 8B omitted: 0.9061.
- Llama-3 MUSE omitted: 0.8364.

Because the retain-utility gate failed, the original pre-E1 E2 contract is
permanently blocked. No later analysis may relabel this E1 decision as PASS.

## 8. Post-gate trade-off audit

Machine-readable outputs:

- `results/org_h200_execution_v1/analysis/e1_tradeoff_audit_v1.json`
- `results/org_h200_execution_v1/analysis/e1_tradeoff_cells_v1.tsv`

Audit status: PASS. It reproduced the gate's checkpoint slacks exactly and
reported 72 cells, 40 checkpoints, 20 cell-level negative slacks, and 15
checkpoint-level negative slacks. The cell-level fraction with both positive D
and non-negative safety slack was 0.625. The Pearson correlation between native
D and safety slack was -0.845098, showing a strong benefit-utility trade-off.

### Algorithm decomposition

| Algorithm | Mean D | Mean slack | Joint positive and safe | Interpretation |
|---|---:|---:|---:|---|
| GA | 0.2652 | 1.0023 | 0.7619 | Useful and usually safe, but heterogeneous |
| GDR | 2.6651 | -2.8425 | 0.3125 | Largest apparent benefit and largest utility damage |
| KLR | 0.00407 | 0.04706 | 0.9048 | Safe but almost no incremental advantage over random |
| NPO | 0.53205 | -0.56934 | 0.3571 | Moderate benefit with frequent utility damage |

For GDR and NPO, native-minus-random retain cost was positive in every
observed cell, with means 2.0539 and 0.98467 respectively. This indicates the
trade-off is specific to the GQDP intervention, not merely the cost of
protecting any matched set of eight matrices.

### Quantizer decomposition

| Quantizer | Mean D | Mean slack | Joint positive and safe | Safe fraction |
|---|---:|---:|---:|---:|
| GPTQ int4 | 0.2796 | 0.0736 | 0.5000 | 0.6250 |
| NF4 | 1.9933 | -1.8802 | 0.4583 | 0.5417 |
| RTN int8 | 0.04978 | 0.49768 | 0.9167 | 1.0000 |

MUSE-NF4 was the harshest group: mean D 3.8215 and mean slack -3.6728. The
worst observed slack was -21.176. RTN int8 was safe in every observed cell but
had a small average ranking advantage. TOFU GPTQ and NF4 had mean slacks 0.1011
and -0.0876; TOFU RTN was uniformly safe.

The worst checkpoint slacks included Llama-2-MUSE GDR seed 5000 (-11.92096),
Llama-2-MUSE GDR seed 2024 (-9.67670), Llama-2-7B NPO seed 2024 (-2.1933),
Llama-3-MUSE GA seed 2024 (-1.17525), and Llama-2-7B NPO seed 5000 (-0.97586).

## 9. Defensible interpretation

E1 supplies strong evidence that harmful recovery exists in this frozen panel
and that the native GQDP ranking at k=8 reduces it more than matched random
protection on average. It also supplies strong evidence that the effect is not
universally safe: the highest apparent gains are entangled with retain-utility
damage, especially for GDR/NPO and NF4/MUSE combinations.

The strongest defensible paper story is therefore a heterogeneous Pareto
frontier in post-unlearning quantization—not a universal mitigation, not a new
unlearning algorithm, and not proof that standard baselines are unnecessary.
E2A is designed to test whether lower budgets move that frontier into a jointly
safe and effective region.

## 10. Status of subsequent stages

- Original E2: blocked forever by the E1 retain-utility failure; preserved as a
  preregistered negative continuation decision.
- E2A: explicitly post-E1 adaptive follow-up, authorized only as a new study.
- E3: not authorized. Its design must be reconsidered after E2A; it cannot be
  inherited automatically from the blocked E2 hierarchy.

## 11. Required preservation

Never overwrite or merge the accepted E1 V3 root, the E1 locks, the 936-file
checksum manifest, or the scientific-gate/trade-off outputs. Preserve failed
and partial roots as failure evidence, clearly labelled and excluded. Any
paper table must be reproducible from the machine-readable E1 gate or trade-off
TSV and must retain the checkpoint-cluster inference unit.
