# ORG E2B B0.5 Historical Behavior Execution Protocol V1

Date frozen: 2026-09-06 (Asia/Taipei)  
Platform: H200 only  
Purpose: certify whether the historical dense Q/DQ signal transports to genuine packed, serialized, freshly reloaded inference.

## Scope and claim boundary

B0.5 is a behavior canary, not the Atlas and not an unlearning-efficacy study. It uses two already existing historical unlearned checkpoints solely to test implementation transport and to measure runtime. Passing B0.5 may authorize construction of the outcome-blind Atlas lock. It does not authorize a paper claim by itself.

## Frozen checkpoint cells

| Canary | Dataset | Model | Algorithm | Seed | Checkpoint |
|---|---|---|---|---|---|
| ordinary | TOFU | `llama2_7b_adamw` | GA | 42 | `checkpoints/llama2_7b_adamw/unlearned/GA/seed_42` |
| adverse | MUSE-News | `llama2_muse_adamw` | GDR | 2024 | `checkpoints/llama2_muse_adamw/unlearned/GDR/seed_2024` |

Each checkpoint is crossed with `nf4`, `gptq_int4`, and `rtn_int8`, producing six independent feature cells and later six independent behavior cells.

## Mandatory two-pass execution

Pass F is outcome-blind feature generation. It recomputes forget and retain mean gradients, the quantizer displacement, signed forget-QDP, and signed retain-QDP for every eligible matrix. It validates the historical forget-QDP reference and freezes the safe-looking singleton, risky-looking singleton, pair, full, and abstention protected sets. No protected-condition outcome is evaluated in Pass F.

Pass P is allowed only after all six Pass-F receipts are certified into one immutable selector lock. Pass P evaluates exactly six conditions per checkpoint/quantizer cell:

1. BF16 unlearned checkpoint;
2. fully packed and freshly reloaded checkpoint;
3. safe-looking singleton from the selector lock;
4. risky-looking singleton from the selector lock;
5. frozen two-matrix set from the selector lock;
6. abstention, an explicit alias of the fully packed artifact and metrics.

The condition runner must preserve genuine packed storage, exact high-precision protected modules, new-process reload, and all paper-facing outcomes must be computed after reload. Dense Q/DQ results are parity evidence only.

## Frozen selector definitions

For a matrix with displacement `epsilon = Q(U) - U`:

- predicted forget benefit of protection is `-g_forget dot epsilon`;
- predicted retain cost of protection is `-g_retain dot epsilon`;
- the safe-looking singleton maximizes positive predicted forget benefit among matrices with non-positive predicted retain cost;
- if that set is empty, the safe-looking condition must abstain and the receipt records why;
- the risky-looking singleton maximizes predicted retain cost, with predicted forget benefit as the deterministic tie-breaker;
- the pair is the two highest positive-benefit safe matrices when available, otherwise the two highest positive-benefit matrices;
- all ties are resolved lexically by exact parameter FQN;
- every non-empty set must protect no more than two matrices and no more than 2% of target-parameter mass.

The labels are prospective predictions, not statements about observed safety or harm.

## Feature acceptance

- exact B0.4 dual-certification prerequisite with `B05_AUTHORIZED=true`;
- exact certified environment and checkpoint-ledger hashes;
- exactly 224 eligible weight matrices for each frozen checkpoint;
- exact forget/retain/calibration manifest identities and sample counts;
- signed-QDP algebra unit tests pass;
- real retain-QDP independent recomputation differs by at most `1e-10` per matrix;
- full-sample forget-QDP agrees with the frozen v4.2 reference within `1e-10` absolute error per matrix;
- no protected-condition outcome is loaded or generated;
- all six immutable feature receipts pass before selector certification.

## Later behavior acceptance

- six conditions per cell with abstention byte-identical to the full-packed condition;
- producer exit and two independent fresh reloads;
- protected module names exact, BF16, and source-tensor exact;
- all other target modules genuinely packed;
- no network, remote code, fallback, CPU/disk offload, NaN, or Inf;
- per-sample pre-save/reload drift no greater than `max(1e-6, 5 * repeatability_noise)` under a tolerance frozen before outcomes;
- at least 90% packed-versus-emulated sign agreement for non-negligible forget and retain contrasts;
- measured total B0.5 allocation no greater than 3 H200 GPU-hours.

## Stop rules

Any feature mismatch, selector ambiguity, packed fallback, source drift, or budget violation stops B0.5. It is recorded as a scientific/engineering result; thresholds or selected matrices are not changed after inspecting protected outcomes.

