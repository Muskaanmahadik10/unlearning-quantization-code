# ORG H200 E1 analysis-method amendment v1

Frozen on 2026-09-05 before inspection of any H200 E1 scientific outcome.

This amendment makes the inference machinery in the E1 scientific contract
fully executable. The analysis unit is the checkpoint `(model_tag, algo,
seed)`. Quantizer-specific observations are averaged within checkpoint before
population inference.

For every scalar checkpoint contrast, report the arithmetic mean, median,
sample standard deviation, standardized mean, sign proportion, and an ordinary
checkpoint-cluster percentile bootstrap confidence interval from 10,000
resamples using seed 20260905. The primary one-sided p-value uses a centered
studentized checkpoint-cluster bootstrap: subtract the observed checkpoint
mean, resample centered checkpoint values, recompute the studentized statistic,
and use `(1 + count(t_star >= t_observed)) / 10001`. Degenerate zero-variance
cases pass only when every checkpoint contrast is strictly positive.

The recovery gate uses `-delta_forget_ce` from the full-quantized condition and
requires a strictly positive two-sided 95% bootstrap lower bound.

The primary native-GQDP gate uses the contract's D contrast and requires a
positive mean, a one-sided centered-bootstrap p-value below 0.025, and a
strictly positive two-sided 95% bootstrap lower bound. Dataset means must both
be positive, the overall median must be positive, and every
leave-one-model-tag-out mean must be non-negative.

For retain utility, define each native-GQDP cell's safety slack as its frozen
non-inferiority margin minus its retain cost relative to the same cell's
full-quantized condition. Aggregate and dataset-specific utility pass only when
the corresponding two-sided 95% checkpoint-bootstrap lower bound for mean
safety slack is non-negative.

No missing cell is imputed. E2 authorization additionally requires PASS
integrity reports for both the stress16 and remaining56 waves, 72 unique cells,
864 result files, 72 receipts, finite required values, and exact E1 source-lock
identity. Secondary endpoints cannot change the authorization decision.

