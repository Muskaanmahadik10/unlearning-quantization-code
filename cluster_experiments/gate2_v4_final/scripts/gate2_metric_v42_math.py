#!/usr/bin/env python3
"""Numerical contracts shared by Gate 2 v4.2 metric producers.

This module is intentionally small and CPU-testable.  It separates the
paper-facing metric definitions from model loading, Slurm, and checkpoint IO.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import torch

from gate2_metric_v42_contracts import (
    GAUSS_LEGENDRE_CONTRACT,
    GQDP_CONTRACT,
    METRIC_SUITE_SCHEMA,
    UPQI_ABS_CONTRACT,
    UPQI_SIGNED_CONTRACT,
    UPQI_STAR_SCHEMA,
)


def gauss_legendre_unit_interval(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1]."""
    if int(order) < 2:
        raise ValueError("Gauss-Legendre order must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite quadrature rule")
    if abs(float(weights.sum()) - 1.0) > 1e-14:
        raise ValueError("Unit-interval quadrature weights do not sum to one")
    return nodes.astype(np.float64), weights.astype(np.float64)


def integrate_gradient_samples(
    gradients: Iterable[torch.Tensor], weights: Iterable[float]
) -> torch.Tensor:
    grads = list(gradients)
    coeffs = [float(value) for value in weights]
    if not grads or len(grads) != len(coeffs):
        raise ValueError("Gradient and quadrature-weight counts must match")
    shape = grads[0].shape
    if any(grad.shape != shape for grad in grads):
        raise ValueError("All quadrature gradients must have the same shape")
    integrated = torch.zeros_like(grads[0], dtype=torch.float64, device="cpu")
    for grad, coeff in zip(grads, coeffs):
        integrated.add_(grad.detach().to(device="cpu", dtype=torch.float64), alpha=coeff)
    if not torch.isfinite(integrated).all():
        raise ValueError("Integrated gradient is non-finite")
    return integrated


def upqi_scores(integrated_gradient: torch.Tensor, displacement: torch.Tensor) -> dict[str, float]:
    grad = integrated_gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    eps = displacement.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if grad.numel() != eps.numel() or grad.numel() == 0:
        raise ValueError("Integrated gradient and displacement must be non-empty and aligned")
    signed = float(torch.dot(grad, eps).item())
    magnitude = float(torch.sum(torch.abs(grad) * torch.abs(eps)).item())
    if not math.isfinite(signed) or not math.isfinite(magnitude):
        raise ValueError("UPQI score is non-finite")
    if magnitude + 1e-12 < abs(signed):
        raise ValueError("UPQI magnitude must upper-bound the signed score")
    return {"upqi_signed": signed, "upqi_abs": magnitude}


def endpoint_metrics(weight: torch.Tensor, gradient: torch.Tensor, displacement: torch.Tensor) -> dict[str, float]:
    w = weight.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    grad = gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    eps = displacement.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not (w.numel() == grad.numel() == eps.numel()) or w.numel() == 0:
        raise ValueError("Weight, gradient, and displacement must be non-empty and aligned")
    g_norm = float(torch.linalg.vector_norm(grad).item())
    e_norm = float(torch.linalg.vector_norm(eps).item())
    w_norm = float(torch.linalg.vector_norm(w).item())
    dot = float(torch.dot(grad, eps).item())
    denom = g_norm * e_norm
    mse = float(torch.mean(eps.square()).item())
    signal = float(torch.sum(w.square()).item())
    noise = float(torch.sum(eps.square()).item())
    sqnr = float(10.0 * math.log10(signal / max(noise, 1e-300)))
    values = {
        "gqdp_signed": dot,
        "gqdp_abs": abs(dot),
        "gqdp_cosine": dot / denom if denom > 1e-30 else 0.0,
        "gqdp_per_param": dot / float(w.numel()),
        "magnitude_product": float(torch.sum(torch.abs(grad) * torch.abs(eps)).item()),
        "grad_l1": float(torch.sum(torch.abs(grad)).item()),
        "grad_l2": g_norm,
        "grad_mean_abs": float(torch.mean(torch.abs(grad)).item()),
        "epsilon_l1": float(torch.sum(torch.abs(eps)).item()),
        "epsilon_l2": e_norm,
        "epsilon_mean_abs": float(torch.mean(torch.abs(eps)).item()),
        "epsilon_max": float(torch.max(torch.abs(eps)).item()),
        "weight_l1": float(torch.sum(torch.abs(w)).item()),
        "weight_l2": w_norm,
        "weight_mean_abs": float(torch.mean(torch.abs(w)).item()),
        "quantization_mse": mse,
        "sqnr_db": sqnr,
        "n_parameters": int(w.numel()),
    }
    if not all(math.isfinite(float(value)) for key, value in values.items() if key != "n_parameters"):
        raise ValueError("Endpoint metric is non-finite")
    return values
