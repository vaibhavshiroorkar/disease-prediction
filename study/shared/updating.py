"""Ways to update a trained model's probabilities with a few local patients.

Every function takes source probabilities, so none of them needs to know what
kind of model made them.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

SHIFT_SD = 2.5
SLOPE_SD = 1.0
EPS = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def fit_shift_slope(p: np.ndarray, y: np.ndarray, learn_slope: bool) -> tuple[float, float]:
    """Penalized maximum likelihood for `shift + slope * logit(p)`.

    The priors keep the answer finite when the sample holds one class, which is
    common for dengue, where about 1 case in 40 is positive.
    """
    z = to_logit(p)
    y = np.asarray(y, dtype=float)

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        shift = theta[0]
        slope = theta[1] if learn_slope else 1.0
        eta = shift + slope * z
        value = np.logaddexp(0.0, eta).sum() - (y * eta).sum() + shift**2 / (2 * SHIFT_SD**2)
        residual = expit(eta) - y
        gradient = [residual.sum() + shift / SHIFT_SD**2]
        if learn_slope:
            value += (slope - 1) ** 2 / (2 * SLOPE_SD**2)
            gradient.append((residual * z).sum() + (slope - 1) / SLOPE_SD**2)
        return value, np.array(gradient)

    start = np.array([0.0, 1.0]) if learn_slope else np.array([0.0])
    result = minimize(loss, start, jac=True, method="BFGS")
    if not np.isfinite(result.x).all() or np.abs(result.jac).max() > 1e-2:
        raise RuntimeError(f"recalibration did not converge: {result.message}")
    return float(result.x[0]), float(result.x[1]) if learn_slope else 1.0


def apply_shift_slope(p: np.ndarray, shift: float, slope: float) -> np.ndarray:
    return expit(shift + slope * to_logit(p))


def local_rate(y: np.ndarray) -> float:
    """Smoothed so a sample with no cases does not predict an impossible 0%."""
    return (float(np.sum(y)) + 0.5) / (len(y) + 1)


def reweight(p: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    return expit(to_logit(p) + logit(target_rate) - logit(source_rate))


def em_rate(p: np.ndarray, source_rate: float, tol: float = 1e-8, max_steps: int = 100_000) -> float:
    """The target rate that best explains unlabelled predictions (Saerens 2002).

    Assumes only the rate changed: a sick patient looks the same everywhere.
    The rate stays off 0 and 1, where the update could not recover.
    """
    rate = source_rate
    for _ in range(max_steps):
        new = float(np.clip(reweight(p, source_rate, rate).mean(), EPS, 1 - EPS))
        if abs(new - rate) < tol:
            return new
        rate = new
    raise RuntimeError("EM did not converge")
