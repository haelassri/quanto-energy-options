"""Vectorised standard bivariate normal CDF.

M(h, k; rho) = P(Z1 <= h, Z2 <= k) for standard normals with correlation rho.

Implemented with Owen's T function (scipy.special.owens_t), which is exact to
machine precision and vectorises cleanly, unlike scipy's multivariate_normal.cdf.
This matters for the hedging simulations, which evaluate the closed-form price
on every path at every rebalancing date.

Identity used (Owen, 1956):
    M(h, k; rho) = 1/2 Phi(h) + 1/2 Phi(k) - T(h, a_h) - T(k, a_k) - beta
    a_h = (k - rho h) / (h sqrt(1 - rho^2)),  a_k = (h - rho k) / (k sqrt(1 - rho^2))
    beta = 0 if h k > 0, or h k = 0 with h + k >= 0;  1/2 otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, owens_t

_EPS = 1e-12


def bivariate_normal_cdf(h, k, rho):
    """Standard bivariate normal CDF M(h, k; rho), broadcasting over all inputs."""
    h, k, rho = np.broadcast_arrays(
        np.asarray(h, dtype=float), np.asarray(k, dtype=float), np.asarray(rho, dtype=float)
    )
    if np.any(np.abs(rho) > 1):
        raise ValueError("correlation must lie in [-1, 1]")

    out = np.empty(h.shape, dtype=float)

    # Degenerate correlations: closed forms.
    pos = rho >= 1 - _EPS
    neg = rho <= -1 + _EPS
    out[pos] = ndtr(np.minimum(h[pos], k[pos]))
    out[neg] = np.maximum(ndtr(h[neg]) + ndtr(k[neg]) - 1.0, 0.0)

    mid = ~(pos | neg)
    if np.any(mid):
        hm, km, rm = h[mid], k[mid], rho[mid]
        # Nudge exact zeros so the Owen's T arguments stay finite; the CDF is
        # continuous in (h, k), so this changes the result by O(1e-15).
        hm = np.where(hm == 0.0, _EPS, hm)
        km = np.where(km == 0.0, _EPS, km)
        s = np.sqrt(1.0 - rm * rm)
        a_h = (km - rm * hm) / (hm * s)
        a_k = (hm - rm * km) / (km * s)
        beta = np.where(hm * km > 0, 0.0, 0.5)
        val = 0.5 * ndtr(hm) + 0.5 * ndtr(km) - owens_t(hm, a_h) - owens_t(km, a_k) - beta
        out[mid] = np.clip(val, 0.0, 1.0)

    return out[()] if out.ndim == 0 else out
