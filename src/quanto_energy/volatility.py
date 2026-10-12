"""Two-factor futures volatility model of Benth, Lange & Myklebust (2015).

For each asset i in {E (energy), I (temperature index)} the futures price on a
delivery period ending at tau2 follows, under Q,

    dF_i(t) / F_i(t) = sigma_i dW_i(t) + sigma'_i exp(-gamma_i (tau2 - t)) dB_i(t)

- a long-term factor W_i with constant volatility sigma_i, and
- a short-term factor B_i whose volatility grows as delivery approaches
  (Samuelson effect), at speed gamma_i.

log F_i(T) - log F_i(t) is Gaussian, so a quanto option is priced from just
three numbers: the two log-return variances sigma_X^2, sigma_Y^2 and their
correlation rho_XY. This module computes them in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorVol:
    """Volatility parameters for one futures curve (annualised)."""

    sigma_long: float  # sigma_i, long-term factor volatility
    sigma_short: float  # sigma'_i, short-term factor volatility at delivery
    gamma: float  # gamma_i, speed at which the short-term volatility decays
    rho_ls: float = 0.0  # corr(W_i, B_i), long vs short factor

    def instantaneous_vol(self, t, tau2):
        """Total instantaneous volatility of the futures at time t."""
        s = self.sigma_short * np.exp(-self.gamma * (tau2 - np.asarray(t, dtype=float)))
        var = self.sigma_long**2 + s**2 + 2 * self.rho_ls * self.sigma_long * s
        return np.sqrt(var)


@dataclass(frozen=True)
class CrossCorrelation:
    """Correlations between the energy (E) and temperature (I) factors."""

    rho_w: float = 0.0  # corr(W_E, W_I)  long-term vs long-term
    rho_b: float = 0.0  # corr(B_E, B_I)  short-term vs short-term
    rho_wb: float = 0.0  # corr(W_E, B_I)
    rho_bw: float = 0.0  # corr(B_E, W_I)


def correlation_matrix(vol_e: FactorVol, vol_i: FactorVol, cross: CrossCorrelation) -> np.ndarray:
    """4x4 correlation matrix of (W_E, B_E, W_I, B_I)."""
    c = np.array(
        [
            [1.0, vol_e.rho_ls, cross.rho_w, cross.rho_wb],
            [vol_e.rho_ls, 1.0, cross.rho_bw, cross.rho_b],
            [cross.rho_w, cross.rho_bw, 1.0, vol_i.rho_ls],
            [cross.rho_wb, cross.rho_b, vol_i.rho_ls, 1.0],
        ]
    )
    if np.linalg.eigvalsh(c).min() < -1e-12:
        raise ValueError("factor correlations do not form a valid (PSD) correlation matrix")
    return c


def _int_exp(g, t, T, tau2):
    """Integral of exp(-g (tau2 - s)) ds over s in [t, T]."""
    g = np.asarray(g, dtype=float)
    t = np.asarray(t, dtype=float)
    small = np.abs(g) < 1e-10
    g_safe = np.where(small, 1.0, g)
    exact = (np.exp(-g_safe * (tau2 - T)) - np.exp(-g_safe * (tau2 - t))) / g_safe
    return np.where(small, T - t, exact)


def log_moments(vol_e: FactorVol, vol_i: FactorVol, cross: CrossCorrelation, t, T, tau2):
    """Return (sigma_X, sigma_Y, rho_XY) of the log futures returns over [t, T].

    t can be an array (used by the hedging simulation). T <= tau2 is the option
    expiry; tau2 is the end of the delivery period that sets the Samuelson decay.
    """
    correlation_matrix(vol_e, vol_i, cross)  # validates the inputs
    if np.any(np.asarray(t) > T) or T > tau2 + 1e-12:
        raise ValueError("need t <= T <= tau2")
    dt = T - np.asarray(t, dtype=float)
    se, sse, ge = vol_e.sigma_long, vol_e.sigma_short, vol_e.gamma
    si, ssi, gi = vol_i.sigma_long, vol_i.sigma_short, vol_i.gamma

    var_x = se**2 * dt + sse**2 * _int_exp(2 * ge, t, T, tau2) + 2 * vol_e.rho_ls * se * sse * _int_exp(ge, t, T, tau2)
    var_y = si**2 * dt + ssi**2 * _int_exp(2 * gi, t, T, tau2) + 2 * vol_i.rho_ls * si * ssi * _int_exp(gi, t, T, tau2)
    cov = (
        cross.rho_w * se * si * dt
        + cross.rho_wb * se * ssi * _int_exp(gi, t, T, tau2)
        + cross.rho_bw * sse * si * _int_exp(ge, t, T, tau2)
        + cross.rho_b * sse * ssi * _int_exp(ge + gi, t, T, tau2)
    )
    sx, sy = np.sqrt(var_x), np.sqrt(var_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(sx * sy > 0, cov / (sx * sy), 0.0)
    return sx, sy, np.clip(rho, -1.0, 1.0)
