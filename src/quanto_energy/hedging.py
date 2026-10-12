"""Delta-hedging simulation for energy-temperature quanto options.

Futures paths are simulated exactly under the two-factor model of
`volatility.py`: over each step [t_k, t_{k+1}] the pair of log returns is
Gaussian with the closed-form moments, so there is no discretisation bias.

A writer sells the option at the model price and rebalances positions in both
futures (or only the energy futures, when temperature futures are too illiquid
to trade, a limitation BLM stress). Futures cost nothing to enter; their daily
gains are credited to a cash account earning r. The terminal hedging error is

    PnL = cash_T - payoff,

which is ~0 for a frequently rebalanced hedge with the right correlation, and
widens when the hedger ignores correlation or drops the temperature leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pricing import quanto_greeks, quanto_price
from .volatility import CrossCorrelation, FactorVol, log_moments

STRATEGIES = ("correlated", "naive", "energy_only")


@dataclass
class HedgeResult:
    strategy: str
    pnl: np.ndarray  # terminal hedging error per path (cash minus payoff)
    premium: float  # price received at t = 0 under the true model

    def summary(self) -> dict:
        q = np.quantile(self.pnl, [0.05, 0.5, 0.95])
        return {
            "strategy": self.strategy,
            "mean": float(self.pnl.mean()),
            "std": float(self.pnl.std(ddof=1)),
            "q05": float(q[0]),
            "median": float(q[1]),
            "q95": float(q[2]),
            "std_pct_premium": float(100 * self.pnl.std(ddof=1) / self.premium),
        }


def simulate_futures(F_E0, F_I0, vol_e: FactorVol, vol_i: FactorVol, cross: CrossCorrelation,
                     T, tau2, n_steps, n_paths, seed=None):
    """Exact simulation of (F_E, F_I) on a uniform grid of n_steps over [0, T].

    Returns times (n_steps + 1,) and two arrays of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, T, n_steps + 1)
    log_e = np.empty((n_paths, n_steps + 1))
    log_i = np.empty((n_paths, n_steps + 1))
    log_e[:, 0], log_i[:, 0] = np.log(F_E0), np.log(F_I0)
    for k in range(n_steps):
        sx, sy, rho = log_moments(vol_e, vol_i, cross, times[k], times[k + 1], tau2)
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        log_e[:, k + 1] = log_e[:, k] - 0.5 * sx**2 + sx * z1
        log_i[:, k + 1] = log_i[:, k] - 0.5 * sy**2 + sy * z2
    return times, np.exp(log_e), np.exp(log_i)


def delta_hedge(F_E, F_I, times, K_E, K_I, vol_e: FactorVol, vol_i: FactorVol, cross: CrossCorrelation,
                tau2, r=0.0, strategy="correlated", kind_e="call", kind_i="call") -> HedgeResult:
    """Run a discrete delta hedge along simulated paths and return the hedging error."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    T = times[-1]
    hedge_cross = CrossCorrelation() if strategy == "naive" else cross

    sx0, sy0, rho0 = log_moments(vol_e, vol_i, cross, 0.0, T, tau2)
    premium = float(quanto_price(F_E[0, 0], F_I[0, 0], K_E, K_I, sx0, sy0, rho0, np.exp(-r * T), kind_e, kind_i))

    cash = np.full(F_E.shape[0], premium)
    for k in range(len(times) - 1):
        t, dt = times[k], times[k + 1] - times[k]
        sx, sy, rho = log_moments(vol_e, vol_i, hedge_cross, t, T, tau2)
        g = quanto_greeks(F_E[:, k], F_I[:, k], K_E, K_I, sx, sy, rho, np.exp(-r * (T - t)), kind_e, kind_i)
        # Tail the futures position so gains, once reinvested at r, match the option delta.
        h_e = g["delta_e"] * np.exp(r * dt)
        h_i = 0.0 if strategy == "energy_only" else g["delta_i"] * np.exp(r * dt)
        cash = cash * np.exp(r * dt) + h_e * (F_E[:, k + 1] - F_E[:, k]) + h_i * (F_I[:, k + 1] - F_I[:, k])

    pe = 1.0 if kind_e == "call" else -1.0
    pi = 1.0 if kind_i == "call" else -1.0
    payoff = np.maximum(pe * (F_E[:, -1] - K_E), 0.0) * np.maximum(pi * (F_I[:, -1] - K_I), 0.0)
    return HedgeResult(strategy, cash - payoff, premium)
