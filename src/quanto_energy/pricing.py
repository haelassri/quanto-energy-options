"""Closed-form and Monte Carlo pricing of energy-temperature quanto options.

Setting (Benth, Lange & Myklebust, 2015): at expiry T the energy futures F_E and
the temperature-index futures F_I are jointly lognormal,

    F_E(T) = F_E exp(-sigma_X^2 / 2 + X),   F_I(T) = F_I exp(-sigma_Y^2 / 2 + Y),
    (X, Y) ~ N(0, [[sigma_X^2, rho sigma_X sigma_Y], [., sigma_Y^2]]).

The option pays the product of two vanilla legs,

    payoff = (phi_E (F_E(T) - K_E))^+ * (phi_I (F_I(T) - K_I))^+ ,   phi = +1 call, -1 put,

so a producer can be paid only when, e.g., it is both cold (high HDD) and power
is expensive. Expanding the product of indicators and changing measure on each
term gives (derivation in docs/derivation.md)

    C = DF phi_E phi_I [ F_E F_I e^{rho sX sY} M(phi_E(y1 + sX + rho sY), phi_I(y2 + sY + rho sX))
                         - F_E K_I M(phi_E(y1 + sX),        phi_I(y2 + rho sX))
                         - F_I K_E M(phi_E(y1 + rho sY),     phi_I(y2 + sY))
                         + K_E K_I M(phi_E y1,               phi_I y2) ]

with y1 = (ln(F_E/K_E) - sX^2/2)/sX, y2 = (ln(F_I/K_I) - sY^2/2)/sY and every M
evaluated at correlation phi_E phi_I rho.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .bivariate import bivariate_normal_cdf as _M

_KIND = {"call": 1.0, "put": -1.0}


def _phi(kind: str) -> float:
    try:
        return _KIND[kind]
    except KeyError as exc:
        raise ValueError("kind must be 'call' or 'put'") from exc


def _terms(F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho, kind_e, kind_i):
    pe, pi = _phi(kind_e), _phi(kind_i)
    F_E, F_I, K_E, K_I, sx, sy, rho = (np.asarray(a, dtype=float) for a in (F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho))
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("sigma_x and sigma_y must be positive (use the payoff directly at expiry)")
    y1 = (np.log(F_E / K_E) - 0.5 * sx**2) / sx
    y2 = (np.log(F_I / K_I) - 0.5 * sy**2) / sy
    r = pe * pi * rho
    m_ei = _M(pe * (y1 + sx + rho * sy), pi * (y2 + sy + rho * sx), r)
    m_e = _M(pe * (y1 + sx), pi * (y2 + rho * sx), r)
    m_i = _M(pe * (y1 + rho * sy), pi * (y2 + sy), r)
    m_0 = _M(pe * y1, pi * y2, r)
    growth = np.exp(rho * sx * sy)
    return pe, pi, F_E, F_I, K_E, K_I, growth, m_ei, m_e, m_i, m_0


def quanto_price(F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho, discount=1.0, kind_e="call", kind_i="call"):
    """Closed-form price of (phi_E(F_E(T)-K_E))^+ (phi_I(F_I(T)-K_I))^+.

    sigma_x, sigma_y are total (not annualised) log-return standard deviations to
    expiry; rho is their correlation; discount = exp(-r (T - t)).
    """
    pe, pi, F_E, F_I, K_E, K_I, g, m_ei, m_e, m_i, m_0 = _terms(F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho, kind_e, kind_i)
    value = F_E * F_I * g * m_ei - F_E * K_I * m_e - F_I * K_E * m_i + K_E * K_I * m_0
    return discount * pe * pi * value


def quanto_greeks(F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho, discount=1.0, kind_e="call", kind_i="call"):
    """Deltas w.r.t. each futures price and the cross-gamma, in closed form.

    Pathwise differentiation of the payoff gives
        dC/dF_E      = DF phi_E phi_I [F_I e^{rho sX sY} M_EI - K_I M_E]
        dC/dF_I      = DF phi_E phi_I [F_E e^{rho sX sY} M_EI - K_E M_I]
        d2C/dF_E dF_I = DF phi_E phi_I e^{rho sX sY} M_EI
    """
    pe, pi, F_E, F_I, K_E, K_I, g, m_ei, m_e, m_i, _ = _terms(F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho, kind_e, kind_i)
    s = discount * pe * pi
    return {
        "delta_e": s * (F_I * g * m_ei - K_I * m_e),
        "delta_i": s * (F_E * g * m_ei - K_E * m_i),
        "cross_gamma": s * g * m_ei,
    }


def black76(F, K, sigma, discount=1.0, kind="call"):
    """Black (1976) price of a call/put on a futures; sigma is the total std dev."""
    phi = _phi(kind)
    F, K, sigma = (np.asarray(a, dtype=float) for a in (F, K, sigma))
    d1 = (np.log(F / K) + 0.5 * sigma**2) / sigma
    d2 = d1 - sigma
    return discount * phi * (F * ndtr(phi * d1) - K * ndtr(phi * d2))


def independent_price(F_E, F_I, K_E, K_I, sigma_x, sigma_y, discount=1.0, kind_e="call", kind_i="call"):
    """Benchmark that ignores correlation: DF x product of two undiscounted Black prices."""
    return discount * black76(F_E, K_E, sigma_x, 1.0, kind_e) * black76(F_I, K_I, sigma_y, 1.0, kind_i)


def tailored_quanto_price(F_E, F_I, strikes, sigma_x, sigma_y, rho, discount=1.0, volume=1.0):
    """BLM eq. (2.5): volume x [call-call(K_E, K_I) + put-put(K'_E, K'_I)].

    Pays when both legs move together in either direction, e.g. a utility hedging
    joint price-and-demand risk. strikes = (K_E, K_I, K_E_put, K_I_put).
    """
    K_E, K_I, K_Ep, K_Ip = strikes
    cc = quanto_price(F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho, discount, "call", "call")
    pp = quanto_price(F_E, F_I, K_Ep, K_Ip, sigma_x, sigma_y, rho, discount, "put", "put")
    return volume * (cc + pp)


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    def ci95(self):
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def quanto_price_mc(F_E, F_I, K_E, K_I, sigma_x, sigma_y, rho, discount=1.0,
                    kind_e="call", kind_i="call", n_paths=1_000_000, seed=None) -> MCResult:
    """Monte Carlo price with antithetic variates (independent check of the closed form)."""
    pe, pi = _phi(kind_e), _phi(kind_i)
    rng = np.random.default_rng(seed)
    half = max(n_paths // 2, 1)
    z1 = rng.standard_normal(half)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(half)

    def payoff(a, b):
        fe = F_E * np.exp(-0.5 * sigma_x**2 + sigma_x * a)
        fi = F_I * np.exp(-0.5 * sigma_y**2 + sigma_y * b)
        return np.maximum(pe * (fe - K_E), 0.0) * np.maximum(pi * (fi - K_I), 0.0)

    pairs = 0.5 * (payoff(z1, z2) + payoff(-z1, -z2))
    return MCResult(discount * pairs.mean(), discount * pairs.std(ddof=1) / np.sqrt(half), 2 * half)
