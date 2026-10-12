"""Spot models for daily temperature and power prices, and their joint calibration.

Both series are modelled the classical way (Benth & Saltyte-Benth for
temperature; Lucia & Schwartz for power):

    x_t = Lambda(t) + Z_t,   Z_t = phi Z_{t-1} + eps_t,

with a deterministic seasonal component

    Lambda(t) = a + b t + sum_k [alpha_k cos(2 pi k t / 365.25) + beta_k sin(2 pi k t / 365.25)]
                (+ day-of-week effects for power),

and an AR(1) (discretely sampled Ornstein-Uhlenbeck) deviation Z_t with
mean-reversion speed kappa = -ln(phi) per day. The correlation between the two
residual series eps_t is the empirical input that drives the quanto premium.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


def degree_days(temperature: pd.Series, base: float = 18.0, kind: str = "HDD") -> pd.Series:
    """Daily heating (HDD) or cooling (CDD) degree days from mean daily temperature (deg C)."""
    if kind == "HDD":
        return (base - temperature).clip(lower=0.0).rename("HDD")
    if kind == "CDD":
        return (temperature - base).clip(lower=0.0).rename("CDD")
    raise ValueError("kind must be 'HDD' or 'CDD'")


def degree_day_index(temperature: pd.Series, start, end, base: float = 18.0, kind: str = "HDD") -> float:
    """Cumulative HDD/CDD over [start, end], the underlying of CME-style temperature futures."""
    return float(degree_days(temperature.loc[start:end], base, kind).sum())


def _design(index: pd.DatetimeIndex, n_harmonics: int, weekday: bool) -> pd.DataFrame:
    t = (index - index[0]).days.to_numpy(dtype=float)
    cols = {"const": np.ones_like(t), "trend": t / DAYS_PER_YEAR}
    for k in range(1, n_harmonics + 1):
        w = 2 * np.pi * k * t / DAYS_PER_YEAR
        cols[f"cos{k}"], cols[f"sin{k}"] = np.cos(w), np.sin(w)
    if weekday:
        dow = index.dayofweek
        for d in range(1, 7):  # Monday is the baseline
            cols[f"dow{d}"] = (dow == d).astype(float)
    return pd.DataFrame(cols, index=index)


@dataclass
class SeasonalOUFit:
    name: str
    coefficients: pd.Series
    phi: float
    residuals: pd.Series  # AR(1) innovations eps_t
    deseasonalised: pd.Series  # Z_t
    fitted_season: pd.Series  # Lambda(t)
    transform: str = "level"
    n_harmonics: int = 1
    weekday: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def kappa_per_day(self) -> float:
        return float(-np.log(self.phi))

    @property
    def half_life_days(self) -> float:
        return float(np.log(2) / self.kappa_per_day)

    @property
    def sigma_daily(self) -> float:
        return float(self.residuals.std(ddof=1))

    def summary(self) -> dict:
        return {
            "series": self.name,
            "transform": self.transform,
            "phi": round(self.phi, 4),
            "half_life_days": round(self.half_life_days, 2),
            "sigma_daily": round(self.sigma_daily, 4),
            "seasonal_amplitude": round(float(np.hypot(self.coefficients.get("cos1", 0), self.coefficients.get("sin1", 0))), 3),
            "n_obs": int(len(self.residuals)),
        }

    def simulate(self, n_days: int, n_paths: int = 1, seed=None) -> np.ndarray:
        """Simulate future values: the fitted season plus Gaussian AR(1) deviations."""
        rng = np.random.default_rng(seed)
        last_index = self.fitted_season.index[-1]
        future = pd.date_range(last_index + pd.Timedelta(days=1), periods=n_days, freq="D")
        full = _design(self.fitted_season.index.append(future), self.n_harmonics, self.weekday)
        season = (full.loc[future] @ self.coefficients).to_numpy()
        z = np.empty((n_paths, n_days))
        prev = np.full(n_paths, float(self.deseasonalised.iloc[-1]))
        for d in range(n_days):
            prev = self.phi * prev + self.sigma_daily * rng.standard_normal(n_paths)
            z[:, d] = prev
        x = season + z
        return np.exp(x) if self.transform == "log" else x


def fit_seasonal_ou(series: pd.Series, n_harmonics: int = 1, weekday: bool = False,
                    transform: str = "level", name: str | None = None) -> SeasonalOUFit:
    """Fit Lambda(t) by OLS, then an AR(1) to the deseasonalised series.

    transform='log' fits log prices (drops non-positive values, which occur in
    European power markets); 'level' fits the raw series (default, safe with
    negative prices).
    """
    s = series.dropna().astype(float).sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    s = s.asfreq("D").interpolate(limit=3).dropna()
    meta = {}
    if transform == "log":
        n_bad = int((s <= 0).sum())
        if n_bad:
            meta["dropped_non_positive"] = n_bad
        s = np.log(s[s > 0])
    elif transform != "level":
        raise ValueError("transform must be 'level' or 'log'")

    X = _design(s.index, n_harmonics, weekday)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), s.to_numpy(), rcond=None)
    coef = pd.Series(beta, index=X.columns)
    season = X @ coef
    z = s - season

    z_lag = z.shift(1)
    mask = z_lag.notna() & z.index.to_series().diff().eq(pd.Timedelta(days=1))
    phi = float(np.dot(z_lag[mask], z[mask]) / np.dot(z_lag[mask], z_lag[mask]))
    if not 0 < phi < 1:
        raise ValueError(f"AR(1) coefficient {phi:.3f} is not mean-reverting; check the data")
    eps = (z - phi * z_lag)[mask]
    return SeasonalOUFit(name or str(series.name), coef, phi, eps.rename("eps"), z, season,
                         transform, n_harmonics, weekday, meta)


def residual_correlation(fit_a: SeasonalOUFit, fit_b: SeasonalOUFit, by_month: bool = False):
    """Correlation of the two AR(1) innovation series on their common dates.

    With by_month=True returns a Series of monthly correlations, to check whether
    the dependence is seasonal (e.g. strong in winter heating months only).
    """
    joined = pd.concat([fit_a.residuals.rename("a"), fit_b.residuals.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 30:
        raise ValueError("fewer than 30 overlapping observations")
    if not by_month:
        return float(joined["a"].corr(joined["b"]))
    return joined.groupby(joined.index.month).apply(lambda g: g["a"].corr(g["b"])).rename("correlation")
