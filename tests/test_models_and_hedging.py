import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid

from quanto_energy import (
    CrossCorrelation,
    FactorVol,
    correlation_matrix,
    degree_day_index,
    degree_days,
    delta_hedge,
    fit_seasonal_ou,
    log_moments,
    residual_correlation,
    simulate_futures,
)
from quanto_energy.data import daily_average, parse_open_meteo
from quanto_energy.params import CROSS_BLM_NY, ENERGY_VOL, F_ENERGY, F_HDD, TEMPERATURE_VOL


# ---------------------------------------------------------------- volatility
def test_constant_vol_limit():
    v = FactorVol(sigma_long=0.3, sigma_short=0.0, gamma=1.0)
    sx, sy, rho = log_moments(v, v, CrossCorrelation(rho_w=0.5), 0.0, 2.0, 2.0)
    assert sx == pytest.approx(0.3 * np.sqrt(2.0))
    assert rho == pytest.approx(0.5)


def test_short_factor_integral_closed_form():
    v = FactorVol(sigma_long=0.0, sigma_short=0.8, gamma=3.0)
    sx, _, _ = log_moments(v, v, CrossCorrelation(), 0.0, 0.5, 1.0)
    s = np.linspace(0.0, 0.5, 200_001)
    numeric = trapezoid((0.8 * np.exp(-3.0 * (1.0 - s))) ** 2, s)
    assert sx**2 == pytest.approx(numeric, rel=1e-8)


def test_samuelson_effect_volatility_rises_into_delivery():
    v = ENERGY_VOL.instantaneous_vol(np.array([0.0, 0.5, 0.99]), tau2=1.0)
    assert np.all(np.diff(v) > 0)


def test_invalid_correlation_matrix_rejected():
    with pytest.raises(ValueError):
        correlation_matrix(FactorVol(0.2, 0.2, 1, 0.99), FactorVol(0.2, 0.2, 1, 0.99), CrossCorrelation(rho_w=-0.99, rho_b=0.99))


# ---------------------------------------------------------------- hedging
def test_simulated_futures_are_martingales():
    T = 0.25
    _, FE, FI = simulate_futures(F_ENERGY, F_HDD, ENERGY_VOL, TEMPERATURE_VOL, CROSS_BLM_NY, T, T, 10, 200_000, seed=0)
    assert FE[:, -1].mean() == pytest.approx(F_ENERGY, rel=5e-3)
    assert FI[:, -1].mean() == pytest.approx(F_HDD, rel=5e-3)


def test_hedge_error_shrinks_with_rebalancing_and_correlation_matters():
    T = 1 / 12
    stds = {}
    for n in (21, 168):
        times, FE, FI = simulate_futures(F_ENERGY, F_HDD, ENERGY_VOL, TEMPERATURE_VOL, CROSS_BLM_NY, T, T, n, 4000, seed=1)
        for s in ("correlated", "naive", "energy_only"):
            res = delta_hedge(FE, FI, times, F_ENERGY, F_HDD, ENERGY_VOL, TEMPERATURE_VOL, CROSS_BLM_NY, T, 0.02, s)
            stds[(s, n)] = res.pnl.std()
    assert stds[("correlated", 168)] < 0.5 * stds[("correlated", 21)]  # ~1/sqrt(n) convergence
    assert stds[("correlated", 168)] < stds[("naive", 168)] < stds[("energy_only", 168)]


# ---------------------------------------------------------------- spot models
def _synthetic(seed=0, rho=0.4, n=4 * 365):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    t = np.arange(n)
    z = rng.standard_normal((n, 2))
    e1, e2 = z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    for i in range(1, n):
        x1[i] = 0.75 * x1[i - 1] + 2.0 * e1[i]
        x2[i] = 0.60 * x2[i - 1] + 8.0 * e2[i]
    temp = 12 - 8 * np.cos(2 * np.pi * t / 365.25) + x1
    price = 60 + 10 * np.cos(2 * np.pi * t / 365.25) + 5 * (idx.dayofweek >= 5) * -1 + x2
    return pd.Series(temp, idx, name="temperature"), pd.Series(price, idx, name="price")


def test_seasonal_ou_recovers_parameters_and_correlation():
    temp, price = _synthetic()
    ft = fit_seasonal_ou(temp, n_harmonics=1)
    fp = fit_seasonal_ou(price, n_harmonics=1, weekday=True)
    assert ft.phi == pytest.approx(0.75, abs=0.05)
    assert fp.phi == pytest.approx(0.60, abs=0.05)
    assert ft.coefficients["cos1"] == pytest.approx(-8.0, abs=0.6)
    assert residual_correlation(ft, fp) == pytest.approx(0.4, abs=0.06)
    assert len(residual_correlation(ft, fp, by_month=True)) == 12
    assert ft.simulate(30, n_paths=5, seed=0).shape == (5, 30)


def test_log_transform_drops_negative_prices():
    _, price = _synthetic()
    price.iloc[10:15] = -5.0
    fit = fit_seasonal_ou(price.clip(lower=-5) + 0.0, transform="log", weekday=True)
    assert fit.meta["dropped_non_positive"] == 5


def test_degree_days():
    t = pd.Series([10.0, 18.0, 25.0], index=pd.date_range("2024-01-01", periods=3))
    assert degree_days(t).tolist() == [8.0, 0.0, 0.0]
    assert degree_days(t, kind="CDD").tolist() == [0.0, 0.0, 7.0]
    assert degree_day_index(t, "2024-01-01", "2024-01-03") == 8.0


# ---------------------------------------------------------------- data parsing (offline)
def test_parse_open_meteo_payload():
    payload = {"daily": {"time": ["2024-01-01", "2024-01-02"], "temperature_2m_mean": [3.1, None]}}
    s = parse_open_meteo(payload)
    assert s.iloc[0] == pytest.approx(3.1) and np.isnan(s.iloc[1])


def test_daily_average_from_hourly():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="Europe/Brussels")
    s = pd.Series(np.r_[np.full(24, 50.0), np.full(24, 70.0)], index=idx)
    out = daily_average(s)
    assert out.tolist() == [50.0, 70.0]
