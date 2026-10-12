"""Monte Carlo pricing of (worst-of) autocallable notes, with optional quanto drift.

Product. On each observation date t_k the worst performance across underlyings,

    W_k = min_i S_i(t_k) / S_i(0),

is compared with three barriers:

* coupon barrier:    W_k >= CB_k pays a coupon c (with memory, also every coupon
                     missed since the last one paid);
* autocall barrier:  W_k >= AB_k (from the first callable date, not at maturity)
                     redeems the note early at 100% of notional;
* knock-in barrier:  if W ever falls below KI (monitored at maturity only, on the
                     observation dates, or daily) and W_T < K at maturity, the
                     note redeems at W_T / K instead of 100%: the investor is short
                     a down-and-in put on the worst performer.

Model. Underlyings follow correlated geometric Brownian motions under the
domestic (payout-currency) risk-neutral measure,

    dS_i / S_i = (r_i - q_i - rho_{i,FX} sigma_i sigma_{FX,i}) dt + sigma_i dW_i,

where r_i is the rate of the asset's own currency and the last term is the quanto
adjustment for an underlying quoted in a foreign currency but paid in the domestic
one (sigma_{FX,i} = 0 for a domestic asset). For a futures underlying set
q_i = r_i so the drift is zero. Cash flows are discounted at the domestic rate r.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

_MONITORING = ("european", "discrete", "daily")


@dataclass(frozen=True)
class Autocallable:
    """Term sheet. Barriers and strike are fractions of the initial fixing.

    Scalars apply to every observation date; sequences give one value per date
    (e.g. a step-down autocall schedule). ``coupon`` is paid per observation
    period, not per annum. Times are in years from today; for a note that is
    already live, pass the remaining dates and its ``initial_fixings``.
    """

    observation_times: Sequence[float]
    coupon: float = 0.02
    autocall_barrier: float | Sequence[float] = 1.0
    coupon_barrier: float | Sequence[float] = 0.7
    knock_in_barrier: float = 0.6
    knock_in_monitoring: str = "european"
    put_strike: float = 1.0
    memory: bool = True
    non_call_periods: int = 0
    notional: float = 1.0
    initial_fixings: Sequence[float] | None = None  # None: fixed today at the current spots

    def __post_init__(self):
        t = np.asarray(self.observation_times, dtype=float)
        if t.ndim != 1 or t.size == 0 or t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("observation_times must be positive and strictly increasing")
        if self.knock_in_monitoring not in _MONITORING:
            raise ValueError(f"knock_in_monitoring must be one of {_MONITORING}")
        for name in ("autocall_barrier", "coupon_barrier"):
            b = np.asarray(getattr(self, name), dtype=float)
            if b.ndim > 1 or (b.ndim == 1 and b.size != t.size):
                raise ValueError(f"{name} must be a scalar or have one value per observation date")
        if self.put_strike <= 0:
            raise ValueError("put_strike must be positive")

    @property
    def maturity(self) -> float:
        return float(self.observation_times[-1])

    def schedule(self, name: str) -> np.ndarray:
        return np.broadcast_to(np.asarray(getattr(self, name), dtype=float), (len(self.observation_times),))


@dataclass(frozen=True)
class Market:
    """Market data for n underlyings; scalars are broadcast to every asset.

    ``rate`` is the domestic (payout-currency) rate used for discounting;
    ``asset_rates`` defaults to it. ``fx_vols`` / ``fx_correlations`` are the
    volatility of each asset's FX rate (domestic per foreign) and its correlation
    with the asset, giving the quanto drift adjustment -rho sigma sigma_FX.
    """

    spots: Sequence[float]
    vols: Sequence[float]
    correlation: Sequence[Sequence[float]] | float = 0.0
    rate: float = 0.0
    dividends: Sequence[float] | float = 0.0
    asset_rates: Sequence[float] | float | None = None
    fx_vols: Sequence[float] | float = 0.0
    fx_correlations: Sequence[float] | float = 0.0

    def __post_init__(self):
        n = self.n_assets
        corr = self.corr_matrix
        if corr.shape != (n, n) or not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation must be a symmetric n x n matrix with unit diagonal")
        if np.any(np.linalg.eigvalsh(corr) <= 0):
            raise ValueError("correlation matrix is not positive definite")
        if np.any(self.spot_array <= 0) or np.any(self.vol_array <= 0):
            raise ValueError("spots and vols must be positive")

    def _vec(self, x) -> np.ndarray:
        return np.broadcast_to(np.asarray(x, dtype=float), (self.n_assets,))

    @property
    def n_assets(self) -> int:
        return np.atleast_1d(self.spots).size

    @property
    def spot_array(self) -> np.ndarray:
        return self._vec(self.spots)

    @property
    def vol_array(self) -> np.ndarray:
        return self._vec(self.vols)

    @property
    def corr_matrix(self) -> np.ndarray:
        corr = np.asarray(self.correlation, dtype=float)
        if corr.ndim == 0:
            corr = np.full((self.n_assets, self.n_assets), float(corr))
            np.fill_diagonal(corr, 1.0)
        return corr

    @property
    def drift(self) -> np.ndarray:
        """Risk-neutral log-drift before the -sigma^2/2 term, including the quanto adjustment."""
        r_i = self.rate if self.asset_rates is None else self.asset_rates
        quanto = self._vec(self.fx_correlations) * self.vol_array * self._vec(self.fx_vols)
        return self._vec(r_i) - self._vec(self.dividends) - quanto


@dataclass
class AutocallResult:
    price: float
    std_error: float
    n_paths: int
    call_probabilities: np.ndarray  # P(redeemed early at each observation date); last entry is 0
    knock_in_probability: float
    loss_probability: float  # P(redemption below 100% at maturity)
    expected_life: float  # years

    @property
    def conf_interval(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def _time_grid(product: Autocallable, steps_per_year: int) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(product.observation_times, dtype=float)
    if product.knock_in_monitoring != "daily":
        return obs, np.arange(obs.size)
    n = max(1, int(np.ceil(product.maturity * steps_per_year)))
    daily = np.linspace(0.0, product.maturity, n + 1)[1:]
    daily = daily[np.min(np.abs(daily[:, None] - obs[None, :]), axis=1) > 1e-9]
    grid = np.sort(np.concatenate([obs, daily]))
    return grid, np.searchsorted(grid, obs)


def simulate_worst_of(product: Autocallable, market: Market, n_paths: int, seed=None,
                      steps_per_year: int = 252, antithetic: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Worst performance on each observation date and its minimum over the knock-in window.

    Log-prices are stepped exactly (no discretisation bias at the grid points), one
    time step at a time so memory is O(n_paths x n_obs). Returns
    ``(worst_obs, worst_min)`` with shapes (n_paths, n_obs) and (n_paths,).
    """
    if antithetic and n_paths % 2:
        n_paths += 1
    rng = np.random.default_rng(seed)
    grid, obs_idx = _time_grid(product, steps_per_year)
    dts = np.diff(grid, prepend=0.0)
    vol, chol, drift = market.vol_array, np.linalg.cholesky(market.corr_matrix), market.drift
    n_draw = n_paths // 2 if antithetic else n_paths
    fixings = market.spot_array if product.initial_fixings is None else np.asarray(product.initial_fixings, dtype=float)
    if fixings.shape != (market.n_assets,):
        raise ValueError("initial_fixings must have one value per underlying")

    log_perf = np.tile(np.log(market.spot_array / fixings), (n_paths, 1))
    worst_obs = np.empty((n_paths, obs_idx.size))
    worst_min = np.full(n_paths, np.inf)
    is_obs = np.zeros(grid.size, dtype=bool)
    is_obs[obs_idx] = True
    k = 0
    for j, dt in enumerate(dts):
        z = rng.standard_normal((n_draw, market.n_assets)) @ chol.T
        if antithetic:
            z = np.concatenate([z, -z])
        log_perf += (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
        worst = np.exp(log_perf.min(axis=1))
        if product.knock_in_monitoring == "daily" or (product.knock_in_monitoring == "discrete" and is_obs[j]):
            np.minimum(worst_min, worst, out=worst_min)
        if is_obs[j]:
            worst_obs[:, k] = worst
            k += 1
    if product.knock_in_monitoring == "european":
        worst_min = worst_obs[:, -1].copy()
    return worst_obs, worst_min


def autocallable_cashflows(product: Autocallable, worst_obs: np.ndarray, worst_min: np.ndarray, rate: float = 0.0):
    """Discounted payoff per path given simulated worst-of performances.

    Returns ``(pv, call_index, knocked_in)`` where call_index is the observation
    index at which each path terminated (the last index if it ran to maturity).
    """
    times = np.asarray(product.observation_times, dtype=float)
    acb, cb = product.schedule("autocall_barrier"), product.schedule("coupon_barrier")
    n_paths, n_obs = worst_obs.shape
    alive = np.ones(n_paths, dtype=bool)
    accrued = np.zeros(n_paths)  # coupon periods not yet paid (memory feature)
    pv = np.zeros(n_paths)
    call_index = np.full(n_paths, n_obs - 1)
    knocked_in = worst_min < product.knock_in_barrier

    for k, t in enumerate(times):
        df = np.exp(-rate * t)
        w = worst_obs[:, k]
        accrued += 1.0
        pays = alive & (w >= cb[k])
        pv += df * product.coupon * np.where(pays, accrued if product.memory else 1.0, 0.0)
        accrued[pays] = 0.0
        if k < n_obs - 1:
            if k >= product.non_call_periods:
                called = alive & (w >= acb[k])
                pv += df * called
                call_index[called] = k
                alive &= ~called
        else:
            loss = knocked_in & (w < product.put_strike)
            pv += df * np.where(alive, np.where(loss, w / product.put_strike, 1.0), 0.0)
    return product.notional * pv, call_index, knocked_in


def autocallable_price_mc(product: Autocallable, market: Market, n_paths: int = 200_000, seed=None,
                          steps_per_year: int = 252, antithetic: bool = True) -> AutocallResult:
    """Monte Carlo price of an autocallable note (per ``product.notional``)."""
    worst_obs, worst_min = simulate_worst_of(product, market, n_paths, seed, steps_per_year, antithetic)
    pv, call_index, knocked_in = autocallable_cashflows(product, worst_obs, worst_min, market.rate)
    n = pv.size
    # antithetic pairs are (i, i + n/2): average them so the standard error uses independent samples
    samples = 0.5 * (pv[: n // 2] + pv[n // 2:]) if antithetic else pv
    times = np.asarray(product.observation_times, dtype=float)
    n_obs = times.size
    matured = call_index == n_obs - 1
    call_probs = np.bincount(call_index[~matured], minlength=n_obs) / n
    loss = matured & knocked_in & (worst_obs[:, -1] < product.put_strike)
    return AutocallResult(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / np.sqrt(samples.size)),
        n_paths=n,
        call_probabilities=call_probs,
        knock_in_probability=float((matured & knocked_in).mean()),
        loss_probability=float(loss.mean()),
        expected_life=float(times[call_index].mean()),
    )


def autocallable_greeks(product: Autocallable, market: Market, spot_bump: float = 0.01, vol_bump: float = 0.01,
                        n_paths: int = 200_000, seed: int = 0, steps_per_year: int = 252) -> dict:
    """Price, per-asset delta (dV/dS_i) and vega (dV/dsigma_i, per unit of vol) by central bumps.

    The initial fixings are frozen at their current values so a spot bump moves
    the performance. Every revaluation reuses the same seed (common random numbers).
    The payoff jumps at the barriers, so Greeks close to a barrier are noisy:
    widen the bumps or add paths if needed.
    """
    if product.initial_fixings is None:
        product = replace(product, initial_fixings=tuple(market.spot_array))

    def price(**changes):
        return autocallable_price_mc(product, replace(market, **changes), n_paths, seed, steps_per_year).price

    spots, vols = market.spot_array, market.vol_array
    delta, vega = np.empty(spots.size), np.empty(vols.size)
    for i in range(spots.size):
        e = np.eye(spots.size)[i]
        h = spot_bump * spots[i]
        delta[i] = (price(spots=spots + h * e) - price(spots=spots - h * e)) / (2 * h)
        vega[i] = (price(vols=vols + vol_bump * e) - price(vols=vols - vol_bump * e)) / (2 * vol_bump)
    return {"price": price(), "delta": delta, "vega": vega}
