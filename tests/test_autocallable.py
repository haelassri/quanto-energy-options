import numpy as np
import pytest
from scipy.special import ndtr

from quanto_energy import Autocallable, Market, autocallable_greeks, autocallable_price_mc, black76

T, R, Q, VOL = 1.0, 0.03, 0.01, 0.25
DF = np.exp(-R * T)


def _within(mc, ref, n_se=4):
    assert abs(mc.price - ref) < n_se * mc.std_error + 1e-12


def _single(**kw):
    return Market(spots=[100.0], vols=[VOL], rate=R, dividends=Q, **kw)


def _d2(F, K, s):
    return (np.log(F / K) - 0.5 * s**2) / s


def test_european_knock_in_matches_closed_form():
    # no coupon, never called: the note is a zero-coupon bond minus a European down-and-in put
    B, K = 0.7, 1.0
    product = Autocallable([T], coupon=0.0, autocall_barrier=np.inf, knock_in_barrier=B, put_strike=K)
    F, s = np.exp((R - Q) * T), VOL * np.sqrt(T)
    ref = DF - (black76(F, B, s, DF, "put") + (K - B) * DF * ndtr(-_d2(F, B, s))) / K
    _within(autocallable_price_mc(product, _single(), n_paths=400_000, seed=1), ref)


@pytest.mark.parametrize("fx_vol,fx_corr", [(0.0, 0.0), (0.12, -0.5), (0.12, 0.6)])
def test_digital_coupon_with_quanto_drift(fx_vol, fx_corr):
    cb, c = 0.9, 0.08
    product = Autocallable([T], coupon=c, coupon_barrier=cb, knock_in_barrier=0.0)
    F = np.exp((R - Q - fx_corr * VOL * fx_vol) * T)
    ref = DF * (1 + c * ndtr(_d2(F, cb, VOL * np.sqrt(T))))
    _within(autocallable_price_mc(product, _single(fx_vols=fx_vol, fx_correlations=fx_corr), n_paths=400_000, seed=2), ref)


def test_certain_first_call():
    product = Autocallable([0.5, 1.0], coupon=0.03, autocall_barrier=0.0, coupon_barrier=0.0)
    res = autocallable_price_mc(product, _single(), n_paths=10_000, seed=0)
    assert res.price == pytest.approx(np.exp(-R * 0.5) * 1.03)
    assert res.call_probabilities[0] == pytest.approx(1.0)
    assert res.expected_life == pytest.approx(0.5)


def test_non_call_periods_and_step_down_schedule():
    times = np.arange(1, 5) / 4
    product = Autocallable(times, coupon=0.01, autocall_barrier=[0.0, 0.0, 0.0, 0.0], coupon_barrier=0.0, non_call_periods=2)
    res = autocallable_price_mc(product, _single(), n_paths=10_000, seed=0)
    assert res.call_probabilities[:3] == pytest.approx([0.0, 0.0, 1.0])
    assert res.price == pytest.approx(sum(np.exp(-R * t) * 0.01 for t in times[:3]) + np.exp(-R * times[2]))


def test_memory_coupon_is_worth_more_and_matches_when_all_coupons_pay():
    times = np.arange(1, 9) / 4
    market = Market(spots=[100.0, 100.0], vols=[0.3, 0.35], correlation=0.5, rate=R)
    kw = dict(coupon=0.02, autocall_barrier=1.05, coupon_barrier=0.8, knock_in_barrier=0.6)
    with_memory = autocallable_price_mc(Autocallable(times, memory=True, **kw), market, n_paths=50_000, seed=3)
    without = autocallable_price_mc(Autocallable(times, memory=False, **kw), market, n_paths=50_000, seed=3)
    assert with_memory.price > without.price
    kw["coupon_barrier"] = 0.0
    a = autocallable_price_mc(Autocallable(times, memory=True, **kw), market, n_paths=50_000, seed=3)
    b = autocallable_price_mc(Autocallable(times, memory=False, **kw), market, n_paths=50_000, seed=3)
    assert a.price == pytest.approx(b.price, rel=1e-12)


def test_knock_in_monitoring_order_and_worst_of_effect():
    times = np.arange(1, 13) / 4
    market = Market(spots=[100.0, 60.0], vols=[0.25, 0.3], correlation=0.5, rate=R, dividends=Q)
    prices = {m: autocallable_price_mc(Autocallable(times, knock_in_monitoring=m), market, n_paths=100_000, seed=4)
              for m in ("european", "discrete", "daily")}
    assert prices["daily"].price < prices["discrete"].price < prices["european"].price
    assert prices["daily"].knock_in_probability > prices["european"].knock_in_probability
    single = autocallable_price_mc(Autocallable(times), Market(spots=[100.0], vols=[0.25], rate=R, dividends=Q),
                                   n_paths=100_000, seed=4)
    assert prices["european"].price < single.price  # adding an underlying to a worst-of cheapens the note


def test_seasoned_note_and_greeks():
    times = np.arange(1, 9) / 4
    market = Market(spots=[100.0, 80.0], vols=[0.25, 0.3], correlation=0.4, rate=R)
    product = Autocallable(times)
    fresh = autocallable_price_mc(product, market, n_paths=20_000, seed=5).price
    fixed = Autocallable(times, initial_fixings=(100.0, 80.0))
    assert autocallable_price_mc(fixed, market, n_paths=20_000, seed=5).price == pytest.approx(fresh, rel=1e-12)
    # an underlying that has already fallen 20% makes the note cheaper
    fallen = Market(spots=[80.0, 80.0], vols=[0.25, 0.3], correlation=0.4, rate=R)
    assert autocallable_price_mc(fixed, fallen, n_paths=20_000, seed=5).price < fresh - 0.02

    g = autocallable_greeks(product, market, n_paths=50_000)
    assert np.all(g["delta"] > 0)  # long the underlyings through the short put and the autocall
    assert np.all(g["vega"] < 0)  # short volatility through the down-and-in put


def test_validation():
    with pytest.raises(ValueError):
        Autocallable([1.0, 0.5])
    with pytest.raises(ValueError):
        Autocallable([0.5, 1.0], autocall_barrier=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        Autocallable([1.0], knock_in_monitoring="continuous")
    with pytest.raises(ValueError):
        Market(spots=[1.0, 1.0], vols=[0.2, 0.2], correlation=[[1.0, 1.2], [1.2, 1.0]])
