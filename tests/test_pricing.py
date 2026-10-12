import numpy as np
import pytest
from scipy.stats import multivariate_normal

from quanto_energy import (
    bivariate_normal_cdf,
    black76,
    independent_price,
    quanto_greeks,
    quanto_price,
    quanto_price_mc,
    tailored_quanto_price,
)

BASE = dict(F_E=5.092, F_I=805.0, K_E=5.0, K_I=800.0, sigma_x=0.45, sigma_y=0.15, discount=np.exp(-0.02))
KINDS = [("call", "call"), ("put", "put"), ("call", "put"), ("put", "call")]


@pytest.mark.parametrize("seed", range(5))
def test_bivariate_cdf_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    h, k = rng.normal(0, 2, 2)
    rho = rng.uniform(-0.99, 0.99)
    ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, k])
    assert bivariate_normal_cdf(h, k, rho) == pytest.approx(ref, abs=1e-10)


def test_bivariate_cdf_edge_cases():
    assert bivariate_normal_cdf(0.0, 0.0, 0.0) == pytest.approx(0.25)
    assert bivariate_normal_cdf(0.0, 0.0, 0.5) == pytest.approx(0.25 + np.arcsin(0.5) / (2 * np.pi))
    assert bivariate_normal_cdf(1.0, 2.0, 1.0) == pytest.approx(0.8413447, abs=1e-6)
    assert bivariate_normal_cdf(0.5, -0.5, -1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kind_e,kind_i", KINDS)
@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.4, 0.9])
def test_closed_form_matches_monte_carlo(kind_e, kind_i, rho):
    cf = quanto_price(rho=rho, kind_e=kind_e, kind_i=kind_i, **BASE)
    mc = quanto_price_mc(rho=rho, kind_e=kind_e, kind_i=kind_i, n_paths=1_000_000, seed=42, **BASE)
    assert abs(cf - mc.price) < 4 * mc.std_error


@pytest.mark.parametrize("kind_e,kind_i", KINDS)
def test_zero_correlation_is_product_of_black_prices(kind_e, kind_i):
    args = {k: v for k, v in BASE.items()}
    cf = quanto_price(rho=0.0, kind_e=kind_e, kind_i=kind_i, **args)
    ind = independent_price(kind_e=kind_e, kind_i=kind_i, **args)
    assert cf == pytest.approx(ind, rel=1e-10)


def test_price_increases_with_correlation_for_call_call():
    prices = [quanto_price(rho=r, **BASE) for r in np.linspace(-0.9, 0.9, 19)]
    assert np.all(np.diff(prices) > 0)


def test_mixed_legs_decrease_with_correlation():
    prices = [quanto_price(rho=r, kind_e="call", kind_i="put", **BASE) for r in np.linspace(-0.9, 0.9, 19)]
    assert np.all(np.diff(prices) < 0)


def test_put_call_parity_on_the_temperature_leg():
    # (E-K)^+ (I-K')^+ - (E-K)^+ (K'-I)^+ = (E-K)^+ (I-K'); expectation known in closed form.
    rho = 0.35
    cc = quanto_price(rho=rho, kind_e="call", kind_i="call", **BASE)
    cp = quanto_price(rho=rho, kind_e="call", kind_i="put", **BASE)
    sx, sy, F_E, F_I, K_E, K_I, df = (BASE[k] for k in ("sigma_x", "sigma_y", "F_E", "F_I", "K_E", "K_I", "discount"))
    # E[(E-K_E)^+ I] = F_I * Black call on E under the measure tilted by I (forward shifted by rho sx sy).
    e_tilted = black76(F_E * np.exp(rho * sx * sy), K_E, sx, 1.0)
    expected = df * (F_I * e_tilted - K_I * black76(F_E, K_E, sx, 1.0))
    assert cc - cp == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("kind_e,kind_i", KINDS)
def test_greeks_match_finite_differences(kind_e, kind_i):
    rho, h = 0.3, 1e-4
    g = quanto_greeks(rho=rho, kind_e=kind_e, kind_i=kind_i, **BASE)

    def p(dE=0.0, dI=0.0):
        a = dict(BASE, F_E=BASE["F_E"] + dE, F_I=BASE["F_I"] + dI)
        return quanto_price(rho=rho, kind_e=kind_e, kind_i=kind_i, **a)

    hE, hI = h * BASE["F_E"], h * BASE["F_I"]
    assert g["delta_e"] == pytest.approx((p(hE) - p(-hE)) / (2 * hE), rel=1e-5)
    assert g["delta_i"] == pytest.approx((p(dI=hI) - p(dI=-hI)) / (2 * hI), rel=1e-5)
    cross = (p(hE, hI) - p(hE, -hI) - p(-hE, hI) + p(-hE, -hI)) / (4 * hE * hI)
    assert g["cross_gamma"] == pytest.approx(cross, rel=1e-4)


def test_tailored_contract_is_sum_of_legs():
    strikes = (5.0, 800.0, 4.5, 700.0)
    total = tailored_quanto_price(BASE["F_E"], BASE["F_I"], strikes, 0.45, 0.15, 0.3, BASE["discount"], volume=2.0)
    cc = quanto_price(rho=0.3, **BASE)
    pp = quanto_price(BASE["F_E"], BASE["F_I"], 4.5, 700.0, 0.45, 0.15, 0.3, BASE["discount"], "put", "put")
    assert total == pytest.approx(2.0 * (cc + pp))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        quanto_price(rho=0.1, kind_e="straddle", **BASE)
    with pytest.raises(ValueError):
        bivariate_normal_cdf(0, 0, 1.5)
