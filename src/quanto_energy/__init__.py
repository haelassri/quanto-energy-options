"""Pricing and hedging of energy-temperature quanto options.

Implements the framework of Benth, Lange & Myklebust (2015), "Pricing and
hedging quanto options in energy markets", Journal of Energy Markets 8(1).
"""

from .bivariate import bivariate_normal_cdf
from .hedging import HedgeResult, delta_hedge, simulate_futures
from .models import SeasonalOUFit, degree_day_index, degree_days, fit_seasonal_ou, residual_correlation
from .pricing import (
    MCResult,
    black76,
    independent_price,
    quanto_greeks,
    quanto_price,
    quanto_price_mc,
    tailored_quanto_price,
)
from .volatility import CrossCorrelation, FactorVol, correlation_matrix, log_moments

__all__ = [
    "bivariate_normal_cdf", "black76", "independent_price", "quanto_price", "quanto_greeks",
    "quanto_price_mc", "tailored_quanto_price", "MCResult", "FactorVol", "CrossCorrelation",
    "correlation_matrix", "log_moments", "simulate_futures", "delta_hedge", "HedgeResult",
    "fit_seasonal_ou", "SeasonalOUFit", "residual_correlation", "degree_days", "degree_day_index",
]
__version__ = "0.1.0"
