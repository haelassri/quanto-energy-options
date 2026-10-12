"""Calibrate the spot models on real French data and measure temperature-power dependence.

    export ENTSOE_API_KEY=...            # free key from transparency.entsoe.eu
    pip install -e ".[data,plots]"
    python examples/calibrate_real_data.py --location paris --zone france --start 2019-01-01 --end 2024-12-31

Steps
1. Download daily mean temperature (Open-Meteo) and daily average day-ahead
   prices (ENTSO-E); both are cached under data/raw/.
2. Fit seasonal + AR(1) models to temperature, HDD and power.
3. Report the residual correlations overall and by month. In heating-driven
   markets like France, cold shocks raise demand and prices, so power residuals
   are expected to correlate negatively with temperature and positively with
   HDD, most strongly in winter. That HDD-power correlation is the empirical
   counterpart of rho in the quanto price.
"""

from __future__ import annotations

import argparse

from quanto_energy import degree_days, fit_seasonal_ou, residual_correlation
from quanto_energy.data import fetch_day_ahead_prices, fetch_temperature


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location", default="paris")
    p.add_argument("--zone", default="france")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--base", type=float, default=18.0, help="HDD base temperature (deg C)")
    args = p.parse_args()

    temp = fetch_temperature(args.location, args.start, args.end)
    price = fetch_day_ahead_prices(args.zone, args.start, args.end)
    hdd = degree_days(temp, base=args.base)

    fit_t = fit_seasonal_ou(temp, n_harmonics=2, name="temperature")
    fit_h = fit_seasonal_ou(hdd, n_harmonics=2, name="HDD")
    fit_p = fit_seasonal_ou(price, n_harmonics=2, weekday=True, name="day-ahead price")

    for f in (fit_t, fit_h, fit_p):
        print(f.summary())

    print(f"\ncorr(temperature, price) residuals: {residual_correlation(fit_t, fit_p):+.3f}")
    print(f"corr(HDD, price) residuals:         {residual_correlation(fit_h, fit_p):+.3f}")
    print("\nBy month, corr(HDD, price):")
    print(residual_correlation(fit_h, fit_p, by_month=True).round(3).to_string())
    print("\nNote: the 2021-2022 energy crisis dominates price volatility; rerun with --start 2023-01-01 to compare.")


if __name__ == "__main__":
    main()
