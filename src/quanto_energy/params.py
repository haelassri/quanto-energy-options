"""Reference parameter set used in the examples.

Cross-asset correlations are the New York estimates reported by Benth, Lange &
Myklebust (2015) for NYMEX natural gas vs CME HDD futures: rho_W = 0.2843,
rho_B = 0.1817. The volatility levels are ILLUSTRATIVE, chosen to match the
shapes they describe (energy futures volatility starting near 50%, HDD futures
volatility above 100% at the short end and falling quickly), not their exact
estimates. Re-calibrate with your own futures data before drawing conclusions.
"""

from .volatility import CrossCorrelation, FactorVol

ENERGY_VOL = FactorVol(sigma_long=0.25, sigma_short=0.40, gamma=1.5, rho_ls=-0.3)
TEMPERATURE_VOL = FactorVol(sigma_long=0.05, sigma_short=1.00, gamma=6.0, rho_ls=0.0)
CROSS_BLM_NY = CrossCorrelation(rho_w=0.2843, rho_b=0.1817)

# Futures levels from BLM's 31 Dec 2010 case study (Dec-2011 contracts).
F_ENERGY = 5.092  # USD/MMBtu
F_HDD = 805.0  # HDD index points
RATE = 0.02
