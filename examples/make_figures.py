"""Reproduce the figures in the README.

    python examples/make_figures.py          # writes figures/*.png and prints the tables

Uses the reference parameters in quanto_energy.params (BLM 2015 correlations,
illustrative volatilities). Runs in about a minute.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quanto_energy import delta_hedge, independent_price, log_moments, quanto_price, simulate_futures
from quanto_energy.params import CROSS_BLM_NY, ENERGY_VOL, F_ENERGY, F_HDD, RATE, TEMPERATURE_VOL

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical palette (3 slots, light surface) and text tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "legend.frameon": False,
})


def correlation_uplift():
    """How much the correlation adds to the price, vs pricing the two legs as independent."""
    moneyness = np.linspace(0.85, 1.15, 31)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = []
    for T, label, colour in [(1 / 12, "1-month option", BLUE), (1.0, "1-year option", ORANGE)]:
        sx, sy, rho = log_moments(ENERGY_VOL, TEMPERATURE_VOL, CROSS_BLM_NY, 0.0, T, T)
        df = np.exp(-RATE * T)
        K_I = moneyness * F_HDD
        corr = quanto_price(F_ENERGY, F_HDD, F_ENERGY, K_I, sx, sy, rho, df)
        ind = independent_price(F_ENERGY, F_HDD, F_ENERGY, K_I, sx, sy, df)
        uplift = 100 * (corr / ind - 1)
        ax.plot(100 * moneyness, uplift, color=colour, label=label)
        ax.annotate(label, (100 * moneyness[-1], uplift[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2)
        rows.append((label, rho, uplift[0], uplift[15], uplift[-1]))
    ax.set_xlabel("HDD strike, % of HDD futures (energy strike at the money)")
    ax.set_ylabel("Price uplift vs. independent legs (%)")
    ax.set_title("Ignoring correlation underprices the quanto by 26-48%")
    ax.set_xlim(85, 122)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_uplift.png", dpi=160)
    plt.close(fig)
    print("\nCorrelation uplift (call-call, ATM energy strike)")
    print(f"{'maturity':<16}{'rho_XY':>8}{'K_I 85%':>10}{'ATM':>8}{'K_I 115%':>10}")
    for label, rho, a, b, c in rows:
        print(f"{label:<16}{rho:>8.3f}{a:>9.1f}%{b:>7.1f}%{c:>9.1f}%")


def hedge_convergence(n_paths=10_000):
    """Std of the terminal hedging error vs number of rebalances, for three hedgers."""
    T = 1 / 12
    steps = [5, 10, 21, 42, 84, 168, 336]
    strategies = [("correlated", "Delta hedge, true correlation", BLUE),
                  ("naive", "Delta hedge, correlation ignored", ORANGE),
                  ("energy_only", "Energy futures only (no HDD leg)", AQUA)]
    stds = {s: [] for s, _, _ in strategies}
    for n in steps:
        times, FE, FI = simulate_futures(F_ENERGY, F_HDD, ENERGY_VOL, TEMPERATURE_VOL, CROSS_BLM_NY, T, T, n, n_paths, seed=7)
        for s, _, _ in strategies:
            res = delta_hedge(FE, FI, times, F_ENERGY, F_HDD, ENERGY_VOL, TEMPERATURE_VOL, CROSS_BLM_NY, T, RATE, s)
            stds[s].append(res.summary()["std_pct_premium"])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for s, label, colour in strategies:
        ax.plot(steps, stds[s], color=colour, marker="o", markersize=5, label=label)
    ax.set_xscale("log")
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.minorticks_off()
    ax.set_xlabel("Rebalances over the option's life (1 month; 21 = daily)")
    ax.set_ylabel("Std of hedging error (% of premium)")
    ax.set_title("Hedge error: correlation matters, and the HDD leg is essential")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "hedge_convergence.png", dpi=160)
    plt.close(fig)

    print("\nHedging error std (% of premium), 1-month ATM call-call")
    print(f"{'rebalances':<12}" + "".join(f"{s:>14}" for s, _, _ in strategies))
    for i, n in enumerate(steps):
        print(f"{n:<12}" + "".join(f"{stds[s][i]:>13.1f}%" for s, _, _ in strategies))


if __name__ == "__main__":
    correlation_uplift()
    hedge_convergence()
    print(f"\nFigures written to {OUT}")
