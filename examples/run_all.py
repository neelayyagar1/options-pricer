"""Regenerate every figure and results table in the README.

Run from the repo root:  python examples/run_all.py

Inputs are an assumed market, not a live quote. Everything is seeded, so the output
is the same on every run.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pricer import black_scholes as bs
from pricer.american import price_american_lsm
from pricer.analytic import geometric_asian_price
from pricer.binomial import binomial_price
from pricer.convergence import measure_convergence
from pricer.greeks import fd_greeks, pathwise_greeks
from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams
from pricer.monte_carlo import simulate_european, simulate_price
from pricer.paths import RandomSource
from pricer.payoffs import AsianKind, AsianPayoff, BarrierPayoff, BarrierStyle
from pricer.variance_reduction import simulate_control_variate

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
MARKET = MarketParams(spot=100.0, volatility=0.20, rate=0.04, dividend_yield=0.015)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
EXACT = float(bs.price_contract(CALL, MARKET))

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
tables: list[str] = []


def convergence() -> None:
    counts = [2**k for k in range(10, 18)]

    def pseudo(n: int, seq: np.random.SeedSequence) -> float:
        return simulate_european(MARKET, CALL, n_paths=n, seed=seq).price

    def sobol(n: int, seq: np.random.SeedSequence) -> float:
        return simulate_european(MARKET, CALL, n_paths=n, seed=seq, source=RandomSource.SOBOL).price

    study_pr = measure_convergence(pseudo, EXACT, counts, replications=40)
    study_qmc = measure_convergence(sobol, EXACT, counts, replications=40)

    big = [2**k for k in range(8, 21, 2)]
    runs = [simulate_european(MARKET, CALL, n_paths=n, seed=7, antithetic=True) for n in big]
    price = np.array([r.price for r in runs])
    half = np.array([1.96 * (r.standard_error or 0.0) for r in runs])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.errorbar(big, price, yerr=half, fmt="o-", capsize=3, label="Monte Carlo (95% CI)")
    a.axhline(EXACT, color="k", ls="--", label=f"Black-Scholes {EXACT:.4f}")
    a.set_xscale("log", base=2)
    a.set(xlabel="paths", ylabel="price", title="Price and confidence interval")
    a.legend()
    for study, name in ((study_pr, "pseudo-random"), (study_qmc, "Sobol")):
        b.loglog(study.path_counts, study.rmse, "o-", label=f"{name}, slope {study.slope:.2f}")
    b.set(xlabel="paths", ylabel="RMSE vs Black-Scholes", title="Error against path count")
    b.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.png")
    plt.close(fig)
    tables.append(
        "### Convergence (ATM call, 40 replications per N)\n\n"
        "| Source | Fitted slope |\n|---|---|\n"
        f"| Pseudo-random | {study_pr.slope:.3f} |\n| Sobol (1-D) | {study_qmc.slope:.3f} |\n"
    )


def variance_reduction() -> None:
    n = 400_000
    steps = 50
    arith = AsianPayoff(100.0, OptionType.CALL, AsianKind.ARITHMETIC)
    geo = AsianPayoff(100.0, OptionType.CALL, AsianKind.GEOMETRIC)
    geo_price = geometric_asian_price(100.0, 100.0, 1.0, 0.04, 0.20, steps, OptionType.CALL, 0.015)

    plain_asian = simulate_price(MARKET, 1.0, arith, n_paths=n, seed=1, n_steps=steps)
    anti_asian = simulate_price(
        MARKET, 1.0, arith, n_paths=n, seed=1, n_steps=steps, antithetic=True
    )
    cv = simulate_control_variate(
        MARKET, 1.0, arith, geo, geo_price, n_paths=n, seed=1, n_steps=steps
    )
    plain_van = simulate_european(MARKET, CALL, n_paths=n, seed=1)
    anti_van = simulate_european(MARKET, CALL, n_paths=n, seed=1, antithetic=True)

    rows = [
        ("Vanilla call", "plain", plain_van, plain_van),
        ("Vanilla call", "antithetic", anti_van, plain_van),
        ("Arithmetic Asian call", "plain", plain_asian, plain_asian),
        ("Arithmetic Asian call", "antithetic", anti_asian, plain_asian),
        ("Arithmetic Asian call", "control variate (geometric)", cv.result, plain_asian),
    ]
    lines = [
        "### Variance reduction (equal path count, 400k paths)\n",
        "| Contract | Method | Price | Std error | Variance reduction | Cost-adjusted speedup |",
        "|---|---|---|---|---|---|",
    ]
    factors = []
    for contract, method, res, base in rows:
        lines.append(
            f"| {contract} | {method} | {res.price:.4f} | {res.standard_error:.5f} | "
            f"{res.variance_reduction_versus(base):.1f}x | {res.efficiency_versus(base):.1f}x |"
        )
        factors.append(res.variance_reduction_versus(base))
    lines.append(
        f"\nPilot correlation between arithmetic and geometric Asian: {cv.correlation:.4f}\n"
    )
    tables.append("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = [
        f"{c.split()[0]} {c.split()[1] if 'Asian' in c else ''}\n{m}".replace("  ", " ")
        for c, m, _, _ in rows
    ]
    ax.bar(range(len(rows)), factors, color=["#999", "#4c78a8", "#999", "#4c78a8", "#f58518"])
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)), labels, fontsize=7)
    ax.set(ylabel="variance reduction (x)", title="Variance reduction at equal path count")
    fig.tight_layout()
    fig.savefig(FIGURES / "variance_reduction.png")
    plt.close(fig)


def barrier() -> None:
    steps = [5, 10, 20, 50, 100, 250, 500]
    out: dict[bool, list[float]] = {True: [], False: []}
    for bridge in (True, False):
        for k in steps:
            payoff = BarrierPayoff(
                100.0, 125.0, OptionType.CALL, BarrierStyle.UP_AND_OUT, 0.20, 1.0, bridge=bridge
            )
            out[bridge].append(
                simulate_price(MARKET, 1.0, payoff, n_paths=300_000, seed=3, n_steps=k).price
            )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogx(steps, out[False], "o-", label="barrier checked on the grid only")
    ax.semilogx(steps, out[True], "o-", label="Brownian bridge correction")
    ax.set(xlabel="time steps", ylabel="up-and-out call price", title="Discrete monitoring bias")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "barrier_bias.png")
    plt.close(fig)
    tables.append(
        "### Up-and-out call, barrier 125 (continuous monitoring)\n\n"
        "| Steps | Grid only | Bridge |\n|---|---|---|\n"
        + "\n".join(
            f"| {k} | {g:.4f} | {b:.4f} |"
            for k, g, b in zip(steps, out[False], out[True], strict=True)
        )
        + "\n"
    )


def greeks() -> None:
    option = EuropeanOption(strike=105.0, maturity=1.0, option_type=OptionType.CALL)
    exact = bs.greeks_contract(option, MARKET)
    fd = fd_greeks(MARKET, option, n_paths=1_000_000, seed=1)
    pw = pathwise_greeks(MARKET, option, n_paths=1_000_000, seed=1)
    rows = [
        ("Delta", float(exact.delta), fd.delta, pw.delta),
        ("Gamma", float(exact.gamma), fd.gamma, None),
        ("Vega", float(exact.vega), fd.vega, pw.vega),
        ("Theta (per year)", float(exact.theta), fd.theta, None),
        ("Rho", float(exact.rho), fd.rho, pw.rho),
    ]
    lines = [
        "### Greeks, call K=105 (1M paths, common random numbers)\n",
        "| Greek | Analytic | Finite difference | Pathwise |",
        "|---|---|---|---|",
    ]
    for name, a, f, p in rows:
        lines.append(f"| {name} | {a:.4f} | {f:.4f} | {'n/a' if p is None else f'{p:.4f}'} |")
    tables.append("\n".join(lines) + "\n")

    bumps = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for crn in (True, False):
        err = []
        for h in bumps:
            e = [
                fd_greeks(
                    MARKET, option, n_paths=100_000, seed=s, rel_bump=h, common_random_numbers=crn
                ).delta
                - float(exact.delta)
                for s in range(6)
            ]
            err.append(math.sqrt(float(np.mean(np.square(e)))))
        ax.loglog(bumps, err, "o-", label="common random numbers" if crn else "independent runs")
    ax.set(
        xlabel="relative bump size",
        ylabel="RMSE of delta",
        title="Finite-difference delta: bump size and CRN",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fd_bump_crn.png")
    plt.close(fig)


def american() -> None:
    lines = [
        "### American put, S=100, sigma=30%, r=5%, T=1 (LSM, 200k paths, 50 dates)\n",
        "| Strike | LSM | Std error | Binomial (2000 steps) | European |",
        "|---|---|---|---|---|",
    ]
    mkt = MarketParams(100.0, 0.30, 0.05, 0.0)
    for strike in (90.0, 100.0, 110.0):
        lsm = price_american_lsm(
            mkt, strike, 1.0, OptionType.PUT, n_paths=200_000, n_steps=50, seed=1
        )
        tree = binomial_price(mkt, strike, 1.0, OptionType.PUT, american=True)
        eu = float(bs.price(100.0, strike, 1.0, 0.05, 0.30, OptionType.PUT))
        lines.append(
            f"| {strike:.0f} | {lsm.price:.4f} | {lsm.standard_error:.4f} | {tree:.4f} | {eu:.4f} |"
        )
    tables.append("\n".join(lines) + "\n")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for step in (convergence, variance_reduction, barrier, greeks, american):
        print("running", step.__name__)
        step()
    (ROOT / "results.md").write_text("\n".join(tables))
    print("wrote results.md and figures/")
