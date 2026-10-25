### Convergence (ATM call, 40 replications per N)

| Source | Fitted slope |
|---|---|
| Pseudo-random | -0.514 |
| Sobol (1-D) | -1.046 |

### Variance reduction (equal path count, 400k paths)

| Contract | Method | Price | Std error | Variance reduction | Cost-adjusted speedup |
|---|---|---|---|---|---|
| Vanilla call | plain | 8.9920 | 0.02167 | 1.0x | 1.0x |
| Vanilla call | antithetic | 9.0041 | 0.01631 | 1.8x | 2.9x |
| Arithmetic Asian call | plain | 5.1653 | 0.01211 | 1.0x | 1.0x |
| Arithmetic Asian call | antithetic | 5.1568 | 0.00894 | 1.8x | 1.6x |
| Arithmetic Asian call | control variate (geometric) | 5.1695 | 0.00032 | 1398.5x | 1229.4x |

Pilot correlation between arithmetic and geometric Asian: 0.9996

### Up-and-out call, barrier 125 (continuous monitoring)

| Steps | Grid only | Bridge |
|---|---|---|
| 5 | 3.1583 | 2.0401 |
| 10 | 2.8738 | 2.0399 |
| 20 | 2.6547 | 2.0369 |
| 50 | 2.4510 | 2.0491 |
| 100 | 2.3226 | 2.0406 |
| 250 | 2.2497 | 2.0608 |
| 500 | 2.1835 | 2.0494 |

### Greeks, call K=105 (1M paths, common random numbers)

| Greek | Analytic | Finite difference | Pathwise |
|---|---|---|---|
| Delta | 0.4851 | 0.4838 | 0.4847 |
| Gamma | 0.0196 | 0.0197 | n/a |
| Vega | 39.2932 | 39.1900 | 39.1572 |
| Theta (per year) | -4.8696 | -4.8570 | n/a |
| Rho | 41.6994 | 41.5955 | 41.6823 |

### American put, S=100, sigma=30%, r=5%, T=1 (LSM, 200k paths, 50 dates)

| Strike | LSM | Std error | Binomial (2000 steps) | European |
|---|---|---|---|---|
| 90 | 5.5595 | 0.0188 | 5.5521 | 5.3081 |
| 100 | 9.8927 | 0.0247 | 9.8704 | 9.3542 |
| 110 | 15.6076 | 0.0295 | 15.6182 | 14.6553 |

### SVI fits, SPY on 2026-10-23 (spot 759.97)

| Expiry | T (years) | Points | Fit RMSE (vol pts) | Min g(k) | Vol at k=-0.10 minus vol at k=+0.10 (pts) |
|---|---|---|---|---|---|
| 2026-11-13 | 0.058 | 123 | 0.14 | 0.080 | 15.0 |
| 2026-11-20 | 0.077 | 144 | 0.15 | 0.105 | 13.4 |
| 2026-11-27 | 0.096 | 112 | 0.13 | 0.118 | 12.7 |
| 2026-11-30 | 0.104 | 230 | 0.13 | 0.120 | 12.3 |
| 2026-12-04 | 0.115 | 113 | 0.12 | 0.138 | 11.9 |
| 2026-12-18 | 0.153 | 156 | 0.08 | 0.179 | 11.2 |
| 2026-12-31 | 0.189 | 270 | 0.08 | 0.188 | 10.7 |
| 2027-01-15 | 0.230 | 60 | 0.06 | 0.204 | 10.1 |
| 2027-01-29 | 0.268 | 149 | 0.07 | 0.226 | 9.8 |
| 2027-02-19 | 0.326 | 63 | 0.06 | 0.245 | 9.1 |
| 2027-02-26 | 0.345 | 139 | 0.06 | 0.256 | 9.0 |
| 2027-03-19 | 0.403 | 69 | 0.05 | 0.282 | 8.4 |
| 2027-03-31 | 0.436 | 133 | 0.57 | 0.256 | 7.9 |
| 2027-06-17 | 0.649 | 76 | 0.07 | 0.377 | 7.1 |
| 2027-06-30 | 0.685 | 142 | 0.28 | 0.403 | 7.2 |
| 2027-09-17 | 0.901 | 75 | 0.09 | 0.449 | 6.3 |
| 2027-09-30 | 0.937 | 139 | 0.09 | 0.454 | 6.0 |

Calendar arbitrage violations between adjacent slices: 0

### Delta hedging a short ATM call

Sold at 25% vol, stock realizes 15%, 126 rebalances, 20k paths.

| Hedge volatility | Mean P&L | Std of P&L |
|---|---|---|
| Implied (25%) | 2.815 | 0.856 |
| Realized (15%) | 2.816 | 0.323 |

Deterministic limit when hedging at realized vol: 2.812

### Variance risk premium, VIX vs realized S&P 500 vol (2005-01-03 to 2026-09-24)

Realized vol is measured over the 21 trading days after each date.

| Measure | Mean | Newey-West t | Non-overlapping t | Share of days positive |
|---|---|---|---|---|
| Volatility premium (VIX - RV, vol points) | 3.47 | 8.7 | 7.5 (n=261) | 84% |
| Variance premium (VIX^2 - RV^2, vol points squared) | 76.43 | 2.3 | 1.7 (n=261) | 84% |

Mean VIX 19.1, mean realized 15.6. Date with the most negative premium (a seller's worst case): 2020-02-20 (-72.0 vol points).

### Walk-forward forecast of 21-day realized vol

Out of sample 2008-02-06 to 2026-09-24, scikit-learn ridge with an expanding window and a 21 day gap between train and test.

| Forecast | RMSE of log vol | QLIKE | Mean log error |
|---|---|---|---|
| VIX as the forecast | 0.452 | 0.378 | +0.282 |
| HAR (ridge) | 0.403 | 0.509 | -0.016 |
| HAR + VIX (ridge) | 0.357 | 0.415 | +0.029 |

### Does the gap between the VIX and the model forecast predict the premium?

| Gap quintile (1 = smallest) | Days | Mean realized vol premium (vol pts) |
|---|---|---|
| 1 | 938 | 4.04 |
| 2 | 937 | 3.95 |
| 3 | 938 | 3.70 |
| 4 | 937 | 3.34 |
| 5 | 938 | 2.98 |

Correlation between the gap and the realized premium: -0.058.
