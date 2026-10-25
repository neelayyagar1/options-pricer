### Convergence (ATM call, 40 replications per N)

| Source | Fitted slope |
|---|---|
| Pseudo-random | -0.514 |
| Sobol (1-D) | -1.046 |

### Variance reduction (equal path count, 400k paths)

| Contract | Method | Price | Std error | Variance reduction | Cost-adjusted speedup |
|---|---|---|---|---|---|
| Vanilla call | plain | 8.9920 | 0.02167 | 1.0x | 1.0x |
| Vanilla call | antithetic | 9.0041 | 0.01631 | 1.8x | 3.0x |
| Arithmetic Asian call | plain | 5.1653 | 0.01211 | 1.0x | 1.0x |
| Arithmetic Asian call | antithetic | 5.1568 | 0.00894 | 1.8x | 1.7x |
| Arithmetic Asian call | control variate (geometric) | 5.1695 | 0.00032 | 1398.5x | 1246.7x |

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
