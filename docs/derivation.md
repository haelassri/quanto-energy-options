# Derivation of the closed-form quanto price

## Setup

Under the pricing measure, the energy and temperature-index futures at the option expiry $T$ are

$$
F_E(T) = F_E\, e^{-\sigma_X^2/2 + X}, \qquad F_I(T) = F_I\, e^{-\sigma_Y^2/2 + Y},
$$

with $(X, Y)$ centred bivariate normal, variances $\sigma_X^2, \sigma_Y^2$ and correlation $\rho$.
Write $X = \sigma_X Z_1$, $Y = \sigma_Y Z_2$ with $(Z_1, Z_2)$ standard normal, correlation $\rho$, and let
$M(h, k; \rho) = \mathbb P(Z_1 \le h, Z_2 \le k)$.

The payoff is $\big(\phi_E(F_E(T) - K_E)\big)^+ \big(\phi_I(F_I(T) - K_I)\big)^+$ with $\phi = +1$ for a call and $-1$ for a put.

## Exercise region

Let $y_1 = \dfrac{\ln(F_E/K_E) - \sigma_X^2/2}{\sigma_X}$ and $y_2 = \dfrac{\ln(F_I/K_I) - \sigma_Y^2/2}{\sigma_Y}$.
The energy leg is in the money iff $\phi_E (Z_1 + y_1) > 0$, i.e. iff $W_1 := -\phi_E Z_1 < \phi_E y_1$; likewise for the
temperature leg with $W_2 := -\phi_I Z_2$. $(W_1, W_2)$ is standard bivariate normal with correlation $\phi_E \phi_I \rho$.

## Four terms

On the exercise region the payoff equals $\phi_E \phi_I \,\big(F_E(T) F_I(T) - K_I F_E(T) - K_E F_I(T) + K_E K_I\big)$.
Each term is an expectation of a lognormal weight times the indicator. A weight $e^{a Z_1 + b Z_2 - \mathrm{Var}(aZ_1 + bZ_2)/2}$
is a change of measure under which $Z_1$ and $Z_2$ gain means $s_1 = a + \rho b$ and $s_2 = b + \rho a$ (Girsanov / Cameron-Martin),
which moves the indicator's thresholds to $\phi_E(y_1 + s_1)$ and $\phi_I(y_2 + s_2)$:

| Term | Weight | Shifts $(s_1, s_2)$ | Prefactor |
|---|---|---|---|
| $F_E(T)F_I(T)$ | $a = \sigma_X,\ b = \sigma_Y$ | $(\sigma_X + \rho\sigma_Y,\ \sigma_Y + \rho\sigma_X)$ | $F_E F_I e^{\rho\sigma_X\sigma_Y}$ |
| $F_E(T)$ | $a = \sigma_X,\ b = 0$ | $(\sigma_X,\ \rho\sigma_X)$ | $F_E$ |
| $F_I(T)$ | $a = 0,\ b = \sigma_Y$ | $(\rho\sigma_Y,\ \sigma_Y)$ | $F_I$ |
| $1$ | none | $(0, 0)$ | $1$ |

The factor $e^{\rho\sigma_X\sigma_Y}$ is $\mathbb E[e^{X - \sigma_X^2/2}\, e^{Y - \sigma_Y^2/2}]$: correlation raises the expected product.

## Result

$$
\begin{aligned}
C = e^{-r(T-t)}\,\phi_E\phi_I \Big[\, & F_E F_I e^{\rho\sigma_X\sigma_Y} M\big(\phi_E(y_1+\sigma_X+\rho\sigma_Y),\ \phi_I(y_2+\sigma_Y+\rho\sigma_X)\big) \\
- \, & F_E K_I\, M\big(\phi_E(y_1+\sigma_X),\ \phi_I(y_2+\rho\sigma_X)\big) \\
- \, & F_I K_E\, M\big(\phi_E(y_1+\rho\sigma_Y),\ \phi_I(y_2+\sigma_Y)\big) \\
+ \, & K_E K_I\, M\big(\phi_E y_1,\ \phi_I y_2\big) \Big],
\end{aligned}
$$

every $M$ at correlation $\phi_E\phi_I\rho$. For $\phi_E = \phi_I = 1$ this is Proposition 3.1 of Benth, Lange & Myklebust (2015).
At $\rho = 0$, $M$ factorises and $C$ is the discounted product of two undiscounted Black-76 prices.

## Greeks

Differentiating the payoff pathwise (it is Lipschitz in each futures price):

$$
\frac{\partial C}{\partial F_E} = e^{-r(T-t)}\phi_E\phi_I\big[F_I e^{\rho\sigma_X\sigma_Y} M_{EI} - K_I M_E\big], \qquad
\frac{\partial^2 C}{\partial F_E\,\partial F_I} = e^{-r(T-t)}\phi_E\phi_I\, e^{\rho\sigma_X\sigma_Y} M_{EI},
$$

with $M_{EI}, M_E$ the first two bivariate terms above, and symmetrically for $\partial C / \partial F_I$.
All of these are checked against Monte Carlo and finite differences in `tests/test_pricing.py`.

## Volatility inputs

With the two-factor dynamics $dF_i/F_i = \sigma_i\,dW_i + \sigma'_i e^{-\gamma_i(\tau_2 - s)}\,dB_i$,

$$
\sigma_X^2 = \int_t^T \big(\sigma_E^2 + \sigma_E'^2 e^{-2\gamma_E(\tau_2-s)} + 2\rho_E\sigma_E\sigma'_E e^{-\gamma_E(\tau_2-s)}\big)\,ds,
$$

and analogously for $\sigma_Y^2$ and $\mathrm{Cov}(X, Y)$, which mixes the four cross-correlations
$\rho_W, \rho_B, \rho_{W_E B_I}, \rho_{B_E W_I}$. Every integral is of the form $\int_t^T e^{-g(\tau_2 - s)}ds = \big(e^{-g(\tau_2-T)} - e^{-g(\tau_2-t)}\big)/g$,
implemented in `volatility.log_moments`.
