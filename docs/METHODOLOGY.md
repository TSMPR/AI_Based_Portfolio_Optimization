# Quantitative methodology

## Input data

Adjusted daily close prices are downloaded at runtime. The loader sorts dates, removes duplicate dates, coerces values to numeric, rejects non-positive prices, forward-fills gaps for at most three rows, and then retains common complete dates. At least 60 overlapping price observations and two assets are required by default.

## Returns and estimates

Simple daily returns are \(r_t=P_t/P_{t-1}-1\). The application uses the arithmetic sample mean multiplied by 252 and the sample covariance multiplied by 252. The covariance is symmetrized for floating-point stability and receives a negligible diagonal ridge.

## Optimization

Both optimization problems use SciPy SLSQP with a fully-invested equality constraint and per-asset bounds. The default lower bound is zero (long-only). Minimum variance minimizes \(w^T\Sigma w\). Maximum Sharpe minimizes the negative Sharpe ratio. The risk-free rate is annual and is not inferred from market data.

## Efficient frontier

A linear program determines the feasible expected-return endpoints under the same budget and box constraints. For each target return, SLSQP minimizes variance subject to budget and target-return equalities. Failed target solves are skipped; the dashboard reports the successfully solved points.

## Interpretation

Expected return is a historical sample estimate, not a forecast. Covariance and correlation are historical estimates and can change during market stress. No optimizer can remove market risk.
