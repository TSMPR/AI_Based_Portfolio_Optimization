"""Portfolio statistics and constrained mean-variance optimization."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        raise ValueError("Price data is empty.")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(returns) < 2:
        raise ValueError("Insufficient observations to calculate returns.")
    return returns

def annualized_inputs(returns: pd.DataFrame, periods: int = TRADING_DAYS):
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Returns must contain at least two assets.")
    mu = returns.mean() * periods
    cov = returns.cov() * periods
    # Symmetrize numerical noise; tiny ridge helps near-singular matrices.
    cov_values = (cov.values + cov.values.T) / 2
    cov_values += np.eye(len(cov_values)) * 1e-10
    cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)
    return mu, cov

def portfolio_metrics(weights, expected_returns, covariance, risk_free_rate=0.02):
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    ret = float(w @ mu)
    variance = float(w @ cov @ w)
    vol = float(np.sqrt(max(variance, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-12 else 0.0
    return {"expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe}

def _validate_inputs(expected_returns, covariance):
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if mu.ndim != 1 or len(mu) < 2 or cov.shape != (len(mu), len(mu)):
        raise ValueError("Expected return vector and covariance matrix dimensions do not match.")
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError("Optimization inputs contain non-finite values.")
    if not np.allclose(cov, cov.T, atol=1e-7):
        raise ValueError("Covariance matrix must be symmetric.")
    return mu, cov

def optimize_portfolio(expected_returns, covariance, objective="max_sharpe",
                       risk_free_rate=0.02, long_only=True, max_weight=1.0):
    """Optimize weights; default is long-only, fully invested, no leverage."""
    mu, cov = _validate_inputs(expected_returns, covariance)
    n = len(mu)
    lower = 0.0 if long_only else -1.0
    if max_weight <= 0 or max_weight * n < 1.0 - 1e-9:
        raise ValueError("Maximum asset weight is infeasible for a fully invested portfolio.")
    bounds = [(lower, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    equal = np.repeat(1.0 / n, n)
    def variance(w): return float(w @ cov @ w)
    if objective == "min_variance":
        fun = variance
    elif objective == "max_sharpe":
        def fun(w):
            vol = np.sqrt(max(variance(w), 1e-16))
            return -float((w @ mu - risk_free_rate) / vol)
    else:
        raise ValueError("objective must be 'min_variance' or 'max_sharpe'")
    result = minimize(fun, equal, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Optimization did not converge: {result.message}")
    weights = np.asarray(result.x)
    weights[np.abs(weights) < 1e-9] = 0.0
    weights /= weights.sum()
    return weights

def efficient_frontier(expected_returns, covariance, points=40, long_only=True, max_weight=1.0):
    """Minimum volatility for target returns over the feasible return interval."""
    mu, cov = _validate_inputs(expected_returns, covariance)
    n = len(mu)
    lo = 0.0 if long_only else -1.0
    bounds = [(lo, max_weight)] * n
    eq = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    equal = np.repeat(1/n, n)
    min_w = optimize_portfolio(mu, cov, "min_variance", long_only=long_only, max_weight=max_weight)
    # A linear program gives the attainable return extrema under box and budget constraints.
    from scipy.optimize import linprog
    lp_min = linprog(mu, A_eq=np.ones((1,n)), b_eq=[1], bounds=bounds, method="highs")
    lp_max = linprog(-mu, A_eq=np.ones((1,n)), b_eq=[1], bounds=bounds, method="highs")
    if not lp_min.success or not lp_max.success:
        raise RuntimeError("Could not determine feasible return range.")
    targets = np.linspace(float(mu @ lp_min.x), float(mu @ lp_max.x), points)
    rows = []
    for target in targets:
        cons = [eq, {"type":"eq", "fun": lambda w, t=target: float(w @ mu - t)}]
        res = minimize(lambda w: float(w @ cov @ w), equal, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 1500, "ftol": 1e-11})
        if res.success:
            vol = np.sqrt(max(float(res.x @ cov @ res.x), 0))
            rows.append({"expected_return": float(res.x @ mu), "volatility": float(vol)})
    return pd.DataFrame(rows)

def allocation_table(weights, tickers):
    result = pd.DataFrame({"Ticker": list(tickers), "Weight": np.asarray(weights, dtype=float)})
    result["Allocation (%)"] = result["Weight"] * 100
    return result.sort_values("Weight", ascending=False).reset_index(drop=True)
