import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from portfolio_optimizer.analytics import calculate_returns, annualized_inputs, portfolio_metrics, optimize_portfolio, efficient_frontier

def sample_returns():
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal([.0004,.0006,.0003], [.01,.015,.008], (500,3)), columns=["A","B","C"])

def test_returns_and_annualization():
    prices = (1 + sample_returns()).cumprod()
    returns = calculate_returns(prices)
    mu, cov = annualized_inputs(returns)
    assert mu.shape == (3,)
    assert cov.shape == (3,3)
    assert np.allclose(cov.values, cov.values.T)

def test_weights_constraints_and_metrics():
    r = sample_returns()
    mu, cov = annualized_inputs(r)
    w = optimize_portfolio(mu.values, cov.values, "max_sharpe", risk_free_rate=.01, max_weight=.8)
    assert np.all(w >= -1e-8)
    assert np.all(w <= .8 + 1e-7)
    assert np.isclose(w.sum(), 1)
    m = portfolio_metrics(w, mu.values, cov.values, .01)
    assert m["volatility"] >= 0
    assert np.isfinite(m["sharpe_ratio"])

def test_min_variance_and_frontier():
    mu = np.array([.08,.11,.06])
    cov = np.array([[.04,.01,.005],[.01,.09,.01],[.005,.01,.025]])
    w = optimize_portfolio(mu, cov, "min_variance")
    assert np.isclose(w.sum(), 1)
    f = efficient_frontier(mu, cov, points=12)
    assert len(f) > 0
    assert (f["volatility"] >= 0).all()

def test_infeasible_max_weight():
    with pytest.raises(ValueError):
        optimize_portfolio([.1,.2,.3], np.eye(3), max_weight=.2)
