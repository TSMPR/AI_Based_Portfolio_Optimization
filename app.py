from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from portfolio_optimizer.data import fetch_yfinance, clean_price_data
from portfolio_optimizer.analytics import (calculate_returns, annualized_inputs,
    optimize_portfolio, portfolio_metrics, efficient_frontier, allocation_table)
from portfolio_optimizer.reporting import build_report, report_json, report_csv

st.set_page_config(page_title="AI-Based Portfolio Optimization", page_icon="📈", layout="wide")
st.title("📈 AI-Based Portfolio Optimization")
st.caption("Quantitative portfolio construction using historical market data — not investment advice.")

with st.sidebar:
    st.header("Portfolio configuration")
    tickers_raw = st.text_input("Asset tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN, NVDA")
    period = st.selectbox("Historical lookback", ["1y", "2y", "5y", "10y"], index=2)
    rf_pct = st.number_input("Annual risk-free rate (%)", min_value=-5.0, max_value=25.0, value=2.0, step=0.25)
    max_weight_pct = st.slider("Maximum weight per asset (%)", 20, 100, 100, 5)
    frontier_points = st.slider("Efficient frontier points", 10, 80, 35, 5)
    run = st.button("Fetch data & optimize", type="primary", use_container_width=True)

tickers = list(dict.fromkeys(x.strip().upper() for x in tickers_raw.split(",") if x.strip()))
if "prices" not in st.session_state:
    st.session_state.prices = None
if run:
    if len(tickers) < 2:
        st.error("Enter at least two different ticker symbols.")
    else:
        with st.spinner("Downloading adjusted prices and optimizing portfolios..."):
            try:
                st.session_state.prices = fetch_yfinance(tickers, period=period)
                st.session_state.last_tickers = tickers
                st.session_state.last_period = period
            except Exception as exc:
                st.error(str(exc))

prices = st.session_state.prices
if prices is None:
    st.info("Choose at least two tickers, then click **Fetch data & optimize**. Internet access is required for live market data.")
    st.markdown("**Example tickers:** `AAPL, MSFT, GOOGL, AMZN, NVDA`")
    st.stop()

try:
    returns = calculate_returns(prices)
    mu, cov = annualized_inputs(returns)
    rf = rf_pct / 100.0
    max_weight = max_weight_pct / 100.0
    if max_weight * len(mu) < 1:
        st.error("The maximum weight is infeasible for the selected number of assets.")
        st.stop()
    names = list(mu.index)
    w_min = optimize_portfolio(mu.values, cov.values, "min_variance", risk_free_rate=rf, max_weight=max_weight)
    w_sharpe = optimize_portfolio(mu.values, cov.values, "max_sharpe", risk_free_rate=rf, max_weight=max_weight)
    w_equal = np.repeat(1 / len(names), len(names))
    strategies = {"Minimum Variance": w_min, "Maximum Sharpe": w_sharpe, "Equal Weight": w_equal}
    metrics = {k: portfolio_metrics(w, mu.values, cov.values, rf) for k, w in strategies.items()}
    frontier = efficient_frontier(mu.values, cov.values, points=frontier_points, max_weight=max_weight)

    st.caption(f"Data window: {prices.index.min().date()} to {prices.index.max().date()} · {len(prices):,} price observations · {len(names)} assets")
    cols = st.columns(4)
    best = metrics["Maximum Sharpe"]
    cols[0].metric("Max Sharpe expected return", f"{best['expected_return']:.2%}")
    cols[1].metric("Max Sharpe volatility", f"{best['volatility']:.2%}")
    cols[2].metric("Max Sharpe ratio", f"{best['sharpe_ratio']:.2f}")
    cols[3].metric("Assets", len(names))

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Efficient frontier", "Risk & diversification", "Data"])
    with tab1:
        st.subheader("Strategy comparison")
        comp = pd.DataFrame([{"Strategy": k, **v} for k,v in metrics.items()])
        st.dataframe(comp.style.format({"expected_return":"{:.2%}", "volatility":"{:.2%}", "sharpe_ratio":"{:.3f}"}), use_container_width=True)
        selected = st.selectbox("Allocation to inspect", list(strategies.keys()), index=1)
        alloc = allocation_table(strategies[selected], names)
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.pie(alloc, names="Ticker", values="Weight", hole=.45, title=f"{selected} allocation"), use_container_width=True)
        c2.plotly_chart(px.bar(alloc, x="Ticker", y="Allocation (%)", title="Asset weights (%)"), use_container_width=True)
        st.dataframe(alloc[["Ticker","Allocation (%)"]].style.format({"Allocation (%)":"{:.2f}%"}), use_container_width=True)
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["expected_return"], mode="lines+markers", name="Efficient frontier"))
        for k,w in strategies.items():
            m=metrics[k]
            fig.add_trace(go.Scatter(x=[m["volatility"]], y=[m["expected_return"]], mode="markers+text",
                text=[k], textposition="top center", name=k, marker={"size":11}))
        fig.update_layout(title="Annualized risk vs. expected return", xaxis_title="Volatility (standard deviation)", yaxis_title="Expected return", yaxis_tickformat=".0%", xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Frontier points solve minimum variance at target expected returns, subject to the selected weight bounds.")
    with tab3:
        st.subheader("Correlation matrix")
        corr = returns.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu", title="Daily return correlations"), use_container_width=True)
        st.subheader("Historical normalized prices")
        normalized = prices / prices.iloc[0] * 100
        st.plotly_chart(px.line(normalized, title="Growth of 100 (price-only, no cash-flow adjustment)"), use_container_width=True)
        st.write("Diversification insight: lower pairwise correlations can reduce portfolio variance, but correlations and risk can rise together during market stress.")
    with tab4:
        st.dataframe(prices.tail(20), use_container_width=True)
        st.dataframe(returns.describe().T, use_container_width=True)

    report = build_report(names, strategies, metrics, rf, st.session_state.get("last_period", period))
    st.download_button("Download allocation CSV", report_csv(strategies, names), "portfolio_allocations.csv", "text/csv")
    st.download_button("Download portfolio report JSON", report_json(report), "portfolio_report.json", "application/json")
    st.warning("Historical estimates are highly sensitive to the sample period and assumptions. They are not forecasts, personalized financial advice, or guaranteed outcomes.")
except Exception as exc:
    st.exception(exc)
