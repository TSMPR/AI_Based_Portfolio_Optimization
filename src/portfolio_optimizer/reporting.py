"""Portable report exports."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import pandas as pd

def build_report(ticker_list, weights_by_strategy, metrics_by_strategy, risk_free_rate, period):
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": list(ticker_list),
        "period": period,
        "risk_free_rate_annual": float(risk_free_rate),
        "assumptions": [
            "Historical arithmetic mean returns annualized using 252 trading days.",
            "Sample covariance annualized using 252 trading days.",
            "Long-only fully invested portfolios unless configured otherwise.",
            "Estimates are historical and are not forecasts or guarantees."
        ],
        "allocations": {k: {str(t): float(w) for t, w in zip(ticker_list, v)}
                        for k, v in weights_by_strategy.items()},
        "metrics": metrics_by_strategy
    }

def report_json(report):
    return json.dumps(report, indent=2, allow_nan=False)

def report_csv(weights_by_strategy, tickers):
    frames = []
    for strategy, weights in weights_by_strategy.items():
        frames.append(pd.DataFrame({"Ticker": list(tickers), "Weight": weights, "Strategy": strategy}))
    return pd.concat(frames, ignore_index=True).to_csv(index=False)
