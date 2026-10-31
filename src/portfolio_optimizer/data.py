"""Market-data loading with validation and graceful provider errors."""
from __future__ import annotations
import logging
from typing import Iterable
import pandas as pd

logger = logging.getLogger(__name__)

def clean_price_data(prices: pd.DataFrame, min_observations: int = 60) -> pd.DataFrame:
    """Align price series, remove unusable columns, and require adequate history."""
    if prices is None or prices.empty:
        raise ValueError("No price data supplied.")
    frame = prices.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(c[0]) for c in frame.columns]
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.where(frame > 0)
    frame = frame.ffill(limit=3).dropna(axis=1, how="all")
    # Use common dates so covariance is calculated from a consistent sample.
    frame = frame.dropna(axis=0, how="any")
    if frame.shape[1] < 2:
        raise ValueError("At least two assets with overlapping valid prices are required.")
    if len(frame) < min_observations:
        raise ValueError(f"Only {len(frame)} overlapping observations; need at least {min_observations}.")
    return frame

def fetch_yfinance(tickers: Iterable[str], period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download adjusted close prices. Requires internet and yfinance."""
    symbols = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if len(symbols) < 2:
        raise ValueError("Enter at least two distinct ticker symbols.")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: pip install -r requirements.txt") from exc
    try:
        raw = yf.download(symbols, period=period, interval=interval,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            raise ValueError("The market-data provider returned no rows.")
        if isinstance(raw.columns, pd.MultiIndex):
            field = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.get_level_values(0)[0]
            prices = raw[field]
        else:
            prices = raw[["Close"]] if "Close" in raw else raw
            if len(symbols) == 1:
                prices.columns = symbols
        prices.columns = [str(c) for c in prices.columns]
        return clean_price_data(prices)
    except Exception as exc:
        logger.exception("Market data download failed")
        raise RuntimeError(f"Could not retrieve market data: {exc}") from exc
