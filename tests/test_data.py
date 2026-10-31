import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import pandas as pd
import pytest
from portfolio_optimizer.data import clean_price_data

def test_clean_price_data_aligns_and_removes_bad_rows():
    idx = pd.date_range("2022-01-01", periods=80, freq="B")
    df = pd.DataFrame({"A": range(100,180), "B": range(200,280)}, index=idx)
    result = clean_price_data(df, min_observations=60)
    assert result.shape == (80,2)

def test_requires_two_assets():
    with pytest.raises(ValueError):
        clean_price_data(pd.DataFrame({"A": range(80)}))
