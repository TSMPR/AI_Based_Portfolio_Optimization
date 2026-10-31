# User guide

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install `requirements.txt`.
4. Run `streamlit run app.py`.
5. Enter at least two Yahoo Finance ticker symbols separated by commas.
6. Choose lookback, risk-free rate, maximum asset weight, and frontier point count.
7. Click **Fetch data & optimize**.
8. Inspect strategy weights and metrics; export CSV and JSON.

If retrieval fails, verify internet access and ticker spelling, then retry later. The provider may rate-limit requests or change its response format.
