# ₿ BTC 1-Hour Range Predictor

A real-time Bitcoin price range prediction dashboard that forecasts the next 1-hour price range using Monte Carlo simulations with **95% verified accuracy** across 720+ hourly predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Live Price Tracking** — Real-time BTCUSDT data from Binance public API
- **95% Confidence Interval** — Statistically calibrated upper/lower price bounds for the next hour
- **Monte Carlo Simulation** — 10,000 path simulations per prediction
- **Adaptive Volatility** — EWMA-based volatility estimation that reacts to market conditions
- **Fat-Tail Modeling** — Student-t distribution to handle extreme crypto price movements
- **30-Day Backtest** — Verified 95.00% coverage across 720 consecutive hourly bars
- **Live Hit/Miss Tracking** — Persistent evaluation log that scores predictions in real-time
- **Interactive Charts** — Candlestick charts with prediction ribbons and Monte Carlo distributions

---

## Dashboard Preview

The dashboard includes:
- Current BTC price with hourly change indicator
- Predicted price range with bullish/bearish drift badge
- Market metrics (drift, volatility, annualized vol)
- Monte Carlo distribution histogram
- Volatility regime indicator (Low / Medium / High)
- Candlestick chart with prediction ribbon overlay
- Live evaluation log with hit/miss history

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mitalijain0606-ux/btc-predictor.git
cd btc-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Run Backtest

To verify model accuracy on the last 30 days of historical data:

```bash
python backtest.py
```

This generates:
- `backtest_summary.json` — Coverage, mean width, and Winkler score
- `backtest_results.jsonl` — Detailed per-bar prediction results

---

## Project Structure

```
btc-predictor/
├── model.py               # Prediction engine (GBM + Monte Carlo)
├── backtest.py             # 30-day historical accuracy test
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── backtest_summary.json   # Latest backtest results
└── .streamlit/
    └── config.toml         # Dashboard theme configuration
```

---

## Backtest Results

| Metric | Value |
|--------|-------|
| **Coverage** | 95.00% ✅ |
| **Mean Range Width** | $1,333 |
| **Mean Winkler Score** | 2,208 |
| **Predictions Tested** | 720 |

---

## How It Works

1. **Data** — Fetches latest 500+ hourly BTCUSDT candles from Binance public API (no API key required)
2. **Volatility** — Estimates current market volatility using an exponentially weighted moving average (24-hour window), giving more weight to recent price action
3. **Simulation** — Runs 10,000 Monte Carlo price paths using a fat-tailed Student-t distribution to account for crypto's extreme price movements
4. **Range** — Extracts the 97.5% confidence interval from simulated prices to produce a calibrated prediction band
5. **Evaluation** — Each prediction is logged and scored against actual outcomes using the Winkler interval score

---

## Tech Stack

- **Language:** Python 3.10+
- **Dashboard:** Streamlit + Plotly
- **Data Source:** Binance Public API
- **Statistics:** SciPy, NumPy, Pandas

---

## License

MIT
