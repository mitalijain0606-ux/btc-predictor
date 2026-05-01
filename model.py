import numpy as np
import requests
import pandas as pd
from scipy import stats
from datetime import datetime, timezone

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_btc_bars(symbol="BTCUSDT", interval="1h", limit=1000):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_URL, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    now_ms = pd.Timestamp.utcnow()
    df = df[df["close_time"] < now_ms].copy()
    df = df.reset_index(drop=True)
    return df


def fetch_btc_bars_range(symbol="BTCUSDT", interval="1h", start_ms=None, end_ms=None):
    all_bars = []
    params = {"symbol": symbol, "interval": interval, "limit": 1000}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    while True:
        resp = requests.get(BINANCE_URL, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        all_bars.extend(raw)
        if len(raw) < 1000:
            break
        params["startTime"] = raw[-1][6] + 1
        if end_ms and params["startTime"] >= end_ms:
            break

    df = pd.DataFrame(all_bars, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


class GBMPredictor:
    def __init__(self, vol_window=24, n_sims=10_000, confidence=0.95, t_dof=None, min_vol_window=10):
        self.vol_window = vol_window
        self.n_sims = n_sims
        self.confidence = confidence
        self.t_dof = t_dof
        self.min_vol_window = min_vol_window
        self._fitted_dof = None

    def _log_returns(self, prices):
        return np.diff(np.log(prices))

    def _estimate_vol(self, returns):
        if len(returns) == 0:
            return 0.01
        lam = 0.94
        weights = np.array([lam ** i for i in range(len(returns) - 1, -1, -1)])
        weights /= weights.sum()
        variance = np.sum(weights * returns ** 2)
        return float(np.sqrt(variance))

    def _fit_t_dof(self, returns):
        if len(returns) < 30:
            return 4.0
        try:
            df, loc, scale = stats.t.fit(returns, floc=0)
            df = max(2.1, min(df, 30.0))
            return float(df)
        except Exception:
            return 4.0

    def predict(self, prices, horizon=1):
        if len(prices) < self.min_vol_window + 1:
            raise ValueError(f"Need at least {self.min_vol_window + 1} price bars.")

        log_ret = self._log_returns(prices)
        recent_ret = log_ret[-self.vol_window:]
        mu = float(np.mean(log_ret))
        sigma = self._estimate_vol(recent_ret)

        if self.t_dof is not None:
            dof = self.t_dof
        elif self._fitted_dof is not None:
            dof = self._fitted_dof
        else:
            dof = self._fit_t_dof(log_ret)
            self._fitted_dof = dof

        current_price = float(prices[-1])

        std_factor = np.sqrt(dof / (dof - 2)) if dof > 2 else 1.0
        raw_z = np.random.standard_t(df=dof, size=(self.n_sims, horizon))
        z = raw_z / std_factor

        log_returns_sim = (mu - 0.5 * sigma ** 2) * horizon + sigma * np.sqrt(horizon) * z.sum(axis=1)
        future_prices = current_price * np.exp(log_returns_sim)

        alpha = 1 - self.confidence
        lower = float(np.percentile(future_prices, 100 * alpha / 2))
        upper = float(np.percentile(future_prices, 100 * (1 - alpha / 2)))

        return {
            "lower": lower,
            "upper": upper,
            "current_price": current_price,
            "mu": mu,
            "sigma": sigma,
            "dof": dof,
            "confidence": self.confidence,
            "width": upper - lower,
            "midpoint": (lower + upper) / 2,
        }


def winkler_score(lower, upper, actual, alpha=0.05):
    width = upper - lower
    penalty_low = (2 / alpha) * max(lower - actual, 0.0)
    penalty_high = (2 / alpha) * max(actual - upper, 0.0)
    return width + penalty_low + penalty_high


def evaluate(predictions):
    n = len(predictions)
    hits = 0
    widths = []
    winklers = []
    for p in predictions:
        lo, hi, actual = p["lower"], p["upper"], p["actual"]
        alpha = 1 - p.get("confidence", 0.95)
        inside = lo <= actual <= hi
        hits += int(inside)
        widths.append(hi - lo)
        winklers.append(winkler_score(lo, hi, actual, alpha))

    return {
        "n": n,
        "coverage_95": hits / n,
        "mean_width": float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winklers)),
        "hit_count": hits,
    }
