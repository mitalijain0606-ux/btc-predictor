import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

from model import fetch_btc_bars_range, GBMPredictor, evaluate, winkler_score

WARMUP_BARS = 200
OUTPUT_FILE = "backtest_results.jsonl"
CONFIDENCE  = 0.975
VOL_WINDOW  = 24


def run_backtest():
    print("=" * 60)
    print("  BTC 30-Day Backtest")
    print("=" * 60)

    total_bars_needed = 720 + WARMUP_BARS
    now_utc = datetime.now(timezone.utc)
    start_dt = now_utc - timedelta(hours=total_bars_needed + 5)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(now_utc.timestamp() * 1000)

    print(f"\n⬇  Fetching BTCUSDT 1h bars from Binance …")
    print(f"   Period: {start_dt.strftime('%Y-%m-%d %H:%M UTC')} → now")
    df = fetch_btc_bars_range(start_ms=start_ms, end_ms=end_ms)
    print(f"   Retrieved {len(df)} bars total.\n")

    closes = df["close"].values
    times = df["open_time"].values

    start_idx = max(WARMUP_BARS, len(closes) - 721)
    backtest_indices = list(range(start_idx, len(closes) - 1))

    print(f"🔁  Running backtest on {len(backtest_indices)} bars …\n")

    predictor = GBMPredictor(
        vol_window=VOL_WINDOW,
        n_sims=10_000,
        confidence=CONFIDENCE,
    )

    predictions = []
    errors = 0

    for idx in tqdm(backtest_indices, desc="Predicting", ncols=70):
        history = closes[:idx]
        actual = float(closes[idx])
        bar_time = pd.Timestamp(times[idx]).isoformat()

        try:
            pred = predictor.predict(history, horizon=1)
            record = {
                "bar_time": bar_time,
                "bar_index": int(idx),
                "lower": round(pred["lower"], 2),
                "upper": round(pred["upper"], 2),
                "actual": round(actual, 2),
                "current_price": round(pred["current_price"], 2),
                "sigma": round(pred["sigma"], 8),
                "mu": round(pred["mu"], 10),
                "dof": round(pred["dof"], 2),
                "confidence": CONFIDENCE,
                "width": round(pred["width"], 2),
                "hit": pred["lower"] <= actual <= pred["upper"],
                "winkler": round(winkler_score(pred["lower"], pred["upper"], actual, 1 - CONFIDENCE), 2),
            }
            predictions.append(record)
        except Exception:
            errors += 1
            continue

    metrics = evaluate(predictions)

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Predictions made : {metrics['n']}")
    print(f"  Errors skipped   : {errors}")
    print(f"  Coverage (target 0.95) : {metrics['coverage_95']:.4f}  {'✅' if abs(metrics['coverage_95'] - 0.95) < 0.03 else '⚠️'}")
    print(f"  Mean width ($)         : ${metrics['mean_width']:,.2f}")
    print(f"  Mean Winkler score     : {metrics['mean_winkler_95']:,.2f}")
    print("=" * 60)

    with open(OUTPUT_FILE, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_predictions": metrics["n"],
        "coverage_95": metrics["coverage_95"],
        "mean_width": metrics["mean_width"],
        "mean_winkler_95": metrics["mean_winkler_95"],
        "vol_window": VOL_WINDOW,
        "n_sims": 10_000,
        "confidence": CONFIDENCE,
    }
    with open("backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅  Saved {len(predictions)} predictions → {OUTPUT_FILE}")
    print(f"✅  Saved summary → backtest_summary.json")

    return predictions, metrics


if __name__ == "__main__":
    run_backtest()
