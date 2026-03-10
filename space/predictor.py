#!/usr/bin/env python3
"""BTC/USDT 5-minute candle predictor v15 — simple momentum, no self-referencing meta-signals."""

import os
import json
import time
import datetime
import requests

SPACE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SPACE_DIR, "state.json")
LOG_FILE = os.path.join(SPACE_DIR, "predictions.log")
BINANCE = "https://api.binance.com"


def fetch_json(url, params=None):
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_price():
    d = fetch_json(f"{BINANCE}/api/v3/ticker/price", {"symbol": "BTCUSDT"})
    return float(d["price"])


def get_klines(interval, limit=20):
    data = fetch_json(f"{BINANCE}/api/v3/klines",
                      {"symbol": "BTCUSDT", "interval": interval, "limit": limit})
    return [{"open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
             "close": float(k[4]), "volume": float(k[5]), "open_time": k[0]} for k in data]


def compute_rsi(klines, period=14):
    if len(klines) < period + 1:
        return 50.0
    closes = [k["close"] for k in klines[-(period + 1):]]
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(d if d > 0 else 0)
        losses.append(-d if d < 0 else 0)
    avg_g = sum(gains) / len(gains)
    avg_l = sum(losses) / len(losses)
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100 - (100 / (1 + rs))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "last_price": None, "last_prediction": None,
        "last_timestamp": None, "last_was_correct": None, "loop_count": 0,
        "strategy_version": "v15", "prediction_history": [], "actual_moves": [],
        "accuracy_history": [], "accuracy_pct": 50.0, "accuracy_last_12": 50.0
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def verify_previous(state):
    """Check if previous prediction's candle has closed, update log and state."""
    if not state.get("last_timestamp") or not state.get("last_price"):
        return state

    klines_5m = get_klines("5m", 3)
    if len(klines_5m) < 2:
        return state

    prev_candle = klines_5m[-2]
    actual = "UP" if prev_candle["close"] > prev_candle["open"] else "DOWN"

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            lines = f.readlines()
        if lines and "PENDING" in lines[-1]:
            last_pred = None
            for part in lines[-1].split("|"):
                if "prediction=" in part:
                    last_pred = part.split("=")[1].strip()
            if last_pred:
                correct = (last_pred == actual)
                lines[-1] = lines[-1].replace("actual=PENDING", f"actual={actual}")
                lines[-1] = lines[-1].replace("correct=PENDING", f"correct={correct}")
                with open(LOG_FILE, "w") as f:
                    f.writelines(lines)

                state["last_was_correct"] = correct
                state["actual_moves"] = (state.get("actual_moves", []) + [actual])[-12:]
                state["accuracy_history"] = (state.get("accuracy_history", []) + [1 if correct else 0])[-12:]
                n = len(state["accuracy_history"])
                state["accuracy_pct"] = (sum(state["accuracy_history"]) / n * 100) if n else 50.0
                last12 = state["accuracy_history"][-12:]
                state["accuracy_last_12"] = (sum(last12) / len(last12) * 100) if last12 else 50.0

    return state


def predict(state):
    """Simple BTC prediction — pure market signals, no self-referencing feedback loops.

    Strategy v15:
    - Look at last 5 completed 5m candles
    - In ranging markets (alternating), predict opposite of last completed candle
    - In trending markets (3+ same direction), follow the trend
    - Use RSI extremes as override
    - Use volume-weighted momentum of current candle
    """
    klines_5m = get_klines("5m", 10)
    price = get_price()

    # Get directions of last 5 completed candles (exclude current incomplete)
    completed = klines_5m[:-1][-5:]  # last 5 completed
    dirs = ["UP" if k["close"] > k["open"] else "DOWN" for k in completed]

    up_score = 0.0
    down_score = 0.0
    signals = []

    # Signal 1: Trend detection — last 3 completed candles
    if len(dirs) >= 3:
        last3 = dirs[-3:]
        up3 = last3.count("UP")
        down3 = last3.count("DOWN")
        if up3 == 3:
            # Strong uptrend - follow it
            signals.append(("UP", 1.5, "trend_3up"))
            up_score += 1.5
        elif down3 == 3:
            # Strong downtrend - follow it
            signals.append(("DOWN", 1.5, "trend_3down"))
            down_score += 1.5
        elif up3 == 2:
            signals.append(("UP", 0.5, "trend_2of3up"))
            up_score += 0.5
        elif down3 == 2:
            signals.append(("DOWN", 0.5, "trend_2of3down"))
            down_score += 0.5

    # Signal 2: Alternation detection
    if len(dirs) >= 3:
        alternations = sum(1 for i in range(len(dirs)-1) if dirs[i] != dirs[i+1])
        alt_rate = alternations / (len(dirs) - 1)
        if alt_rate >= 0.6:
            # Alternating market — predict opposite of last completed
            last_dir = dirs[-1]
            opp = "DOWN" if last_dir == "UP" else "UP"
            w = 1.2
            signals.append((opp, w, f"alternation_{alt_rate:.0%}"))
            if opp == "UP":
                up_score += w
            else:
                down_score += w

    # Signal 3: Current candle momentum (intra-candle)
    cur = klines_5m[-1]
    if cur["open"] > 0:
        cur_pct = (cur["close"] - cur["open"]) / cur["open"] * 100
        if abs(cur_pct) > 0.01:  # meaningful move
            # Momentum continuation
            w = min(abs(cur_pct) * 50, 1.0)
            d = "UP" if cur_pct > 0 else "DOWN"
            signals.append((d, w, f"cur_momentum_{cur_pct:+.4f}%"))
            if d == "UP":
                up_score += w
            else:
                down_score += w

    # Signal 4: RSI on 5m candles (mean reversion at extremes)
    rsi = compute_rsi(klines_5m, 14)
    if rsi > 70:
        w = min((rsi - 70) / 20, 1.5)
        signals.append(("DOWN", w, f"rsi_high_{rsi:.1f}"))
        down_score += w
    elif rsi < 30:
        w = min((30 - rsi) / 20, 1.5)
        signals.append(("UP", w, f"rsi_low_{rsi:.1f}"))
        up_score += w

    # Signal 5: Volume spike detection
    if len(completed) >= 5:
        volumes = [k["volume"] for k in completed]
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
        if avg_vol > 0 and volumes[-1] > avg_vol * 1.5:
            # High volume on last candle — continuation is more likely
            last_dir = dirs[-1]
            w = 0.8
            signals.append((last_dir, w, f"vol_spike_{volumes[-1]/avg_vol:.1f}x"))
            if last_dir == "UP":
                up_score += w
            else:
                down_score += w

    # Decision
    prediction = "UP" if up_score >= down_score else "DOWN"
    total = up_score + down_score
    confidence = abs(up_score - down_score) / total if total > 0 else 0.0

    details = {
        "regime": "v15_simple",
        "up_score": round(up_score, 3), "down_score": round(down_score, 3),
        "num_signals": len(signals), "confidence": round(confidence, 3),
        "rsi": round(rsi, 1),
        "recent_dirs": dirs,
        "signals": [[s[0], round(s[1], 3), s[2]] for s in signals]
    }

    return prediction, confidence, price, details


def log_prediction(price, prediction, confidence):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    line = f"{ts} | price={price:.2f} | prediction={prediction} | confidence={confidence:.3f} | actual=PENDING | correct=PENDING\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    return ts


def main():
    state = load_state()
    state["loop_count"] = state.get("loop_count", 0) + 1
    state["strategy_version"] = "v15"

    print(f"[v15] Loop count: {state['loop_count']}")

    # Verify previous prediction
    state = verify_previous(state)
    print(f"[v15] Accuracy: {state.get('accuracy_pct', 'N/A')}% overall, "
          f"{state.get('accuracy_last_12', 'N/A')}% last 12")

    # Make prediction
    prediction, confidence, price, details = predict(state)

    # Log it
    ts = log_prediction(price, prediction, confidence)

    # Update state
    state["last_price"] = price
    state["last_prediction"] = prediction
    state["last_timestamp"] = ts
    state["last_details"] = details
    state["prediction_history"] = (state.get("prediction_history", []) + [prediction])[-12:]

    save_state(state)

    print(f"[v15] BTC=${price:.2f} | Pred={prediction} | Conf={confidence:.3f}")
    for s in details["signals"]:
        print(f"  {s[0]:>4} {s[1]:.3f}  {s[2]}")


if __name__ == "__main__":
    main()
