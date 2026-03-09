#!/usr/bin/env python3
"""BTC/USDT 5-minute candle predictor v14.4 — regime-adaptive with hysteresis dampening."""

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


def efficiency_ratio(klines, period=14):
    if len(klines) < period + 1:
        return 0.5
    closes = [k["close"] for k in klines[-(period + 1):]]
    net = abs(closes[-1] - closes[0])
    total = sum(abs(closes[i + 1] - closes[i]) for i in range(len(closes) - 1))
    return net / total if total > 0 else 0.0


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


def compute_ema(closes, period):
    if not closes:
        return 0
    k = 2 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)
    return ema


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "dampening_engaged": False, "last_price": None, "last_prediction": None,
        "last_timestamp": None, "last_was_correct": None, "loop_count": 0,
        "strategy_version": "v14.4", "prediction_history": [], "actual_moves": [],
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

    # Update predictions.log — replace last PENDING line
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

                # Hysteresis dampening
                if state["accuracy_last_12"] < 40:
                    state["dampening_engaged"] = True
                elif state["accuracy_last_12"] > 50:
                    state["dampening_engaged"] = False

    return state


def predict(state):
    """Make BTC prediction using v14.4 strategy."""
    klines_5m = get_klines("5m", 20)
    klines_1m = get_klines("1m", 5)
    klines_15m = get_klines("15m", 5)
    price = get_price()

    er = efficiency_ratio(klines_5m)
    rsi = compute_rsi(klines_1m, 14)

    # Regime and lag_scale
    if er < 0.2:
        lag_scale = 0.0
    elif er < 0.75:
        lag_scale = (er - 0.2) / 0.55  # ramp 0→1
    else:
        lag_scale = 1.0

    # Dampening hysteresis
    if state.get("dampening_engaged") and lag_scale > 0:
        acc = state.get("accuracy_last_12", 50)
        lag_scale *= min(acc / 50.0, 1.0)

    regime = "TRENDING" if er > 0.45 else "RANGING"
    signals = []
    up_score = 0.0
    down_score = 0.0

    # --- Wrong-streak meta-signal ---
    pred_hist = state.get("prediction_history", [])
    acc_hist = state.get("accuracy_history", [])
    wrong_streak = 0
    for a in reversed(acc_hist):
        if a == 0:
            wrong_streak += 1
        else:
            break
    if wrong_streak >= 2 and state.get("last_prediction"):
        nudge_dir = "UP" if state["last_prediction"] == "DOWN" else "DOWN"
        w = min(0.5 * wrong_streak, 2.0)
        signals.append((nudge_dir, w, f"wrong_streak_{wrong_streak}_nudge"))
        if nudge_dir == "UP":
            up_score += w
        else:
            down_score += w

    # --- Poor accuracy meta-signal ---
    acc_12 = state.get("accuracy_last_12", 50.0)
    actual_moves = state.get("actual_moves", [])
    if acc_12 < 45 and len(actual_moves) >= 3:
        up_count = actual_moves[-12:].count("UP")
        down_count = actual_moves[-12:].count("DOWN")
        if abs(up_count - down_count) > 1:  # skip when balanced
            severity = max(0, (45 - acc_12) / 20)  # 0 at 45%, 1 at 25%
            w = 1.0 + severity * 0.8  # 1.0 to 1.8
            nudge_dir = "UP" if up_count > down_count else "DOWN"
            signals.append((nudge_dir, w, f"poor_acc_{acc_12:.0f}%_nudge_{nudge_dir}"))
            if nudge_dir == "UP":
                up_score += w
            else:
                down_score += w

    # --- Dir-bias meta-signal ---
    if len(pred_hist) >= 12:
        last12 = pred_hist[-12:]
        up_p = last12.count("UP")
        down_p = last12.count("DOWN")
        bias = max(up_p, down_p)
        if bias >= 9:
            gate = (acc_12 < 45) or (acc_12 >= 50)
            if gate:
                w = 1.5 if acc_12 < 45 else 1.2
                nudge_dir = "DOWN" if up_p > down_p else "UP"
                signals.append((nudge_dir, w, f"dir_bias_{bias}/12_{nudge_dir}"))
                if nudge_dir == "UP":
                    up_score += w
                else:
                    down_score += w

    # --- RSI extreme ---
    if rsi > 80:
        w = min((rsi - 80) / 20, 1.0)
        signals.append(("DOWN", w, f"rsi_overbought_{rsi:.1f}"))
        down_score += w
    elif rsi < 20:
        w = min((20 - rsi) / 20, 1.0)
        signals.append(("UP", w, f"rsi_oversold_{rsi:.1f}"))
        up_score += w

    # --- Candle 5m momentum ---
    if klines_5m:
        cur = klines_5m[-1]
        candle_pct = (cur["close"] - cur["open"]) / cur["open"] * 100 if cur["open"] else 0
        threshold = 0.001  # very low threshold
        if abs(candle_pct) > threshold:
            w = min(abs(candle_pct) * 100, 2.0)  # up to 2.0
            d = "UP" if candle_pct > 0 else "DOWN"
            signals.append((d, w, f"candle_5m_{candle_pct:+.4f}%"))
            if d == "UP":
                up_score += w
            else:
                down_score += w

    # --- Kline alternation pattern (only in ranging, ER < 0.45) ---
    if er < 0.45 and len(klines_5m) >= 4:
        completed = klines_5m[-4:-1]  # last 3 completed
        dirs = ["UP" if k["close"] > k["open"] else "DOWN" for k in completed]
        alt_count = sum(1 for i in range(len(dirs) - 1) if dirs[i] != dirs[i + 1])
        alt_pct = alt_count / (len(dirs) - 1) if len(dirs) > 1 else 0
        if alt_pct > 0.5:
            # Alternation pattern — predict opposite of last completed candle
            last_dir = dirs[-1]
            pred_dir = "DOWN" if last_dir == "UP" else "UP"
            w = 1.2
            signals.append((pred_dir, w, f"kline_alt_{alt_pct*100:.0f}%_{last_dir}→{pred_dir}"))
            if pred_dir == "UP":
                up_score += w
            else:
                down_score += w

    # --- EMA crossover (ER-scaled) ---
    if len(klines_5m) >= 10:
        closes = [k["close"] for k in klines_5m]
        ema_fast = compute_ema(closes, 5)
        ema_slow = compute_ema(closes, 10)
        cross_pct = (ema_fast - ema_slow) / ema_slow * 100 if ema_slow else 0
        w = min(abs(cross_pct) * 30, 1.5) * lag_scale
        if w > 0.01:
            d = "UP" if cross_pct > 0 else "DOWN"
            signals.append((d, w, f"ema_cross_{cross_pct:+.4f}%"))
            if d == "UP":
                up_score += w
            else:
                down_score += w

    # --- Trend 15m (ER-scaled) ---
    if len(klines_15m) >= 3:
        trend_pct = (klines_15m[-1]["close"] - klines_15m[-3]["close"]) / klines_15m[-3]["close"] * 100
        w = min(abs(trend_pct) * 7, 0.7) * lag_scale
        if w > 0.01:
            d = "UP" if trend_pct > 0 else "DOWN"
            signals.append((d, w, f"trend_15m_{trend_pct:+.4f}%"))
            if d == "UP":
                up_score += w
            else:
                down_score += w

    # --- Persistence ---
    base_thresh = 0.15 if regime == "RANGING" else 0.30
    threshold = base_thresh * max(lag_scale, 0.3)  # minimum threshold even in deep range
    net = up_score - down_score

    if state.get("last_prediction") and abs(net) < threshold:
        prediction = state["last_prediction"]
        persisted = True
    else:
        prediction = "UP" if up_score >= down_score else "DOWN"
        persisted = False

    total = up_score + down_score
    confidence = abs(net) / total if total > 0 else 0.5

    details = {
        "regime": regime, "efficiency_ratio": round(er, 3),
        "lag_scale": round(lag_scale, 2),
        "up_score": round(up_score, 3), "down_score": round(down_score, 3),
        "num_signals": len(signals), "confidence": round(confidence, 3),
        "rsi": round(rsi, 1), "persisted": persisted,
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

    print(f"[v14.4] Loop count: {state['loop_count']}")

    # Verify previous prediction
    state = verify_previous(state)
    print(f"[v14.4] Accuracy: {state.get('accuracy_pct', 'N/A')}% overall, "
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

    print(f"[v14.4] BTC=${price:.2f} | Pred={prediction} | Conf={confidence:.3f} | "
          f"Regime={details['regime']} (ER={details['efficiency_ratio']})")
    for s in details["signals"]:
        print(f"  {s[0]:>4} {s[1]:.3f}  {s[2]}")


if __name__ == "__main__":
    main()
