#!/usr/bin/env python3
"""BTC/USDT 5-minute candle predictor for Avolution survival loop."""

import json
import re
import time
import sys
import os
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.error import URLError

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(WORKSPACE, 'state.json')
PREDICTIONS_LOG = os.path.join(WORKSPACE, 'predictions.log')
BINANCE_URL = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=20'


def fetch_candles():
    """Fetch recent 5m candles from Binance."""
    try:
        resp = urlopen(BINANCE_URL, timeout=10)
        data = json.loads(resp.read())
        candles = []
        for c in data:
            candles.append({
                'open_time': c[0],
                'open': float(c[1]),
                'high': float(c[2]),
                'low': float(c[3]),
                'close': float(c[4]),
                'volume': float(c[5]),
                'close_time': c[6],
            })
        return candles
    except (URLError, Exception) as e:
        print(f"Error fetching candles: {e}", file=sys.stderr)
        return None


def get_window_info():
    """Calculate current prediction window."""
    now = time.time()
    window_start = int((now // 300) * 300)
    deadline = window_start + 270
    target_open = window_start + 300
    target_close = window_start + 600

    return {
        'now': now,
        'window_start': window_start,
        'deadline': deadline,
        'target_open': target_open,
        'target_close': target_close,
        'window_label': datetime.fromtimestamp(window_start, tz=timezone.utc).strftime('%H:%M'),
        'target_label': datetime.fromtimestamp(target_open, tz=timezone.utc).strftime('%H:%M'),
        'secs_to_deadline': deadline - now,
    }


def load_state():
    """Load previous state."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    """Save state to disk."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def calc_rsi(closes, period=14):
    """Calculate RSI from close prices."""
    if len(closes) < period + 1:
        return 50.0  # neutral default
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(values, period):
    """Calculate EMA."""
    if not values:
        return 0
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def calc_bollinger(closes, period=20, std_mult=2.0):
    """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
    if len(closes) < period:
        period = len(closes)
    if period < 2:
        return closes[-1] + 1, closes[-1], closes[-1] - 1
    recent = closes[-period:]
    middle = sum(recent) / len(recent)
    variance = sum((x - middle) ** 2 for x in recent) / len(recent)
    std = variance ** 0.5
    return middle + std_mult * std, middle, middle - std_mult * std


def predict(candles):
    """
    Predict next candle direction using multiple signals.
    Returns (prediction, confidence, reason).
    """
    if not candles or len(candles) < 5:
        return 'UP', 0.5, 'insufficient_data'

    # Only use closed candles (exclude the last one which may be current/incomplete)
    closed = candles[:-1]
    closes = [c['close'] for c in closed]

    # Signal 1: Recent momentum (last 3 candles direction)
    recent = closed[-3:]
    recent_dirs = [1 if c['close'] > c['open'] else -1 for c in recent]
    momentum = sum(recent_dirs) / len(recent_dirs)

    # Signal 2: Streak detection (mean reversion)
    last5_dirs = [1 if c['close'] > c['open'] else -1 for c in closed[-5:]]
    streak_up = sum(1 for d in last5_dirs if d == 1)
    streak_down = sum(1 for d in last5_dirs if d == -1)

    # Signal 3: Price vs short-term SMA
    sma_closes = closes[-10:]
    sma = sum(sma_closes) / len(sma_closes)
    last_price = closes[-1]
    price_vs_sma = (last_price - sma) / sma

    # Signal 4: Volume analysis
    vols = [c['volume'] for c in closed[-5:]]
    avg_vol = sum(vols) / len(vols)
    last_vol = closed[-1]['volume']
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

    # Signal 5: Candle body analysis
    last_body = closed[-1]['close'] - closed[-1]['open']
    last_range = closed[-1]['high'] - closed[-1]['low']
    body_ratio = abs(last_body) / last_range if last_range > 0 else 0

    # Signal 6: RSI
    rsi = calc_rsi(closes)

    # Signal 7: EMA crossover (fast vs slow)
    ema_fast = calc_ema(closes, 5)
    ema_slow = calc_ema(closes, 12)
    ema_diff = (ema_fast - ema_slow) / ema_slow if ema_slow else 0

    # Signal 8: Bollinger Bands position
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes, 15)
    bb_range = bb_upper - bb_lower
    bb_pos = (last_price - bb_lower) / bb_range if bb_range > 0 else 0.5

    # Signal 9: Current (incomplete) candle momentum
    current = candles[-1]  # incomplete candle
    cur_move = (current['close'] - current['open']) / current['open'] if current['open'] else 0
    cur_vol = current['volume']
    cur_vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

    # Signal 10: Alternation pattern (UP/DOWN/UP/DOWN)
    if len(closed) >= 4:
        last4_dirs = [1 if c['close'] > c['open'] else -1 for c in closed[-4:]]
        alternating = all(last4_dirs[i] != last4_dirs[i+1] for i in range(3))
    else:
        alternating = False

    # Signal 11: Price trend (close-to-close over last 5 candles)
    if len(closes) >= 6:
        price_change_5 = (closes[-1] - closes[-6]) / closes[-6]
    else:
        price_change_5 = 0

    # Signal 12: Price trend (close-to-close over last 10 candles)
    if len(closes) >= 11:
        price_change_10 = (closes[-1] - closes[-11]) / closes[-11]
    else:
        price_change_10 = 0

    # Signal 13: Last candle reversal (simple mean reversion)
    last_dir = 1 if closed[-1]['close'] > closed[-1]['open'] else -1

    # Combine signals — continuous (no dead zones)
    signals = {}
    score = 0.0

    # Mean reversion on strong streaks (4+ same direction)
    # Reduce weight when trend signals confirm the streak direction (trend > mean reversion)
    if streak_down >= 4:
        s = 0.10 if (price_change_5 < -0.0005 or price_change_10 < -0.0005) else 0.20
        score += s
        signals['mr_dn'] = s
    elif streak_up >= 4:
        s = -0.10 if (price_change_5 > 0.0005 or price_change_10 > 0.0005) else -0.20
        score += s
        signals['mr_up'] = s

    # Short momentum (last 3 candle direction) — continuous
    s = momentum * 0.10
    score += s
    signals['mom'] = s

    # Price trend 5-candle — continuous, proportional
    # EMERGENCY v17.6: cap reduced 0.14→0.10 to prevent overriding reversal
    s = max(-0.10, min(0.10, price_change_5 * 120))
    score += s
    signals['t5'] = s

    # Price trend 10-candle — continuous, proportional (cap raised)
    s = max(-0.14, min(0.14, price_change_10 * 80))
    score += s
    signals['t10'] = s

    # SMA deviation (mean revert) — continuous
    # v17.8: dampen during strong trends to stop fighting trend
    sma_mult = 15 if (streak_up >= 4 or streak_down >= 4) else 40
    s = max(-0.20, min(0.20, -price_vs_sma * sma_mult))
    score += s
    signals['sma'] = s

    # RSI — continuous, centered at 50 (multiplier reduced to balance vs trend)
    # v17.8: dampen during strong trends
    rsi_mult = 0.08 if (streak_up >= 4 or streak_down >= 4) else 0.22
    s = -(rsi - 50) / 100 * rsi_mult
    score += s
    signals['rsi'] = s

    # EMA crossover — continuous
    s = max(-0.15, min(0.15, ema_diff * 40))
    score += s
    signals['ema'] = s

    # Bollinger Bands — continuous, centered at 0.5 (multiplier reduced to balance vs trend)
    # v17.8: dampen during strong trends
    bb_mult = 0.08 if (streak_up >= 4 or streak_down >= 4) else 0.22
    s = -(bb_pos - 0.5) * bb_mult
    score += s
    signals['bb'] = s

    # Current candle momentum (incomplete) — continuous
    if cur_vol_ratio > 0.15:
        s = max(-0.10, min(0.10, cur_move * 80))
        score += s
        signals['cur'] = s

    # Alternation pattern
    if alternating and len(closed) >= 4:
        last_dir = 1 if closed[-1]['close'] > closed[-1]['open'] else -1
        s = -last_dir * 0.10
        score += s
        signals['alt'] = s

    # Last candle reversal (mean reversion — 5m candles alternate ~60-67%)
    # v17.7: trend-adaptive rev — reduce during strong trends (4+ streak)
    # When market is clearly trending, alternation breaks down, rev fights trend
    if streak_up >= 4 or streak_down >= 4:
        rev_weight = 0.05  # heavily reduced during strong trends
    else:
        rev_weight = 0.20  # full weight during normal alternation
    s = -last_dir * rev_weight
    score += s
    signals['rev'] = s

    # High volume + strong body = continuation
    if vol_ratio > 1.3 and body_ratio > 0.5:
        s = 0.08 if last_body > 0 else -0.08
        score += s
        signals['vol'] = s

    # Debug: print all signal contributions
    sig_str = ' '.join(f'{k}={v:+.3f}' for k, v in sorted(signals.items()))
    print(f"  Signals: {sig_str}")
    print(f"  Score: {score:+.4f} | RSI={rsi:.1f} BB={bb_pos:.2f} EMA_diff={ema_diff:.5f}")
    print(f"  Trend5={price_change_5:+.5f} Trend10={price_change_10:+.5f}")

    # Final decision
    if score > 0:
        prediction = 'UP'
    elif score < 0:
        prediction = 'DOWN'
    else:
        prediction = 'UP' if closed[-1]['close'] > closed[-1]['open'] else 'DOWN'
        signals['tie'] = 0

    confidence = min(0.5 + abs(score) * 0.2, 0.85)
    reason = '+'.join(f'{k}' for k in signals.keys())

    return prediction, confidence, reason


def verify_all_pending(candles):
    """Scan predictions.log for all PENDING entries and verify against candle data."""
    if not candles or not os.path.exists(PREDICTIONS_LOG):
        return 0

    # Build lookup: candle open_time HH:MM -> candle data
    candle_map = {}
    for c in candles:
        label = datetime.fromtimestamp(c['open_time'] / 1000, tz=timezone.utc).strftime('%H:%M')
        candle_map[label] = c

    lines = []
    with open(PREDICTIONS_LOG) as f:
        lines = f.readlines()

    updated_count = 0
    now = time.time()

    for i, line in enumerate(lines):
        if 'actual=PENDING' not in line:
            continue

        # Extract target label from line
        target_match = re.search(r'target=(\d{2}:\d{2})', line)
        pred_match = re.search(r'prediction=(UP|DOWN)', line)
        if not target_match or not pred_match:
            continue

        target_label = target_match.group(1)
        prediction = pred_match.group(1)

        # Check if this target candle exists and is closed
        if target_label not in candle_map:
            continue

        candle = candle_map[target_label]
        # The candle closes 300s after it opens
        candle_close_time = candle['open_time'] / 1000 + 300
        if now < candle_close_time:
            continue  # Candle not yet closed

        actual = 'UP' if candle['close'] > candle['open'] else 'DOWN'
        correct = actual == prediction

        lines[i] = line.replace('actual=PENDING', f'actual={actual}')
        lines[i] = lines[i].replace('correct=PENDING', f'correct={correct}')
        updated_count += 1
        print(f"VERIFIED: target={target_label} pred={prediction} actual={actual} correct={correct}")

    if updated_count > 0:
        with open(PREDICTIONS_LOG, 'w') as f:
            f.writelines(lines)

    return updated_count


def log_prediction(window_info, prediction, confidence, reason):
    """Log prediction. Updates existing PENDING entry if prediction changed."""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    new_line = (f"{ts} | window={window_info['window_label']} | "
                f"target={window_info['target_label']} | prediction={prediction} | "
                f"actual=PENDING | correct=PENDING\n")

    if os.path.exists(PREDICTIONS_LOG):
        lines = []
        with open(PREDICTIONS_LOG) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if f"target={window_info['target_label']}" in line and 'actual=PENDING' in line:
                # Check if prediction changed
                if f"prediction={prediction}" in line:
                    print(f"Already predicted {prediction} for target={window_info['target_label']}, no change")
                    return
                # Update with new prediction
                lines[i] = new_line
                with open(PREDICTIONS_LOG, 'w') as f:
                    f.writelines(lines)
                print(f"Updated prediction for target={window_info['target_label']} to {prediction}")
                return

    with open(PREDICTIONS_LOG, 'a') as f:
        f.write(new_line)
    print(f"Logged: {new_line.strip()}")


def print_accuracy():
    """Print accuracy stats from log (read-only, never influences predictions)."""
    if not os.path.exists(PREDICTIONS_LOG):
        return
    correct = 0
    wrong = 0
    pending = 0
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            if 'correct=True' in line:
                correct += 1
            elif 'correct=False' in line:
                wrong += 1
            elif 'actual=PENDING' in line:
                pending += 1
    total = correct + wrong
    if total > 0:
        pct = correct / total * 100
        print(f"ACCURACY: {correct}/{total} ({pct:.0f}%) + {pending} pending")
    else:
        print(f"ACCURACY: no verified predictions yet, {pending} pending")


def main():
    print("=" * 60)
    print("PREDICTOR RUN")
    print("=" * 60)

    # Get window info
    wi = get_window_info()
    print(f"Window: {wi['window_label']} | Target: {wi['target_label']} | "
          f"Secs to deadline: {wi['secs_to_deadline']:.0f}")

    # Load state
    state = load_state()

    # Fetch candles (retry once on failure)
    candles = fetch_candles()
    if not candles:
        print("WARN: First fetch failed, retrying...")
        time.sleep(2)
        candles = fetch_candles()
    if not candles:
        # Fallback: use reversal of last prediction (our best edge)
        last_pred = state.get('prediction', 'UP')
        prediction = 'DOWN' if last_pred == 'UP' else 'UP'
        confidence, reason = 0.5, 'no_data_reversal'
        print(f"WARN: Could not fetch candles, fallback reversal of {last_pred}")
    else:
        # Verify ALL pending predictions in the log
        verified = verify_all_pending(candles)
        if verified:
            print(f"Verified {verified} pending prediction(s)")

        # Make prediction
        prediction, confidence, reason = predict(candles)

    # Show accuracy stats (read-only)
    print_accuracy()

    print(f"PREDICTION: {prediction} (confidence={confidence:.2f}, reason={reason})")

    # Log prediction
    log_prediction(wi, prediction, confidence, reason)

    # Save state
    new_state = {
        'window_start': wi['window_start'],
        'window_label': wi['window_label'],
        'target_label': wi['target_label'],
        'target_open_ts': wi['target_open'],
        'target_close_ts': wi['target_close'],
        'prediction': prediction,
        'confidence': confidence,
        'reason': reason,
        'predicted_at': datetime.now(timezone.utc).isoformat(),
    }
    if candles:
        new_state['last_price'] = candles[-1]['close']
    save_state(new_state)
    print("State saved.")

    return prediction


if __name__ == '__main__':
    main()
