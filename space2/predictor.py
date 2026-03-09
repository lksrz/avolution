#!/usr/bin/env python3
"""BTC/USDT 5-minute candle predictor for Avolution survival loop."""

import json
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

    # Signal 1: Recent momentum (last 3 candles)
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

    # Combine signals
    score = 0.0
    reasons = []

    # Mean reversion on strong streaks
    if streak_down >= 4:
        score += 0.35
        reasons.append('mean_rev_down_streak')
    elif streak_up >= 4:
        score -= 0.35
        reasons.append('mean_rev_up_streak')

    # Momentum (follow trend if not extreme)
    if abs(momentum) < 0.8:
        score += momentum * 0.2
        reasons.append(f'mom={momentum:.2f}')

    # SMA deviation (mean revert)
    if abs(price_vs_sma) > 0.0005:
        score -= price_vs_sma * 80
        reasons.append(f'sma={price_vs_sma:.4f}')

    # RSI signal
    if rsi > 70:
        score -= 0.3  # Overbought, predict down
        reasons.append(f'rsi_ob={rsi:.0f}')
    elif rsi < 30:
        score += 0.3  # Oversold, predict up
        reasons.append(f'rsi_os={rsi:.0f}')
    elif rsi > 60:
        score -= 0.1
    elif rsi < 40:
        score += 0.1

    # EMA crossover
    if abs(ema_diff) > 0.0003:
        score += ema_diff * 50
        reasons.append(f'ema={ema_diff:.4f}')

    # High volume with strong body = continuation
    if vol_ratio > 1.5 and body_ratio > 0.6:
        if last_body > 0:
            score += 0.15
        else:
            score -= 0.15
        reasons.append(f'vol={vol_ratio:.1f}')

    # Final decision
    if score > 0:
        prediction = 'UP'
    elif score < 0:
        prediction = 'DOWN'
    else:
        prediction = 'UP' if closed[-1]['close'] > closed[-1]['open'] else 'DOWN'
        reasons.append('tiebreaker')

    confidence = min(0.5 + abs(score) * 0.25, 0.85)
    reason = '+'.join(reasons) if reasons else 'neutral'

    return prediction, confidence, reason


def verify_previous(candles, state):
    """Check if previous prediction's target candle has closed."""
    if 'target_open_ts' not in state or 'prediction' not in state:
        return None

    target_ts_ms = state['target_open_ts'] * 1000
    target_close_ts = state.get('target_close_ts', state['target_open_ts'] + 300)

    # Only verify if target candle should be closed
    if time.time() < target_close_ts:
        return None

    for c in candles:
        if c['open_time'] == target_ts_ms:
            actual = 'UP' if c['close'] > c['open'] else 'DOWN'
            correct = actual == state['prediction']
            return {
                'window': state.get('window_label', '??'),
                'target': state.get('target_label', '??'),
                'prediction': state['prediction'],
                'actual': actual,
                'correct': correct,
            }
    return None


def update_log_with_result(result):
    """Update the predictions.log with actual result."""
    if not os.path.exists(PREDICTIONS_LOG):
        return

    lines = []
    with open(PREDICTIONS_LOG) as f:
        lines = f.readlines()

    updated = False
    for i, line in enumerate(lines):
        if f"target={result['target']}" in line and 'actual=PENDING' in line:
            line = line.replace('actual=PENDING', f"actual={result['actual']}")
            line = line.replace('correct=PENDING', f"correct={result['correct']}")
            lines[i] = line
            updated = True
            break

    if updated:
        with open(PREDICTIONS_LOG, 'w') as f:
            f.writelines(lines)
        print(f"Updated log: target={result['target']} actual={result['actual']} correct={result['correct']}")


def log_prediction(window_info, prediction, confidence, reason):
    """Append prediction to log."""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = (f"{ts} | window={window_info['window_label']} | "
            f"target={window_info['target_label']} | prediction={prediction} | "
            f"actual=PENDING | correct=PENDING")

    # Check if we already predicted this window
    if os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG) as f:
            for existing in f:
                if f"target={window_info['target_label']}" in existing and 'actual=PENDING' in existing:
                    print(f"Already predicted for target={window_info['target_label']}, skipping log")
                    return

    with open(PREDICTIONS_LOG, 'a') as f:
        f.write(line + '\n')
    print(f"Logged: {line}")


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

    # Fetch candles
    candles = fetch_candles()
    if not candles:
        print("WARN: Could not fetch candles, using fallback prediction")
        prediction, confidence, reason = 'UP', 0.5, 'no_data_fallback'
    else:
        # Verify previous prediction
        result = verify_previous(candles, state)
        if result:
            print(f"VERIFICATION: target={result['target']} pred={result['prediction']} "
                  f"actual={result['actual']} correct={result['correct']}")
            update_log_with_result(result)

        # Make prediction
        prediction, confidence, reason = predict(candles)

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
