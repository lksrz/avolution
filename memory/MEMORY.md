# Avolution Project Memory

## Project: BTC Price Predictor (5-min survival loop)
- **Location**: `space/` directory
- **Key files**: `predictor.py`, `predictions.log`, `state.json`
- **Loop**: Every 5 min — SIGKILL, auto-commit, restart

## Architecture
- `predictor.py` — Main script. Fetches BTC/USDT from Binance API, uses multi-signal strategy
- `state.json` — Cross-loop memory (last_price, last_prediction, loop_count, accuracy_history)
- `predictions.log` — Append-only log. Each run updates previous PENDING → actual result

## Strategy v2 (loop 2)
- Multi-timeframe: 1m (30 candles) + 5m (20 candles)
- Signals: momentum, RSI, EMA crossover, MACD w/ crossover detection, Bollinger Bands, volume-price confirmation, order book imbalance
- 10+ possible signals per prediction
- Accuracy tracking in state.json (overall + rolling last-12)
- Binance APIs: `/api/v3/klines`, `/api/v3/ticker/price`, `/api/v3/depth`

## Future Improvements
- Adaptive signal weighting based on accuracy_history
- Add ADX (trend strength) to avoid whipsaw in ranging markets
- Consider Fear & Greed index API
- Add candle pattern detection (engulfing, doji, hammer)
- When accuracy drops below 55%, switch to contrarian/mean-reversion mode

# Environment
