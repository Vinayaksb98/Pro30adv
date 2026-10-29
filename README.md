# Nifty Pro Momentum Scanner — Nifty 500

Streamlit app for scanning individual Nifty 500 constituents using daily/intraday OHLCV history from Yahoo Finance. Includes EMA/RSI momentum screening, 20-candle breakout flags, volume confirmation, ATR-based trade planning, charts, and a single-stock illustrative backtest.

## Deploy on Streamlit Community Cloud

1. Upload the contents of this folder to the **root** of a GitHub repository.
2. In Streamlit Community Cloud, choose that repository and set **Main file path** to `app.py`.
3. Deploy. The app fetches the Nifty 500 symbol list from a public CSV mirror on first load and caches it for 24 hours.

## Universe list and data limitations

- The app no longer has a small hard-coded Nifty 50 dictionary. It loads a constituent list with 400+ symbols and defaults to **All Nifty 500**.
- It tries two public CSV mirrors. To pin/update the universe yourself, add `nifty500_symbols.csv` beside `app.py` with a `Symbol` column containing NSE symbols (without `.NS`). A local file with at least 400 valid unique symbols takes precedence.
- Index membership changes. Check the source list against the official NSE/Nifty Indices constituent download after index rebalances.
- Yahoo Finance may not provide data for every symbol, may rename symbols, or may throttle requests. The app reports symbols with unavailable/insufficient data and does not silently count them as scanned.
- A full-universe scan may take several minutes. Daily candles are recommended; Yahoo Finance intraday history has stricter limits.
- The included backtest is illustrative, not a production-grade portfolio backtest: costs, realistic order fills, slippage, taxes, and point-in-time membership are not modeled.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```


## Notes on signals and filters
- `Trend + Momentum` is a descriptive classification (`BULLISH`, `MODERATE`, `NEUTRAL`, or `INSUFFICIENT DATA`), not a guaranteed buy recommendation.
- `Setup Status` separately identifies the scanner's EMA/RSI setup condition.
- The confirmed-breakout filter requires both a 20-candle breakout and volume above the average of the prior 20 completed candles.
- The universe is obtained from public CSV mirrors and may be stale. Check the loaded symbol count and unavailable-data list before interpreting a scan as complete.
- Backtests are educational and do not include transaction costs, slippage, taxes, or survivorship-bias corrections.
