import io
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty Pro Momentum Scanner", page_icon="📈", layout="wide")

# Constituent symbols are fetched at runtime so the app is not permanently limited to a small hard-coded list.
# The source is a public GitHub mirror of NSE Nifty 500 symbols. Put a current nifty500_symbols.csv
# (with a Symbol column) beside app.py to override the online mirror.
SYMBOLS_URLS = [
    "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data/main/nifty500_symbols.csv",
    "https://raw.githubusercontent.com/BKKB20/nse-index-history/main/current_members/NIFTY_500.csv",
]

@st.cache_data(ttl=86400, show_spinner=False)
def load_universe() -> tuple[pd.DataFrame, str]:
    """Load Nifty 500 constituents from local CSV or public mirror; never silently fall back to Nifty 50."""
    local_path = "nifty500_symbols.csv"
    try:
        local = pd.read_csv(local_path)
        col = next((c for c in local.columns if str(c).strip().lower() in {"symbol", "ticker", "tradingsymbol"}), None)
        if col:
            syms = local[col].astype(str).str.strip().str.upper()
            syms = syms[syms.str.match(r"^[A-Z0-9&.-]+$")].drop_duplicates()
            if len(syms) >= 400:
                return pd.DataFrame({"Symbol": syms.tolist()}), f"Local CSV: {local_path}"
    except (FileNotFoundError, pd.errors.ParserError, OSError, ValueError):
        pass

    errors = []
    for url in SYMBOLS_URLS:
        try:
            response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 NiftyScanner/1.0"})
            response.raise_for_status()
            table = pd.read_csv(io.StringIO(response.text))
            col = next((c for c in table.columns if str(c).strip().lower() in {"symbol", "ticker", "tradingsymbol"}), None)
            if col is None:
                # Some CSV mirrors have an unnamed index column plus a Symbol column.
                col = next((c for c in table.columns if "symbol" in str(c).lower()), None)
            if col:
                syms = table[col].astype(str).str.strip().str.upper()
                syms = syms[syms.str.match(r"^[A-Z0-9&.-]+$")].drop_duplicates()
                if len(syms) >= 400:
                    return pd.DataFrame({"Symbol": syms.tolist()}), url
            errors.append(f"{url}: CSV did not contain at least 400 symbols")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not load the Nifty 500 constituent list. Check internet access, or upload a current `nifty500_symbols.csv` beside app.py. Details: " + " | ".join(errors))


def yahoo_ticker(symbol: str) -> str:
    return f"{symbol}.NS"

@st.cache_data(ttl=300, show_spinner=False)
def download_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.loc[:, ~data.columns.duplicated()].copy()
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not set(needed).issubset(data.columns):
            return pd.DataFrame()
        return data[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def download_universe(period: str, interval: str, symbols: tuple[str, ...]) -> tuple[dict, list]:
    """Download in chunks to reduce per-symbol requests and return data plus symbols with no usable data."""
    histories = {}
    failed = []
    chunk_size = 40
    for start in range(0, len(symbols), chunk_size):
        chunk = list(symbols[start:start + chunk_size])
        tickers = [yahoo_ticker(s) for s in chunk]
        try:
            bulk = yf.download(
                tickers=tickers, period=period, interval=interval, auto_adjust=True,
                group_by="ticker", progress=False, threads=True, timeout=25
            )
        except Exception:
            bulk = pd.DataFrame()
        for symbol, ticker in zip(chunk, tickers):
            hist = pd.DataFrame()
            try:
                if not bulk.empty:
                    if isinstance(bulk.columns, pd.MultiIndex):
                        level0 = bulk.columns.get_level_values(0)
                        level1 = bulk.columns.get_level_values(1)
                        if ticker in level0:
                            hist = bulk[ticker].copy()
                        elif ticker in level1:
                            hist = bulk.xs(ticker, axis=1, level=1).copy()
                    elif len(chunk) == 1:
                        hist = bulk.copy()
                if not hist.empty:
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)
                    needed = ["Open", "High", "Low", "Close", "Volume"]
                    if set(needed).issubset(hist.columns):
                        hist = hist[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["High", "Low", "Close"])
                    else:
                        hist = pd.DataFrame()
            except Exception:
                hist = pd.DataFrame()
            # If bulk extraction fails for a ticker, try a single-symbol request once.
            if hist.empty and not bulk.empty:
                hist = download_history(ticker, period, interval)
            if not hist.empty:
                histories[symbol] = hist
            else:
                failed.append(symbol)
    return histories, failed


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    prev_close = close.shift(1)
    tr = pd.concat([(out["High"] - out["Low"]), (out["High"] - prev_close).abs(), (out["Low"] - prev_close).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    out["VolumeAvg20"] = out["Volume"].rolling(20, min_periods=20).mean().shift(1)
    out["PriorHigh20"] = out["High"].rolling(20, min_periods=20).max().shift(1)
    out["Return1D%"] = close.pct_change() * 100
    return out


def evaluate_trend_momentum(row: pd.Series) -> str:
    """Classify trend/momentum without treating every positive trend as a buy signal."""
    close = row.get("Close")
    ema20 = row.get("EMA20")
    ema50 = row.get("EMA50")
    rsi = row.get("RSI14")
    values = [close, ema20, ema50, rsi]
    if any(pd.isna(value) for value in values):
        return "INSUFFICIENT DATA"
    if close > ema20 > ema50 and 55 <= rsi <= 70:
        return "BULLISH"
    if close > ema20:
        return "MODERATE"
    return "NEUTRAL"


def calculate_swing_targets(df: pd.DataFrame, entry_price: float, atr_multiplier: float = 1.5) -> dict:
    required = {"High", "Low", "Close"}
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Historical price data is empty.")
    if not required.issubset(df.columns):
        raise ValueError("Data must contain High, Low, and Close columns.")
    if not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("Entry price must be positive.")
    if not np.isfinite(atr_multiplier) or atr_multiplier <= 0:
        raise ValueError("ATR multiplier must be positive.")
    prices = df[["High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(prices) < 15:
        raise ValueError("At least 15 valid candles are required for ATR.")
    if (prices["Low"] > prices["High"]).any() or (prices["Close"] <= 0).any():
        raise ValueError("Invalid OHLC values detected.")
    prev_close = prices["Close"].shift(1)
    tr = pd.concat([(prices["High"] - prices["Low"]), (prices["High"] - prev_close).abs(), (prices["Low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("Unable to calculate valid ATR.")
    stop = min(float(prices["Low"].tail(5).min()), entry_price - atr * atr_multiplier)
    risk = entry_price - stop
    if risk <= 0:
        raise ValueError("Calculated stop-loss is not below entry price.")
    last10 = prices.tail(10)
    height = float(last10["High"].max() - last10["Low"].min())
    return {"Entry Price": round(float(entry_price), 2), "ATR (14)": round(atr, 2), "ATR Multiplier": atr_multiplier,
            "Stop Loss": round(stop, 2), "Risk Distance (1R)": round(risk, 2),
            "Target 1 (1:2 R:R)": round(entry_price + 2*risk, 2),
            "Target 2 (1:3 R:R)": round(entry_price + 3*risk, 2),
            "Measured Move Target": round(entry_price + height, 2)}


def run_backtest(df: pd.DataFrame, initial_capital: float = 100000.0) -> tuple[pd.DataFrame, dict]:
    data = add_indicators(df).dropna(subset=["EMA20", "EMA50", "RSI14"]).copy()
    if len(data) < 5:
        raise ValueError("Not enough historical candles for backtesting.")
    raw = (data["EMA20"] > data["EMA50"]) & data["RSI14"].between(50, 70)
    position = raw.shift(1).fillna(False).astype(bool)
    daily_returns = data["Close"].pct_change().fillna(0)
    strategy_returns = daily_returns * position.astype(float)
    equity = initial_capital * (1 + strategy_returns).cumprod()
    data["Position"], data["Strategy Return"], data["Equity"] = position.astype(int), strategy_returns, equity
    data["Buy & Hold Equity"] = initial_capital * (1 + daily_returns).cumprod()
    trades = int((position & ~position.shift(1).fillna(False)).sum())
    peak = equity.cummax()
    stats = {"Strategy Return %": round(float((equity.iloc[-1]/initial_capital-1)*100), 2),
             "Buy & Hold Return %": round(float((data["Buy & Hold Equity"].iloc[-1]/initial_capital-1)*100), 2),
             "Max Drawdown %": round(float(((equity/peak)-1).min()*100), 2), "Entry Signals": trades,
             "Ending Capital": round(float(equity.iloc[-1]), 2)}
    return data, stats

st.title("📈 Nifty Pro Momentum Scanner — Nifty 500")
st.caption("Scans individual NSE-listed constituents, not just the Nifty index ETF. Yahoo Finance data can be delayed or unavailable for some symbols.")

try:
    universe_df, universe_source = load_universe()
    symbols_all = universe_df["Symbol"].tolist()
    universe = {s: yahoo_ticker(s) for s in symbols_all}
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Scanner Settings")
    st.success(f"Loaded {len(universe)} constituent symbols")
    st.caption(f"Universe source: {universe_source}")
    scan_scope = st.radio("Universe", ["All Nifty 500", "Choose stocks"], index=0)
    if scan_scope == "Choose stocks":
        chosen_symbols = st.multiselect("Stocks to scan", symbols_all, default=symbols_all[:10])
    else:
        chosen_symbols = symbols_all
    period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Candle interval", ["1d", "1h", "30m", "15m"], index=0, help="Yahoo Finance limits intraday history; daily candles are recommended for a full-universe scan.")
    rsi_min = st.slider("Minimum RSI", 0, 70, 50)
    rsi_max = st.slider("Maximum RSI", 30, 100, 70)
    volume_filter = st.checkbox("Require volume above prior 20-candle average", value=False)
    confirmed_only = st.checkbox("Show confirmed breakouts only", value=False,
                                 help="Requires both a 20-candle breakout and volume confirmation.")
    scan_btn = st.button("🔍 Run Scanner", type="primary", use_container_width=True)

if rsi_min > rsi_max:
    st.error("Minimum RSI cannot exceed maximum RSI.")
    st.stop()
if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()
if "scan_failed" not in st.session_state:
    st.session_state.scan_failed = []

if scan_btn:
    if not chosen_symbols:
        st.warning("Select at least one stock.")
    else:
        status = st.status(f"Downloading data for {len(chosen_symbols)} stocks...", expanded=True)
        with st.spinner("Yahoo Finance is retrieving price history. A full-universe scan can take several minutes."):
            histories, failed = download_universe(period, interval, tuple(chosen_symbols))
        status.write(f"Usable price histories: {len(histories)} / {len(chosen_symbols)}")
        rows = []
        for i, (symbol, hist) in enumerate(histories.items(), start=1):
            if len(hist) < 20:
                failed.append(symbol)
                continue
            ind = add_indicators(hist).dropna(subset=["EMA20", "EMA50", "RSI14"])
            if ind.empty:
                failed.append(symbol)
                continue
            last = ind.iloc[-1]
            breakout = bool(pd.notna(last["PriorHigh20"]) and last["Close"] > last["PriorHigh20"])
            volume_ok = bool(pd.notna(last["VolumeAvg20"]) and last["Volume"] > last["VolumeAvg20"])
            trend = bool(last["Close"] > last["EMA20"] > last["EMA50"])
            momentum = bool(rsi_min <= last["RSI14"] <= rsi_max)
            passed = trend and momentum and (volume_ok if volume_filter else True)
            trend_momentum = evaluate_trend_momentum(last)
            try:
                targets = calculate_swing_targets(hist, float(last["Close"]))
            except ValueError:
                targets = {}
            rows.append({"Symbol": symbol, "Ticker": universe[symbol], "Close": round(float(last["Close"]), 2),
                "1D Change %": round(float(last["Return1D%"]), 2) if pd.notna(last["Return1D%"]) else np.nan,
                "RSI (14)": round(float(last["RSI14"]), 2), "EMA20": round(float(last["EMA20"]), 2), "EMA50": round(float(last["EMA50"]), 2),
                "Trend + Momentum": trend_momentum, "Setup Status": "BUY SETUP" if passed else "—",
                "20-Candle Breakout": "YES" if breakout else "NO",
                "Volume Confirmed": "YES" if volume_ok else "NO", "Stop Loss": targets.get("Stop Loss", np.nan),
                "Target 1 (2R)": targets.get("Target 1 (1:2 R:R)", np.nan), "Target 2 (3R)": targets.get("Target 2 (1:3 R:R)", np.nan)})
        st.session_state.scan_results = pd.DataFrame(rows)
        st.session_state.scan_failed = sorted(set(failed))
        status.update(label="Scan complete", state="complete", expanded=False)

results = st.session_state.scan_results
if not results.empty:
    st.subheader("Scanner Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requested", len(chosen_symbols) if "chosen_symbols" in locals() else len(results) + len(st.session_state.scan_failed))
    c2.metric("Stocks with usable data", len(results))
    c3.metric("Momentum setups", int((results["Setup Status"] == "BUY SETUP").sum()))
    c4.metric("20-candle breakouts", int((results["20-Candle Breakout"] == "YES").sum()))
    display_df = results.copy()
    if confirmed_only:
        display_df = display_df[
            (display_df["20-Candle Breakout"].astype(str).str.upper() == "YES")
            & (display_df["Volume Confirmed"].astype(str).str.upper() == "YES")
        ]
    if display_df.empty:
        st.info("No stocks match the selected execution filters.")
    else:
        trend_order = {"BULLISH": 0, "MODERATE": 1, "NEUTRAL": 2, "INSUFFICIENT DATA": 3}
        display_df["_trend_order"] = display_df["Trend + Momentum"].map(trend_order).fillna(4)
        display_df = display_df.sort_values(["_trend_order", "1D Change %"], ascending=[True, False]).drop(columns="_trend_order")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button("Download all scan results CSV", results.to_csv(index=False).encode("utf-8"), "nifty500_momentum_scan.csv", "text/csv")
    failed = st.session_state.scan_failed
    if failed:
        with st.expander(f"Symbols with unavailable/insufficient data ({len(failed)})"):
            st.caption("These symbols were not silently counted as scanned. Some Yahoo Finance symbols may be unsupported, renamed, newly listed, or temporarily unavailable.")
            st.code(", ".join(failed))
else:
    st.info("The app loads the Nifty 500 constituent universe automatically. Click **Run Scanner** to scan all loaded symbols.")

st.divider()
st.subheader("🎯 Individual Stock Analysis & Trade Targets")
col1, col2 = st.columns(2)
with col1:
    analysis_symbol = st.selectbox("Choose stock", symbols_all, index=symbols_all.index("RELIANCE") if "RELIANCE" in symbols_all else 0)
    analysis_period = st.selectbox("Analysis history", ["6mo", "1y", "2y", "5y"], index=1)
with col2:
    entry_override = st.number_input("Entry price (0 = latest close)", min_value=0.0, value=0.0, step=1.0)
    atr_mult = st.slider("ATR stop multiplier", 0.5, 4.0, 1.5, 0.25)
if st.button("Load Analysis", type="primary"):
    hist = download_history(universe[analysis_symbol], analysis_period, "1d")
    if hist.empty:
        st.error("No market data returned. Try again later or choose another symbol.")
    else:
        ind = add_indicators(hist)
        latest = ind.iloc[-1]
        entry = float(entry_override if entry_override > 0 else latest["Close"])
        a, b, c, d = st.columns(4)
        a.metric("Latest close", f"₹{latest['Close']:,.2f}")
        b.metric("RSI (14)", f"{latest['RSI14']:.1f}" if pd.notna(latest["RSI14"]) else "N/A")
        c.metric("EMA20", f"₹{latest['EMA20']:,.2f}")
        d.metric("EMA50", f"₹{latest['EMA50']:,.2f}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ind.index, open=ind["Open"], high=ind["High"], low=ind["Low"], close=ind["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=ind.index, y=ind["EMA20"], name="EMA20", mode="lines"))
        fig.add_trace(go.Scatter(x=ind.index, y=ind["EMA50"], name="EMA50", mode="lines"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        try:
            plan = calculate_swing_targets(hist, entry, atr_mult)
            st.dataframe(pd.DataFrame({"Metric": list(plan.keys()), "Value": list(plan.values())}), hide_index=True, use_container_width=True)
        except ValueError as exc:
            st.warning(str(exc))

st.divider()
st.subheader("🧪 Historical Strategy Backtest")
st.caption("Illustrative EMA20/EMA50 + RSI strategy. Signals are shifted one candle to reduce look-ahead bias. This simplified backtest does not model brokerage, taxes, slippage, realistic intraday fills, or survivorship bias.")
backtest_symbol = st.selectbox("Backtest stock", symbols_all, index=symbols_all.index("RELIANCE") if "RELIANCE" in symbols_all else 0, key="backtest_symbol")
backtest_period = st.selectbox("Backtest period", ["6mo", "1y", "2y", "5y"], index=2, key="backtest_period")
capital = st.number_input("Initial capital (₹)", min_value=1000.0, value=100000.0, step=10000.0)
if st.button("Run Backtest"):
    bt_hist = download_history(universe[backtest_symbol], backtest_period, "1d")
    if bt_hist.empty:
        st.error("No data available for this backtest.")
    else:
        try:
            bt, stats = run_backtest(bt_hist, capital)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Strategy return", f"{stats['Strategy Return %']:.2f}%")
            m2.metric("Buy & hold", f"{stats['Buy & Hold Return %']:.2f}%")
            m3.metric("Max drawdown", f"{stats['Max Drawdown %']:.2f}%")
            m4.metric("Entry signals", str(stats["Entry Signals"]))
            st.metric("Ending capital", f"₹{stats['Ending Capital']:,.2f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], name="Strategy Equity", mode="lines"))
            fig.add_trace(go.Scatter(x=bt.index, y=bt["Buy & Hold Equity"], name="Buy & Hold", mode="lines"))
            fig.update_layout(height=400, yaxis_title="Portfolio value (₹)", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download backtest CSV", bt.to_csv().encode("utf-8"), "nifty_backtest.csv", "text/csv")
        except ValueError as exc:
            st.error(str(exc))

st.caption("Educational research only; not investment advice. A missing Yahoo Finance history is reported and is not counted as a successful scan.")
