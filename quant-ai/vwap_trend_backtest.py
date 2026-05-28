from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


NY_TZ = "America/New_York"
TRADING_DAYS = 252
DEFAULT_TICKERS = ("QQQ", "SPY", "TQQQ", "UPRO", "SSO", "DIA", "IWM")
ARTIFACT_DIR = Path("artifacts/vwap_trend")
DUKASCOPY_TICKER_MAP = {
    "QQQ": "qqqususd",
    "SPY": "spyususd",
    "DIA": "diaususd",
    "IWM": "iwmususd",
    "EEM": "eemususd",
    "EFA": "efaususd",
    "GLD": "gldususd",
    "IBB": "ibbususd",
    "TLT": "tltususd",
    "USO": "usoususd",
    "XLE": "xleususd",
    "XLF": "xlfususd",
    "XLI": "xliususd",
    "XLK": "xlkususd",
    "XLP": "xlpususd",
    "XLU": "xluususd",
    "XLV": "xlvususd",
    "XLY": "xlyususd",
    "XOP": "xopususd",
}
DUKASCOPY_INTERVAL_MAP = {
    "1m": "m1",
    "5m": "m5",
    "15m": "m15",
    "30m": "m30",
    "60m": "h1",
}


@dataclass
class BacktestResult:
    ticker: str
    daily_returns: pd.Series
    daily_equity: pd.Series
    buyhold_returns: pd.Series
    buyhold_equity: pd.Series
    trades: pd.DataFrame
    bars: pd.DataFrame


def flatten_yahoo_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = [col[0] for col in df.columns]
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing Yahoo columns {missing}")
    return df[required].dropna(subset=["Open", "High", "Low", "Close"])


def normalize_intraday_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert(NY_TZ)
    out = out.sort_index()
    out = out.between_time("09:30", "15:59")
    out["session"] = out.index.date
    return out


def drop_incomplete_sessions(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    min_bars = max(5, int((390 / interval_minutes) * 0.8))
    close_label = (pd.Timestamp("2000-01-01 16:00") - pd.Timedelta(minutes=interval_minutes)).time()
    keep_sessions = []
    for session, day in df.groupby("session", sort=True):
        first_time = day.index[0].time()
        last_time = day.index[-1].time()
        has_open = first_time <= datetime.strptime("09:35", "%H:%M").time()
        has_close = last_time >= close_label
        if len(day) >= min_bars and has_open and has_close:
            keep_sessions.append(session)
    return df[df["session"].isin(keep_sessions)].copy()


def yahoo_download_chunk(ticker: str, start: datetime, end: datetime, interval: str, retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                prepost=False,
                progress=False,
                threads=False,
                timeout=30,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(1.0 + attempt)
    else:
        if last_error:
            raise last_error
        df = pd.DataFrame()
    if df.empty:
        return df
    return flatten_yahoo_columns(df, ticker)


def download_yahoo_intraday(
    ticker: str,
    start: str | None,
    end: str | None,
    interval: str,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    interval_minutes = parse_interval_minutes(interval)
    now_utc = datetime.now(timezone.utc)
    end_dt = pd.Timestamp(end).to_pydatetime().replace(tzinfo=timezone.utc) if end else now_utc
    if start:
        start_dt = pd.Timestamp(start).to_pydatetime().replace(tzinfo=timezone.utc)
    elif interval == "1m":
        start_dt = end_dt - timedelta(days=29)
    else:
        start_dt = end_dt - timedelta(days=59)

    frames = []
    if interval == "1m":
        # Yahoo allows 1-minute requests in <=8 day slices and only within roughly 30 days.
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=7), end_dt)
            frames.append(yahoo_download_chunk(ticker, chunk_start, chunk_end, interval))
            chunk_start = chunk_end
            time.sleep(sleep_seconds)
    else:
        frames.append(yahoo_download_chunk(ticker, start_dt, end_dt, interval))

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"{ticker}: Yahoo returned no {interval} data for the requested window")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = normalize_intraday_index(df)
    df = drop_incomplete_sessions(df, interval_minutes)
    if df.empty:
        raise ValueError(f"{ticker}: no complete regular-hours sessions after filtering")
    return df


def load_intraday_csv(path: Path, ticker: str | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?<!^)(?<!\n)(?=\d{13},)", "\n", text)
        from io import StringIO

        df = pd.read_csv(StringIO(text))
    lower = {col.lower(): col for col in df.columns}
    datetime_col = next((lower[name] for name in ("datetime", "timestamp", "date") if name in lower), None)
    if datetime_col is None:
        raise ValueError(f"{path}: expected datetime/timestamp/date column")
    if ticker and "ticker" in lower:
        df = df[df[lower["ticker"]].astype(str).str.upper() == ticker.upper()]
    rename = {}
    for name in ("open", "high", "low", "close", "volume"):
        if name not in lower:
            raise ValueError(f"{path}: missing required column {name}")
        rename[lower[name]] = name.title()
    df = df.rename(columns=rename)
    timestamp_values = df[datetime_col]
    if pd.api.types.is_numeric_dtype(timestamp_values):
        median_abs = timestamp_values.dropna().abs().median()
        unit = "ms" if median_abs > 10_000_000_000 else "s"
        df.index = pd.to_datetime(timestamp_values, unit=unit, utc=True, errors="coerce")
    else:
        df.index = pd.to_datetime(timestamp_values, utc=True, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return normalize_intraday_index(df[["Open", "High", "Low", "Close", "Volume"]])


def npx_command() -> str:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise RuntimeError("npx was not found; install Node.js/npm to use Dukascopy downloads")
    return npx


def dukascopy_instrument_for_ticker(ticker: str) -> str:
    instrument = DUKASCOPY_TICKER_MAP.get(ticker.upper())
    if not instrument:
        supported = ", ".join(sorted(DUKASCOPY_TICKER_MAP))
        raise ValueError(f"{ticker}: no Dukascopy mapping. Supported ETF tickers: {supported}")
    return instrument


def download_dukascopy_intraday(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    out_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    if interval not in DUKASCOPY_INTERVAL_MAP:
        supported = ", ".join(DUKASCOPY_INTERVAL_MAP)
        raise ValueError(f"Dukascopy source supports these intervals: {supported}")
    if not start or not end:
        raise ValueError("Dukascopy source requires --start and --end dates")

    instrument = dukascopy_instrument_for_ticker(ticker)
    duk_interval = DUKASCOPY_INTERVAL_MAP[interval]
    raw_dir = out_dir / "dukascopy_csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"{ticker.upper()}_{instrument}_{interval}_{start}_{end}".replace(":", "-")
    csv_path = raw_dir / f"{file_stem}.csv"
    if refresh or not csv_path.exists():
        command = [
            npx_command(),
            "--yes",
            "dukascopy-node",
            "-i",
            instrument,
            "-from",
            start,
            "-to",
            end,
            "-t",
            duk_interval,
            "-p",
            "bid",
            "-v",
            "-vu",
            "units",
            "-f",
            "csv",
            "-dir",
            str(raw_dir),
            "-fn",
            file_stem,
            "-s",
            "-r",
            "5",
            "-re",
            "-fr",
        ]
        subprocess.run(command, check=True)
    df = load_intraday_csv(csv_path)
    df["source_symbol"] = instrument
    return df


def parse_interval_minutes(interval: str) -> int:
    if not interval.endswith("m"):
        raise ValueError("This VWAP backtest expects minute intervals such as 1m or 5m")
    return int(interval[:-1])


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3.0
    dollar_volume = typical_price * out["Volume"].fillna(0.0)
    grouped = out.groupby("session", sort=False)
    cum_dollar_volume = dollar_volume.groupby(out["session"], sort=False).cumsum()
    cum_volume = grouped["Volume"].cumsum().replace(0, np.nan)
    out["VWAP"] = cum_dollar_volume / cum_volume
    out["VWAP"] = out.groupby("session", sort=False)["VWAP"].ffill()
    return out.dropna(subset=["VWAP"])


def close_position(
    *,
    equity: float,
    position: int,
    shares: int,
    entry_price: float,
    entry_time: pd.Timestamp,
    entry_equity: float,
    entry_commission: float,
    exit_price: float,
    exit_time: pd.Timestamp,
    commission_per_share: float,
    ticker: str,
    trade_rows: list[dict],
) -> tuple[float, int]:
    if position == 0 or shares == 0:
        return equity, 0
    exit_commission = shares * commission_per_share
    gross_pnl = position * shares * (exit_price - entry_price)
    net_pnl = gross_pnl - exit_commission
    equity += net_pnl
    total_commission = entry_commission + exit_commission
    trade_rows.append(
        {
            "ticker": ticker,
            "side": "long" if position > 0 else "short",
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "gross_pnl": gross_pnl,
            "commission": total_commission,
            "net_pnl": gross_pnl - total_commission,
            "return": (gross_pnl - total_commission) / entry_equity if entry_equity else np.nan,
        }
    )
    return equity, 0


def backtest_vwap_trend(
    ticker: str,
    bars: pd.DataFrame,
    initial_capital: float,
    commission_per_share: float,
) -> BacktestResult:
    bars = add_vwap(bars)
    equity = initial_capital
    daily_equity = {}
    daily_returns = {}
    trade_rows: list[dict] = []

    for session, day in bars.groupby("session", sort=True):
        day = day.sort_index()
        if len(day) < 3:
            continue
        start_equity = equity
        position = 0
        shares = 0
        entry_price = math.nan
        entry_time = None
        entry_equity = math.nan
        entry_commission = 0.0

        for i in range(1, len(day)):
            prev = day.iloc[i - 1]
            current = day.iloc[i]
            if prev["Close"] > prev["VWAP"]:
                target = 1
            elif prev["Close"] < prev["VWAP"]:
                target = -1
            else:
                target = position

            if target != position:
                if position != 0:
                    equity, shares = close_position(
                        equity=equity,
                        position=position,
                        shares=shares,
                        entry_price=entry_price,
                        entry_time=entry_time,
                        entry_equity=entry_equity,
                        entry_commission=entry_commission,
                        exit_price=float(current["Open"]),
                        exit_time=current.name,
                        commission_per_share=commission_per_share,
                        ticker=ticker,
                        trade_rows=trade_rows,
                    )
                    position = 0
                if target != 0 and equity > 0:
                    entry_price = float(current["Open"])
                    shares = int(equity // entry_price)
                    if shares > 0:
                        entry_time = current.name
                        entry_equity = equity
                        entry_commission = shares * commission_per_share
                        equity -= entry_commission
                        position = target

        if position != 0:
            last = day.iloc[-1]
            equity, shares = close_position(
                equity=equity,
                position=position,
                shares=shares,
                entry_price=entry_price,
                entry_time=entry_time,
                entry_equity=entry_equity,
                entry_commission=entry_commission,
                exit_price=float(last["Close"]),
                exit_time=last.name + pd.Timedelta(minutes=parse_index_step_minutes(day.index)),
                commission_per_share=commission_per_share,
                ticker=ticker,
                trade_rows=trade_rows,
            )

        daily_equity[pd.Timestamp(session)] = equity
        daily_returns[pd.Timestamp(session)] = equity / start_equity - 1.0 if start_equity else np.nan

    daily_equity_series = pd.Series(daily_equity, name=ticker).sort_index()
    daily_return_series = pd.Series(daily_returns, name=ticker).sort_index()
    buyhold_returns, buyhold_equity = build_buyhold_series(bars, initial_capital)
    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades.insert(0, "trade_id", range(1, len(trades) + 1))
    return BacktestResult(
        ticker=ticker,
        daily_returns=daily_return_series,
        daily_equity=daily_equity_series,
        buyhold_returns=buyhold_returns,
        buyhold_equity=buyhold_equity,
        trades=trades,
        bars=bars,
    )


def parse_index_step_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    return int(deltas.mode().iloc[0]) if not deltas.empty else 1


def build_buyhold_series(bars: pd.DataFrame, initial_capital: float) -> tuple[pd.Series, pd.Series]:
    daily = bars.groupby("session", sort=True).agg(Open=("Open", "first"), Close=("Close", "last"))
    first_open = float(daily["Open"].iloc[0])
    equity = initial_capital * (daily["Close"] / first_open)
    equity.index = pd.to_datetime(equity.index)
    returns = equity.pct_change()
    returns.iloc[0] = float(daily["Close"].iloc[0] / daily["Open"].iloc[0] - 1.0)
    returns.name = "buyhold"
    equity.name = "buyhold"
    return returns, equity


def equity_with_initial(equity: pd.Series, initial_capital: float) -> pd.Series:
    equity = equity.dropna().sort_index()
    if equity.empty:
        return equity
    start_marker = equity.index[0] - pd.Timedelta(days=1)
    return pd.concat([pd.Series([initial_capital], index=[start_marker]), equity])


def max_drawdown(equity: pd.Series, initial_capital: float | None = None) -> float:
    if equity.empty:
        return np.nan
    if initial_capital is not None:
        equity = equity_with_initial(equity, initial_capital)
    return float((equity / equity.cummax() - 1.0).min())


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float(returns.mean() * TRADING_DAYS)


def cagr(equity: pd.Series, initial_capital: float) -> float:
    equity = equity.dropna()
    if equity.empty:
        return np.nan
    start_marker = equity.index[0] - pd.Timedelta(days=1)
    years = max((equity.index[-1] - start_marker).days / 365.25, 1 / TRADING_DAYS)
    return float((equity.iloc[-1] / initial_capital) ** (1 / years) - 1.0)


def summarize_result(result: BacktestResult, initial_capital: float) -> dict:
    returns = result.daily_returns.dropna()
    equity = result.daily_equity.dropna()
    trades = result.trades
    if returns.empty or equity.empty:
        raise ValueError(f"{result.ticker}: no returns generated")

    ann_vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))
    ann_return = annualized_return(returns)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    buyhold_total = result.buyhold_equity.iloc[-1] / initial_capital - 1.0
    strategy_returns_aligned, buyhold_returns_aligned = returns.align(result.buyhold_returns, join="inner")
    beta = np.nan
    alpha = np.nan
    if len(strategy_returns_aligned) > 2 and buyhold_returns_aligned.var(ddof=0) > 0:
        beta = float(np.cov(strategy_returns_aligned, buyhold_returns_aligned, ddof=0)[0, 1] / buyhold_returns_aligned.var(ddof=0))
        alpha = float((strategy_returns_aligned.mean() - beta * buyhold_returns_aligned.mean()) * TRADING_DAYS)

    wins = trades[trades["return"] > 0]["return"] if not trades.empty else pd.Series(dtype=float)
    losses = trades[trades["return"] < 0]["return"] if not trades.empty else pd.Series(dtype=float)
    gain_loss = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan
    return {
        "ticker": result.ticker,
        "start": equity.index[0].date().isoformat(),
        "end": equity.index[-1].date().isoformat(),
        "sessions": int(len(equity)),
        "start_capital": initial_capital,
        "end_capital": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / initial_capital - 1.0),
        "ann_return_mean": ann_return,
        "cagr": cagr(equity, initial_capital),
        "volatility": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown(equity, initial_capital),
        "buyhold_total_return": float(buyhold_total),
        "buyhold_max_drawdown": max_drawdown(result.buyhold_equity, initial_capital),
        "alpha_vs_buyhold": alpha,
        "beta_vs_buyhold": beta,
        "trades": int(len(trades)),
        "shares_traded": int(trades["shares"].sum() * 2) if not trades.empty else 0,
        "commission_paid": float(trades["commission"].sum()) if not trades.empty else 0.0,
        "trade_hit_ratio": float((trades["return"] > 0).mean()) if not trades.empty else np.nan,
        "trade_gain_loss": gain_loss,
        "max_trade_gain": float(trades["return"].max()) if not trades.empty else np.nan,
        "max_trade_loss": float(trades["return"].min()) if not trades.empty else np.nan,
        "max_daily_gain": float(returns.max()),
        "max_daily_loss": float(returns.min()),
    }


def plot_equity(results: list[BacktestResult], out_path: Path, initial_capital: float) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    for result in results:
        normalized = result.daily_equity / initial_capital
        normalized.plot(ax=ax, linewidth=1.7, label=f"{result.ticker} VWAP TT")
        if result.ticker in {"QQQ", "SPY"}:
            (result.buyhold_equity / initial_capital).plot(
                ax=ax,
                linewidth=1.2,
                linestyle="--",
                label=f"{result.ticker} buy-hold",
            )
    ax.set_title("VWAP Trend Trading Validation")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2%}"


def money(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"${value:,.2f}"


def build_report(metrics: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> None:
    lines = [
        "# VWAP Trend Trading Replication",
        "",
        "## Strategy Rules Replicated",
        "",
        "- Data uses regular trading hours only, 9:30-16:00 New York time.",
        "- VWAP resets every session and uses typical price `(high + low + close) / 3` weighted by volume.",
        "- The first possible trade is entered at the next bar after the first completed bar.",
        "- If the prior bar closes above VWAP, the portfolio targets 100% long exposure.",
        "- If the prior bar closes below VWAP, the portfolio targets 100% short exposure.",
        "- The strategy flips only after a completed candle closes across VWAP.",
        "- Any open position is closed at the session close; no overnight exposure is held.",
        f"- Commissions are `{args.commission_per_share}` dollars per share per order, matching the paper's $0.0005 default when unchanged.",
        "",
        "## Data Caveats",
        "",
        "- Yahoo Finance 1-minute data is limited to roughly the last 30 calendar days and must be requested in short chunks.",
        "- The SSRN paper used paid IQFeed/Interactive Brokers 1-minute data from 2018-01-02 through 2023-09-28, so this is a public-data validation, not a full historical reproduction.",
        "- To run the same engine on longer Dukascopy/IB/exported data, use `--csv path.csv` with columns datetime/open/high/low/close/volume and optional ticker.",
        "- To fetch Dukascopy directly, use `--source dukascopy --start YYYY-MM-DD --end YYYY-MM-DD`; raw CSVs are cached under the output folder.",
        "",
        "## Results",
        "",
    ]
    display_cols = [
        "ticker",
        "start",
        "end",
        "sessions",
        "end_capital",
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "buyhold_total_return",
        "trades",
        "trade_hit_ratio",
        "trade_gain_loss",
        "commission_paid",
    ]
    formatted = metrics[display_cols].copy()
    for col in ["total_return", "cagr", "volatility", "max_drawdown", "buyhold_total_return", "trade_hit_ratio"]:
        formatted[col] = formatted[col].map(pct)
    formatted["end_capital"] = formatted["end_capital"].map(money)
    formatted["commission_paid"] = formatted["commission_paid"].map(money)
    formatted["sharpe"] = formatted["sharpe"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    formatted["trade_gain_loss"] = formatted["trade_gain_loss"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    lines.append(formatted.to_markdown(index=False))
    lines.extend(
        [
            "",
            "Short samples can annualize into extreme CAGR/Sharpe values; the most useful columns for this run are total return, drawdown, trade count, hit ratio, and comparison with buy-and-hold over the same sessions.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate the SSRN VWAP Trend Trading strategy.")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS), help="Yahoo tickers to test.")
    parser.add_argument("--interval", default="1m", help="Minute interval, usually 1m or 5m.")
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=None,
        help="Run several minute intervals in one batch, e.g. --intervals 1m 2m 5m 15m.",
    )
    parser.add_argument("--start", default=None, help="Optional UTC-ish start date, e.g. 2026-04-01.")
    parser.add_argument("--end", default=None, help="Optional UTC-ish end date, e.g. 2026-04-28.")
    parser.add_argument("--initial-capital", type=float, default=25_000.0)
    parser.add_argument("--commission-per-share", type=float, default=0.0005)
    parser.add_argument("--source", choices=["yahoo", "dukascopy", "csv"], default="yahoo")
    parser.add_argument("--refresh-data", action="store_true", help="Redownload cached Dukascopy CSVs.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local OHLCV CSV instead of Yahoo.")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT_DIR)
    return parser.parse_args()


def run_one_interval(args: argparse.Namespace, interval: str, out_dir: Path) -> pd.DataFrame:
    parse_interval_minutes(interval)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[BacktestResult] = []
    errors = []
    for ticker in args.tickers:
        try:
            if args.source == "csv" or args.csv:
                if not args.csv:
                    raise ValueError("--source csv requires --csv path.csv")
                bars = load_intraday_csv(args.csv, ticker)
                bars = drop_incomplete_sessions(bars, parse_interval_minutes(interval))
            elif args.source == "dukascopy":
                bars = download_dukascopy_intraday(
                    ticker=ticker,
                    start=args.start,
                    end=args.end,
                    interval=interval,
                    out_dir=out_dir,
                    refresh=args.refresh_data,
                )
            else:
                bars = download_yahoo_intraday(ticker, args.start, args.end, interval)
            result = backtest_vwap_trend(ticker, bars, args.initial_capital, args.commission_per_share)
            results.append(result)
            print(f"{ticker}: {len(result.daily_equity)} sessions, {len(result.trades)} trades, end {result.daily_equity.iloc[-1]:,.2f}")
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            print(f"{ticker}: ERROR {exc}")

    if not results:
        raise ValueError(f"{interval}: no successful backtests")

    metrics = pd.DataFrame([summarize_result(result, args.initial_capital) for result in results])
    metrics.insert(1, "interval", interval)
    metrics_path = out_dir / "metrics.csv"
    trades_path = out_dir / "trades.csv"
    daily_path = out_dir / "daily_returns.csv"
    equity_path = out_dir / "daily_equity.csv"
    errors_path = out_dir / "errors.csv"
    plot_path = out_dir / "equity_curve.png"
    report_path = out_dir / "report.md"

    all_trades = pd.concat([result.trades for result in results if not result.trades.empty], ignore_index=True)
    daily_returns = pd.concat([result.daily_returns.rename(result.ticker) for result in results], axis=1)
    daily_equity = pd.concat([result.daily_equity.rename(result.ticker) for result in results], axis=1)

    metrics.to_csv(metrics_path, index=False)
    all_trades.to_csv(trades_path, index=False)
    daily_returns.to_csv(daily_path, index_label="date")
    daily_equity.to_csv(equity_path, index_label="date")
    if errors:
        pd.DataFrame(errors).to_csv(errors_path, index=False)
    plot_equity(results, plot_path, args.initial_capital)
    old_interval = args.interval
    args.interval = interval
    build_report(metrics, args, report_path)
    args.interval = old_interval

    print(f"Wrote {metrics_path}")
    print(f"Wrote {trades_path}")
    print(f"Wrote {report_path}")
    return metrics


def build_multitimeframe_report(metrics: pd.DataFrame, out_path: Path) -> None:
    display_cols = [
        "interval",
        "ticker",
        "sessions",
        "end_capital",
        "total_return",
        "max_drawdown",
        "buyhold_total_return",
        "trades",
        "trade_hit_ratio",
        "trade_gain_loss",
        "commission_paid",
    ]
    formatted = metrics[display_cols].copy()
    for col in ["total_return", "max_drawdown", "buyhold_total_return", "trade_hit_ratio"]:
        formatted[col] = formatted[col].map(pct)
    formatted["end_capital"] = formatted["end_capital"].map(money)
    formatted["commission_paid"] = formatted["commission_paid"].map(money)
    formatted["trade_gain_loss"] = formatted["trade_gain_loss"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    lines = [
        "# VWAP Trend Multi-Timeframe Comparison",
        "",
        "Each row is the same VWAP trend/reversal engine run on a different candle interval.",
        "",
        formatted.to_markdown(index=False),
        "",
        "Yahoo data availability differs by interval. For the most accurate full-history test, use a local minute CSV and run the same intervals from that source.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_multitimeframe_equity_plots(
    base_out_dir: Path,
    intervals: Iterable[str],
    tickers: Iterable[str],
    initial_capital: float,
) -> None:
    plot_dir = base_out_dir / "equity_by_ticker"
    plot_dir.mkdir(parents=True, exist_ok=True)

    equity_by_interval = {}
    for interval in intervals:
        equity_path = base_out_dir / interval / "daily_equity.csv"
        if equity_path.exists():
            equity_by_interval[interval] = pd.read_csv(equity_path, index_col="date", parse_dates=True)

    if not equity_by_interval:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    for ticker in tickers:
        fig, ax = plt.subplots(figsize=(12, 7))
        plotted = False
        for interval, equity in equity_by_interval.items():
            if ticker not in equity.columns:
                continue
            series = equity[ticker].dropna() / initial_capital
            if series.empty:
                continue
            series.plot(ax=ax, linewidth=1.8, label=interval)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.45)
        ax.set_title(f"{ticker} VWAP Trend Equity by Timeframe")
        ax.set_ylabel("Growth of $1")
        ax.set_xlabel("")
        ax.legend(title="Timeframe", frameon=True, ncol=3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{ticker}_equity_by_timeframe.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    intervals = args.intervals or [args.interval]
    all_metrics = []
    for interval in intervals:
        print(f"\n=== Running {interval} ===")
        out_dir = args.out_dir if len(intervals) == 1 else args.out_dir / interval
        try:
            all_metrics.append(run_one_interval(args, interval, out_dir))
        except Exception as exc:
            print(f"{interval}: skipped ({exc})")

    if len(all_metrics) > 1:
        combined = pd.concat(all_metrics, ignore_index=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        combined_path = args.out_dir / "multi_timeframe_metrics.csv"
        report_path = args.out_dir / "multi_timeframe_report.md"
        combined.to_csv(combined_path, index=False)
        build_multitimeframe_report(combined, report_path)
        build_multitimeframe_equity_plots(args.out_dir, intervals, args.tickers, args.initial_capital)
        print(f"Wrote {combined_path}")
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
