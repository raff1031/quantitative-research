"""Build a public-data S&P 500 point-in-time proxy universe CSV.

This is a bootstrap dataset, not an institutional-grade PIT database. It uses:

- historical S&P 500 membership from fja05680/sp500
- current GICS sectors from the same repo's Wikipedia-derived current list
- Yahoo Finance trailing dollar volume as a point-in-time ranking proxy

The output has the columns required by ``--universe-csv``:
date,ticker,sector,market_cap,volume,dollar_volume

``market_cap`` intentionally stores the ranking proxy because free reliable
historical market capitalization is not available from the public sources used
here. The metadata JSON records that caveat.
"""

from __future__ import annotations

import argparse
import json
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


HISTORICAL_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%2801-17-2026%29.csv"
)
CURRENT_CONSTITUENTS_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"


def yahoo_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def quarterly_schedule(start_date: str, end_date: str, cadence_months: int) -> list[pd.Timestamp]:
    dates = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while current <= end:
        dates.append(current)
        current = current + pd.DateOffset(months=cadence_months)
    return dates


def download_csv(url: str) -> pd.DataFrame:
    text = pd.read_csv(url).to_csv(index=False)
    return pd.read_csv(StringIO(text))


def load_membership_snapshots(start_date: str, end_date: str, cadence_months: int) -> pd.DataFrame:
    history = download_csv(HISTORICAL_CONSTITUENTS_URL)
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")

    rows = []
    for scheduled in quarterly_schedule(start_date, end_date, cadence_months):
        eligible = history.loc[history["date"] <= scheduled]
        if eligible.empty:
            raise RuntimeError(f"No S&P 500 membership snapshot on or before {scheduled.date()}")
        snapshot = eligible.iloc[-1]
        tickers = [yahoo_symbol(ticker) for ticker in str(snapshot["tickers"]).split(",") if ticker.strip()]
        for ticker in sorted(set(tickers)):
            rows.append(
                {
                    "date": scheduled.strftime("%Y-%m-%d"),
                    "membership_snapshot_date": snapshot["date"].strftime("%Y-%m-%d"),
                    "ticker": ticker,
                }
            )
    return pd.DataFrame(rows)


def load_sector_map() -> dict[str, str]:
    current = download_csv(CURRENT_CONSTITUENTS_URL)
    return {
        yahoo_symbol(row["Symbol"]): row["GICS Sector"]
        for _, row in current.iterrows()
        if isinstance(row.get("Symbol"), str) and isinstance(row.get("GICS Sector"), str)
    }


def download_ohlcv_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return raw


def extract_ticker_frame(raw: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if single_ticker:
        frame = raw.copy()
    else:
        if not isinstance(raw.columns, pd.MultiIndex) or ticker not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        frame = raw[ticker].copy()
    if frame.empty or "Close" not in frame or "Volume" not in frame:
        return pd.DataFrame()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame[["Close", "Volume"]].dropna(how="all")


def download_ohlcv(tickers: list[str], start: str, end: str, batch_size: int, pause: float) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]
        print(f"[universe] downloading Yahoo prices {offset + 1}-{offset + len(batch)} / {len(tickers)}", flush=True)
        raw = download_ohlcv_batch(batch, start, end)
        single = len(batch) == 1
        for ticker in batch:
            frame = extract_ticker_frame(raw, ticker, single)
            if not frame.empty:
                data[ticker] = frame
        time.sleep(pause)
    return data


def trailing_metrics(frame: pd.DataFrame, asof: str, lookback_days: int) -> tuple[float | None, float | None]:
    history = frame.loc[frame.index <= pd.Timestamp(asof)].dropna()
    if len(history) < max(5, lookback_days // 3):
        return None, None
    tail = history.tail(lookback_days)
    close = tail["Close"]
    volume = tail["Volume"]
    avg_volume = float(volume.mean())
    dollar_volume = float((close * volume).mean())
    return avg_volume, dollar_volume


def build_universe(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    membership = load_membership_snapshots(args.start_date, args.end_date, args.cadence_months)
    sector_map = load_sector_map()
    membership["sector"] = membership["ticker"].map(sector_map).fillna(args.unknown_sector_name)

    tickers = sorted(membership["ticker"].unique())
    price_start = (pd.Timestamp(args.start_date) - pd.DateOffset(days=max(160, args.lookback_days * 3))).strftime("%Y-%m-%d")
    price_end = (pd.Timestamp(args.end_date) + pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    prices = download_ohlcv(tickers, price_start, price_end, args.batch_size, args.pause)

    rows = []
    missing_price = set()
    for _, row in membership.iterrows():
        ticker = row["ticker"]
        frame = prices.get(ticker)
        if frame is None:
            missing_price.add(ticker)
            if args.keep_missing_price:
                rows.append(
                    {
                        "date": row["date"],
                        "ticker": ticker,
                        "sector": row["sector"],
                        "market_cap": 1.0,
                        "volume": 0.0,
                        "dollar_volume": 1.0,
                        "membership_snapshot_date": row["membership_snapshot_date"],
                        "market_cap_proxy": "missing_price_placeholder",
                    }
                )
            continue
        volume, dollar_volume = trailing_metrics(frame, row["date"], args.lookback_days)
        if volume is None or dollar_volume is None or dollar_volume <= 0:
            missing_price.add(ticker)
            if not args.keep_missing_price:
                continue
            volume, dollar_volume = 0.0, 1.0
        rows.append(
            {
                "date": row["date"],
                "ticker": ticker,
                "sector": row["sector"],
                "market_cap": dollar_volume,
                "volume": volume,
                "dollar_volume": dollar_volume,
                "membership_snapshot_date": row["membership_snapshot_date"],
                "market_cap_proxy": f"trailing_{args.lookback_days}d_avg_dollar_volume",
            }
        )

    output = pd.DataFrame(rows)
    metadata = {
        "created_from": {
            "historical_constituents": HISTORICAL_CONSTITUENTS_URL,
            "current_sector_map": CURRENT_CONSTITUENTS_URL,
            "prices": "Yahoo Finance via yfinance",
        },
        "start_date": args.start_date,
        "end_date": args.end_date,
        "cadence_months": args.cadence_months,
        "lookback_days": args.lookback_days,
        "rank_field": "market_cap",
        "rank_field_is_proxy": True,
        "rank_proxy": f"trailing_{args.lookback_days}d_avg_dollar_volume",
        "known_limitations": [
            "Historical membership is public-source and not an official S&P licensed feed.",
            "Sector labels come from the current S&P 500 list; removed historical names may be Unknown.",
            "market_cap is a point-in-time dollar-volume ranking proxy, not true historical market cap.",
            "Yahoo Finance may omit delisted or renamed tickers, so this is not fully survivorship-bias-free.",
        ],
        "rows": int(len(output)),
        "snapshots": int(output["date"].nunique()) if not output.empty else 0,
        "tickers_seen": int(membership["ticker"].nunique()),
        "tickers_with_prices": int(len(prices)),
        "tickers_missing_price": sorted(missing_price),
    }
    return output, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build public S&P 500 PIT proxy universe CSV.")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--cadence-months", type=int, default=3)
    parser.add_argument("--lookback-days", type=int, default=63)
    parser.add_argument("--output", type=Path, default=Path("data") / "universe_pit_public_proxy.csv")
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--unknown-sector-name", default="Unknown")
    parser.add_argument("--keep-missing-price", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, metadata = build_universe(args)
    if output.empty:
        raise RuntimeError("No rows built for public S&P 500 universe.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metadata": str(metadata_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
