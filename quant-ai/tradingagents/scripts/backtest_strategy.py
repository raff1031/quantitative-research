"""Backtest strategy artifacts produced by scripts/strategy_lab.py.

This is intentionally simple and auditable: each TradingAgents decision is
treated as an independent event, entered after the signal date, and measured
over fixed holding windows versus a benchmark.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Signal:
    strategy_name: str
    ticker: str
    trade_date: str
    rating: str
    weight: float


def load_signals(paths: list[Path], mode: str) -> list[Signal]:
    signals: list[Signal] = []
    weight_key = "target_weight_long_only" if mode == "long_only" else "target_weight_long_short"

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for decision in payload["decisions"]:
            signals.append(
                Signal(
                    strategy_name=payload.get("name", path.stem),
                    ticker=decision["ticker"],
                    trade_date=decision.get("trade_date", payload["trade_date"]),
                    rating=decision["rating"],
                    weight=float(decision.get(weight_key, 0.0)),
                )
            )
    return signals


def download_ohlc(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in sorted(set(tickers)):
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if frame.empty:
            raise RuntimeError(f"No price data returned for {ticker}")
        data[ticker] = frame.dropna(how="all")
    return data


def next_bar_index(index: pd.DatetimeIndex, trade_date: str, execution: str) -> int | None:
    signal_ts = pd.Timestamp(trade_date)
    if execution == "same_close":
        candidates = index[index >= signal_ts]
    else:
        candidates = index[index > signal_ts]
    if len(candidates) == 0:
        return None
    return int(index.get_loc(candidates[0]))


def scalar_price(value: Any) -> float:
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


def run_event_backtest(args: argparse.Namespace) -> dict[str, Any]:
    signals = load_signals(args.strategy_json, args.mode)
    if not signals:
        raise RuntimeError("No decisions found in strategy JSON input.")

    min_date = min(datetime.strptime(s.trade_date, "%Y-%m-%d") for s in signals)
    max_holding = max(args.holding_days)
    data_start = (min_date - timedelta(days=10)).strftime("%Y-%m-%d")
    data_end = args.end_date or (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    tickers = [s.ticker for s in signals] + [args.benchmark]
    data = download_ohlc(tickers, data_start, data_end)

    rows: list[dict[str, Any]] = []
    round_trip_cost = 2.0 * (args.transaction_cost_bps / 10_000.0)

    for signal in signals:
        prices = data[signal.ticker]
        benchmark = data[args.benchmark]
        entry_i = next_bar_index(prices.index, signal.trade_date, args.execution)
        bench_i = next_bar_index(benchmark.index, signal.trade_date, args.execution)
        if entry_i is None or bench_i is None:
            continue

        entry_date = prices.index[entry_i]
        entry_price = scalar_price(prices["Close"].iloc[entry_i])
        bench_entry_price = scalar_price(benchmark["Close"].iloc[bench_i])

        for holding_days in args.holding_days:
            exit_i = min(entry_i + holding_days, len(prices) - 1)
            bench_exit_i = min(bench_i + holding_days, len(benchmark) - 1)
            if exit_i <= entry_i or bench_exit_i <= bench_i:
                continue

            exit_date = prices.index[exit_i]
            exit_price = scalar_price(prices["Close"].iloc[exit_i])
            bench_exit_price = scalar_price(benchmark["Close"].iloc[bench_exit_i])

            asset_return = exit_price / entry_price - 1.0
            strategy_return = signal.weight * asset_return - abs(signal.weight) * round_trip_cost
            benchmark_return = bench_exit_price / bench_entry_price - 1.0

            rows.append(
                {
                    "strategy_name": signal.strategy_name,
                    "ticker": signal.ticker,
                    "rating": signal.rating,
                    "mode": args.mode,
                    "weight": signal.weight,
                    "signal_date": signal.trade_date,
                    "entry_date": entry_date.date().isoformat(),
                    "exit_date": exit_date.date().isoformat(),
                    "holding_days": holding_days,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "asset_return": asset_return,
                    "strategy_return": strategy_return,
                    "benchmark": args.benchmark,
                    "benchmark_return": benchmark_return,
                    "alpha": strategy_return - benchmark_return,
                }
            )

    if not rows:
        raise RuntimeError("No completed backtest rows. Try a shorter holding window or later end date.")

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("holding_days", as_index=False)
        .agg(
            trades=("ticker", "count"),
            avg_strategy_return=("strategy_return", "mean"),
            avg_benchmark_return=("benchmark_return", "mean"),
            avg_alpha=("alpha", "mean"),
            win_rate=("strategy_return", lambda s: float((s > 0).mean())),
            alpha_win_rate=("alpha", lambda s: float((s > 0).mean())),
        )
        .sort_values("holding_days")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "_".join(path.stem for path in args.strategy_json)
    results_path = args.output_dir / f"{stem}_{args.mode}_events.csv"
    summary_path = args.output_dir / f"{stem}_{args.mode}_summary.csv"
    markdown_path = args.output_dir / f"{stem}_{args.mode}_backtest.md"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(render_markdown(args, results, summary), encoding="utf-8")

    return {
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "summary": summary.to_dict(orient="records"),
    }


def render_markdown(args: argparse.Namespace, results: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# TradingAgents Event Backtest",
        "",
        f"- Mode: {args.mode}",
        f"- Execution: {args.execution}",
        f"- Benchmark: {args.benchmark}",
        f"- Transaction cost: {args.transaction_cost_bps:.2f} bps per side",
        "",
        "## Summary",
        "",
        "| Holding Days | Trades | Strategy Return | Benchmark Return | Alpha | Win Rate | Alpha Win Rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| {holding_days:.0f} | {trades:.0f} | {avg_strategy_return:.2%} | "
            "{avg_benchmark_return:.2%} | {avg_alpha:.2%} | {win_rate:.2%} | "
            "{alpha_win_rate:.2%} |".format(**row)
        )

    lines.extend(["", "## Events", ""])
    lines.extend(render_events_table(results))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is an event study, not a full portfolio rebalance simulation.",
            "- Entry is at the adjusted close selected by the execution mode.",
            "- Stops, slippage, borrow costs, and liquidity constraints are not modeled.",
        ]
    )
    return "\n".join(lines)


def render_events_table(results: pd.DataFrame) -> list[str]:
    headers = [
        "Ticker",
        "Rating",
        "Weight",
        "Signal",
        "Entry",
        "Exit",
        "Days",
        "Asset Ret.",
        "Strategy Ret.",
        "Bench Ret.",
        "Alpha",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in results.to_dict(orient="records"):
        values = [
            row["ticker"],
            row["rating"],
            f"{row['weight']:.1%}",
            row["signal_date"],
            row["entry_date"],
            row["exit_date"],
            str(row["holding_days"]),
            f"{row['asset_return']:.2%}",
            f"{row['strategy_return']:.2%}",
            f"{row['benchmark_return']:.2%}",
            f"{row['alpha']:.2%}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest TradingAgents strategy JSON files.")
    parser.add_argument("strategy_json", nargs="+", type=Path)
    parser.add_argument("--mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--holding-days", nargs="+", type=int, default=[5, 20, 60, 120])
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--execution", choices=["next_close", "same_close"], default="next_close")
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--end-date")
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    output = run_event_backtest(parse_args())
    print(json.dumps(output, indent=2))
