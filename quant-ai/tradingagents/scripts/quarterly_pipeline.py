"""Quarterly TradingAgents signal generation and portfolio backtest.

The pipeline is deliberately strict about look-ahead bias:

- signal dates are processed chronologically
- each signal run gets an isolated memory log, so future decisions cannot leak
  into earlier prompts on reruns
- strict mode allows only the market analyst, whose yfinance indicator path is
  filtered to the trade date in stockstats_utils.load_ohlcv
- portfolio weights are applied after the signal date, at the next close
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import yfinance as yf

from strategy_lab import run_strategy
from universe_selection import build_plan_universes


LOOKAHEAD_PRONE_ANALYSTS = {"news", "social", "fundamentals"}


@dataclass(frozen=True)
class RebalancePlan:
    scheduled_date: str
    signal_date: str
    tickers: tuple[str, ...]
    universe_rows: list[dict[str, Any]]
    output_dir: Path
    json_path: Path


def download_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        sorted(set(tickers)),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("No price data returned from yfinance.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if len(set(tickers)) == 1:
        close.columns = [tickers[0]]
    return close.dropna(how="all")


def quarterly_schedule(start_date: str, end_date: str, cadence_months: int) -> list[str]:
    dates = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current = current + pd.DateOffset(months=cadence_months)
    return dates


def previous_trading_day(calendar: pd.DatetimeIndex, scheduled_date: str) -> str:
    candidates = calendar[calendar <= pd.Timestamp(scheduled_date)]
    if len(candidates) == 0:
        raise RuntimeError(f"No trading day found on or before {scheduled_date}")
    return candidates[-1].strftime("%Y-%m-%d")


def next_trading_day(calendar: pd.DatetimeIndex, signal_date: str) -> pd.Timestamp | None:
    candidates = calendar[calendar > pd.Timestamp(signal_date)]
    if len(candidates) == 0:
        return None
    return candidates[0]


def build_rebalance_plan(args: argparse.Namespace) -> list[RebalancePlan]:
    if not args.tickers and args.universe_csv is None and not args.backtest_only:
        raise ValueError("Pass --tickers or --universe-csv when generating signals.")

    scheduled = quarterly_schedule(args.start_date, args.end_date, args.cadence_months)
    if args.max_periods is not None:
        scheduled = scheduled[: args.max_periods]
    cal_start = (pd.Timestamp(args.start_date) - pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    cal_end = (pd.Timestamp(args.end_date) + pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    benchmark_close = download_close([args.benchmark], cal_start, cal_end)
    calendar = benchmark_close.index

    plans = []
    signal_name = args.signal_name or args.name
    signal_dates = [previous_trading_day(calendar, scheduled_date) for scheduled_date in scheduled]
    universe_by_date = {}
    if args.universe_csv:
        universe_by_date = build_plan_universes(
            args.universe_csv,
            signal_dates,
            top_n_per_sector=args.universe_top_n_per_sector,
            max_age_days=args.universe_max_age_days,
            sectors=args.universe_sectors,
            min_market_cap=args.universe_min_market_cap,
            min_volume=args.universe_min_volume,
            min_dollar_volume=args.universe_min_dollar_volume,
        )

    for idx, scheduled_date in enumerate(scheduled):
        signal_date = signal_dates[idx]
        if universe_by_date:
            selected = universe_by_date[signal_date]
            tickers = tuple(selected["ticker"].tolist())
            universe_rows = json.loads(selected.to_json(orient="records", date_format="iso"))
        else:
            tickers = tuple(args.tickers or [])
            universe_rows = []
        period_dir = args.signals_dir / signal_date
        name = f"{signal_name} {signal_date}"
        slug = f"{name.lower().replace(' ', '_')}_{signal_date}"
        plans.append(
            RebalancePlan(
                scheduled_date=scheduled_date,
                signal_date=signal_date,
                tickers=tickers,
                universe_rows=universe_rows,
                output_dir=period_dir,
                json_path=period_dir / f"{slug}.json",
            )
        )
    return plans


def validate_no_lookahead_settings(args: argparse.Namespace) -> None:
    selected = {a.lower() for a in args.analysts}
    risky = selected & LOOKAHEAD_PRONE_ANALYSTS
    if args.strict_no_lookahead and risky:
        names = ", ".join(sorted(risky))
        raise ValueError(
            "Strict no-lookahead mode rejects these analysts because their default "
            f"Yahoo paths can use current snapshots instead of point-in-time data: {names}. "
            "Use --analysts market, or pass --allow-lookahead-prone-data only for exploratory runs."
        )


def strategy_file_is_complete(path: Path, tickers: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    completed = {decision.get("ticker") for decision in payload.get("decisions", [])}
    return set(tickers).issubset(completed)


def generate_signals(args: argparse.Namespace, plans: list[RebalancePlan]) -> list[Path]:
    validate_no_lookahead_settings(args)
    paths = []

    for plan in plans:
        if args.resume and strategy_file_is_complete(plan.json_path, list(plan.tickers)):
            print(f"skip complete {plan.json_path}", flush=True)
            paths.append(plan.json_path)
            continue

        if args.dry_run:
            print(f"dry-run signal {plan.signal_date}: {', '.join(plan.tickers)} -> {plan.json_path}")
            paths.append(plan.json_path)
            continue

        print(
            f"[pipeline] signal {plan.signal_date}: {len(plan.tickers)} ticker -> {plan.json_path}",
            flush=True,
        )
        signal_args = SimpleNamespace(
            name=f"{args.name} {plan.signal_date}",
            tickers=list(plan.tickers),
            trade_date=plan.signal_date,
            provider=args.provider,
            quick_model=args.quick_model,
            deep_model=args.deep_model,
            analysts=args.analysts,
            debate_rounds=args.debate_rounds,
            risk_rounds=args.risk_rounds,
            output_language=args.output_language,
            output_dir=plan.output_dir,
            cache_dir=args.cache_dir,
            checkpoint=False,
            debug=False,
            resume_ticker_logs=args.resume,
        )
        result = run_strategy(signal_args)
        paths.append(Path(result["json_path"]))
        print(f"[pipeline] signal {plan.signal_date}: JSON completato", flush=True)
    return paths


def load_signal_weights(path: Path, mode: str) -> tuple[str, dict[str, float], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weight_key = "target_weight_long_only" if mode == "long_only" else "target_weight_long_short"
    weights = {
        decision["ticker"]: float(decision.get(weight_key, 0.0))
        for decision in payload["decisions"]
    }
    ratings = {decision["ticker"]: decision["rating"] for decision in payload["decisions"]}
    return payload["trade_date"], weights, ratings


def cap_gross_exposure(weights: dict[str, float], cap: float) -> dict[str, float]:
    gross = sum(abs(v) for v in weights.values())
    if gross <= cap or gross == 0:
        return weights
    scale = cap / gross
    return {ticker: weight * scale for ticker, weight in weights.items()}


def tsmom_regime(
    close: pd.DataFrame,
    signal_date: str,
    asset: str,
    lookback_days: int,
    threshold: float,
) -> tuple[bool, float | None]:
    history = close.loc[close.index <= pd.Timestamp(signal_date), asset].dropna()
    if len(history) <= lookback_days:
        return True, None
    momentum = float(history.iloc[-1] / history.iloc[-lookback_days - 1] - 1.0)
    return momentum > threshold, momentum


def tsmom_value(close: pd.DataFrame, asof_date: str, asset: str, lookback_days: int) -> float | None:
    history = close.loc[close.index <= pd.Timestamp(asof_date), asset].dropna()
    if len(history) <= lookback_days:
        return None
    return float(history.iloc[-1] / history.iloc[-lookback_days - 1] - 1.0)


def choose_risk_off_asset(
    close: pd.DataFrame,
    asof_date: str,
    assets: list[str],
    lookback_days: int,
    threshold: float,
    require_positive: bool,
) -> tuple[str | None, dict[str, float | None]]:
    momentums = {
        asset: tsmom_value(close, asof_date, asset, lookback_days)
        for asset in assets
    }
    valid = {asset: mom for asset, mom in momentums.items() if mom is not None}
    if not valid:
        return None, momentums
    best_asset, best_momentum = max(valid.items(), key=lambda item: item[1])
    if require_positive and best_momentum <= threshold:
        return None, momentums
    return best_asset, momentums


def previous_close_date(calendar: pd.DatetimeIndex, current_date: pd.Timestamp) -> pd.Timestamp | None:
    candidates = calendar[calendar < current_date]
    if len(candidates) == 0:
        return None
    return candidates[-1]


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def perf_summary(daily: pd.DataFrame, benchmark: str) -> dict[str, float]:
    strategy_ret = daily["strategy_return"]
    benchmark_ret = daily["benchmark_return"]
    universe_ret = daily["universe_equal_weight_return"] if "universe_equal_weight_return" in daily else None
    equity = daily["strategy_equity"]
    benchmark_equity = daily["benchmark_equity"]
    universe_equity = daily["universe_equal_weight_equity"] if "universe_equal_weight_equity" in daily else None

    ann_factor = 252
    periods = max(len(daily), 1)
    total = float(equity.iloc[-1] - 1.0)
    bench_total = float(benchmark_equity.iloc[-1] - 1.0)
    ann = float((1.0 + total) ** (ann_factor / periods) - 1.0)
    bench_ann = float((1.0 + bench_total) ** (ann_factor / periods) - 1.0)
    vol = float(strategy_ret.std(ddof=0) * math.sqrt(ann_factor))
    bench_vol = float(benchmark_ret.std(ddof=0) * math.sqrt(ann_factor))

    summary = {
        "strategy_total_return": total,
        "benchmark_total_return": bench_total,
        "alpha_total": total - bench_total,
        "strategy_ann_return": ann,
        "benchmark_ann_return": bench_ann,
        "strategy_ann_vol": vol,
        "benchmark_ann_vol": bench_vol,
        "strategy_sharpe_0rf": ann / vol if vol else 0.0,
        "benchmark_sharpe_0rf": bench_ann / bench_vol if bench_vol else 0.0,
        "strategy_max_drawdown": max_drawdown(equity),
        "benchmark_max_drawdown": max_drawdown(benchmark_equity),
        "daily_observations": float(periods),
    }
    if universe_ret is not None and universe_equity is not None:
        universe_total = float(universe_equity.iloc[-1] - 1.0)
        universe_ann = float((1.0 + universe_total) ** (ann_factor / periods) - 1.0)
        universe_vol = float(universe_ret.std(ddof=0) * math.sqrt(ann_factor))
        summary.update(
            {
                "universe_equal_weight_total_return": universe_total,
                "universe_equal_weight_ann_return": universe_ann,
                "universe_equal_weight_ann_vol": universe_vol,
                "universe_equal_weight_sharpe_0rf": universe_ann / universe_vol if universe_vol else 0.0,
                "universe_equal_weight_max_drawdown": max_drawdown(universe_equity),
                "alpha_vs_universe_equal_weight": total - universe_total,
            }
        )
    return summary


def run_portfolio_backtest(args: argparse.Namespace, signal_paths: list[Path]) -> dict[str, Any]:
    existing_paths = [p for p in signal_paths if p.exists()]
    if not existing_paths:
        raise RuntimeError("No generated signal JSON files found for backtest.")

    signals = [load_signal_weights(path, args.mode) for path in sorted(existing_paths)]
    signal_dates = [date for date, _, _ in signals]
    tickers = sorted({ticker for _, weights, _ in signals for ticker in weights})
    risk_off_assets = sorted(set(args.risk_off_assets or []))
    strategy_assets = sorted(set(tickers + risk_off_assets))

    longest_overlay_lookback = max(args.tsmom_lookback_days, args.risk_off_lookback_days)
    overlay_enabled = args.tsmom_filter or bool(risk_off_assets)
    price_buffer_days = max(10, int(longest_overlay_lookback * 1.8) + 20 if overlay_enabled else 10)
    price_start = (pd.Timestamp(min(signal_dates)) - pd.DateOffset(days=price_buffer_days)).strftime("%Y-%m-%d")
    price_end = (pd.Timestamp(args.end_date) + pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    price_tickers = strategy_assets + [args.benchmark]
    if args.tsmom_filter:
        price_tickers.append(args.tsmom_asset)
    close = download_close(price_tickers, price_start, price_end).ffill()
    returns = close.pct_change().fillna(0.0)
    calendar = close.index
    final_exit_date = pd.Timestamp(previous_trading_day(calendar, args.end_date))

    rebalances = []
    daily_parts = []
    previous_weights = {ticker: 0.0 for ticker in strategy_assets}

    for idx, (signal_date, raw_weights, ratings) in enumerate(signals):
        entry_date = next_trading_day(calendar, signal_date)
        if entry_date is None:
            continue

        next_entry_date = None
        if idx + 1 < len(signals):
            next_entry_date = next_trading_day(calendar, signals[idx + 1][0])

        period_end_date = min(next_entry_date, final_exit_date) if next_entry_date is not None else final_exit_date
        period_index = returns.index[(returns.index > entry_date) & (returns.index <= period_end_date)]
        if len(period_index) == 0:
            continue

        weights = cap_gross_exposure(raw_weights, args.gross_exposure_cap)
        base_weights = pd.Series({ticker: weights.get(ticker, 0.0) for ticker in tickers})
        tsmom_risk_on = True
        tsmom_momentum = None
        tsmom_events = []

        if args.tsmom_filter and args.tsmom_update_frequency == "signal":
            tsmom_risk_on, tsmom_momentum = tsmom_regime(
                close,
                signal_date,
                args.tsmom_asset,
                args.tsmom_lookback_days,
                args.tsmom_threshold,
            )
            if not tsmom_risk_on:
                weights = {ticker: weight * args.tsmom_off_exposure for ticker, weight in weights.items()}
            base_weights = pd.Series({ticker: weights.get(ticker, 0.0) for ticker in tickers})

        asset_returns = returns.loc[period_index, strategy_assets].fillna(0.0)
        cost_series = pd.Series(0.0, index=period_index)
        weights_by_day = []
        previous_effective = pd.Series(previous_weights)
        period_turnover = 0.0
        last_tsmom_month = None
        last_risk_off_month = None
        current_scale = 1.0
        current_risk_off_asset = None
        current_risk_off_momentums: dict[str, float | None] = {}

        for day in period_index:
            asof = previous_close_date(calendar, day)
            if args.tsmom_filter and args.tsmom_update_frequency == "monthly":
                month_key = (day.year, day.month)
                if last_tsmom_month != month_key:
                    if asof is not None:
                        tsmom_risk_on, tsmom_momentum = tsmom_regime(
                            close,
                            asof.strftime("%Y-%m-%d"),
                            args.tsmom_asset,
                            args.tsmom_lookback_days,
                            args.tsmom_threshold,
                        )
                        current_scale = 1.0 if tsmom_risk_on else args.tsmom_off_exposure
                        tsmom_events.append(
                            f"{day.strftime('%Y-%m-%d')}:{'' if tsmom_momentum is None else f'{tsmom_momentum:.2%}'}:{current_scale:.0%}"
                        )
                    last_tsmom_month = month_key

            effective_weights = pd.Series(0.0, index=strategy_assets)
            effective_weights.loc[tickers] = base_weights * current_scale

            uninvested_weight = max(0.0, args.gross_exposure_cap - float(effective_weights.abs().sum()))
            if risk_off_assets and uninvested_weight > 0 and asof is not None:
                month_key = (day.year, day.month)
                if (
                    args.risk_off_update_frequency == "daily"
                    or current_risk_off_asset is None
                    or last_risk_off_month != month_key
                ):
                    current_risk_off_asset, current_risk_off_momentums = choose_risk_off_asset(
                        close,
                        asof.strftime("%Y-%m-%d"),
                        risk_off_assets,
                        args.risk_off_lookback_days,
                        args.risk_off_threshold,
                        args.risk_off_require_positive,
                    )
                    last_risk_off_month = month_key
                if current_risk_off_asset:
                    effective_weights.loc[current_risk_off_asset] += uninvested_weight

            turnover_day = float((effective_weights - previous_effective).abs().sum())
            if turnover_day:
                cost_series.loc[day] = turnover_day * args.transaction_cost_bps / 10_000.0
                period_turnover += turnover_day
                previous_effective = effective_weights
            weights_by_day.append(effective_weights)

        weights_frame = pd.DataFrame(weights_by_day, index=period_index)
        strategy_returns = asset_returns.mul(weights_frame, axis=0).sum(axis=1) - cost_series
        aligned_weights = weights_by_day[-1]
        turnover = period_turnover
        cost = float(cost_series.sum())

        period = pd.DataFrame(
            {
                "strategy_return": strategy_returns,
                "benchmark_return": returns.loc[period_index, args.benchmark],
                "universe_equal_weight_return": returns.loc[period_index, sorted(raw_weights)].fillna(0.0).mean(axis=1),
                "signal_date": signal_date,
                "entry_date": entry_date.strftime("%Y-%m-%d"),
            }
        )
        daily_parts.append(period)

        rebalances.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "period_end": period_index[-1].strftime("%Y-%m-%d"),
                "turnover": turnover,
                "transaction_cost": cost,
                "gross_exposure": float(aligned_weights.abs().sum()),
                "tsmom_risk_on": tsmom_risk_on,
                "tsmom_momentum": tsmom_momentum,
                "tsmom_events": ";".join(tsmom_events),
                "risk_off_asset": current_risk_off_asset or "",
                "risk_off_momentums": json.dumps(current_risk_off_momentums),
                **{f"weight_{ticker}": float(aligned_weights.get(ticker, 0.0)) for ticker in tickers},
                **{f"riskoff_weight_{asset}": float(aligned_weights.get(asset, 0.0)) for asset in risk_off_assets},
                **{f"rating_{ticker}": ratings.get(ticker, "") for ticker in tickers},
            }
        )
        previous_weights = dict(aligned_weights)

    if not daily_parts:
        raise RuntimeError("No daily return periods could be built.")

    daily = pd.concat(daily_parts).sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    daily["strategy_equity"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_equity"] = (1.0 + daily["benchmark_return"]).cumprod()
    daily["universe_equal_weight_equity"] = (1.0 + daily["universe_equal_weight_return"]).cumprod()

    summary = perf_summary(daily, args.benchmark)
    summary["rebalances"] = float(len(rebalances))

    args.backtest_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.name.lower().replace(' ', '_')}_{args.start_date}_{args.end_date}_{args.mode}"
    daily_path = args.backtest_dir / f"{prefix}_daily.csv"
    rebalances_path = args.backtest_dir / f"{prefix}_rebalances.csv"
    summary_path = args.backtest_dir / f"{prefix}_summary.json"
    markdown_path = args.backtest_dir / f"{prefix}_report.md"

    daily.to_csv(daily_path, index_label="date")
    pd.DataFrame(rebalances).to_csv(rebalances_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(render_report(args, summary, rebalances), encoding="utf-8")

    return {
        "daily_path": str(daily_path),
        "rebalances_path": str(rebalances_path),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "summary": summary,
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def render_report(args: argparse.Namespace, summary: dict[str, float], rebalances: list[dict[str, Any]]) -> str:
    universe_label = "dynamic point-in-time universe" if args.universe_csv or not args.tickers else ", ".join(args.tickers)
    lines = [
        "# Quarterly TradingAgents Pipeline Backtest",
        "",
        f"- Tickers: {universe_label}",
        f"- Dates: {args.start_date} to {args.end_date}",
        f"- Cadence: every {args.cadence_months} months",
        f"- Mode: {args.mode}",
        f"- Benchmark: {args.benchmark}",
        f"- Execution: next close after signal date",
        f"- Gross exposure cap: {args.gross_exposure_cap:.1%}",
        f"- Transaction cost: {args.transaction_cost_bps:.2f} bps per turnover",
        f"- TSMOM filter: {'on' if args.tsmom_filter else 'off'}",
        f"- TSMOM update frequency: {args.tsmom_update_frequency}",
        f"- Risk-off assets: {', '.join(args.risk_off_assets) if args.risk_off_assets else 'cash'}",
        "",
        "## Performance",
        "",
        f"- Strategy total return: {pct(summary['strategy_total_return'])}",
        f"- Benchmark total return: {pct(summary['benchmark_total_return'])}",
        f"- Total alpha: {pct(summary['alpha_total'])}",
        f"- Equal-weight universe total return: {pct(summary.get('universe_equal_weight_total_return', 0.0))}",
        f"- Alpha vs equal-weight universe: {pct(summary.get('alpha_vs_universe_equal_weight', 0.0))}",
        f"- Strategy annualized return: {pct(summary['strategy_ann_return'])}",
        f"- Strategy annualized vol: {pct(summary['strategy_ann_vol'])}",
        f"- Strategy Sharpe, 0 rf: {summary['strategy_sharpe_0rf']:.2f}",
        f"- Strategy max drawdown: {pct(summary['strategy_max_drawdown'])}",
        "",
        "## Rebalances",
        "",
        "| Signal | Entry | Period End | TSMOM | Gross | Turnover | Cost | Weights / Ratings |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rebalances:
        pairs = []
        active_tickers = sorted(
            key.replace("rating_", "")
            for key, value in row.items()
            if key.startswith("rating_") and value
        )
        for ticker in active_tickers:
            weight = row.get(f"weight_{ticker}", 0.0)
            rating = row.get(f"rating_{ticker}", "")
            pairs.append(f"{ticker}: {weight:.1%} ({rating})")
        for asset in args.risk_off_assets or []:
            weight = row.get(f"riskoff_weight_{asset}", 0.0)
            if weight:
                pairs.append(f"{asset}: {weight:.1%} (risk-off)")
        lines.append(
            f"| {row['signal_date']} | {row['entry_date']} | {row['period_end']} | "
            f"{'' if row.get('tsmom_momentum') is None else pct(row['tsmom_momentum'])} | "
            f"{row['gross_exposure']:.1%} | {row['turnover']:.1%} | "
            f"{row['transaction_cost']:.2%} | {'; '.join(pairs)} |"
        )

    lines.extend(
        [
            "",
            "## Look-Ahead Guards",
            "",
            "- Signals are generated in chronological order.",
            "- Each rebalance date uses an isolated memory log.",
            "- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.",
            "- Portfolio weights start after the signal date, at the next close.",
            "- Current/future rows in technical indicator data are filtered by trade date.",
            "- TSMOM, when enabled, is computed from closes available on or before the signal date.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quarterly TradingAgents pipeline.")
    parser.add_argument("--name", default="Quarterly TA")
    parser.add_argument("--signal-name", help="Existing signal strategy name to read when backtesting variants.")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--cadence-months", type=int, default=3)
    parser.add_argument("--universe-csv", type=Path, help="Point-in-time CSV with date,ticker,sector,market_cap columns.")
    parser.add_argument("--universe-top-n-per-sector", type=int, default=20)
    parser.add_argument("--universe-max-age-days", type=int, default=120)
    parser.add_argument("--universe-sectors", nargs="*")
    parser.add_argument("--universe-min-market-cap", type=float, default=0.0)
    parser.add_argument("--universe-min-volume", type=float, default=0.0)
    parser.add_argument("--universe-min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--quick-model", default="gpt-5.4-mini")
    parser.add_argument("--deep-model", default="gpt-5.4-mini")
    parser.add_argument("--analysts", nargs="+", default=["market"])
    parser.add_argument("--debate-rounds", type=int, default=1)
    parser.add_argument("--risk-rounds", type=int, default=1)
    parser.add_argument("--output-language", default="Italian")
    parser.add_argument("--mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--gross-exposure-cap", type=float, default=1.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--tsmom-filter", action="store_true", help="Scale exposure by a point-in-time time-series momentum regime filter.")
    parser.add_argument("--tsmom-asset", default="SPY")
    parser.add_argument("--tsmom-lookback-days", type=int, default=252)
    parser.add_argument("--tsmom-threshold", type=float, default=0.0)
    parser.add_argument("--tsmom-off-exposure", type=float, default=0.0)
    parser.add_argument("--tsmom-update-frequency", choices=["signal", "monthly"], default="signal")
    parser.add_argument("--risk-off-assets", nargs="*", default=[])
    parser.add_argument("--risk-off-lookback-days", type=int, default=63)
    parser.add_argument("--risk-off-threshold", type=float, default=0.0)
    parser.add_argument("--risk-off-update-frequency", choices=["monthly", "daily"], default="monthly")
    parser.add_argument("--risk-off-require-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--signals-dir", type=Path, default=Path("quarterly_pipeline") / "signals")
    parser.add_argument("--backtest-dir", type=Path, default=Path("quarterly_pipeline") / "backtests")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache") / "quarterly_pipeline")
    parser.add_argument("--max-periods", type=int, help="Limit the number of scheduled rebalance dates.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--signals-only", action="store_true")
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-lookahead-prone-data", dest="strict_no_lookahead", action="store_false")
    parser.set_defaults(strict_no_lookahead=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = build_rebalance_plan(args)

    if args.dry_run:
        for plan in plans:
            print(f"{plan.scheduled_date} -> signal {plan.signal_date} -> {plan.json_path}")

    signal_paths = [plan.json_path for plan in plans]
    if not args.backtest_only:
        signal_paths = generate_signals(args, plans)

    if args.signals_only or args.dry_run:
        return

    result = run_portfolio_backtest(args, signal_paths)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
