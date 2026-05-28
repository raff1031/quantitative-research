"""Feature-only LLM signal pipeline with anonymized assets and dates.

This script is designed to reduce historical look-ahead leakage from LLM
latent knowledge. The model never sees real tickers, real dates, news, or
fundamental snapshots. It receives only point-in-time numeric features and
returns ratings for anonymous asset IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from tradingagents.llm_clients import create_llm_client

from universe_selection import build_plan_universes


RATING_TO_WEIGHT = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": -0.5,
    "Sell": -1.0,
}

RATING_TO_LONG_ONLY_WEIGHT = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": 0.0,
    "Sell": 0.0,
}

VALID_RATINGS = set(RATING_TO_WEIGHT)


@dataclass(frozen=True)
class SignalPlan:
    period_id: str
    scheduled_date: str
    signal_date: str
    tickers: tuple[str, ...]
    universe_rows: list[dict[str, Any]]
    output_dir: Path
    json_path: Path


def download_ohlcv(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = pd.DataFrame()
        last_error = None
        for attempt in range(4):
            try:
                frame = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    multi_level_index=False,
                )
                if frame.empty:
                    frame = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
                if not frame.empty:
                    break
            except Exception as exc:
                last_error = exc
            time.sleep(1.5 * (attempt + 1))
        if frame.empty:
            raise RuntimeError(f"No data returned for {ticker}: {last_error}")
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        data[ticker] = frame.dropna(how="all")
    return data


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
        raise RuntimeError(f"No trading day on or before {scheduled_date}")
    return candidates[-1].strftime("%Y-%m-%d")


def build_plan(args: argparse.Namespace) -> list[SignalPlan]:
    if not args.tickers and args.universe_csv is None:
        raise ValueError("Pass --tickers or --universe-csv.")

    cal_start = (pd.Timestamp(args.start_date) - pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    cal_end = (pd.Timestamp(args.end_date) + pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    spy = yf.download(args.calendar_asset, start=cal_start, end=cal_end, auto_adjust=True, progress=False)
    if spy.empty:
        raise RuntimeError(f"No calendar data returned for {args.calendar_asset}")
    calendar = spy.index

    plans = []
    scheduled = quarterly_schedule(args.start_date, args.end_date, args.cadence_months)
    if args.max_periods:
        scheduled = scheduled[: args.max_periods]

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

    for idx, scheduled_date in enumerate(scheduled, start=1):
        period_id = f"Period_{idx:03d}"
        signal_date = signal_dates[idx - 1]
        if universe_by_date:
            selected = universe_by_date[signal_date]
            tickers = tuple(selected["ticker"].tolist())
            universe_rows = json.loads(selected.to_json(orient="records", date_format="iso"))
        else:
            tickers = tuple(args.tickers or [])
            universe_rows = []
        output_dir = args.signals_dir / signal_date
        slug = f"{args.name.lower().replace(' ', '_')}_{signal_date}_{signal_date}"
        plans.append(
            SignalPlan(
                period_id=period_id,
                scheduled_date=scheduled_date,
                signal_date=signal_date,
                tickers=tickers,
                universe_rows=universe_rows,
                output_dir=output_dir,
                json_path=output_dir / f"{slug}.json",
            )
        )
    return plans


def rsi(close: pd.Series, window: int = 14) -> float | None:
    delta = close.diff().dropna()
    if len(delta) < window + 1:
        return None
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, math.nan)
    value = 100 - (100 / (1 + rs.iloc[-1]))
    if pd.isna(value):
        return None
    return float(value)


def max_drawdown(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    segment = close.iloc[-window:]
    drawdown = segment / segment.cummax() - 1.0
    return float(drawdown.min())


def pct_change(close: pd.Series, lookback: int) -> float | None:
    if len(close) <= lookback:
        return None
    return float(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0)


def feature_row(frame: pd.DataFrame, signal_date: str) -> dict[str, float | None]:
    history = frame.loc[frame.index <= pd.Timestamp(signal_date)].copy()
    close = history["Close"].dropna()
    volume = history["Volume"].dropna() if "Volume" in history else pd.Series(dtype=float)
    if len(close) < 260:
        return {}

    daily = close.pct_change().dropna()
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    vol63 = daily.iloc[-63:].std(ddof=0) * math.sqrt(252) if len(daily) >= 63 else None

    vol_trend = None
    if len(volume) >= 126:
        vol_fast = volume.iloc[-21:].mean()
        vol_slow = volume.iloc[-126:].mean()
        vol_trend = float(vol_fast / vol_slow - 1.0) if vol_slow else None

    return {
        "ret_1m": pct_change(close, 21),
        "ret_3m": pct_change(close, 63),
        "ret_6m": pct_change(close, 126),
        "ret_12m": pct_change(close, 252),
        "vol_3m_ann": float(vol63) if vol63 is not None else None,
        "drawdown_6m": max_drawdown(close, 126),
        "rsi_14": rsi(close, 14),
        "dist_sma_50": float(close.iloc[-1] / sma50 - 1.0) if not pd.isna(sma50) else None,
        "dist_sma_200": float(close.iloc[-1] / sma200 - 1.0) if not pd.isna(sma200) else None,
        "volume_trend_1m_vs_6m": vol_trend,
    }


def compute_features(args: argparse.Namespace, plans: list[SignalPlan]) -> dict[str, dict[str, Any]]:
    first = pd.Timestamp(plans[0].signal_date) - pd.DateOffset(days=520)
    last = pd.Timestamp(args.end_date) + pd.DateOffset(days=10)
    all_tickers = sorted({ticker for plan in plans for ticker in plan.tickers})
    data = download_ohlcv(all_tickers, first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))

    features: dict[str, dict[str, Any]] = {}
    for plan in plans:
        rows = {}
        asset_to_ticker = {}
        skipped = []
        for ticker in plan.tickers:
            frame = data.get(ticker)
            if frame is None or frame.empty:
                skipped.append({"ticker": ticker, "reason": "missing_price_data"})
                continue
            row = feature_row(frame, plan.signal_date)
            if not row:
                skipped.append({"ticker": ticker, "reason": "insufficient_history"})
                continue
            asset_id = f"Asset_{len(rows) + 1:03d}"
            rows[asset_id] = row
            asset_to_ticker[asset_id] = ticker
        if len(rows) < args.min_assets_per_signal:
            raise RuntimeError(
                f"Only {len(rows)} assets with usable features for {plan.signal_date}; "
                f"minimum is {args.min_assets_per_signal}."
            )
        if skipped:
            print(
                f"[feature-signal] {plan.signal_date}: skipped {len(skipped)} assets without usable PIT features",
                flush=True,
            )
        features[plan.signal_date] = {
            "feature_table": rows,
            "asset_to_ticker": asset_to_ticker,
            "skipped_assets": skipped,
        }
    return features


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_prompt(period_id: str, feature_table: dict[str, dict[str, float | None]]) -> str:
    payload = {
        "period_id": period_id,
        "feature_definitions": {
            "ret_1m/3m/6m/12m": "Trailing close-to-close returns ending at the as-of close.",
            "vol_3m_ann": "Annualized daily volatility over the last 63 trading days.",
            "drawdown_6m": "Worst drawdown over the last 126 trading days.",
            "rsi_14": "14-day RSI.",
            "dist_sma_50/200": "Close divided by SMA minus 1.",
            "volume_trend_1m_vs_6m": "21-day average volume divided by 126-day average volume minus 1.",
        },
        "assets": feature_table,
    }
    return (
        "You are ranking anonymous assets using only the numeric features below. "
        "You do not know the real date, ticker, company, sector, or future returns. "
        "Do not infer external events. Use only this table.\n\n"
        "Return strict JSON only, with this shape:\n"
        "{\"decisions\":[{\"asset_id\":\"Asset_001\",\"rating\":\"Buy|Overweight|Hold|Underweight|Sell\",\"rationale\":\"short feature-based reason\"}]}\n\n"
        f"Feature payload:\n{json.dumps(payload, indent=2)}"
    )


def parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:300]}")
    return json.loads(match.group(0))


def call_llm(args: argparse.Namespace, prompt: str) -> str:
    llm = create_llm_client(args.provider, args.model).get_llm()
    response = llm.invoke(prompt)
    return str(getattr(response, "content", response))


def write_strategy(
    args: argparse.Namespace,
    plan: SignalPlan,
    feature_table: dict[str, dict[str, float | None]],
    asset_to_ticker: dict[str, str],
    skipped_assets: list[dict[str, str]],
    raw_response: str,
    parsed: dict[str, Any],
) -> None:
    decisions = []
    for item in parsed.get("decisions", []):
        asset_id = item.get("asset_id")
        ticker = asset_to_ticker.get(asset_id)
        rating = item.get("rating", "Hold")
        if ticker is None:
            continue
        if rating not in VALID_RATINGS:
            rating = "Hold"
        decisions.append(
            {
                "ticker": ticker,
                "asset_id": asset_id,
                "trade_date": plan.signal_date,
                "rating": rating,
                "target_weight_long_short": RATING_TO_WEIGHT[rating],
                "target_weight_long_only": RATING_TO_LONG_ONLY_WEIGHT[rating],
                "feature_only": True,
                "rationale": item.get("rationale", ""),
                "features": feature_table[asset_id],
                "final_trade_decision": f"**Rating**: {rating}\n\n**Investment Thesis**: {item.get('rationale', '')}",
                "trader_plan": f"**Action**: {'Buy' if RATING_TO_LONG_ONLY_WEIGHT[rating] > 0 else 'Hold'}\n\n**Reasoning**: Feature-only anonymous rating.",
            }
        )

    strategy = {
        "name": f"{args.name} {plan.signal_date}",
        "created_on": date.today().isoformat(),
        "trade_date": plan.signal_date,
        "provider": args.provider,
        "quick_model": args.model,
        "deep_model": args.model,
        "analysts": ["feature_only_anonymous"],
        "rating_to_weight": RATING_TO_WEIGHT,
        "rating_to_long_only_weight": RATING_TO_LONG_ONLY_WEIGHT,
        "lookahead_controls": {
            "anonymized_tickers": True,
            "anonymized_dates": True,
            "tool_calls": False,
            "point_in_time_universe": bool(args.universe_csv),
            "universe_csv": str(args.universe_csv) if args.universe_csv else None,
            "feature_hash": hashlib.sha256(canonical_json(feature_table).encode("utf-8")).hexdigest(),
            "skipped_assets": skipped_assets,
        },
        "decisions": decisions,
    }

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(plan.period_id, feature_table)
    audit = {
        "period_id": plan.period_id,
        "signal_date": plan.signal_date,
        "asset_mapping": asset_to_ticker,
        "universe_rows": plan.universe_rows,
        "skipped_assets": skipped_assets,
        "feature_table": feature_table,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_response": parsed,
    }
    plan.json_path.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
    plan.json_path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def signals_complete(path: Path, tickers: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return {d.get("ticker") for d in payload.get("decisions", [])} >= set(tickers)


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    plans = build_plan(args)
    if args.dry_run:
        for plan in plans:
            print(f"{plan.scheduled_date} -> {plan.signal_date} -> {plan.json_path}")
        return

    features = compute_features(args, plans)
    for plan in plans:
        feature_payload = features[plan.signal_date]
        feature_table = feature_payload["feature_table"]
        asset_to_ticker = feature_payload["asset_to_ticker"]
        skipped_assets = feature_payload["skipped_assets"]
        active_tickers = list(asset_to_ticker.values())
        if args.resume and signals_complete(plan.json_path, active_tickers):
            print(f"skip complete {plan.json_path}", flush=True)
            continue
        prompt = build_prompt(plan.period_id, feature_table)
        print(f"[feature-signal] {plan.signal_date}: LLM anonymous feature rating ({len(active_tickers)} assets)", flush=True)
        raw = call_llm(args, prompt)
        parsed = parse_llm_json(raw)
        write_strategy(args, plan, feature_table, asset_to_ticker, skipped_assets, raw, parsed)
        print(f"[feature-signal] {plan.signal_date}: wrote {plan.json_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anonymous feature-only LLM signals.")
    parser.add_argument("--name", default="Feature Only TA")
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
    parser.add_argument("--min-assets-per-signal", type=int, default=1)
    parser.add_argument("--calendar-asset", default="SPY")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--signals-dir", type=Path, default=Path("feature_pipeline") / "signals")
    parser.add_argument("--max-periods", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
