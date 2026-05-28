"""Run TradingAgents as a repeatable strategy research harness.

The framework produces prose reports and a final five-tier portfolio rating.
This script turns that output into a concrete strategy artifact:

- one row per ticker with rating, target weights, price target, and rationale
- a Markdown investment memo
- a JSON file suitable for later backtests or dashboards
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


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


def _parse_optional_float(label: str, text: str) -> float | None:
    match = re.search(rf"\*\*{re.escape(label)}\*\*:\s*\$?(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _parse_optional_text(label: str, text: str) -> str | None:
    pattern = rf"\*\*{re.escape(label)}\*\*:\s*(.+?)(?=\n\n(?:\*\*|FINAL TRANSACTION PROPOSAL)|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "llm_provider": args.provider,
            "quick_think_llm": args.quick_model,
            "deep_think_llm": args.deep_model,
            "max_debate_rounds": args.debate_rounds,
            "max_risk_discuss_rounds": args.risk_rounds,
            "output_language": args.output_language,
            "results_dir": str(args.output_dir),
            "data_cache_dir": str(args.cache_dir),
            "memory_log_path": str(args.output_dir / "memory" / "trading_memory.md"),
            "checkpoint_enabled": args.checkpoint,
        }
    )
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    return config


def _strategy_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    slug = f"{args.name.lower().replace(' ', '_')}_{args.trade_date}"
    return args.output_dir / f"{slug}.json", args.output_dir / f"{slug}.md"


def _ticker_state_log_path(args: argparse.Namespace, ticker: str) -> Path:
    safe_ticker = safe_ticker_component(ticker)
    return (
        args.output_dir
        / safe_ticker
        / "TradingAgentsStrategy_logs"
        / f"full_states_log_{args.trade_date}.json"
    )


def _decision_from_state(ticker: str, trade_date: str, state: dict[str, Any]) -> dict[str, Any]:
    final_decision = state["final_trade_decision"]
    trader_plan = state.get("trader_investment_plan") or state.get("trader_investment_decision", "")
    rating = parse_rating(final_decision)
    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "rating": rating,
        "target_weight_long_short": RATING_TO_WEIGHT.get(rating, 0.0),
        "target_weight_long_only": RATING_TO_LONG_ONLY_WEIGHT.get(rating, 0.0),
        "price_target": _parse_optional_float("Price Target", final_decision),
        "time_horizon": _parse_optional_text("Time Horizon", final_decision),
        "entry_price": _parse_optional_float("Entry Price", trader_plan),
        "stop_loss": _parse_optional_float("Stop Loss", trader_plan),
        "position_sizing": _parse_optional_text("Position Sizing", trader_plan),
        "final_trade_decision": final_decision,
        "trader_plan": trader_plan,
    }


def _build_strategy(args: argparse.Namespace, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": args.name,
        "created_on": date.today().isoformat(),
        "trade_date": args.trade_date,
        "provider": args.provider,
        "quick_model": args.quick_model,
        "deep_model": args.deep_model,
        "analysts": args.analysts,
        "rating_to_weight": RATING_TO_WEIGHT,
        "rating_to_long_only_weight": RATING_TO_LONG_ONLY_WEIGHT,
        "decisions": decisions,
    }


def _write_strategy_outputs(args: argparse.Namespace, decisions: list[dict[str, Any]]) -> tuple[Path, Path]:
    strategy = _build_strategy(args, decisions)
    json_path, md_path = _strategy_paths(args)
    json_path.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(strategy), encoding="utf-8")
    return json_path, md_path


def run_strategy(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    load_dotenv(".env.enterprise", override=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(args)
    graph = TradingAgentsGraph(
        selected_analysts=args.analysts,
        debug=args.debug,
        config=config,
    )

    decisions = []
    total = len(args.tickers)
    for idx, ticker in enumerate(args.tickers, start=1):
        state_log_path = _ticker_state_log_path(args, ticker)
        if getattr(args, "resume_ticker_logs", False) and state_log_path.exists():
            print(
                f"[strategy] {args.trade_date} {idx}/{total} {ticker}: resume da log",
                flush=True,
            )
            state = json.loads(state_log_path.read_text(encoding="utf-8"))
        else:
            print(
                f"[strategy] {args.trade_date} {idx}/{total} {ticker}: analisi TradingAgents...",
                flush=True,
            )
            state, _ = graph.propagate(ticker, args.trade_date)

        decisions.append(_decision_from_state(ticker, args.trade_date, state))
        _write_strategy_outputs(args, decisions)
        print(
            f"[strategy] {args.trade_date} {idx}/{total} {ticker}: completato "
            f"({decisions[-1]['rating']})",
            flush=True,
        )

    strategy = _build_strategy(args, decisions)
    json_path, md_path = _write_strategy_outputs(args, decisions)

    return {"strategy": strategy, "json_path": str(json_path), "markdown_path": str(md_path)}


def render_markdown(strategy: dict[str, Any]) -> str:
    lines = [
        f"# {strategy['name']}",
        "",
        f"- Trade date: {strategy['trade_date']}",
        f"- Provider: {strategy['provider']}",
        f"- Models: quick={strategy['quick_model']}, deep={strategy['deep_model']}",
        f"- Analysts: {', '.join(strategy['analysts'])}",
        "",
        "## Portfolio Rules",
        "",
        "- Long/short weights: Buy +100%, Overweight +50%, Hold 0%, Underweight -50%, Sell -100%.",
        "- Long-only weights: Buy 100%, Overweight 50%, Hold/Underweight/Sell 0%.",
        "- Treat every output as a research signal; validate with out-of-sample backtests before trading.",
        "",
        "## Decisions",
        "",
    ]
    for item in strategy["decisions"]:
        lines.extend(
            [
                f"### {item['ticker']} - {item['rating']}",
                "",
                f"- Long/short target weight: {item['target_weight_long_short']:.1%}",
                f"- Long-only target weight: {item['target_weight_long_only']:.1%}",
                f"- Entry price: {item['entry_price'] if item['entry_price'] is not None else 'n/a'}",
                f"- Stop loss: {item['stop_loss'] if item['stop_loss'] is not None else 'n/a'}",
                f"- Price target: {item['price_target'] if item['price_target'] is not None else 'n/a'}",
                f"- Time horizon: {item['time_horizon'] or 'n/a'}",
                "",
                "#### Final Decision",
                "",
                item["final_trade_decision"],
                "",
                "#### Trader Plan",
                "",
                item["trader_plan"],
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a TradingAgents strategy artifact.")
    parser.add_argument("--name", default="TradingAgents Strategy")
    parser.add_argument("--tickers", nargs="+", default=["NVDA"])
    parser.add_argument("--trade-date", default="2024-05-10")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--quick-model", default="deepseek-chat")
    parser.add_argument("--deep-model", default="deepseek-chat")
    parser.add_argument("--analysts", nargs="+", default=["market"])
    parser.add_argument("--debate-rounds", type=int, default=1)
    parser.add_argument("--risk-rounds", type=int, default=1)
    parser.add_argument("--output-language", default="Italian")
    parser.add_argument("--output-dir", type=Path, default=Path("strategy_outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache") / "tradingagents")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume-ticker-logs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run_strategy(parse_args())
    print(json.dumps(result, indent=2))
