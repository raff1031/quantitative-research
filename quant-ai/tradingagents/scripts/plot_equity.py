"""Render an equity line HTML from a quarterly_pipeline daily CSV.

No plotting dependencies are required. The output is a standalone HTML file
with an inline SVG chart for strategy vs benchmark equity.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


def scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high == low:
        return (out_low + out_high) / 2.0
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def polyline(points: list[tuple[float, float]], color: str) -> str:
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{data}" fill="none" stroke="{color}" stroke-width="2.4" />'


def pct(value: float) -> str:
    return f"{value:.2%}"


def render_equity_html(daily: pd.DataFrame, title: str) -> str:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    width = 1180
    height = 680
    left = 78
    right = 32
    top = 54
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = pd.concat([daily["strategy_equity"], daily["benchmark_equity"]])
    low = float(values.min())
    high = float(values.max())
    pad = (high - low) * 0.08 if high > low else 0.1
    y_low = max(0.0, low - pad)
    y_high = high + pad

    n = len(daily)
    xs = [scale(i, 0, max(n - 1, 1), left, left + plot_w) for i in range(n)]
    strategy_points = [
        (x, scale(float(v), y_low, y_high, top + plot_h, top))
        for x, v in zip(xs, daily["strategy_equity"])
    ]
    benchmark_points = [
        (x, scale(float(v), y_low, y_high, top + plot_h, top))
        for x, v in zip(xs, daily["benchmark_equity"])
    ]

    y_ticks = []
    for i in range(6):
        val = y_low + (y_high - y_low) * i / 5
        y = scale(val, y_low, y_high, top + plot_h, top)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e6e8eb" />'
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{val:.2f}x</text>'
        )

    x_ticks = []
    tick_count = min(8, n)
    for i in range(tick_count):
        idx = round(i * (n - 1) / max(tick_count - 1, 1))
        x = xs[idx]
        label = daily["date"].iloc[idx].strftime("%Y-%m-%d")
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#8b949e" />'
            f'<text x="{x:.2f}" y="{top + plot_h + 26}" text-anchor="middle">{label}</text>'
        )

    rebalance_lines = []
    for signal_date, group in daily.groupby("signal_date", sort=True):
        idx = int(group.index[0])
        positional_idx = daily.index.get_loc(idx)
        x = xs[positional_idx]
        rebalance_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            'stroke="#b6bec8" stroke-dasharray="5 5" stroke-width="1" />'
        )

    strategy_total = float(daily["strategy_equity"].iloc[-1] - 1.0)
    benchmark_total = float(daily["benchmark_equity"].iloc[-1] - 1.0)
    alpha_total = strategy_total - benchmark_total

    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; color: #18212f; background: #f7f8fa; }}
    main {{ max-width: {width}px; margin: 28px auto; padding: 0 18px; }}
    h1 {{ font-size: 24px; margin: 0 0 12px; }}
    .stats {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }}
    .stat {{ background: white; border: 1px solid #dde2e8; border-radius: 8px; padding: 12px 14px; min-width: 165px; }}
    .label {{ color: #657080; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 21px; font-weight: 700; margin-top: 4px; }}
    svg {{ background: white; border: 1px solid #dde2e8; border-radius: 8px; width: 100%; height: auto; }}
    text {{ font-size: 12px; fill: #657080; }}
    .legend text {{ font-size: 14px; fill: #18212f; }}
  </style>
</head>
<body>
<main>
  <h1>{escaped_title}</h1>
  <section class="stats">
    <div class="stat"><div class="label">Strategy Total</div><div class="value">{pct(strategy_total)}</div></div>
    <div class="stat"><div class="label">Benchmark Total</div><div class="value">{pct(benchmark_total)}</div></div>
    <div class="stat"><div class="label">Alpha</div><div class="value">{pct(alpha_total)}</div></div>
  </section>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity line chart">
    <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" />
    {''.join(y_ticks)}
    {''.join(rebalance_lines)}
    <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#8b949e" />
    <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#8b949e" />
    {''.join(x_ticks)}
    {polyline(benchmark_points, "#6b7280")}
    {polyline(strategy_points, "#0f6bff")}
    <g class="legend">
      <line x1="{left}" y1="28" x2="{left + 30}" y2="28" stroke="#0f6bff" stroke-width="3" />
      <text x="{left + 38}" y="33">Strategy</text>
      <line x1="{left + 130}" y1="28" x2="{left + 160}" y2="28" stroke="#6b7280" stroke-width="3" />
      <text x="{left + 168}" y="33">Benchmark</text>
      <line x1="{left + 300}" y1="28" x2="{left + 330}" y2="28" stroke="#b6bec8" stroke-dasharray="5 5" />
      <text x="{left + 338}" y="33">Rebalance</text>
    </g>
  </svg>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot strategy and benchmark equity lines.")
    parser.add_argument("daily_csv", type=Path)
    parser.add_argument("--title", default="TradingAgents Equity Line")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = pd.read_csv(args.daily_csv)
    output = args.output or args.daily_csv.with_name(args.daily_csv.stem + "_equity.html")
    output.write_text(render_equity_html(daily, args.title), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
