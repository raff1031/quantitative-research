from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf


START_DATE = "2013-01-01"
UNIVERSE_URL = "https://en.wikipedia.org/wiki/S%26P_100"
ARTIFACT_DIR = Path("artifacts/recent_quant_papers")
NOTEBOOK_DIR = Path("output/jupyter-notebook")
COST_BPS_ONE_WAY = 10.0
LONG_SHORT_BUCKET_FRAC = 0.2


PAPER_NOTES = [
    {
        "strategy_id": "same_weekday_momentum",
        "paper": "Same-Weekday Momentum",
        "year": 2024,
        "source": "SSRN",
        "url": "https://ssrn.com/abstract=4806275",
        "translation": (
            "Cross-sectional daily long-short strategy. Rank stocks using average same-weekday "
            "returns over prior same weekdays, skip the most recent four same weekdays, then "
            "trade the next same day from open to close."
        ),
    },
    {
        "strategy_id": "smad_reversal",
        "paper": "Short-Term Moving Average Distance and the Cross-Section of Stock Returns",
        "year": 2025,
        "source": "Financial Analysts Journal",
        "url": "https://rpc.cfainstitute.org/research/financial-analysts-journal/2025/short-term-moving-average-distance-and-the-cross-section-of-stock-returns",
        "translation": (
            "Monthly cross-sectional reversal. Rank stocks by price distance to the trailing 10-day "
            "moving average at month-end; long the most negative distances and short the most "
            "positive distances."
        ),
    },
    {
        "strategy_id": "dispersion_switch",
        "paper": "When Retail Investors Strike: Return Dispersion, Momentum Crashes, and Reversals",
        "year": 2025,
        "source": "SSRN",
        "url": "https://ssrn.com/abstract=5428094",
        "translation": (
            "Monthly regime-switched signal inspired by return-dispersion evidence. Use standard "
            "12-1 momentum when cross-sectional dispersion is calm; switch to 1-month reversal "
            "when dispersion is elevated."
        ),
    },
]


@dataclass
class StrategyResult:
    daily_returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame


def month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.Series(index, index=index).groupby(index.to_period("M")).last().to_list()
    )


def fetch_sp100_constituents() -> pd.DataFrame:
    response = requests.get(UNIVERSE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    constituents = tables[2].copy()
    constituents["Symbol"] = constituents["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return constituents.rename(columns={"Symbol": "ticker", "Name": "name", "Sector": "sector"})


def download_prices(tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys(tickers))
    panel = yf.download(
        tickers,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    fields = {}
    for field in ("Open", "Close"):
        frame = panel[field].copy()
        frame = frame.dropna(how="all").sort_index()
        fields[field.lower()] = frame
    return fields


def build_ls_weights(signal: pd.DataFrame, bucket_frac: float = LONG_SHORT_BUCKET_FRAC) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    if signal.empty:
        return weights
    bucket_count = max(1, int(np.floor(signal.shape[1] * bucket_frac)))
    for dt, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < 2 * bucket_count:
            continue
        longs = valid.nlargest(bucket_count).index
        shorts = valid.nsmallest(bucket_count).index
        weights.loc[dt, longs] = 0.5 / len(longs)
        weights.loc[dt, shorts] = -0.5 / len(shorts)
    return weights


def expand_snapshot_weights(snapshots: pd.DataFrame, dates: pd.DatetimeIndex, lag_days: int = 1) -> pd.DataFrame:
    weights = snapshots.reindex(dates).ffill()
    if lag_days:
        weights = weights.shift(lag_days)
    return weights.fillna(0.0)


def apply_costs(weights: pd.DataFrame, asset_returns: pd.DataFrame, one_way_bps: float = COST_BPS_ONE_WAY) -> StrategyResult:
    weights = weights.reindex(asset_returns.index).fillna(0.0)
    gross = (weights * asset_returns).sum(axis=1).fillna(0.0)
    turnover = 0.5 * weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1) * 0.5)
    costs = turnover * (one_way_bps / 10_000.0)
    net = gross - costs
    return StrategyResult(daily_returns=net, turnover=turnover, weights=weights)


def same_weekday_signal(daily_close_rets: pd.DataFrame, lookback_weeks: int = 52, skip_weeks: int = 4) -> pd.DataFrame:
    signal = pd.DataFrame(index=daily_close_rets.index, columns=daily_close_rets.columns, dtype=float)
    weekdays = pd.Series(daily_close_rets.index.weekday, index=daily_close_rets.index)
    for weekday in range(5):
        mask = weekdays == weekday
        weekday_rets = daily_close_rets.loc[mask]
        weekday_sig = weekday_rets.shift(skip_weeks).rolling(lookback_weeks, min_periods=max(8, lookback_weeks // 4)).mean()
        signal.loc[mask] = weekday_sig.values
    return signal


def smad_signal(close: pd.DataFrame, monthly_idx: pd.DatetimeIndex, window: int = 10) -> pd.DataFrame:
    ma = close.rolling(window, min_periods=window).mean()
    smad = close / ma - 1.0
    return -smad.loc[monthly_idx]


def momentum_signal(close: pd.DataFrame, monthly_idx: pd.DatetimeIndex) -> pd.DataFrame:
    score = close.shift(21) / close.shift(252) - 1.0
    return score.loc[monthly_idx]


def reversal_signal(close: pd.DataFrame, monthly_idx: pd.DatetimeIndex) -> pd.DataFrame:
    score = -(close / close.shift(21) - 1.0)
    return score.loc[monthly_idx]


def dispersion_switch_signal(close: pd.DataFrame, monthly_idx: pd.DatetimeIndex) -> pd.DataFrame:
    daily_rets = close.pct_change()
    dispersion = daily_rets.std(axis=1)
    current_disp = dispersion.rolling(21, min_periods=10).mean()
    baseline = current_disp.rolling(252, min_periods=126).median()
    high_disp = (current_disp > baseline).reindex(monthly_idx)
    mom = momentum_signal(close, monthly_idx)
    rev = reversal_signal(close, monthly_idx)
    combined = mom.copy()
    combined.loc[high_disp.fillna(False)] = rev.loc[high_disp.fillna(False)]
    return combined


def compute_metrics(returns: pd.Series) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {k: np.nan for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "cagr")}
    equity = (1.0 + returns).cumprod()
    ann_return = returns.mean() * 252
    ann_vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_drawdown = (equity / equity.cummax() - 1.0).min()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 252)
    cagr = equity.iloc[-1] ** (1 / years) - 1.0
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "cagr": cagr,
    }


def plot_equity_curves(curves: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    curves.plot(ax=ax, linewidth=1.8)
    ax.set_title("Recent Quant Paper Strategies via Yahoo Finance")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_report(metrics: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Recent Quant Papers to Yahoo Strategies",
        "",
        "This lab translates recent quant-style papers into practical Yahoo Finance strategies.",
        "",
        "## Universe and Caveats",
        "",
        "- Universe: current S&P 100 constituents fetched from Wikipedia and normalized for Yahoo tickers.",
        "- Data source: Yahoo Finance adjusted open/close data via `yfinance`.",
        "- Bias note: this remains exposed to survivorship bias because the universe uses current constituents.",
        "- Costs: 10 bps one-way turnover cost.",
        "- Translation note: these are implementable proxies inspired by the papers, not exact replications of the authors' datasets or portfolio construction conventions.",
        "",
        "## Paper Translations",
        "",
    ]
    for note in PAPER_NOTES:
        lines.extend(
            [
                f"### {note['paper']} ({note['year']}, {note['source']})",
                f"- Source: {note['url']}",
                f"- Strategy translation: {note['translation']}",
                "",
            ]
        )
    lines.extend(["## Metrics", "", metrics.to_markdown(index=False), ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    constituents = fetch_sp100_constituents()
    universe = constituents["ticker"].tolist()
    universe_with_benchmark = universe + ["SPY"]

    prices = download_prices(universe_with_benchmark)
    open_px = prices["open"]
    close_px = prices["close"]
    benchmark = "SPY"

    valid_ratio = close_px[universe].notna().mean()
    keep = valid_ratio[valid_ratio >= 0.80].index.tolist()
    close_px = close_px[keep + [benchmark]]
    open_px = open_px[keep + [benchmark]]
    constituents = constituents.loc[constituents["ticker"].isin(keep)].copy()
    constituents.to_csv(ARTIFACT_DIR / "universe_constituents.csv", index=False)

    stock_close = close_px[keep]
    stock_open = open_px[keep]
    dates = stock_close.index
    monthly_idx = month_end_index(dates)

    oo_rets = stock_open.pct_change().shift(-1)
    intraday_rets = stock_close / stock_open - 1.0
    daily_close_rets = stock_close.pct_change()
    benchmark_oo = open_px[benchmark].pct_change().shift(-1).fillna(0.0)

    # Strategy 1: Same-weekday momentum, signal available before the open and traded intraday.
    swm_sig = same_weekday_signal(daily_close_rets)
    swm_weights = build_ls_weights(swm_sig)
    swm = apply_costs(swm_weights, intraday_rets)

    # Strategy 2: Short-term moving-average distance reversal, monthly snapshots then next-day open-to-open.
    smad_sig = smad_signal(stock_close, monthly_idx)
    smad_snapshots = build_ls_weights(smad_sig)
    smad_weights = expand_snapshot_weights(smad_snapshots, dates, lag_days=1)
    smad = apply_costs(smad_weights, oo_rets)

    # Strategy 3: Dispersion-switched momentum/reversal.
    disp_sig = dispersion_switch_signal(stock_close, monthly_idx)
    disp_snapshots = build_ls_weights(disp_sig)
    disp_weights = expand_snapshot_weights(disp_snapshots, dates, lag_days=1)
    disp = apply_costs(disp_weights, oo_rets)

    strategy_returns = pd.DataFrame(
        {
            "SameWeekdayMomentum": swm.daily_returns,
            "SMADReversal": smad.daily_returns,
            "DispersionSwitch": disp.daily_returns,
            "SPY_OpenToOpen": benchmark_oo,
        }
    ).dropna(how="all")
    equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    equity.to_csv(ARTIFACT_DIR / "equity_curves.csv")
    strategy_returns.to_csv(ARTIFACT_DIR / "daily_returns.csv")

    metrics_rows = []
    for name in strategy_returns.columns:
        stats = compute_metrics(strategy_returns[name].fillna(0.0))
        if name == "SameWeekdayMomentum":
            turnover = swm.turnover.mean() * 252
        elif name == "SMADReversal":
            turnover = smad.turnover.mean() * 252
        elif name == "DispersionSwitch":
            turnover = disp.turnover.mean() * 252
        else:
            turnover = 0.0
        metrics_rows.append({"strategy": name, **stats, "annual_turnover": turnover})
    metrics = pd.DataFrame(metrics_rows).sort_values("sharpe", ascending=False)
    metrics.to_csv(ARTIFACT_DIR / "summary_metrics.csv", index=False)

    plot_equity_curves(equity, ARTIFACT_DIR / "equity_curves.png")
    build_report(metrics, ARTIFACT_DIR / "paper_strategy_report.md")


if __name__ == "__main__":
    main()
