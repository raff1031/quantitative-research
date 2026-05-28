import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from universe_selection import load_universe_csv, select_universe_for_date  # noqa: E402
import quarterly_pipeline  # noqa: E402


@pytest.mark.unit
def test_universe_selection_uses_latest_snapshot_not_future_rows(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text(
        "\n".join(
            [
                "date,ticker,sector,market_cap",
                "2024-01-03,AAPL,Technology,100",
                "2024-04-03,FUTR,Technology,999",
            ]
        ),
        encoding="utf-8",
    )

    universe = load_universe_csv(path)
    selected = select_universe_for_date(universe, "2024-01-03", top_n_per_sector=5)

    assert selected["ticker"].tolist() == ["AAPL"]
    assert selected["snapshot_date"].unique().tolist() == ["2024-01-03"]


@pytest.mark.unit
def test_universe_selection_takes_top_n_per_sector(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text(
        "\n".join(
            [
                "date,ticker,sector,market_cap",
                "2024-01-03,AAPL,Technology,100",
                "2024-01-03,MSFT,Technology,90",
                "2024-01-03,SMALL,Technology,10",
                "2024-01-03,LLY,Healthcare,80",
                "2024-01-03,PFE,Healthcare,40",
            ]
        ),
        encoding="utf-8",
    )

    universe = load_universe_csv(path)
    selected = select_universe_for_date(universe, "2024-01-04", top_n_per_sector=2)

    assert selected["ticker"].tolist() == ["LLY", "PFE", "AAPL", "MSFT"]
    assert selected["sector_rank"].tolist() == [1, 2, 1, 2]


@pytest.mark.unit
def test_universe_selection_rejects_stale_snapshot(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text("date,ticker,sector,market_cap\n2024-01-03,AAPL,Technology,100\n", encoding="utf-8")

    universe = load_universe_csv(path)
    selected = select_universe_for_date(universe, "2024-06-03", top_n_per_sector=5, max_age_days=30)

    assert selected.empty


@pytest.mark.unit
def test_backtest_equal_weight_uses_period_tickers_not_global_union(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "trade_date": "2024-01-02",
                "decisions": [{"ticker": "AAPL", "rating": "Buy", "target_weight_long_only": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "trade_date": "2024-01-04",
                "decisions": [{"ticker": "MSFT", "rating": "Buy", "target_weight_long_only": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    close = pd.DataFrame(
        {
            "AAPL": [100.0, 100.0, 110.0, 121.0, 121.0],
            "MSFT": [100.0, 100.0, 100.0, 100.0, 120.0],
            "SPY": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    monkeypatch.setattr(quarterly_pipeline, "download_close", lambda *args, **kwargs: close)

    args = SimpleNamespace(
        name="Dynamic Test",
        start_date="2024-01-02",
        end_date="2024-01-08",
        cadence_months=3,
        tickers=[],
        universe_csv=None,
        mode="long_only",
        benchmark="SPY",
        gross_exposure_cap=1.0,
        transaction_cost_bps=0.0,
        tsmom_filter=False,
        tsmom_asset="SPY",
        tsmom_lookback_days=63,
        tsmom_threshold=0.0,
        tsmom_off_exposure=0.0,
        tsmom_update_frequency="signal",
        risk_off_assets=[],
        risk_off_lookback_days=63,
        risk_off_threshold=0.0,
        risk_off_update_frequency="monthly",
        risk_off_require_positive=True,
        backtest_dir=tmp_path / "backtests",
    )

    result = quarterly_pipeline.run_portfolio_backtest(args, [first, second])
    daily = pd.read_csv(result["daily_path"])

    assert daily.loc[daily["date"] == "2024-01-04", "universe_equal_weight_return"].iloc[0] == pytest.approx(0.10)
    assert daily.loc[daily["date"] == "2024-01-05", "universe_equal_weight_return"].iloc[0] == pytest.approx(0.10)
    assert daily.loc[daily["date"] == "2024-01-08", "universe_equal_weight_return"].iloc[0] == pytest.approx(0.20)
