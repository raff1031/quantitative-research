"""Point-in-time universe selection helpers.

The selector expects an input table where each row is a snapshot known on
``date``. For a rebalance at ``signal_date`` it only uses rows with
``date <= signal_date``, then uses the latest complete snapshot and ranks
within each sector. This avoids carrying old ticker rows forward forever.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"date", "ticker", "sector", "market_cap"}


def load_universe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Universe CSV not found: {path}. Create it with columns "
            "date,ticker,sector,market_cap and one complete snapshot per date. "
            "See data/universe_pit_example.csv."
        )
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Universe CSV missing required columns: {names}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["sector"] = frame["sector"].astype(str).str.strip()
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    for optional in ("volume", "dollar_volume"):
        if optional in frame.columns:
            frame[optional] = pd.to_numeric(frame[optional], errors="coerce")

    frame = frame.dropna(subset=["date", "ticker", "sector", "market_cap"])
    frame = frame[(frame["ticker"] != "") & (frame["sector"] != "") & (frame["market_cap"] > 0)]
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def select_universe_for_date(
    universe: pd.DataFrame,
    signal_date: str,
    top_n_per_sector: int,
    max_age_days: int | None = None,
    sectors: list[str] | None = None,
    min_market_cap: float = 0.0,
    min_volume: float = 0.0,
    min_dollar_volume: float = 0.0,
) -> pd.DataFrame:
    if top_n_per_sector <= 0:
        raise ValueError("--universe-top-n-per-sector must be greater than zero")

    asof = pd.Timestamp(signal_date)
    historical = universe.loc[universe["date"] <= asof].copy()
    if historical.empty:
        return historical

    snapshot_date = historical["date"].max()
    if max_age_days is not None and snapshot_date < asof - pd.Timedelta(days=max_age_days):
        return historical.iloc[0:0]

    eligible = historical.loc[historical["date"] == snapshot_date].copy()
    if sectors:
        allowed = {sector.strip() for sector in sectors}
        eligible = eligible.loc[eligible["sector"].isin(allowed)]
    if min_market_cap:
        eligible = eligible.loc[eligible["market_cap"] >= min_market_cap]
    if min_volume and "volume" in eligible.columns:
        eligible = eligible.loc[eligible["volume"].fillna(0) >= min_volume]
    if min_dollar_volume and "dollar_volume" in eligible.columns:
        eligible = eligible.loc[eligible["dollar_volume"].fillna(0) >= min_dollar_volume]

    if eligible.empty:
        return eligible

    ranked = eligible.sort_values(["sector", "market_cap", "ticker"], ascending=[True, False, True]).copy()
    ranked["sector_rank"] = ranked.groupby("sector").cumcount() + 1
    selected = ranked.loc[ranked["sector_rank"] <= top_n_per_sector].copy()
    selected["snapshot_date"] = snapshot_date.strftime("%Y-%m-%d")
    selected["signal_date"] = asof.strftime("%Y-%m-%d")
    return selected.sort_values(["sector", "sector_rank", "ticker"]).reset_index(drop=True)


def build_plan_universes(
    universe_csv: Path,
    signal_dates: list[str],
    top_n_per_sector: int,
    max_age_days: int | None = None,
    sectors: list[str] | None = None,
    min_market_cap: float = 0.0,
    min_volume: float = 0.0,
    min_dollar_volume: float = 0.0,
) -> dict[str, pd.DataFrame]:
    universe = load_universe_csv(universe_csv)
    selections: dict[str, pd.DataFrame] = {}
    for signal_date in signal_dates:
        selected = select_universe_for_date(
            universe,
            signal_date,
            top_n_per_sector=top_n_per_sector,
            max_age_days=max_age_days,
            sectors=sectors,
            min_market_cap=min_market_cap,
            min_volume=min_volume,
            min_dollar_volume=min_dollar_volume,
        )
        if selected.empty:
            raise RuntimeError(f"No universe members selected for {signal_date}")
        selections[signal_date] = selected
    return selections
