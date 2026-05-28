from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_LINK_SCORES = (0, 1, 2)
DEFAULT_FPI = "1"
DEFAULT_HOLD_DAYS = 20


@dataclass
class WrdsExportPaths:
    estimates: Path
    actuals: Path
    adjustments: Path
    links: Path
    crsp_daily: Path
    output_dir: Path
    identifiers: Optional[Path] = None


def _read_csv(path: Path, parse_dates: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    for col in parse_dates or ():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _rename_first_match(df: pd.DataFrame, aliases: dict, table_name: str) -> pd.DataFrame:
    rename_map = {}
    missing = []
    for canonical, options in aliases.items():
        match = next((col for col in options if col in df.columns), None)
        if match is None:
            missing.append(canonical)
            continue
        rename_map[match] = canonical
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")
    return df.rename(columns=rename_map)


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _choose_latest_per_analyst(estimates: pd.DataFrame) -> pd.DataFrame:
    estimates = estimates.copy()
    estimates["analyst_key"] = (
        estimates["analys"].fillna("").astype(str).str.strip()
        + "|"
        + estimates["broker"].fillna("").astype(str).str.strip()
    )
    estimates = estimates.sort_values(
        ["ibes_ticker", "fpedats", "estdats", "analyst_key", "revdats"]
    )
    return estimates.drop_duplicates(
        subset=["ibes_ticker", "fpedats", "estdats", "analyst_key"], keep="last"
    )


def _calendar_asof_lags(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    lag_days: Sequence[int],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    out = []
    for _, group in frame.groupby(list(group_cols), sort=False):
        group = group.sort_values(date_col).copy()
        base = group[[date_col] + list(value_cols)].copy()
        for lag in lag_days:
            target_col = f"_target_{lag}d"
            right_date = f"_matched_{lag}d"
            group[target_col] = group[date_col] - pd.Timedelta(days=lag)
            lagged = base.rename(columns={date_col: right_date})
            renamed_values = {
                col: f"{col}_lag_{lag}d" for col in value_cols if col in lagged.columns
            }
            lagged = lagged.rename(columns=renamed_values)
            group = pd.merge_asof(
                group.sort_values(target_col),
                lagged.sort_values(right_date),
                left_on=target_col,
                right_on=right_date,
                direction="backward",
            )
        out.append(group)
    if not out:
        return frame.iloc[0:0].copy()
    combined = pd.concat(out, ignore_index=True)
    drop_cols = [col for col in combined.columns if col.startswith("_target_") or col.startswith("_matched_")]
    return combined.drop(columns=drop_cols)


def _factor_asof(
    left: pd.DataFrame,
    adjustments: pd.DataFrame,
    date_col: str,
    out_col: str,
) -> pd.DataFrame:
    left = left.copy()
    left["_row_id"] = np.arange(len(left))
    out = []
    for ticker, group in left.groupby("ibes_ticker", sort=False):
        group = group.copy()
        adj = adjustments.loc[adjustments["ibes_ticker"] == ticker, ["spdates", "adj_factor"]].copy()
        missing_dates = group[group[date_col].isna()].copy()
        if not missing_dates.empty:
            missing_dates[out_col] = 1.0
        ready = group[group[date_col].notna()].sort_values(date_col).copy()
        if ready.empty:
            out.append(missing_dates)
            continue
        if adj.empty:
            ready[out_col] = 1.0
        else:
            adj = adj.sort_values("spdates")
            merged = pd.merge_asof(
                ready,
                adj,
                left_on=date_col,
                right_on="spdates",
                direction="backward",
            )
            ready[out_col] = merged["adj_factor"].fillna(1.0).to_numpy()
        out.append(pd.concat([ready, missing_dates], ignore_index=True))
    if not out:
        return left.drop(columns="_row_id")
    combined = pd.concat(out, ignore_index=True).sort_values("_row_id")
    return combined.drop(columns="_row_id").reset_index(drop=True)


def _next_trading_day(signal_dates: pd.Series, trading_dates: pd.Series) -> pd.Series:
    calendar = pd.DataFrame({"trade_date": pd.Index(pd.to_datetime(trading_dates).dropna().unique()).sort_values()})
    if calendar.empty:
        return pd.Series(pd.NaT, index=signal_dates.index, dtype="datetime64[ns]")
    lookup = pd.DataFrame({"signal_date": pd.to_datetime(signal_dates)})
    lookup["_row"] = np.arange(len(lookup))
    ready = lookup[lookup["signal_date"].notna()].copy()
    ready["_target_trade"] = ready["signal_date"] + pd.Timedelta(days=1)
    if ready.empty:
        return pd.Series(pd.NaT, index=signal_dates.index, dtype="datetime64[ns]")
    merged = pd.merge_asof(
        ready.sort_values("_target_trade"),
        calendar,
        left_on="_target_trade",
        right_on="trade_date",
        direction="forward",
    )
    result = lookup[["_row"]].merge(merged[["_row", "trade_date"]], on="_row", how="left")
    return result.sort_values("_row")["trade_date"].reset_index(drop=True)


def _link_permno(
    signals: pd.DataFrame,
    links: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    candidates = signals.copy()
    candidates["_row_id"] = np.arange(len(candidates))
    merged = candidates.merge(links, on="ibes_ticker", how="left")
    valid = merged.loc[
        (merged[date_col] >= merged["link_start"])
        & (
            merged["link_end"].isna()
            | (merged[date_col] <= merged["link_end"])
        )
    ].copy()
    if valid.empty:
        signals = signals.copy()
        signals["permno"] = np.nan
        signals["link_score"] = np.nan
        return signals
    valid = valid.sort_values(["_row_id", "link_score", "link_start"], ascending=[True, True, False])
    best = valid.drop_duplicates(subset="_row_id", keep="first")
    mapped = candidates.merge(
        best[["_row_id", "permno", "link_score"]],
        on="_row_id",
        how="left",
    )
    return mapped.drop(columns="_row_id")


def _prev_close_before_event(events: pd.DataFrame, crsp: pd.DataFrame, date_col: str) -> pd.Series:
    base = events[[date_col]].copy()
    base["_row_id"] = np.arange(len(base))
    eligible = events.copy()
    eligible["_row_id"] = np.arange(len(eligible))
    eligible = eligible.loc[eligible["permno"].notna() & eligible[date_col].notna()].copy()
    out = []
    for permno, group in eligible.groupby("permno", sort=False):
        group = group.sort_values(date_col).copy()
        px = crsp.loc[crsp["permno"] == permno, ["date", "close"]].copy().sort_values("date")
        if px.empty:
            group["prev_close"] = np.nan
        else:
            group["_target_prev_close"] = group[date_col] - pd.Timedelta(days=1)
            merged = pd.merge_asof(
                group.sort_values("_target_prev_close"),
                px,
                left_on="_target_prev_close",
                right_on="date",
                direction="backward",
            )
            group["prev_close"] = merged["close"].to_numpy()
            group = group.drop(columns="_target_prev_close")
        out.append(group[["_row_id", "prev_close"]])
    if not out:
        return pd.Series(np.nan, index=events.index, dtype=float)
    mapped = base.merge(pd.concat(out, ignore_index=True), on="_row_id", how="left")
    return mapped.sort_values("_row_id")["prev_close"].reset_index(drop=True)


def load_estimates(path: Path, fpi: str = DEFAULT_FPI) -> pd.DataFrame:
    df = _read_csv(path, parse_dates=("fpedats", "estdats", "revdats"))
    df = _rename_first_match(
        df,
        {
            "ibes_ticker": ("ticker",),
            "fpedats": ("fpedats",),
            "estdats": ("estdats", "statpers"),
            "forecast_value": ("value", "meanest", "medest"),
            "measure": ("measure",),
            "fpi": ("fpi",),
            "usfirm": ("usfirm",),
            "analys": ("analys",),
            "broker": ("broker",),
            "revdats": ("revdats",),
        },
        table_name="IBES estimates",
    )
    df = _coerce_numeric(df, ("forecast_value", "usfirm"))
    df = df.loc[(df["usfirm"] == 1) & (df["measure"].astype(str).str.upper() == "EPS")].copy()
    if fpi:
        df = df.loc[df["fpi"].astype(str) == str(fpi)].copy()
    df = df.dropna(subset=["ibes_ticker", "fpedats", "estdats", "forecast_value"])
    return _choose_latest_per_analyst(df)


def load_actuals(path: Path) -> pd.DataFrame:
    df = _read_csv(path, parse_dates=("pends", "repdats"))
    df = _rename_first_match(
        df,
        {
            "ibes_ticker": ("ticker",),
            "pends": ("pends", "fpedats"),
            "repdats": ("repdats",),
            "actual_value": ("value", "actual"),
            "usfirm": ("usfirm",),
            "pdicity": ("pdicity",),
        },
        table_name="IBES actuals",
    )
    df = _coerce_numeric(df, ("actual_value", "usfirm"))
    df = df.loc[df["usfirm"] == 1].copy()
    df = df.dropna(subset=["ibes_ticker", "pends", "repdats", "actual_value"])
    df = df.sort_values(["ibes_ticker", "pends", "repdats"]).drop_duplicates(
        subset=["ibes_ticker", "pends", "repdats"], keep="first"
    )
    return df


def load_adjustments(path: Path) -> pd.DataFrame:
    df = _read_csv(path, parse_dates=("spdates",))
    df = _rename_first_match(
        df,
        {
            "ibes_ticker": ("ticker",),
            "spdates": ("spdates",),
            "adj_factor": ("adj",),
            "usfirm": ("usfirm",),
        },
        table_name="IBES adjustments",
    )
    df = _coerce_numeric(df, ("adj_factor", "usfirm"))
    df = df.loc[df["usfirm"] == 1].copy()
    df = df.dropna(subset=["ibes_ticker", "spdates", "adj_factor"])
    return df.sort_values(["ibes_ticker", "spdates"])


def load_links(path: Path, allowed_scores: Sequence[int] = DEFAULT_LINK_SCORES) -> pd.DataFrame:
    df = _read_csv(path, parse_dates=("sdate", "edate", "linkdt", "linkenddt"))
    df = _rename_first_match(
        df,
        {
            "ibes_ticker": ("ticker",),
            "permno": ("permno",),
            "link_start": ("sdate", "linkdt"),
            "link_end": ("edate", "linkenddt"),
            "link_score": ("score",),
        },
        table_name="IBES-CRSP link",
    )
    df = _coerce_numeric(df, ("permno", "link_score"))
    df = df.loc[df["link_score"].isin(list(allowed_scores))].copy()
    return df.dropna(subset=["ibes_ticker", "permno", "link_start"])


def load_crsp_daily(path: Path) -> pd.DataFrame:
    df = _read_csv(path, parse_dates=("date",))
    df = _rename_first_match(
        df,
        {
            "permno": ("permno",),
            "date": ("date",),
            "ret": ("ret",),
            "close": ("prc", "altprc", "close"),
        },
        table_name="CRSP daily",
    )
    optional = {}
    if "dlret" in df.columns:
        optional["dlret"] = "dlret"
    if optional:
        df = df.rename(columns=optional)
    df = _coerce_numeric(df, ("permno", "ret", "close", "dlret"))
    if "dlret" not in df.columns:
        df["dlret"] = np.nan
    gross = (1.0 + df["ret"].fillna(0.0)) * (1.0 + df["dlret"].fillna(0.0)) - 1.0
    missing = df["ret"].isna() & df["dlret"].isna()
    df["ret_total"] = gross.mask(missing)
    df["close"] = df["close"].abs()
    return df.dropna(subset=["permno", "date"]).sort_values(["permno", "date"])


def build_revision_signals(
    estimates: pd.DataFrame,
    links: pd.DataFrame,
    crsp: pd.DataFrame,
    hold_days: int = DEFAULT_HOLD_DAYS,
) -> pd.DataFrame:
    consensus = (
        estimates.groupby(["ibes_ticker", "fpedats", "estdats"], as_index=False)
        .agg(
            medest=("forecast_value", "median"),
            meanest=("forecast_value", "mean"),
            stdev=("forecast_value", "std"),
            n_estimates=("forecast_value", "size"),
        )
        .sort_values(["ibes_ticker", "estdats", "fpedats"])
    )
    # Use the nearest-quarter horizon already enforced through FPI filtering.
    consensus = consensus.drop_duplicates(subset=["ibes_ticker", "estdats"], keep="first")
    consensus = _calendar_asof_lags(
        consensus,
        group_cols=("ibes_ticker",),
        date_col="estdats",
        lag_days=(21, 63),
        value_cols=("medest", "stdev"),
    )
    consensus["rev_21d"] = consensus["medest"] - consensus["medest_lag_21d"]
    consensus["rev_63d"] = consensus["medest"] - consensus["medest_lag_63d"]
    consensus["rev_pct_21d"] = np.where(
        consensus["medest_lag_21d"].abs() > 0,
        consensus["medest"] / consensus["medest_lag_21d"].abs() - 1.0,
        np.nan,
    )
    consensus["dispersion_change_21d"] = consensus["stdev"] - consensus["stdev_lag_21d"]
    consensus["stale_days"] = (
        consensus.groupby("ibes_ticker")["estdats"].diff().dt.days.fillna(np.nan)
    )
    consensus["signal_date"] = consensus["estdats"]
    consensus["trade_date"] = _next_trading_day(consensus["signal_date"], crsp["date"])
    consensus["exit_date_hint"] = consensus["trade_date"] + pd.to_timedelta(hold_days, unit="D")
    consensus = _link_permno(consensus, links, date_col="signal_date")
    return consensus.sort_values(["signal_date", "ibes_ticker"]).reset_index(drop=True)


def build_pead_events(
    estimates: pd.DataFrame,
    actuals: pd.DataFrame,
    adjustments: pd.DataFrame,
    links: pd.DataFrame,
    crsp: pd.DataFrame,
    hold_days: int = DEFAULT_HOLD_DAYS,
) -> pd.DataFrame:
    consensus = (
        estimates.groupby(["ibes_ticker", "fpedats", "estdats"], as_index=False)
        .agg(
            medest=("forecast_value", "median"),
            meanest=("forecast_value", "mean"),
            stdev=("forecast_value", "std"),
            n_estimates=("forecast_value", "size"),
        )
        .sort_values(["ibes_ticker", "fpedats", "estdats"])
    )
    actuals = actuals.rename(columns={"pends": "fpedats"})
    actuals = actuals.sort_values(["ibes_ticker", "fpedats", "repdats"]).copy()
    aligned = []
    for (ticker, fpedats), group in actuals.groupby(["ibes_ticker", "fpedats"], sort=False):
        group = group.sort_values("repdats").copy()
        est = consensus.loc[
            (consensus["ibes_ticker"] == ticker) & (consensus["fpedats"] == fpedats)
        ].copy()
        if est.empty:
            group["estdats"] = pd.NaT
            group["medest"] = np.nan
            group["stdev"] = np.nan
            group["n_estimates"] = np.nan
            aligned.append(group)
            continue
        est = est.sort_values("estdats")
        merged = pd.merge_asof(
            group,
            est[["estdats", "medest", "stdev", "n_estimates"]],
            left_on="repdats",
            right_on="estdats",
            direction="backward",
            allow_exact_matches=False,
        )
        aligned.append(merged)
    pead = pd.concat(aligned, ignore_index=True) if aligned else actuals.iloc[0:0].copy()
    pead = _factor_asof(pead, adjustments, date_col="repdats", out_col="adj_factor_rep")
    pead = _factor_asof(pead, adjustments, date_col="estdats", out_col="adj_factor_est")
    pead["actual_value_aligned"] = pead["actual_value"] * (
        pead["adj_factor_est"].fillna(1.0) / pead["adj_factor_rep"].fillna(1.0)
    )
    pead["surprise_raw"] = pead["actual_value_aligned"] - pead["medest"]
    pead["surprise_to_dispersion"] = pead["surprise_raw"] / pead["stdev"].replace(0.0, np.nan)
    pead["signal_date"] = pead["repdats"]
    pead["trade_date"] = _next_trading_day(pead["signal_date"], crsp["date"])
    pead["exit_date_hint"] = pead["trade_date"] + pd.to_timedelta(hold_days, unit="D")
    pead = _link_permno(pead, links, date_col="signal_date")
    pead["prev_close"] = _prev_close_before_event(pead, crsp, date_col="signal_date")
    pead["surprise_to_price"] = pead["surprise_raw"] / pead["prev_close"].replace(0.0, np.nan)
    return pead.sort_values(["signal_date", "ibes_ticker"]).reset_index(drop=True)


def build_manifest(paths: WrdsExportPaths, revisions: pd.DataFrame, pead: pd.DataFrame) -> dict:
    return {
        "source_files": {
            "estimates": str(paths.estimates),
            "actuals": str(paths.actuals),
            "adjustments": str(paths.adjustments),
            "links": str(paths.links),
            "crsp_daily": str(paths.crsp_daily),
            "identifiers": str(paths.identifiers) if paths.identifiers else None,
        },
        "row_counts": {
            "revision_signals": int(len(revisions)),
            "pead_events": int(len(pead)),
        },
        "date_ranges": {
            "revision_signal_start": str(revisions["signal_date"].min().date()) if not revisions.empty else None,
            "revision_signal_end": str(revisions["signal_date"].max().date()) if not revisions.empty else None,
            "pead_signal_start": str(pead["signal_date"].min().date()) if not pead.empty else None,
            "pead_signal_end": str(pead["signal_date"].max().date()) if not pead.empty else None,
        },
        "timing_rules": {
            "revisions_trade_date": "next trading day after estimate snapshot date",
            "pead_trade_date": "next trading day after repdats because announcement timestamps are not assumed",
            "split_adjustment": "actual_aligned = actual_value * adj_factor_est / adj_factor_rep",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-in-time IBES revision and PEAD signal tables from WRDS exports."
    )
    parser.add_argument("--estimates", required=True, help="Path to ibes.detu CSV export")
    parser.add_argument("--actuals", required=True, help="Path to ibes.actu CSV export")
    parser.add_argument("--adjustments", required=True, help="Path to ibes.adj CSV export")
    parser.add_argument("--links", required=True, help="Path to IBES-CRSP link CSV export")
    parser.add_argument("--crsp-daily", required=True, help="Path to CRSP daily CSV export")
    parser.add_argument("--identifiers", default=None, help="Optional ibes.id CSV export")
    parser.add_argument("--output-dir", default="artifacts/wrds_pead", help="Where to write outputs")
    parser.add_argument("--fpi", default=DEFAULT_FPI, help="Quarterly forecast horizon to keep")
    parser.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS, help="Reference holding period")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = WrdsExportPaths(
        estimates=Path(args.estimates),
        actuals=Path(args.actuals),
        adjustments=Path(args.adjustments),
        links=Path(args.links),
        crsp_daily=Path(args.crsp_daily),
        identifiers=Path(args.identifiers) if args.identifiers else None,
        output_dir=Path(args.output_dir),
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    estimates = load_estimates(paths.estimates, fpi=args.fpi)
    actuals = load_actuals(paths.actuals)
    adjustments = load_adjustments(paths.adjustments)
    links = load_links(paths.links)
    crsp = load_crsp_daily(paths.crsp_daily)

    revisions = build_revision_signals(estimates, links, crsp, hold_days=args.hold_days)
    pead = build_pead_events(estimates, actuals, adjustments, links, crsp, hold_days=args.hold_days)
    manifest = build_manifest(paths, revisions, pead)

    revisions.to_csv(paths.output_dir / "revision_signals.csv", index=False)
    pead.to_csv(paths.output_dir / "pead_events.csv", index=False)
    (paths.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
