from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from difflib import get_close_matches
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import re
import subprocess
import time
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf


START_DATE = "2010-01-01"
END_DATE = "2026-01-01"
TRADING_DAYS = 252

DEVELOPMENT_END = "2018-12-31"
VALIDATION_END = "2021-12-31"
TEST_START = "2022-01-01"

BENCHMARK = "SPY"
TOP_HOLDINGS_PER_ETF = 8
DEFAULT_REBALANCE_RULE = "monthly"
DEFAULT_SCORE_HYSTERESIS = 0.0
STRATEGY_VARIANTS = (
    {"label": "monthly", "rebalance_rule": "monthly", "score_hysteresis": 0.0},
    {"label": "10d", "rebalance_rule": "10d", "score_hysteresis": 0.0},
    {"label": "10d_guarded", "rebalance_rule": "10d", "score_hysteresis": 0.5},
)
WAYBACK_TIMEOUT = 60
WAYBACK_RETRY_DELAY_SECONDS = 2.0

COMMISSION_BPS = 0.50
SLIPPAGE_BPS = 1.00
STRESS_COMMISSION_BPS = 0.75
STRESS_SLIPPAGE_BPS = 2.25

ARTIFACT_DIR = Path("artifacts")
CACHE_DIR = ARTIFACT_DIR / "wayback_cache"
CDX_CACHE_DIR = CACHE_DIR / "cdx"
SNAPSHOT_CACHE_DIR = CACHE_DIR / "snapshots"
SEC_CACHE_DIR = CACHE_DIR / "sec"
SEC_NPORT_CACHE_DIR = SEC_CACHE_DIR / "nport_filings"
SEC_SUBMISSIONS_CACHE_PATH = SEC_CACHE_DIR / "select_sector_spdr_submissions.json"
SEC_COMPANY_TICKERS_CACHE_PATH = SEC_CACHE_DIR / "company_tickers_exchange.json"

PLOT_PATH = ARTIFACT_DIR / "etf_stockpicking_equity_curve.png"
METRICS_PATH = ARTIFACT_DIR / "etf_stockpicking_metrics.csv"
SELECTIONS_PATH = ARTIFACT_DIR / "etf_stockpicking_selections.csv"
DIAGNOSTICS_PATH = ARTIFACT_DIR / "etf_stockpicking_diagnostics.csv"
HOLDINGS_HISTORY_PATH = ARTIFACT_DIR / "historical_etf_holdings_wayback.csv"
HOLDINGS_METADATA_PATH = ARTIFACT_DIR / "historical_etf_holdings_wayback_metadata.json"
REBALANCE_COMPARISON_PATH = ARTIFACT_DIR / "rebalance_frequency_comparison.csv"

MODEL_NOTE = (
    "Monthly stock-picking model with ETF universes reconstructed from a hybrid source stack: "
    "Wayback snapshots of the Select Sector SPDR holdings export plus SEC N-PORT-P filings used "
    "causally from filing date onward. This removes most current-membership survivorship in the "
    "universe, but delisted-price bias can still remain when Yahoo lacks historical data for old symbols."
)

WAYBACK_EXPORT_URL_TEMPLATES = (
    "http://www.sectorspdr.com/sectorspdr/IDCO.Client.Spdrs.Holdings/Export/ExportCsv?symbol={symbol}",
    "https://www.sectorspdrs.com/sectorspdr/IDCO.Client.Spdrs.Holdings/Export/ExportCsv?symbol={symbol}",
)

YAHOO_SYMBOL_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "FB": "META",
    "PCLN": "BKNG",
}

SEC_HEADERS = {"User-Agent": "quant-research@local.invalid"}
HOLDINGS_HISTORY_MODE = "hybrid_wayback_sec_v2"
SEC_NAMESPACE = {"n": "http://www.sec.gov/edgar/nport"}

SEC_SERIES_TO_ETF = {
    "The Technology Select Sector SPDR Fund": "XLK",
    "The Financial Select Sector SPDR Fund": "XLF",
    "The Energy Select Sector SPDR Fund": "XLE",
    "The Health Care Select Sector SPDR Fund": "XLV",
    "The Industrial Select Sector SPDR Fund": "XLI",
    "The Industrials Select Sector SPDR Fund": "XLI",
    "The Consumer Staples Select Sector SPDR Fund": "XLP",
    "The Consumer Discretionary Select Sector SPDR Fund": "XLY",
    "The Materials Select Sector SPDR Fund": "XLB",
    "The Utilities Select Sector SPDR Fund": "XLU",
}


@dataclass(frozen=True)
class ETFUniverseConfig:
    etf: str
    top_n: int = TOP_HOLDINGS_PER_ETF


ETF_UNIVERSES: Tuple[ETFUniverseConfig, ...] = (
    ETFUniverseConfig("XLK"),
    ETFUniverseConfig("XLF"),
    ETFUniverseConfig("XLE"),
    ETFUniverseConfig("XLV"),
    ETFUniverseConfig("XLI"),
    ETFUniverseConfig("XLP"),
    ETFUniverseConfig("XLY"),
    ETFUniverseConfig("XLB"),
    ETFUniverseConfig("XLU"),
)


def business_month_ends(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="BME")


def sample_slices() -> Dict[str, slice]:
    return {
        "development": slice("2012-01-01", DEVELOPMENT_END),
        "validation": slice("2019-01-01", VALIDATION_END),
        "test": slice(TEST_START, "2025-12-31"),
        "full_live_window": slice("2012-01-01", "2025-12-31"),
    }


def ensure_cache_dirs() -> None:
    for directory in [ARTIFACT_DIR, CACHE_DIR, CDX_CACHE_DIR, SNAPSHOT_CACHE_DIR, SEC_CACHE_DIR, SEC_NPORT_CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def request_json(url: str, params: Dict[str, str]) -> List[List[str]]:
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=WAYBACK_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception:
            try:
                full_url = f"{url}?{urlencode(params)}"
                result = subprocess.run(
                    (
                        f'curl.exe -A "Mozilla/5.0" "{full_url}" --silent --show-error '
                        f'--retry 4 --retry-all-errors --retry-delay 2 --connect-timeout 20 --max-time {WAYBACK_TIMEOUT}'
                    ),
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return json.loads(result.stdout)
            except Exception:
                pass
            if attempt == 2:
                raise
            time.sleep(WAYBACK_RETRY_DELAY_SECONDS * (attempt + 1))
    raise RuntimeError("Unreachable retry loop")


def request_text(url: str) -> str:
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=WAYBACK_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception:
            try:
                result = subprocess.run(
                    (
                        f'curl.exe -A "Mozilla/5.0" "{url}" --silent --show-error '
                        f'--retry 4 --retry-all-errors --retry-delay 2 --connect-timeout 20 --max-time {WAYBACK_TIMEOUT}'
                    ),
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout
            except Exception:
                pass
            if attempt == 2:
                raise
            time.sleep(WAYBACK_RETRY_DELAY_SECONDS * (attempt + 1))
    raise RuntimeError("Unreachable retry loop")


def get_export_urls(etf: str) -> List[str]:
    return [template.format(symbol=etf.lower()) for template in WAYBACK_EXPORT_URL_TEMPLATES]


def load_or_fetch_cdx_index(etf: str) -> List[Dict[str, str]]:
    ensure_cache_dirs()
    cache_path = CDX_CACHE_DIR / f"{etf.lower()}_cdx.json"
    cached_records: List[Dict[str, str]] = []
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached and isinstance(cached[0], str):
            legacy_url = get_export_urls(etf)[0]
            cached_records = [{"timestamp": timestamp, "original": legacy_url} for timestamp in cached]
        else:
            cached_records = cached

    existing_urls = {record["original"] for record in cached_records}
    urls_to_fetch = [export_url for export_url in get_export_urls(etf) if export_url not in existing_urls]
    if not urls_to_fetch and cached_records:
        return cached_records

    rows: List[Dict[str, str]] = list(cached_records)
    for export_url in urls_to_fetch:
        try:
            cdx_rows = request_json(
                "https://web.archive.org/cdx/search/cdx",
                {
                    "url": export_url,
                    "output": "json",
                    "fl": "timestamp,original",
                    "filter": "statuscode:200",
                    "limit": "5000",
                    "from": "201001",
                    "to": "202612",
                },
            )
        except Exception:
            continue
        rows.extend({"timestamp": row[0], "original": row[1]} for row in cdx_rows[1:])

    deduped = {(row["timestamp"], row["original"]): row for row in rows}
    records = sorted(deduped.values(), key=lambda row: row["timestamp"])
    cache_path.write_text(json.dumps(records))
    return records


def holdings_cache_matches(configs: Iterable[ETFUniverseConfig]) -> bool:
    if not HOLDINGS_HISTORY_PATH.exists() or not HOLDINGS_METADATA_PATH.exists():
        return False

    try:
        metadata = json.loads(HOLDINGS_METADATA_PATH.read_text())
    except Exception:
        return False

    requested = {cfg.etf: cfg.top_n for cfg in configs}
    cached = metadata.get("requested_universe", {})
    return cached == requested and metadata.get("history_mode") == HOLDINGS_HISTORY_MODE


def normalize_yahoo_symbol(symbol: str) -> str:
    clean = str(symbol).strip().upper()
    return YAHOO_SYMBOL_MAP.get(clean, clean)


def normalize_company_name(name: str) -> str:
    clean = str(name).upper().replace("&", " AND ")
    clean = clean.replace("/THE", " ").replace("/DE/", " ").replace("/MD", " ")
    clean = clean.replace(" COS ", " COMPANIES ").replace(" COS", " COMPANIES")
    clean = re.sub(r"[^A-Z0-9 ]+", " ", clean)
    stopwords = {
        "INC",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LTD",
        "PLC",
        "HOLDINGS",
        "HOLDING",
        "THE",
        "CLASS",
        "A",
        "B",
        "C",
        "COMMON",
        "NEW",
    }
    tokens = [token for token in clean.split() if token not in stopwords]
    return " ".join(tokens)


def load_or_fetch_sec_submissions() -> Dict[str, object]:
    ensure_cache_dirs()
    if SEC_SUBMISSIONS_CACHE_PATH.exists():
        return json.loads(SEC_SUBMISSIONS_CACHE_PATH.read_text())

    response = requests.get("https://data.sec.gov/submissions/CIK0001064641.json", headers=SEC_HEADERS, timeout=WAYBACK_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    SEC_SUBMISSIONS_CACHE_PATH.write_text(json.dumps(payload))
    return payload


def load_or_fetch_sec_company_tickers() -> List[List[object]]:
    ensure_cache_dirs()
    if SEC_COMPANY_TICKERS_CACHE_PATH.exists():
        payload = json.loads(SEC_COMPANY_TICKERS_CACHE_PATH.read_text())
        return payload["data"]

    response = requests.get("https://www.sec.gov/files/company_tickers_exchange.json", headers=SEC_HEADERS, timeout=WAYBACK_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    SEC_COMPANY_TICKERS_CACHE_PATH.write_text(json.dumps(payload))
    return payload["data"]


def load_or_fetch_sec_nport_filing(accession_compact: str) -> Dict[str, object]:
    ensure_cache_dirs()
    cache_path = SEC_NPORT_CACHE_DIR / f"{accession_compact}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        if payload.get("period_source") == "index_headers":
            return payload

    url = f"https://www.sec.gov/Archives/edgar/data/1064641/{accession_compact}/primary_doc.xml"
    response = requests.get(url, headers=SEC_HEADERS, timeout=WAYBACK_TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    index_headers_url = f"https://www.sec.gov/Archives/edgar/data/1064641/{accession_compact}/{accession_compact[:10]}-{accession_compact[10:12]}-{accession_compact[12:]}-index-headers.html"
    index_headers = requests.get(index_headers_url, headers=SEC_HEADERS, timeout=WAYBACK_TIMEOUT)
    index_headers.raise_for_status()
    period_match = re.search(r"<PERIOD>(\d{8})", index_headers.text)
    report_date = None
    if period_match:
        report_date = pd.Timestamp(period_match.group(1))

    series_name = root.findtext(".//n:seriesName", default="", namespaces=SEC_NAMESPACE)
    holdings_rows: List[Dict[str, object]] = []
    for security in root.findall(".//n:invstOrSec", SEC_NAMESPACE):
        company_name = (security.findtext("n:name", default="", namespaces=SEC_NAMESPACE) or "").strip()
        if not company_name:
            continue
        asset_category = (security.findtext("n:assetCat", default="", namespaces=SEC_NAMESPACE) or "").strip()
        payoff_profile = (security.findtext("n:payoffProfile", default="", namespaces=SEC_NAMESPACE) or "").strip()
        if asset_category != "EC" or payoff_profile != "Long":
            continue
        pct_value = pd.to_numeric(security.findtext("n:pctVal", default="0", namespaces=SEC_NAMESPACE), errors="coerce")
        if pd.isna(pct_value):
            continue
        holdings_rows.append(
            {
                "company_name": company_name,
                "index_weight": float(pct_value) / 100.0,
            }
        )

    payload = {
        "series_name": series_name,
        "as_of_date": report_date.strftime("%Y-%m-%d") if report_date is not None else None,
        "period_source": "index_headers",
        "rows": holdings_rows,
    }
    cache_path.write_text(json.dumps(payload))
    return payload


def build_company_name_symbol_map(wayback_history: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    name_to_symbol: Dict[str, str] = {}

    if not wayback_history.empty:
        distinct_names = wayback_history[["company_name", "symbol"]].dropna().drop_duplicates()
        for _, row in distinct_names.iterrows():
            normalized = normalize_company_name(row["company_name"])
            if normalized:
                name_to_symbol.setdefault(normalized, str(row["symbol"]).upper())

    for _, company_name, ticker, _ in load_or_fetch_sec_company_tickers():
        normalized = normalize_company_name(company_name)
        if normalized:
            name_to_symbol.setdefault(normalized, str(ticker).upper())

    return name_to_symbol, list(name_to_symbol.keys())


def resolve_symbol_from_company_name(company_name: str, name_to_symbol: Dict[str, str], all_names: List[str]) -> str | None:
    normalized = normalize_company_name(company_name)
    if not normalized:
        return None
    exact = name_to_symbol.get(normalized)
    if exact:
        return exact
    close_match = get_close_matches(normalized, all_names, n=1, cutoff=0.80)
    if not close_match:
        return None
    return name_to_symbol[close_match[0]]


def parse_snapshot_csv(csv_text: str) -> Tuple[pd.Timestamp | None, pd.DataFrame]:
    lines = [line.replace("\ufeff", "").strip() for line in csv_text.splitlines() if line.strip()]
    if len(lines) < 3:
        return None, pd.DataFrame(columns=["Symbol", "Company Name", "Index Weight"])

    as_of_date = None
    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", lines[0])
    if date_match:
        as_of_date = pd.to_datetime(date_match.group(1), format="%m/%d/%Y")

    table_text = "\n".join(lines[1:])
    frame = pd.read_csv(StringIO(table_text))
    frame.columns = [str(col).strip().strip('"') for col in frame.columns]

    needed = ["Symbol", "Company Name", "Index Weight"]
    missing_cols = [col for col in needed if col not in frame.columns]
    if missing_cols:
        return as_of_date, pd.DataFrame(columns=needed)

    frame = frame[needed].copy()
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip().str.upper()
    frame["Company Name"] = frame["Company Name"].astype(str).str.strip()
    frame["Index Weight"] = (
        frame["Index Weight"].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    )
    frame["Index Weight"] = pd.to_numeric(frame["Index Weight"], errors="coerce") / 100.0
    frame = frame.dropna(subset=["Symbol", "Index Weight"])
    frame = frame[frame["Symbol"].ne("NAN")]
    frame = frame.sort_values("Index Weight", ascending=False).reset_index(drop=True)
    return as_of_date, frame


def load_or_fetch_snapshot(etf: str, timestamp: str, original_url: str) -> Tuple[pd.Timestamp | None, pd.DataFrame]:
    ensure_cache_dirs()
    source_key = "modern" if "sectorspdrs.com" in original_url else "legacy"
    cache_path = SNAPSHOT_CACHE_DIR / f"{etf.lower()}_{timestamp}_{source_key}.json"
    legacy_cache_path = SNAPSHOT_CACHE_DIR / f"{etf.lower()}_{timestamp}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        as_of = pd.Timestamp(payload["as_of_date"]) if payload["as_of_date"] else None
        return as_of, pd.DataFrame(payload["rows"])
    if legacy_cache_path.exists():
        payload = json.loads(legacy_cache_path.read_text())
        as_of = pd.Timestamp(payload["as_of_date"]) if payload["as_of_date"] else None
        return as_of, pd.DataFrame(payload["rows"])

    archived_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
    try:
        text = request_text(archived_url)
        as_of_date, frame = parse_snapshot_csv(text)
    except Exception:
        payload = {
            "as_of_date": None,
            "rows": [],
            "fetch_failed": True,
        }
        cache_path.write_text(json.dumps(payload))
        return None, pd.DataFrame(columns=["Symbol", "Company Name", "Index Weight"])

    payload = {
        "as_of_date": as_of_date.strftime("%Y-%m-%d") if as_of_date is not None else None,
        "rows": frame.to_dict(orient="records"),
    }
    cache_path.write_text(json.dumps(payload))
    return as_of_date, frame


def choose_snapshot_for_date(
    etf: str,
    target_date: pd.Timestamp,
    snapshot_records: List[Dict[str, str]],
) -> Tuple[str, str, pd.Timestamp | None, pd.DataFrame] | None:
    cutoff = target_date.strftime("%Y%m%d") + "235959"
    timestamps = [record["timestamp"] for record in snapshot_records]
    idx = bisect_right(timestamps, cutoff) - 1

    while idx >= 0:
        snapshot_record = snapshot_records[idx]
        timestamp = snapshot_record["timestamp"]
        original_url = snapshot_record["original"]
        try:
            as_of_date, frame = load_or_fetch_snapshot(etf, timestamp, original_url)
        except Exception:
            idx -= 1
            continue
        if as_of_date is not None and as_of_date <= target_date and not frame.empty:
            return timestamp, original_url, as_of_date, frame
        idx -= 1

    return None


def build_wayback_holdings_history(configs: Iterable[ETFUniverseConfig]) -> pd.DataFrame:
    rebalance_dates = business_month_ends(START_DATE, END_DATE)
    rows: List[Dict[str, object]] = []

    for cfg in configs:
        snapshot_records = load_or_fetch_cdx_index(cfg.etf)
        for rebalance_date in rebalance_dates:
            snapshot_info = choose_snapshot_for_date(cfg.etf, rebalance_date, snapshot_records)
            if snapshot_info is None:
                continue

            snapshot_timestamp, original_url, as_of_date, frame = snapshot_info
            top_holdings = frame.head(cfg.top_n).copy()
            for _, row in top_holdings.iterrows():
                rows.append(
                    {
                        "rebalance_date": rebalance_date,
                        "source_etf": cfg.etf,
                        "source_type": "wayback_ssga",
                        "available_date": pd.Timestamp(snapshot_timestamp[:8]),
                        "snapshot_timestamp": snapshot_timestamp,
                        "snapshot_original_url": original_url,
                        "as_of_date": as_of_date,
                        "symbol": row["Symbol"],
                        "company_name": row["Company Name"],
                        "index_weight": float(row["Index Weight"]),
                        "snapshot_age_days": int((rebalance_date - as_of_date).days),
                    }
                )

    return pd.DataFrame(rows).sort_values(["rebalance_date", "source_etf", "index_weight"], ascending=[True, True, False])


def build_sec_holdings_events(configs: Iterable[ETFUniverseConfig], wayback_history: pd.DataFrame) -> pd.DataFrame:
    target_etfs = {cfg.etf for cfg in configs}
    submissions = load_or_fetch_sec_submissions()
    recent = submissions["filings"]["recent"]
    name_to_symbol, known_names = build_company_name_symbol_map(wayback_history)

    rows: List[Dict[str, object]] = []
    for form, accession, filing_date in zip(recent["form"], recent["accessionNumber"], recent["filingDate"]):
        if form != "NPORT-P":
            continue
        payload = load_or_fetch_sec_nport_filing(accession.replace("-", ""))
        series_name = str(payload.get("series_name", "")).strip()
        etf = SEC_SERIES_TO_ETF.get(series_name)
        if etf not in target_etfs:
            continue

        as_of_date = pd.Timestamp(payload["as_of_date"]) if payload.get("as_of_date") else None
        filing_timestamp = pd.Timestamp(filing_date)
        holdings_rows = []
        for security in payload.get("rows", []):
            symbol = resolve_symbol_from_company_name(security["company_name"], name_to_symbol, known_names)
            if symbol is None:
                continue
            holdings_rows.append(
                {
                    "source_etf": etf,
                    "source_type": "sec_nport",
                    "available_date": filing_timestamp,
                    "snapshot_timestamp": accession.replace("-", ""),
                    "snapshot_original_url": f"https://www.sec.gov/Archives/edgar/data/1064641/{accession.replace('-', '')}/primary_doc.xml",
                    "as_of_date": as_of_date,
                    "symbol": symbol,
                    "company_name": security["company_name"],
                    "index_weight": float(security["index_weight"]),
                }
            )

        if not holdings_rows:
            continue

        frame = pd.DataFrame(holdings_rows).sort_values("index_weight", ascending=False)
        top_n = next(cfg.top_n for cfg in configs if cfg.etf == etf)
        rows.extend(frame.head(top_n).to_dict(orient="records"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_etf",
                "source_type",
                "available_date",
                "snapshot_timestamp",
                "snapshot_original_url",
                "as_of_date",
                "symbol",
                "company_name",
                "index_weight",
            ]
        )

    return pd.DataFrame(rows).sort_values(["source_etf", "available_date", "index_weight"], ascending=[True, True, False])


def build_historical_holdings(configs: Iterable[ETFUniverseConfig]) -> pd.DataFrame:
    ensure_cache_dirs()
    configs = tuple(configs)
    if holdings_cache_matches(configs):
        history = pd.read_csv(HOLDINGS_HISTORY_PATH, parse_dates=["rebalance_date", "available_date", "as_of_date"])
        return history

    wayback_history = build_wayback_holdings_history(configs)
    sec_events = build_sec_holdings_events(configs, wayback_history)

    rebalance_dates = business_month_ends(START_DATE, END_DATE)
    rows: List[Dict[str, object]] = []

    for cfg in configs:
        wayback_by_rebalance = {
            date: frame.copy()
            for date, frame in wayback_history.loc[wayback_history["source_etf"] == cfg.etf].groupby("rebalance_date")
        }
        sec_etf_events = sec_events.loc[sec_events["source_etf"] == cfg.etf].copy()
        sec_event_dates = sorted(sec_etf_events["available_date"].dropna().unique().tolist()) if not sec_etf_events.empty else []

        for rebalance_date in rebalance_dates:
            candidate_frames: List[pd.DataFrame] = []
            if rebalance_date in wayback_by_rebalance:
                candidate_frames.append(wayback_by_rebalance[rebalance_date])

            if sec_event_dates:
                cutoff_idx = bisect_right(sec_event_dates, rebalance_date.to_datetime64()) - 1
                if cutoff_idx >= 0:
                    latest_sec_date = pd.Timestamp(sec_event_dates[cutoff_idx])
                    sec_frame = sec_etf_events.loc[sec_etf_events["available_date"] == latest_sec_date].copy()
                    if not sec_frame.empty:
                        candidate_frames.append(sec_frame)

            if not candidate_frames:
                continue

            selected = max(candidate_frames, key=lambda frame: pd.Timestamp(frame["available_date"].iloc[0]))
            selected = selected.sort_values("index_weight", ascending=False).head(cfg.top_n).copy()
            selected["rebalance_date"] = rebalance_date
            selected["snapshot_age_days"] = selected["as_of_date"].apply(lambda value: int((rebalance_date - pd.Timestamp(value)).days))
            rows.extend(selected.to_dict(orient="records"))

    history = pd.DataFrame(rows).sort_values(["rebalance_date", "source_etf", "index_weight"], ascending=[True, True, False])
    history.to_csv(HOLDINGS_HISTORY_PATH, index=False)
    HOLDINGS_METADATA_PATH.write_text(
        json.dumps(
            {
                "history_mode": HOLDINGS_HISTORY_MODE,
                "requested_universe": {cfg.etf: cfg.top_n for cfg in configs},
                "row_count": int(len(history)),
                "rebalance_start": str(history["rebalance_date"].min().date()) if not history.empty else None,
                "rebalance_end": str(history["rebalance_date"].max().date()) if not history.empty else None,
            },
            indent=2,
        )
    )
    return history


def download_prices(symbols: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    requested_symbols = sorted({str(symbol).strip().upper() for symbol in symbols})
    yahoo_symbols = [normalize_yahoo_symbol(symbol) for symbol in requested_symbols]
    rename_map = {normalize_yahoo_symbol(symbol): symbol for symbol in requested_symbols}

    raw = yf.download(
        yahoo_symbols,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    close_px = raw["Adj Close"].copy().rename(columns=rename_map)
    adj_factor = raw["Adj Close"] / raw["Close"].replace(0, np.nan)
    open_px = (raw["Open"] * adj_factor).rename(columns=rename_map)

    close_px = close_px.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    open_px = open_px.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    common_index = close_px.index.intersection(open_px.index)
    return close_px.loc[common_index], open_px.loc[common_index]


def zscore_series(values: pd.Series) -> pd.Series:
    std = values.std(ddof=1)
    if len(values) < 2 or std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def get_rebalance_dates(index: pd.Index, rebalance_rule: str) -> List[pd.Timestamp]:
    if rebalance_rule == "monthly":
        return index.to_series().groupby(index.to_period("M")).tail(1).index.tolist()

    if rebalance_rule.endswith("d"):
        step = int(rebalance_rule[:-1])
        dates = index[::step].tolist()
        if dates[-1] != index[-1]:
            dates.append(index[-1])
        return dates

    raise ValueError(f"Unsupported rebalance rule: {rebalance_rule}")


def latest_holdings_for_rebalance(holdings_history: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    exact = holdings_history.loc[holdings_history["rebalance_date"] == rebalance_date.normalize()]
    if not exact.empty:
        return exact

    eligible = holdings_history.loc[holdings_history["rebalance_date"] <= rebalance_date.normalize()].copy()
    if eligible.empty:
        return eligible

    latest_dates = eligible.groupby("source_etf")["rebalance_date"].transform("max")
    return eligible.loc[eligible["rebalance_date"] == latest_dates]


def compute_cross_sectional_features(
    close_px: pd.DataFrame,
    etf_symbol: str,
    stock_symbol: str,
    loc: int,
) -> Dict[str, float] | None:
    stock_history = close_px[stock_symbol].iloc[: loc + 1]
    etf_history = close_px[etf_symbol].iloc[: loc + 1]
    if len(stock_history) < 252 or stock_history.tail(252).isna().any() or etf_history.tail(252).isna().any():
        return None

    stock_returns = stock_history.pct_change(fill_method=None)
    etf_returns = etf_history.pct_change(fill_method=None)

    momentum_12_1 = stock_history.iloc[-21] / stock_history.iloc[-252] - 1.0
    etf_momentum_12_1 = etf_history.iloc[-21] / etf_history.iloc[-252] - 1.0
    sector_relative_momentum_12_1 = momentum_12_1 - etf_momentum_12_1
    reversal_1m = stock_history.iloc[-1] / stock_history.iloc[-21] - 1.0
    trend_200d = stock_history.iloc[-1] / stock_history.iloc[-200:].mean() - 1.0
    high_52w = stock_history.iloc[-252:].max()
    proximity_52w_high = stock_history.iloc[-1] / high_52w - 1.0 if high_52w and np.isfinite(high_52w) else np.nan

    beta_window_stock = stock_returns.iloc[-126:]
    beta_window_etf = etf_returns.iloc[-126:]
    beta = beta_window_stock.cov(beta_window_etf) / beta_window_etf.var()
    if np.isnan(beta):
        return None

    residual_returns = beta_window_stock - beta * beta_window_etf
    idio_vol_6m = residual_returns.std(ddof=1)
    residual_momentum_12_1 = residual_returns.iloc[-126:-21].mean() / residual_returns.iloc[-126:-21].std(ddof=1)

    return {
        "momentum_12_1": float(momentum_12_1),
        "sector_relative_momentum_12_1": float(sector_relative_momentum_12_1),
        "residual_momentum_12_1": float(residual_momentum_12_1) if np.isfinite(residual_momentum_12_1) else np.nan,
        "reversal_1m": float(reversal_1m),
        "trend_200d": float(trend_200d),
        "proximity_52w_high": float(proximity_52w_high) if np.isfinite(proximity_52w_high) else np.nan,
        "beta_6m": float(beta),
        "idiosyncratic_vol_6m": float(idio_vol_6m) if np.isfinite(idio_vol_6m) else np.nan,
    }


def build_positions(
    close_px: pd.DataFrame,
    holdings_history: pd.DataFrame,
    rebalance_rule: str = DEFAULT_REBALANCE_RULE,
    score_hysteresis: float = DEFAULT_SCORE_HYSTERESIS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    positions = pd.DataFrame(0.0, index=close_px.index, columns=close_px.columns)
    selection_rows: List[Dict[str, object]] = []
    previous_symbol_by_etf: Dict[str, str] = {}

    rebalance_dates = get_rebalance_dates(close_px.index, rebalance_rule)

    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        loc = close_px.index.get_loc(rebalance_date)
        if loc < 252 or loc >= len(close_px.index) - 2:
            continue

        active_holdings = latest_holdings_for_rebalance(holdings_history, rebalance_date)
        if active_holdings.empty:
            continue

        raw_picks: List[Dict[str, object]] = []
        for etf_symbol, group in active_holdings.groupby("source_etf"):
            feature_rows: List[Dict[str, object]] = []
            for _, holding in group.iterrows():
                symbol = holding["symbol"]
                if symbol not in close_px.columns:
                    continue
                features = compute_cross_sectional_features(close_px, etf_symbol, symbol, loc)
                if features is None:
                    continue
                feature_rows.append(
                    {
                        "source_etf": etf_symbol,
                        "symbol": symbol,
                        "snapshot_age_days": holding["snapshot_age_days"],
                        "index_weight": holding["index_weight"],
                        **features,
                    }
                )

            if not feature_rows:
                continue

            feature_frame = pd.DataFrame(feature_rows).set_index("symbol")
            feature_frame["z_momentum"] = zscore_series(feature_frame["momentum_12_1"])
            feature_frame["z_sector_relative_momentum"] = zscore_series(feature_frame["sector_relative_momentum_12_1"])
            feature_frame["z_residual_momentum"] = zscore_series(feature_frame["residual_momentum_12_1"])
            feature_frame["z_trend"] = zscore_series(feature_frame["trend_200d"])
            feature_frame["z_52w_high"] = zscore_series(feature_frame["proximity_52w_high"])
            feature_frame["z_reversal"] = zscore_series(feature_frame["reversal_1m"])
            feature_frame["z_idio_vol"] = zscore_series(feature_frame["idiosyncratic_vol_6m"])
            feature_frame["z_beta"] = zscore_series(feature_frame["beta_6m"].abs())

            feature_frame["score"] = (
                1.00 * feature_frame["z_sector_relative_momentum"]
                + 0.75 * feature_frame["z_residual_momentum"]
                + 0.50 * feature_frame["z_momentum"]
                + 0.35 * feature_frame["z_52w_high"]
                + 0.30 * feature_frame["z_trend"]
                - 0.40 * feature_frame["z_reversal"]
                - 0.35 * feature_frame["z_idio_vol"]
                - 0.15 * feature_frame["z_beta"]
            )

            best_symbol = feature_frame["score"].idxmax()
            chosen_symbol = best_symbol
            previous_symbol = previous_symbol_by_etf.get(etf_symbol)
            if score_hysteresis > 0.0 and previous_symbol in feature_frame.index:
                best_score = float(feature_frame.loc[best_symbol, "score"])
                previous_score = float(feature_frame.loc[previous_symbol, "score"])
                if best_score - previous_score < score_hysteresis:
                    chosen_symbol = previous_symbol

            previous_symbol_by_etf[etf_symbol] = chosen_symbol
            best_row = feature_frame.loc[chosen_symbol]
            raw_picks.append(
                {
                    "source_etf": etf_symbol,
                    "symbol": chosen_symbol,
                    "score": float(best_row["score"]),
                    "momentum_12_1": float(best_row["momentum_12_1"]),
                    "sector_relative_momentum_12_1": float(best_row["sector_relative_momentum_12_1"]),
                    "residual_momentum_12_1": float(best_row["residual_momentum_12_1"]),
                    "reversal_1m": float(best_row["reversal_1m"]),
                    "trend_200d": float(best_row["trend_200d"]),
                    "proximity_52w_high": float(best_row["proximity_52w_high"]),
                    "beta_6m": float(best_row["beta_6m"]),
                    "idiosyncratic_vol_6m": float(best_row["idiosyncratic_vol_6m"]),
                    "snapshot_age_days": int(best_row["snapshot_age_days"]),
                    "index_weight": float(best_row["index_weight"]),
                }
            )

        if not raw_picks:
            continue

        # Deduplicate symbols shared across ETF universes.
        best_by_symbol: Dict[str, Dict[str, object]] = {}
        for pick in raw_picks:
            symbol = str(pick["symbol"])
            if symbol not in best_by_symbol or float(pick["score"]) > float(best_by_symbol[symbol]["score"]):
                best_by_symbol[symbol] = pick

        picks = list(best_by_symbol.values())
        next_rebalance_date = rebalance_dates[i + 1]
        trade_dates = close_px.index[(close_px.index > rebalance_date) & (close_px.index <= next_rebalance_date)]
        if len(trade_dates) == 0:
            continue

        weight = 1.0 / len(picks)
        for trade_date in trade_dates:
            for pick in picks:
                positions.loc[trade_date, str(pick["symbol"])] = weight

        for pick in picks:
            selection_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "rebalance_rule": rebalance_rule,
                    "score_hysteresis": score_hysteresis,
                    "source_etf": pick["source_etf"],
                    "symbol": pick["symbol"],
                    "portfolio_weight": weight,
                    **pick,
                }
            )

    return positions, pd.DataFrame(selection_rows)


def compute_strategy_returns(
    positions: pd.DataFrame,
    open_px: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
) -> pd.Series:
    open_returns = open_px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    next_open_returns = open_returns.shift(-1).fillna(0.0)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    gross_returns = (positions * next_open_returns[positions.columns]).sum(axis=1)
    cost = turnover * ((commission_bps + slippage_bps) / 10000.0)
    return gross_returns - cost


def compute_metrics(returns: pd.Series, annual_turnover: float | None = None) -> Dict[str, float]:
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "sharpe": returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "cagr": equity.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1.0,
        "max_drawdown": drawdown.min(),
        "total_return": equity.iloc[-1] - 1.0,
        "annual_turnover": annual_turnover if annual_turnover is not None else np.nan,
    }


def compute_activity_metrics(positions: pd.DataFrame, turnover: pd.Series, index: pd.Index) -> Dict[str, float]:
    window_positions = positions.loc[index]
    gross = window_positions.abs().sum(axis=1)
    window_turnover = turnover.loc[index]
    return {
        "annual_turnover": window_turnover.mean() * TRADING_DAYS if len(window_turnover) else np.nan,
        "active_ratio": float((gross > 0.0).mean()) if len(gross) else np.nan,
        "trade_ratio": float((window_turnover > 0.0).mean()) if len(window_turnover) else np.nan,
        "avg_gross_exposure": float(gross.mean()) if len(gross) else np.nan,
    }


def build_report_table(
    strategy_returns: pd.Series,
    stress_returns: pd.Series,
    benchmark_returns: pd.Series,
    positions: pd.DataFrame,
    turnover: pd.Series,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for label, slc in sample_slices().items():
        strat = strategy_returns.loc[slc]
        stress = stress_returns.loc[slc]
        bench = benchmark_returns.loc[slc]

        strat_activity = compute_activity_metrics(positions, turnover, strat.index)
        stress_activity = compute_activity_metrics(positions, turnover, stress.index)

        strat_metrics = {**compute_metrics(strat, strat_activity["annual_turnover"]), **strat_activity}
        stress_metrics = {**compute_metrics(stress, stress_activity["annual_turnover"]), **stress_activity}
        bench_metrics = compute_metrics(bench)

        records.extend(
            [
                {"sample": label, "scenario": "strategy_base", **strat_metrics},
                {"sample": label, "scenario": "strategy_stress", **stress_metrics},
                {"sample": label, "scenario": "benchmark_spy", **bench_metrics},
            ]
        )

    return pd.DataFrame(records)


def build_diagnostics(holdings_history: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    age = holdings_history.groupby("source_etf")["snapshot_age_days"].agg(["mean", "max", "count"]).reset_index()
    for _, row in age.iterrows():
        rows.append({"type": "snapshot_age_mean", "name": row["source_etf"], "value": float(row["mean"])})
        rows.append({"type": "snapshot_age_max", "name": row["source_etf"], "value": int(row["max"])})
        rows.append({"type": "snapshot_obs_count", "name": row["source_etf"], "value": int(row["count"])})

    if "source_type" in holdings_history.columns:
        source_counts = holdings_history.groupby(["source_etf", "source_type"]).size().reset_index(name="count")
        for _, row in source_counts.iterrows():
            rows.append({"type": "source_rows", "name": f"{row['source_etf']}::{row['source_type']}", "value": int(row["count"])})

    if not selections.empty:
        for symbol, count in selections.groupby("symbol").size().sort_values(ascending=False).items():
            rows.append({"type": "selection_count", "name": symbol, "value": int(count)})

    return pd.DataFrame(rows)


def evaluate_strategy_variants(
    close_px: pd.DataFrame,
    open_px: pd.DataFrame,
    holdings_history: pd.DataFrame,
    benchmark_returns: pd.Series,
    variants: Iterable[Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for variant in variants:
        label = str(variant["label"])
        rebalance_rule = str(variant["rebalance_rule"])
        score_hysteresis = float(variant.get("score_hysteresis", 0.0))

        positions, selections = build_positions(
            close_px,
            holdings_history,
            rebalance_rule=rebalance_rule,
            score_hysteresis=score_hysteresis,
        )
        turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
        base_returns = compute_strategy_returns(positions, open_px, COMMISSION_BPS, SLIPPAGE_BPS)
        stress_returns = compute_strategy_returns(positions, open_px, STRESS_COMMISSION_BPS, STRESS_SLIPPAGE_BPS)
        metrics = build_report_table(base_returns, stress_returns, benchmark_returns, positions, turnover)

        validation = metrics.loc[(metrics["sample"] == "validation") & (metrics["scenario"] == "strategy_base")].iloc[0]
        test = metrics.loc[(metrics["sample"] == "test") & (metrics["scenario"] == "strategy_base")].iloc[0]
        rows.append(
            {
                "strategy_variant": label,
                "rebalance_rule": rebalance_rule,
                "score_hysteresis": score_hysteresis,
                "validation_sharpe": float(validation["sharpe"]),
                "validation_cagr": float(validation["cagr"]),
                "validation_turnover": float(validation["annual_turnover"]),
                "test_sharpe": float(test["sharpe"]),
                "test_cagr": float(test["cagr"]),
                "test_turnover": float(test["annual_turnover"]),
                "test_trade_ratio": float(test["trade_ratio"]),
                "selection_count": int(len(selections)),
            }
        )

    return pd.DataFrame(rows).sort_values(["test_cagr", "validation_sharpe"], ascending=[False, False])


def plot_results(strategy_returns: pd.Series, benchmark_returns: pd.Series, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_strategy = strategy_returns.loc[TEST_START:"2025-12-31"].dropna()
    test_benchmark = benchmark_returns.loc[test_strategy.index]

    strategy_equity = (1.0 + test_strategy).cumprod()
    benchmark_equity = (1.0 + test_benchmark).cumprod()
    drawdown = strategy_equity / strategy_equity.cummax() - 1.0

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(strategy_equity.index, strategy_equity, label="ETF Basket Stockpicking", linewidth=2.2, color="#1f77b4")
    ax_top.plot(benchmark_equity.index, benchmark_equity, label=BENCHMARK, linewidth=1.8, color="#ff7f0e")
    ax_top.set_title("Wayback ETF Stockpicking vs SPY")
    ax_top.set_ylabel("Growth of $1")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="upper left")

    ax_bottom.fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.35)
    ax_bottom.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=1.1)
    ax_bottom.set_title("Strategy Drawdown")
    ax_bottom.set_ylabel("Drawdown")
    ax_bottom.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ensure_cache_dirs()

    holdings_history = build_historical_holdings(ETF_UNIVERSES)
    symbols = (
        holdings_history["symbol"].dropna().astype(str).str.upper().unique().tolist()
        + [cfg.etf for cfg in ETF_UNIVERSES]
        + [BENCHMARK]
    )

    close_px, open_px = download_prices(symbols)
    benchmark_returns = open_px[BENCHMARK].pct_change(fill_method=None).shift(-1).fillna(0.0)
    rebalance_comparison = evaluate_strategy_variants(
        close_px,
        open_px,
        holdings_history,
        benchmark_returns,
        STRATEGY_VARIANTS,
    )
    positions, selections = build_positions(close_px, holdings_history, rebalance_rule=DEFAULT_REBALANCE_RULE)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))

    base_returns = compute_strategy_returns(positions, open_px, COMMISSION_BPS, SLIPPAGE_BPS)
    stress_returns = compute_strategy_returns(positions, open_px, STRESS_COMMISSION_BPS, STRESS_SLIPPAGE_BPS)

    metrics = build_report_table(base_returns, stress_returns, benchmark_returns, positions, turnover)
    diagnostics = build_diagnostics(holdings_history, selections)

    metrics.to_csv(METRICS_PATH, index=False)
    selections.to_csv(SELECTIONS_PATH, index=False)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    rebalance_comparison.to_csv(REBALANCE_COMPARISON_PATH, index=False)
    plot_results(base_returns, benchmark_returns, PLOT_PATH)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("MODEL")
    print(MODEL_NOTE)
    print()

    print("HOLDINGS SAMPLE")
    print(holdings_history.head(20).to_string(index=False))
    print()

    print("METRICS")
    print(metrics.round(6).to_string(index=False))
    print()

    print("DIAGNOSTICS")
    print(diagnostics.head(40).to_string(index=False))
    print()

    print("REBALANCE COMPARISON")
    print(rebalance_comparison.round(6).to_string(index=False))
    print()

    print(f"Saved holdings history to {HOLDINGS_HISTORY_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved selections to {SELECTIONS_PATH}")
    print(f"Saved diagnostics to {DIAGNOSTICS_PATH}")
    print(f"Saved rebalance comparison to {REBALANCE_COMPARISON_PATH}")
    print(f"Saved plot to {PLOT_PATH}")
