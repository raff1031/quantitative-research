"""
Free Data Sources Integration
==============================
Pulls congressional trades, insider activity, and institutional data
from 100% free sources — no API keys needed.

Sources:
  - Senate trades: GitHub repo (timothycarambat/senate-stock-watcher-data)
  - House trades:  GitHub repo (housestockwatcher via S3 bulk download)
  - Insider trades: SEC EDGAR Form 4 (via OpenInsider scrape)
  - Capitol Trades: capitoltrades.com (web scrape, backup source)

Usage:
    from data_sources import FreeDataPipeline
    
    pipeline = FreeDataPipeline()
    pipeline.fetch_all()
    
    # Get congressional trades for a specific ticker
    trades = pipeline.get_congress_trades("NVDA")
    
    # Get insider activity
    insiders = pipeline.get_insider_trades("NVDA")
    
    # Get everything formatted for the screener
    screener_data = pipeline.get_screener_input(["NVDA", "MSFT", "LMT"])
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import time
import os
import re

# ============================================================================
# CONFIG
# ============================================================================

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CACHE_EXPIRY_HOURS = 12  # re-fetch after 12h

# Congressional leadership & committee info for weighting
NOTABLE_MEMBERS = {
    # Name variations → canonical info
    "Nancy Pelosi": {"party": "D", "role": "leadership", "weight": 3.0},
    "Pelosi, Nancy": {"party": "D", "role": "leadership", "weight": 3.0},
    "Tommy Tuberville": {"party": "R", "role": "committee", "weight": 2.5},
    "Tuberville, Tommy": {"party": "R", "role": "committee", "weight": 2.5},
    "Michael McCaul": {"party": "R", "role": "committee_chair", "weight": 2.5},
    "McCaul, Michael T.": {"party": "R", "role": "committee_chair", "weight": 2.5},
    "Dan Crenshaw": {"party": "R", "role": "committee", "weight": 2.0},
    "Crenshaw, Dan": {"party": "R", "role": "committee", "weight": 2.0},
    "Josh Gottheimer": {"party": "D", "role": "committee", "weight": 2.0},
    "Gottheimer, Josh": {"party": "D", "role": "committee", "weight": 2.0},
    "Marjorie Taylor Greene": {"party": "R", "role": "rank", "weight": 1.5},
    "Greene, Marjorie Taylor": {"party": "R", "role": "rank", "weight": 1.5},
    "Mark Green": {"party": "R", "role": "committee", "weight": 2.0},
    "Green, Mark E.": {"party": "R", "role": "committee", "weight": 2.0},
    "Ro Khanna": {"party": "D", "role": "rank", "weight": 1.5},
    "Khanna, Ro": {"party": "D", "role": "rank", "weight": 1.5},
    "Markwayne Mullin": {"party": "R", "role": "committee", "weight": 2.5},
    "Mullin, Markwayne": {"party": "R", "role": "committee", "weight": 2.5},
    "David Rouzer": {"party": "R", "role": "rank", "weight": 1.5},
    "Rouzer, David": {"party": "R", "role": "rank", "weight": 1.5},
}


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_valid(filepath):
    """Check if cache file exists and is fresh enough."""
    if not os.path.exists(filepath):
        return False
    age = time.time() - os.path.getmtime(filepath)
    return age < (CACHE_EXPIRY_HOURS * 3600)


def _normalize_name(name):
    """Try to match name to NOTABLE_MEMBERS."""
    if name in NOTABLE_MEMBERS:
        return name, NOTABLE_MEMBERS[name]
    # Try "Last, First" → "First Last"
    if "," in name:
        parts = name.split(",", 1)
        flipped = f"{parts[1].strip()} {parts[0].strip()}"
        if flipped in NOTABLE_MEMBERS:
            return flipped, NOTABLE_MEMBERS[flipped]
    # Try partial match
    name_lower = name.lower()
    for key, info in NOTABLE_MEMBERS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return key, info
    return name, None


# ============================================================================
# SENATE DATA (GitHub Repo — JSON)
# ============================================================================

class SenateDataFetcher:
    """
    Fetches senate trading data from timothycarambat/senate-stock-watcher-data.
    The repo has an 'aggregate' folder with pre-compiled JSON.
    Also fetches from senatestockwatcher.com API as backup.
    """
    
    AGGREGATE_URL = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json"
    
    def fetch(self) -> pd.DataFrame:
        """Fetch all senate transactions, return as DataFrame."""
        _ensure_cache_dir()
        cache_file = os.path.join(CACHE_DIR, "senate_trades.parquet")
        
        if _cache_valid(cache_file):
            print("  [Senate] Loading from cache...")
            return pd.read_parquet(cache_file)
        
        print("  [Senate] Fetching from GitHub...")
        try:
            resp = requests.get(self.AGGREGATE_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [Senate] GitHub failed: {e}")
            print("  [Senate] Trying senatestockwatcher.com API...")
            try:
                resp = requests.get(
                    "https://senatestockwatcher.com/api",
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e2:
                print(f"  [Senate] All sources failed: {e2}")
                return pd.DataFrame()
        
        # Parse JSON into flat records
        records = []
        for entry in data:
            senator = f"{entry.get('first_name', '')} {entry.get('last_name', '')}".strip()
            for tx in entry.get("transactions", []):
                ticker = tx.get("ticker", "")
                if ticker == "--" or not ticker:
                    continue
                records.append({
                    "member": senator,
                    "chamber": "Senate",
                    "ticker": ticker.upper(),
                    "asset_description": tx.get("asset_description", ""),
                    "asset_type": tx.get("asset_type", "Stock"),
                    "type": tx.get("type", "").lower(),  # purchase / sale / exchange
                    "amount": tx.get("amount", ""),
                    "transaction_date": tx.get("transaction_date", ""),
                    "owner": tx.get("owner", ""),
                    "comment": tx.get("comment", ""),
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
            df = df.dropna(subset=["transaction_date"])
            df = df.sort_values("transaction_date", ascending=False)
            df.to_parquet(cache_file, index=False)
        
        print(f"  [Senate] Loaded {len(df)} transactions")
        return df


# ============================================================================
# HOUSE DATA (S3 Bulk Download — JSON)
# ============================================================================

class HouseDataFetcher:
    """
    Fetches House representative trading data from the public S3 bucket
    used by housestockwatcher.com (same data as the website API).
    """
    
    BULK_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
    
    def fetch(self) -> pd.DataFrame:
        """Fetch all house transactions."""
        _ensure_cache_dir()
        cache_file = os.path.join(CACHE_DIR, "house_trades.parquet")
        
        if _cache_valid(cache_file):
            print("  [House] Loading from cache...")
            return pd.read_parquet(cache_file)
        
        print("  [House] Fetching from S3 bulk download...")
        try:
            resp = requests.get(self.BULK_URL, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [House] S3 failed: {e}")
            return pd.DataFrame()
        
        records = []
        for tx in data:
            ticker = tx.get("ticker", "")
            if ticker == "--" or not ticker or ticker == "N/A":
                continue
            
            records.append({
                "member": tx.get("representative", ""),
                "chamber": "House",
                "ticker": ticker.upper().strip(),
                "asset_description": tx.get("asset_description", ""),
                "asset_type": tx.get("type", "Stock"),  # purchase / sale / etc
                "type": tx.get("type", "").lower(),
                "amount": tx.get("amount", ""),
                "transaction_date": tx.get("transaction_date", ""),
                "disclosure_date": tx.get("disclosure_date", ""),
                "owner": tx.get("owner", ""),
                "district": tx.get("district", ""),
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
            df = df.dropna(subset=["transaction_date"])
            df = df.sort_values("transaction_date", ascending=False)
            df.to_parquet(cache_file, index=False)
        
        print(f"  [House] Loaded {len(df)} transactions")
        return df


# ============================================================================
# INSIDER TRADES (OpenInsider Scrape — SEC Form 4)
# ============================================================================

class InsiderDataFetcher:
    """
    Scrapes insider trading data from OpenInsider.com (free, no auth).
    SEC Form 4 filings — corporate insiders (CEO, CFO, directors, 10%+ owners).
    """
    
    BASE_URL = "http://openinsider.com/screener"
    
    def fetch_for_ticker(self, ticker: str, days_back: int = 180) -> pd.DataFrame:
        """Fetch insider trades for a specific ticker."""
        _ensure_cache_dir()
        cache_file = os.path.join(CACHE_DIR, f"insider_{ticker}.parquet")
        
        if _cache_valid(cache_file):
            return pd.read_parquet(cache_file)
        
        print(f"  [Insider] Fetching {ticker} from OpenInsider...")
        
        params = {
            "s": ticker,
            "o": "",
            "pl": "",
            "ph": "",
            "st": "",
            "td": "0",
            "tdr": "",
            "fdlyl": "",
            "fdlyh": "",
            "dtefrom": "",
            "dteto": "",
            "xp": "1",  # exclude planned trades
            "vtefrom": "",
            "vteto": "",
            "tefrom": "",
            "teto": "",
            "hession": "",
            "sortcol": "0",
            "maxresults": "100",
            "start": "0",
        }
        
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15,
                              headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception as e:
            print(f"  [Insider] Failed for {ticker}: {e}")
            return pd.DataFrame()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        
        if not table:
            return pd.DataFrame()
        
        records = []
        rows = table.find_all("tr")[1:]  # skip header
        
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 12:
                continue
            try:
                records.append({
                    "ticker": ticker.upper(),
                    "filing_date": cols[1].text.strip(),
                    "trade_date": cols[2].text.strip(),
                    "insider_name": cols[4].text.strip(),
                    "title": cols[5].text.strip(),
                    "trade_type": cols[6].text.strip(),  # P (Purchase) / S (Sale)
                    "price": cols[7].text.strip().replace("$", "").replace(",", ""),
                    "qty": cols[8].text.strip().replace(",", "").replace("+", ""),
                    "owned": cols[9].text.strip().replace(",", ""),
                    "value": cols[11].text.strip().replace("$", "").replace(",", "").replace("+", ""),
                })
            except (IndexError, ValueError):
                continue
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
            df = df.dropna(subset=["trade_date"])
            
            # Filter to recent
            cutoff = datetime.now() - timedelta(days=days_back)
            df = df[df["trade_date"] >= cutoff]
            df.to_parquet(cache_file, index=False)
        
        return df
    
    def get_net_sentiment(self, df: pd.DataFrame) -> dict:
        """Summarize insider activity into net sentiment."""
        if df.empty:
            return {"active": False, "direction": None, "net_shares": 0, "details": []}
        
        buys = df[df["trade_type"].str.upper().str.startswith("P")]
        sells = df[df["trade_type"].str.upper().str.startswith("S")]
        
        buy_count = len(buys)
        sell_count = len(sells)
        
        try:
            buy_value = buys["value"].astype(float).sum()
            sell_value = sells["value"].astype(float).sum()
        except:
            buy_value = 0
            sell_value = 0
        
        try:
            net_shares = buys["qty"].astype(float).sum() - sells["qty"].astype(float).sum()
        except:
            net_shares = 0
        
        if buy_count > sell_count and buy_value > sell_value:
            direction = "BUY"
        elif sell_count > buy_count:
            direction = "SELL"
        else:
            direction = "MIXED"
        
        return {
            "active": True,
            "direction": direction,
            "net_shares": int(net_shares),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "notable_insiders": df["insider_name"].head(5).tolist(),
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class FreeDataPipeline:
    """
    Orchestrates all free data sources and produces
    screener-ready input data.
    """
    
    def __init__(self):
        self.senate_fetcher = SenateDataFetcher()
        self.house_fetcher = HouseDataFetcher()
        self.insider_fetcher = InsiderDataFetcher()
        
        self.senate_df = pd.DataFrame()
        self.house_df = pd.DataFrame()
        self.congress_df = pd.DataFrame()  # combined
    
    def fetch_all(self):
        """Fetch all congressional data (senate + house)."""
        print("\n📡 Fetching free data sources...\n")
        
        self.senate_df = self.senate_fetcher.fetch()
        
        # Rate limit courtesy
        time.sleep(1)
        
        self.house_df = self.house_fetcher.fetch()
        
        # Combine
        frames = []
        if not self.senate_df.empty:
            frames.append(self.senate_df)
        if not self.house_df.empty:
            frames.append(self.house_df)
        
        if frames:
            # Align columns
            common_cols = ["member", "chamber", "ticker", "asset_description",
                          "type", "amount", "transaction_date", "owner"]
            for f in frames:
                for col in common_cols:
                    if col not in f.columns:
                        f[col] = ""
            
            self.congress_df = pd.concat([f[common_cols] for f in frames], ignore_index=True)
            self.congress_df = self.congress_df.sort_values("transaction_date", ascending=False)
        
        total = len(self.congress_df)
        print(f"\n✅ Total congressional trades loaded: {total:,}")
        
        if total > 0:
            newest = self.congress_df["transaction_date"].max()
            oldest = self.congress_df["transaction_date"].min()
            print(f"   Date range: {oldest.strftime('%Y-%m-%d')} → {newest.strftime('%Y-%m-%d')}")
            unique_members = self.congress_df["member"].nunique()
            unique_tickers = self.congress_df["ticker"].nunique()
            print(f"   {unique_members} unique members, {unique_tickers} unique tickers")
        
        return self
    
    def get_congress_trades(self, ticker: str, days_back: int = 365) -> list:
        """
        Get congressional trades for a ticker, formatted for the screener.
        Returns list of dicts compatible with CongressionalAnalyzer.
        """
        if self.congress_df.empty:
            return []
        
        cutoff = datetime.now() - timedelta(days=days_back)
        mask = (
            (self.congress_df["ticker"] == ticker.upper()) &
            (self.congress_df["transaction_date"] >= cutoff)
        )
        trades = self.congress_df[mask]
        
        result = []
        for _, row in trades.iterrows():
            member_name = row["member"]
            canonical_name, member_info = _normalize_name(member_name)
            
            # Determine instrument type from asset description
            desc = str(row.get("asset_description", "")).lower()
            if "call" in desc or "option" in desc:
                instrument = "option_call"
            elif "put" in desc:
                instrument = "option_put"
            else:
                instrument = "stock"
            
            # Normalize trade type
            trade_type = str(row["type"]).lower()
            if "purchase" in trade_type or "buy" in trade_type:
                normalized_type = "purchase"
            elif "sale" in trade_type or "sell" in trade_type:
                normalized_type = "sale"
            else:
                normalized_type = trade_type
            
            result.append({
                "member": canonical_name,
                "type": normalized_type,
                "amount": row.get("amount", ""),
                "date": row["transaction_date"].strftime("%Y-%m-%d"),
                "instrument": instrument,
                "chamber": row.get("chamber", ""),
                "owner": row.get("owner", ""),
                "is_notable": member_info is not None,
                "role": member_info["role"] if member_info else "rank",
                "weight": member_info["weight"] if member_info else 1.0,
            })
        
        return result
    
    def get_insider_trades(self, ticker: str) -> dict:
        """Get insider trading sentiment for a ticker."""
        df = self.insider_fetcher.fetch_for_ticker(ticker)
        return self.insider_fetcher.get_net_sentiment(df)
    
    def get_screener_input(self, tickers: list, days_back: int = 365) -> dict:
        """
        Get all data formatted for the MultiFactorScreener.
        Returns dict: {ticker: list_of_trades}
        """
        result = {}
        for ticker in tickers:
            trades = self.get_congress_trades(ticker, days_back)
            if trades:
                result[ticker] = trades
        return result
    
    def get_full_convergence_data(self, ticker: str) -> dict:
        """
        Get full convergence data for a single ticker:
        congressional + insider activity.
        """
        congress = self.get_congress_trades(ticker)
        insider = self.get_insider_trades(ticker)
        
        # Congressional summary
        buys = [t for t in congress if t["type"] == "purchase"]
        sells = [t for t in congress if t["type"] == "sale"]
        notable = [t for t in congress if t["is_notable"]]
        
        congress_summary = {
            "active": len(congress) > 0,
            "direction": "BUY" if len(buys) > len(sells) else "SELL" if len(sells) > len(buys) else "MIXED",
            "total_trades": len(congress),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "notable_members": list(set(t["member"] for t in notable)),
            "has_options": any(t["instrument"].startswith("option") for t in congress),
            "trades": congress,
        }
        
        return {
            "ticker": ticker,
            "congressional": congress_summary,
            "insider": insider,
        }
    
    def summary(self):
        """Print a summary of most-traded tickers by congress."""
        if self.congress_df.empty:
            print("No data loaded. Run fetch_all() first.")
            return
        
        # Last 180 days
        cutoff = datetime.now() - timedelta(days=180)
        recent = self.congress_df[self.congress_df["transaction_date"] >= cutoff]
        
        print(f"\n{'='*60}")
        print(f"  TOP 20 MOST TRADED TICKERS BY CONGRESS (last 180d)")
        print(f"{'='*60}\n")
        
        ticker_stats = []
        for ticker, group in recent.groupby("ticker"):
            buys = group[group["type"].str.contains("purchase|buy", case=False, na=False)]
            sells = group[group["type"].str.contains("sale|sell", case=False, na=False)]
            
            notable_count = 0
            for _, row in group.iterrows():
                _, info = _normalize_name(row["member"])
                if info:
                    notable_count += 1
            
            ticker_stats.append({
                "ticker": ticker,
                "total": len(group),
                "buys": len(buys),
                "sells": len(sells),
                "unique_members": group["member"].nunique(),
                "notable_trades": notable_count,
                "net_sentiment": "🟢 BUY" if len(buys) > len(sells) else "🔴 SELL" if len(sells) > len(buys) else "🟡 MIXED",
                "last_trade": group["transaction_date"].max().strftime("%Y-%m-%d"),
            })
        
        stats_df = pd.DataFrame(ticker_stats)
        stats_df = stats_df.sort_values("total", ascending=False).head(20)
        
        for i, row in stats_df.iterrows():
            notable_flag = " ⭐" if row["notable_trades"] > 0 else ""
            print(f"  {row['ticker']:6s}  {row['total']:3d} trades  "
                  f"(Buy:{row['buys']:2d} Sell:{row['sells']:2d})  "
                  f"{row['unique_members']:2d} members  "
                  f"{row['net_sentiment']}  "
                  f"Last: {row['last_trade']}{notable_flag}")
        
        print(f"\n  ⭐ = Notable trader (leadership/committee) involved")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    pipeline = FreeDataPipeline()
    pipeline.fetch_all()
    pipeline.summary()
    
    # If tickers passed as args, show convergence data
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["NVDA", "MSFT", "GOOGL", "LMT", "TSLA"]
    
    print(f"\n{'='*60}")
    print(f"  CONVERGENCE DATA")
    print(f"{'='*60}")
    
    for ticker in tickers:
        data = pipeline.get_full_convergence_data(ticker)
        cong = data["congressional"]
        ins = data["insider"]
        
        print(f"\n  {ticker}")
        print(f"  {'─'*40}")
        
        if cong["active"]:
            print(f"  Congress: {cong['direction']} "
                  f"({cong['buy_count']} buys, {cong['sell_count']} sells)")
            if cong["notable_members"]:
                print(f"  Notable:  {', '.join(cong['notable_members'])}")
            if cong["has_options"]:
                print(f"  ⚡ Options activity detected (higher conviction)")
        else:
            print(f"  Congress: No recent activity")
        
        if ins.get("active"):
            print(f"  Insider:  {ins['direction']} "
                  f"(Net shares: {ins['net_shares']:+,})")
        else:
            print(f"  Insider:  Fetching skipped (run with --insider flag)")
    
    print()
