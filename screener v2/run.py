"""
Edge Scanner — Full Pipeline Runner
=====================================
Connects free data sources → multi-factor screener → trade signals.

Usage:
    python run.py                          # Screen default watchlist
    python run.py NVDA MSFT LMT TSLA      # Screen specific tickers
    python run.py --top-congress           # Screen most-traded by congress
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from data_sources import FreeDataPipeline
from screener import MultiFactorScreener

def main():
    # ── Parse args ──────────────────────────────────────────────
    if "--top-congress" in sys.argv:
        mode = "top_congress"
        tickers = []
    elif len(sys.argv) > 1:
        mode = "custom"
        tickers = [t.upper() for t in sys.argv[1:] if not t.startswith("--")]
    else:
        mode = "default"
        tickers = [
            "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL",
            "TSLA", "META", "LMT", "RTX", "JPM",
            "GS", "XOM", "PLTR", "JNJ", "UNH",
        ]
    
    # ── Step 1: Fetch free data ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  EDGE SCANNER — FREE DATA PIPELINE")
    print("=" * 60)
    
    pipeline = FreeDataPipeline()
    pipeline.fetch_all()
    
    # If top-congress mode, get most-traded tickers
    if mode == "top_congress":
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=180)
        recent = pipeline.congress_df[pipeline.congress_df["transaction_date"] >= cutoff]
        top = recent.groupby("ticker").size().sort_values(ascending=False).head(20)
        tickers = top.index.tolist()
        print(f"\n  Using top {len(tickers)} congress-traded tickers: {', '.join(tickers)}")
    
    # Show congress summary
    pipeline.summary()
    
    # ── Step 2: Get congressional data for screener ─────────────
    print(f"\n{'='*60}")
    print(f"  PREPARING SCREENER DATA")
    print(f"{'='*60}\n")
    
    congressional_data = pipeline.get_screener_input(tickers)
    
    for ticker in tickers:
        trades = congressional_data.get(ticker, [])
        if trades:
            buys = sum(1 for t in trades if t["type"] == "purchase")
            sells = sum(1 for t in trades if t["type"] == "sale")
            notable = [t["member"] for t in trades if t["is_notable"]]
            flag = f" ⭐ {', '.join(set(notable))}" if notable else ""
            print(f"  {ticker:6s}: {len(trades)} trades (B:{buys} S:{sells}){flag}")
        else:
            print(f"  {ticker:6s}: No congressional trades")
    
    # ── Step 3: Run screener ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RUNNING MULTI-FACTOR SCREENER")
    print(f"{'='*60}\n")
    
    # Weights: congressional gets 10%, rest split among others
    weights = {
        'fundamental': 0.30,
        'technical': 0.25,
        'relative_strength': 0.20,
        'congressional': 0.10,
        'quality': 0.15,
    }
    
    screener = MultiFactorScreener(weights=weights)
    df, full_results = screener.screen(tickers, congressional_data)
    
    if df.empty:
        print("  No results. Check internet connection.")
        return
    
    # ── Step 4: Display results ─────────────────────────────────
    from screener import format_market_cap
    
    display_df = df.copy()
    display_df['Market Cap'] = display_df['Market Cap'].apply(format_market_cap)
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.2f}" if x else 'N/A')
    
    cols = ['Ticker', 'Name', 'Price', 'COMPOSITE', 'Signal',
            'Fundamental', 'Technical', 'Rel. Strength', 'Quality',
            'Congressional', 'F-Score', 'RSI']
    
    print("\n" + "=" * 60)
    print("  FINAL RANKING")
    print("=" * 60 + "\n")
    print(display_df[cols].to_string())
    
    # ── Step 5: Convergence summary ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONVERGENCE SIGNALS — TOP ACTIONABLE")
    print(f"{'='*60}")
    
    for r in full_results[:5]:
        if "error" in r:
            continue
        
        ticker = r["ticker"]
        cong = r["congressional"]
        
        # Count convergence signals
        signals = 0
        signal_labels = []
        
        # Congressional signal
        if cong["trade_count"] > 0 and cong["net_sentiment"] == "bullish":
            signals += 1
            signal_labels.append("Congress BUY")
        elif cong["trade_count"] > 0 and cong["net_sentiment"] == "bearish":
            signal_labels.append("Congress SELL")
        
        # Technical signal
        tech = r["technical"]["indicators"]
        if tech.get("above_sma200") and tech.get("rsi_14", 50) > 40:
            signals += 1
            signal_labels.append("Tech ✓")
        
        # Fundamental signal
        if r["fundamental"]["composite"] > 60:
            signals += 1
            signal_labels.append("Fund ✓")
        
        # Quality signal
        if r["quality"]["f_score"] >= 7:
            signals += 1
            signal_labels.append("Quality ✓")
        
        # RS signal
        if r["relative_strength"]["vs_spy"].get("mansfield_rs", 0) > 0:
            signals += 1
            signal_labels.append("RS ✓")
        
        bar = "█" * signals + "░" * (5 - signals)
        
        print(f"\n  {ticker:6s} [{bar}] {signals}/5 convergence")
        print(f"          Score: {r['composite_score']}/100 → {r['signal']}")
        print(f"          Signals: {' | '.join(signal_labels)}")
        
        if cong["notable_traders"]:
            print(f"          Notable: {', '.join(cong['notable_traders'])}")
    
    print(f"\n{'='*60}")
    print(f"  Weights: Fund={weights['fundamental']:.0%} | "
          f"Tech={weights['technical']:.0%} | "
          f"RS={weights['relative_strength']:.0%} | "
          f"Congress={weights['congressional']:.0%} | "
          f"Quality={weights['quality']:.0%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
