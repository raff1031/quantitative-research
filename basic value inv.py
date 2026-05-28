# comprehensive_comparison.py - Confronto completo di approcci
import os
import time
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================= CONFIG =================================

VERBOSE = True
YF_SLEEP = 0.02

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    per_trade_usd: float = 50.0
    start_year: int = 2010                # Riduciamo a 2010 per avere dati più affidabili
    rolling_window: int = 60
    min_periods: int = 30
    entry_k_std: float = 0.0
    exit_k_std: float = 1.0
    accumulate_daily: bool = True
    fractional_shares: bool = True
    max_alloc_per_ticker_frac: float = 0.02  # Ridotto a 2% se usiamo tutti i titoli
    
    # Parametri per confronto
    rebalance_frequency_years: int = 2    # Ogni 2 anni
    min_trading_days: int = 100
    min_market_cap_billion: float = 1.0   # Min 1B market cap per essere realistico
    
    transaction_cost_per_trade: float = 0.0
    random_seed: int = 42

# ============================= REALISTIC UNIVERSE ===================

# Universo più realistico con focus su titoli che esistevano davvero
REALISTIC_LARGE_CAP = [
    # Tech che esisteva nel 2010
    "AAPL","MSFT","AMZN","GOOGL","GOOG","ORCL","CSCO","INTC","IBM","HPQ","DELL","QCOM","AMAT","ADI","NVDA","AMD",
    
    # Finance solidi
    "JPM","BAC","WFC","GS","MS","C","USB","PNC","COF","AXP","BLK","SCHW","BBT","SunTrust","KEY","CMA","FITB","RF",
    
    # Healthcare
    "JNJ","PFE","MRK","ABBV","LLY","UNH","CVS","WYE","BMY","ABT","GILD","AMGN","MDT","TMO","DHR","BSX","ISRG","REGN",
    
    # Consumer & Retail
    "WMT","HD","TGT","COST","LOW","KO","PEP","PG","MCD","SBUX","NKE","TJX","KSS","M","JCP","GPS","ANF","AEO",
    
    # Energy (pre-shale boom)
    "XOM","CVX","COP","SLB","HAL","OXY","APA","DVN","EOG","NBR","RIG","DO",
    
    # Industrials
    "GE","CAT","MMM","HON","UTX","BA","LMT","NOC","GD","UPS","FDX","DE","EMR","ITW","DOV",
    
    # Materials
    "DD","DOW","LYB","APD","PPG","SHW","ECL","CLF","X","AKS","FCX","NEM",
    
    # Utilities
    "GE","D","SO","NEE","DUK","EXC","PCG","ED","AEP","XEL",
    
    # REITs
    "AMT","PLD","PSA","EQR","AVB","UDR","CPT","MAA","ESS","BXP","SPG","REG","KIM",
    
    # Media & Telecom
    "DIS","TWX","CBS","VIA","T","VZ","S","WIN","CMCSA","CHTR","DISH"
]

def get_realistic_universe() -> List[str]:
    return REALISTIC_LARGE_CAP

# ============================= UTILS ==================================

def today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def years_ago_date(years: int) -> str:
    return (datetime.utcnow() - timedelta(days=int(years*365.25))).strftime("%Y-%m-%d")

# ========================= Data fetching ===============================

def fetch_prices_yf(symbol: str, start: str, end: str) -> pd.Series:
    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        if data is None or len(data) == 0:
            return pd.Series(dtype=float)
        
        close = data["Close"] if "Close" in data.columns else data["Adj Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.name = "price"
        return close
    except Exception:
        return pd.Series(dtype=float)

def fetch_yf_fundamentals(symbol: str) -> Tuple[pd.Series, float]:
    try:
        ticker = yf.Ticker(symbol)
        
        # Priorità ai dati annual
        annual_financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
        
        financials = None
        
        if annual_financials is not None and not annual_financials.empty:
            financials = annual_financials
        elif quarterly_financials is not None and not quarterly_financials.empty:
            financials = quarterly_financials
        
        if financials is None or financials.empty:
            return pd.Series(dtype=float), np.nan
        
        # Cerca ricavi
        revenue_row = None
        for idx in financials.index:
            idx_str = str(idx).lower()
            if any(term in idx_str for term in ["total revenue", "revenue", "net sales", "sales"]):
                revenue_row = idx
                break
        
        if revenue_row is None:
            return pd.Series(dtype=float), np.nan
        
        revenue_series = financials.loc[revenue_row]
        revenue_series = pd.to_numeric(revenue_series, errors="coerce").dropna()
        revenue_series = revenue_series.sort_index()
        
        # Shares outstanding
        info = ticker.info
        shares = info.get("sharesOutstanding", None)
        if shares is None:
            shares = info.get("impliedSharesOutstanding", None)
        if shares is None:
            shares = info.get("floatShares", None)
        
        shares = float(shares) if shares and np.isfinite(float(shares)) else np.nan
        
        return revenue_series, shares
        
    except Exception:
        return pd.Series(dtype=float), np.nan

def compute_ps_from_yf_data(price: pd.Series, revenue_series: pd.Series, shares_outstanding: float) -> pd.Series:
    if revenue_series.empty or not np.isfinite(shares_outstanding) or shares_outstanding <= 0:
        return pd.Series(index=price.index, dtype=float)
    
    # Gestisce sia quarterly che annual
    if len(revenue_series) >= 4:
        time_diffs = revenue_series.index.to_series().diff().dt.days.dropna()
        avg_diff = time_diffs.mean()
        
        if avg_diff < 150:  # quarterly
            revenue_use = revenue_series.rolling(window=4, min_periods=1).sum()
        else:  # annual
            revenue_use = revenue_series
    else:
        revenue_use = revenue_series
    
    revenue_use = revenue_use.dropna()
    if revenue_use.empty:
        return pd.Series(index=price.index, dtype=float)
    
    # Revenue per share
    revps = revenue_use / shares_outstanding
    revps_daily = revps.reindex(price.index, method="ffill")
    
    # P/S giornaliero
    ps_daily = price / revps_daily
    ps_daily = ps_daily.replace([np.inf, -np.inf], np.nan)
    ps_daily = ps_daily[(ps_daily > 0) & np.isfinite(ps_daily)]
    
    return ps_daily

def get_approximate_market_cap(ticker: str, price: float) -> float:
    """Stima approssimativa market cap"""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        shares = info.get("sharesOutstanding", None)
        if shares is None:
            shares = info.get("impliedSharesOutstanding", None)
        if shares is None:
            shares = info.get("floatShares", None)
            
        if shares and np.isfinite(float(shares)):
            return float(shares) * price / 1e9  # in billions
        else:
            return np.nan
    except:
        return np.nan

# ========================= Selection Strategies ========================

def select_all_available(all_price_data: Dict[str, pd.Series], 
                        period_start: pd.Timestamp, 
                        period_end: pd.Timestamp,
                        cfg: BacktestConfig) -> List[str]:
    """APPROACH A: Usa TUTTI i titoli disponibili"""
    
    candidates = []
    
    for ticker, price_series in all_price_data.items():
        period_prices = price_series.loc[
            (price_series.index >= period_start) & 
            (price_series.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= cfg.min_trading_days:
            candidates.append(ticker)
    
    print(f"  APPROACH A - Tutti disponibili: {len(candidates)} titoli")
    return candidates

def select_top_market_cap(all_price_data: Dict[str, pd.Series], 
                         period_start: pd.Timestamp, 
                         period_end: pd.Timestamp,
                         cfg: BacktestConfig,
                         top_n: int = 50) -> List[str]:
    """APPROACH B: Solo top N per market cap (simula S&P 500)"""
    
    candidates = []
    
    for ticker, price_series in all_price_data.items():
        period_prices = price_series.loc[
            (price_series.index >= period_start) & 
            (price_series.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= cfg.min_trading_days:
            # Usa prezzo medio del periodo per stima market cap
            avg_price = period_prices.mean()
            market_cap = get_approximate_market_cap(ticker, avg_price)
            
            if pd.notna(market_cap) and market_cap >= cfg.min_market_cap_billion:
                candidates.append({
                    'ticker': ticker,
                    'market_cap': market_cap,
                    'avg_price': avg_price
                })
    
    # Ordina per market cap e prendi top N
    candidates_df = pd.DataFrame(candidates)
    if candidates_df.empty:
        print(f"  APPROACH B - Nessun candidato valido")
        return []
    
    candidates_df = candidates_df.sort_values(['market_cap'], ascending=False)
    selected = candidates_df.head(top_n)['ticker'].tolist()
    
    print(f"  APPROACH B - Top {top_n} market cap: {len(selected)} titoli (da {len(candidates)} candidati)")
    return selected

# ========================= Backtest Logic ==============================

def compute_signals_from_ps(price: pd.Series, ps_series: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    df["ps"] = ps_series.reindex(df.index)
    df.loc[~np.isfinite(df["ps"]), "ps"] = np.nan

    # Rolling stats
    df["ps_ma"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).mean()
    df["ps_std"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).std()

    # Soglie
    df["entry_threshold"] = df["ps_ma"] + cfg.entry_k_std * df["ps_std"]
    df["exit_threshold"] = df["ps_ma"] + cfg.exit_k_std * df["ps_std"]

    # Segnali base
    df["entry_cond"] = (df["ps"] < df["entry_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    df["exit_cond"] = (df["ps"] > df["exit_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    
    return df

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    action: str
    price: float
    qty: float
    cash_after: float

def simple_backtest(signal_map: Dict[str, pd.DataFrame], cfg: BacktestConfig, strategy_name: str = "") -> Tuple[pd.DataFrame, List[Trade]]:
    """Backtest semplificato"""
    
    first_idx = min(df.index.min() for df in signal_map.values())
    last_idx = max(df.index.max() for df in signal_map.values())
    
    spy = fetch_prices_yf("SPY", start=first_idx.strftime("%Y-%m-%d"), end=last_idx.strftime("%Y-%m-%d"))
    calendar = spy.index

    cash = cfg.initial_capital
    holdings_qty: Dict[str, float] = {t: 0.0 for t in signal_map.keys()}
    trades: List[Trade] = []
    equity_curve = []

    price_map: Dict[str, pd.Series] = {t: df["price"].reindex(calendar) for t, df in signal_map.items()}
    np.random.seed(cfg.random_seed)

    def portfolio_value(day: pd.Timestamp) -> float:
        val = cash
        for t, qty in holdings_qty.items():
            if qty == 0:
                continue
            p = price_map[t].get(day, np.nan)
            if np.isfinite(p):
                val += qty * p
        return val

    for day in calendar:
        port_val = portfolio_value(day)
        
        # 1) SELL prima
        for t, df in signal_map.items():
            price = price_map[t].get(day, np.nan)
            if not np.isfinite(price):
                continue
            if holdings_qty[t] > 0 and bool(df["exit_cond"].reindex(calendar).get(day, False)):
                qty = holdings_qty[t]
                proceeds = qty * price
                cash += proceeds
                holdings_qty[t] = 0.0
                trades.append(Trade(day, t, "SELL", float(price), float(qty), float(cash)))

        # Aggiorna port_val dopo vendite
        port_val = portfolio_value(day)

        # 2) BUY 
        for t, df in signal_map.items():
            price = price_map[t].get(day, np.nan)
            if not np.isfinite(price):
                continue

            entry_signal = bool(df["entry_cond"].reindex(calendar).get(day, False))
            if not entry_signal:
                continue
            
            # Controlli allocazione ticker
            max_alloc_for_ticker = cfg.max_alloc_per_ticker_frac * port_val
            current_alloc = holdings_qty[t] * price if holdings_qty[t] > 0 else 0
            remaining_alloc = max(0.0, max_alloc_for_ticker - current_alloc)
            amount = min(cfg.per_trade_usd, cash, remaining_alloc)
            
            if amount <= 0:
                continue
                
            qty = amount / price if cfg.fractional_shares else math.floor(amount / price)
            if qty <= 0:
                continue
                
            cash -= qty * price
            holdings_qty[t] += qty
            trades.append(Trade(day, t, "BUY", float(price), float(qty), float(cash)))

        equity_curve.append({"date": day, "equity": portfolio_value(day), "cash": cash})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    
    if VERBOSE and strategy_name:
        eq = equity_df["equity"].dropna()
        if len(eq) > 1:
            total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
            print(f"    {strategy_name}: {total_return:+.1%} total return, {len(trades)} trades")
    
    return equity_df, trades

def buy_hold_benchmark(start_date: str, end_date: str, initial_capital: float = 10000) -> pd.DataFrame:
    """Buy & Hold SPY benchmark"""
    spy = fetch_prices_yf("SPY", start_date, end_date)
    if spy.empty:
        return pd.DataFrame()
    
    # Compra tutto SPY all'inizio
    initial_shares = initial_capital / spy.iloc[0]
    equity_curve = spy * initial_shares
    
    return pd.DataFrame({"equity": equity_curve})

# ========================= COMPREHENSIVE COMPARISON ====================

def run_comprehensive_comparison(cfg: BacktestConfig):
    """Confronto completo: Tutti vs Top50 vs Buy&Hold"""
    print("=== CONFRONTO COMPLETO: 3 APPROCCI ===")
    
    # 1. Scarica dati universo realistico
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    universe_tickers = get_realistic_universe()
    print(f"Scarico dati per {len(universe_tickers)} titoli dal {cfg.start_year}...")
    
    all_price_data = {}
    
    for sym in tqdm(universe_tickers):
        price = fetch_prices_yf(sym, start_date, end_date)
        if not price.empty and len(price) > 200:  # Almeno 200 giorni
            all_price_data[sym] = price
        time.sleep(YF_SLEEP)
    
    print(f"Dati prezzi raccolti per {len(all_price_data)} titoli")
    
    # 2. Crea periodi
    periods = []
    current_year = cfg.start_year
    while current_year < 2025:
        end_year = min(current_year + cfg.rebalance_frequency_years, 2025)
        periods.append((current_year, end_year))
        current_year = end_year
    
    print(f"Creati {len(periods)} periodi di {cfg.rebalance_frequency_years} anni")
    
    # 3. Per ogni periodo, testa 3 approcci
    all_results = {'approach_a': [], 'approach_b': [], 'approach_c': []}
    
    for period_idx, (start_year, end_year) in enumerate(periods):
        print(f"\n--- Periodo {period_idx+1}: {start_year} to {end_year} ---")
        
        period_start = pd.to_datetime(f"{start_year}-01-01")
        period_end = pd.to_datetime(f"{end_year}-01-01")
        
        # APPROACH A: Tutti i titoli disponibili
        universe_all = select_all_available(all_price_data, period_start, period_end, cfg)
        
        # APPROACH B: Top 50 market cap
        universe_top50 = select_top_market_cap(all_price_data, period_start, period_end, cfg, top_n=50)
        
        # Testa entrambi gli approcci
        for approach_name, universe in [("approach_a", universe_all), ("approach_b", universe_top50)]:
            if len(universe) < 5:
                print(f"  {approach_name.upper()}: troppi pochi titoli, skip")
                continue
            
            # Raccogli signal_map
            signal_map = {}
            
            for ticker in universe:
                if ticker not in all_price_data:
                    continue
                    
                price = all_price_data[ticker]
                revenue_series, shares = fetch_yf_fundamentals(ticker)
                
                if revenue_series.empty or not np.isfinite(shares):
                    continue
                    
                ps_series = compute_ps_from_yf_data(price, revenue_series, shares)
                if ps_series.empty:
                    continue
                    
                df = compute_signals_from_ps(price, ps_series, cfg)
                if df["ps_ma"].dropna().empty:
                    continue
                    
                signal_map[ticker] = df
                time.sleep(YF_SLEEP * 0.02)
            
            if len(signal_map) < 3:
                print(f"  {approach_name.upper()}: troppi pochi segnali, skip")
                continue
            
            # Backtest
            equity, trades = simple_backtest(signal_map, cfg, approach_name.upper())
            
            # Calcola performance
            eq = equity["equity"].dropna()
            if len(eq) > 1:
                period_return = eq.iloc[-1] / eq.iloc[0] - 1.0
                annualized_return = (eq.iloc[-1] / eq.iloc[0]) ** (1/cfg.rebalance_frequency_years) - 1.0
            else:
                period_return = 0.0
                annualized_return = 0.0
            
            all_results[approach_name].append({
                'period': f"{start_year}-{end_year}",
                'start_year': start_year,
                'universe_size': len(universe),
                'signals_count': len(signal_map),
                'period_return': period_return,
                'annualized_return': annualized_return,
                'trades_count': len(trades),
                'equity': equity
            })
        
        # APPROACH C: Buy & Hold SPY
        spy_equity = buy_hold_benchmark(period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"), cfg.initial_capital)
        if not spy_equity.empty:
            eq_spy = spy_equity["equity"].dropna()
            if len(eq_spy) > 1:
                spy_return = eq_spy.iloc[-1] / eq_spy.iloc[0] - 1.0
                spy_annualized = (eq_spy.iloc[-1] / eq_spy.iloc[0]) ** (1/cfg.rebalance_frequency_years) - 1.0
            else:
                spy_return = 0.0
                spy_annualized = 0.0
            
            all_results['approach_c'].append({
                'period': f"{start_year}-{end_year}",
                'start_year': start_year,
                'universe_size': 1,
                'signals_count': 1,
                'period_return': spy_return,
                'annualized_return': spy_annualized,
                'trades_count': 1,
                'equity': spy_equity
            })
            
            print(f"    APPROACH C (SPY): {spy_return:+.1%} total return")
    
    # 4. Analisi comparativa
    print(f"\n=== CONFRONTO FINALE ===")
    
    approaches = {
        'A - Tutti i titoli': all_results['approach_a'],
        'B - Top 50 market cap': all_results['approach_b'], 
        'C - Buy & Hold SPY': all_results['approach_c']
    }
    
    # Plot comparativo
    plot_comprehensive_comparison(approaches)
    
    # Statistiche finali
    for name, results in approaches.items():
        if not results:
            continue
            
        returns = [r['annualized_return'] * 100 for r in results]
        trades = [r['trades_count'] for r in results]
        
        print(f"\n{name}:")
        print(f"  Rendimento annualizzato medio: {np.mean(returns):.1f}%")
        print(f"  Rendimento mediano: {np.median(returns):.1f}%")
        print(f"  Volatilità: {np.std(returns):.1f}%")
        print(f"  Trades medi per periodo: {np.mean(trades):.0f}")
        print(f"  Periodi: {len(results)}")
    
    return all_results

def plot_comprehensive_comparison(approaches: Dict[str, List[dict]]):
    """Plot comparativo dei 3 approcci"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green']
    
    # 1. Rendimenti annualizzati per approccio
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        periods = [r['period'] for r in results]
        returns = [r['annualized_return'] * 100 for r in results]
        
        ax1.plot(range(len(periods)), returns, 'o-', color=colors[i], 
                label=name, linewidth=2, markersize=6)
    
    ax1.set_title('Rendimenti Annualizzati per Periodo')
    ax1.set_ylabel('Rendimento Annualizzato (%)')
    ax1.set_xlabel('Periodo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 2. Box plot dei rendimenti
    returns_data = []
    labels = []
    for name, results in approaches.items():
        if results:
            returns_data.append([r['annualized_return'] * 100 for r in results])
            labels.append(name.split(' - ')[0])  # Solo la lettera
    
    if returns_data:
        ax2.boxplot(returns_data, labels=labels)
        ax2.set_title('Distribuzione Rendimenti')
        ax2.set_ylabel('Rendimento Annualizzato (%)')
        ax2.grid(True, alpha=0.3)
    
    # 3. Numero trades
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        periods = [r['period'] for r in results]
        trades = [r['trades_count'] for r in results]
        
        ax3.bar([x + i*0.25 for x in range(len(periods))], trades, 
               width=0.25, alpha=0.7, color=colors[i], label=name)
    
    ax3.set_title('Numero Trades per Periodo')
    ax3.set_ylabel('Numero Trades')
    ax3.set_xlabel('Periodo')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Universo size
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        periods = [r['period'] for r in results]
        universe_sizes = [r['universe_size'] for r in results]
        
        ax4.plot(range(len(periods)), universe_sizes, 'o-', 
                color=colors[i], label=name, linewidth=2, markersize=6)
    
    ax4.set_title('Dimensione Universo')
    ax4.set_ylabel('Numero Titoli')
    ax4.set_xlabel('Periodo')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ====================== MAIN ==========================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        per_trade_usd=50.0,
        start_year=2010,                     # Dal 2010 per dati più affidabili
        rolling_window=60,
        min_periods=30,
        entry_k_std=0.0,
        exit_k_std=1.0,
        accumulate_daily=True,
        fractional_shares=True,
        max_alloc_per_ticker_frac=0.02,      # 2% per titolo se usiamo tutti
        rebalance_frequency_years=2,         # Ogni 2 anni
        min_trading_days=100,
        min_market_cap_billion=1.0,          # Min 1B market cap
    )
    
    print("Avvio confronto completo: Tutti vs Top50 vs SPY...")
    results = run_comprehensive_comparison(cfg)
    
    print(f"\nConfronto completato!")# comprehensive_comparison.py - Confronto completo di approcci
import os
import time
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================= CONFIG =================================

VERBOSE = True
YF_SLEEP = 0.02

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    per_trade_usd: float = 50.0
    start_year: int = 2010                # Riduciamo a 2010 per avere dati più affidabili
    rolling_window: int = 60
    min_periods: int = 30
    entry_k_std: float = 0.0
    exit_k_std: float = 1.0
    accumulate_daily: bool = True
    fractional_shares: bool = True
    max_alloc_per_ticker_frac: float = 0.02  # Ridotto a 2% se usiamo tutti i titoli
    
    # Parametri per confronto
    rebalance_frequency_years: int = 2    # Ogni 2 anni
    min_trading_days: int = 100
    min_market_cap_billion: float = 1.0   # Min 1B market cap per essere realistico
    
    transaction_cost_per_trade: float = 0.0
    random_seed: int = 42

# ============================= REALISTIC UNIVERSE ===================

# Universo più realistico con focus su titoli che esistevano davvero
REALISTIC_LARGE_CAP = [
    # Tech che esisteva nel 2010
    "AAPL","MSFT","AMZN","GOOGL","GOOG","ORCL","CSCO","INTC","IBM","HPQ","DELL","QCOM","AMAT","ADI","NVDA","AMD",
    
    # Finance solidi
    "JPM","BAC","WFC","GS","MS","C","USB","PNC","COF","AXP","BLK","SCHW","BBT","SunTrust","KEY","CMA","FITB","RF",
    
    # Healthcare
    "JNJ","PFE","MRK","ABBV","LLY","UNH","CVS","WYE","BMY","ABT","GILD","AMGN","MDT","TMO","DHR","BSX","ISRG","REGN",
    
    # Consumer & Retail
    "WMT","HD","TGT","COST","LOW","KO","PEP","PG","MCD","SBUX","NKE","TJX","KSS","M","JCP","GPS","ANF","AEO",
    
    # Energy (pre-shale boom)
    "XOM","CVX","COP","SLB","HAL","OXY","APA","DVN","EOG","NBR","RIG","DO",
    
    # Industrials
    "GE","CAT","MMM","HON","UTX","BA","LMT","NOC","GD","UPS","FDX","DE","EMR","ITW","DOV",
    
    # Materials
    "DD","DOW","LYB","APD","PPG","SHW","ECL","CLF","X","AKS","FCX","NEM",
    
    # Utilities
    "GE","D","SO","NEE","DUK","EXC","PCG","ED","AEP","XEL",
    
    # REITs
    "AMT","PLD","PSA","EQR","AVB","UDR","CPT","MAA","ESS","BXP","SPG","REG","KIM",
    
    # Media & Telecom
    "DIS","TWX","CBS","VIA","T","VZ","S","WIN","CMCSA","CHTR","DISH"
]

def get_realistic_universe() -> List[str]:
    return REALISTIC_LARGE_CAP

# ============================= UTILS ==================================

def today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def years_ago_date(years: int) -> str:
    return (datetime.utcnow() - timedelta(days=int(years*365.25))).strftime("%Y-%m-%d")

# ========================= Data fetching ===============================

def fetch_prices_yf(symbol: str, start: str, end: str) -> pd.Series:
    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        if data is None or len(data) == 0:
            return pd.Series(dtype=float)
        
        close = data["Close"] if "Close" in data.columns else data["Adj Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.name = "price"
        return close
    except Exception:
        return pd.Series(dtype=float)

def fetch_yf_fundamentals(symbol: str) -> Tuple[pd.Series, float]:
    try:
        ticker = yf.Ticker(symbol)
        
        # Priorità ai dati annual
        annual_financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
        
        financials = None
        
        if annual_financials is not None and not annual_financials.empty:
            financials = annual_financials
        elif quarterly_financials is not None and not quarterly_financials.empty:
            financials = quarterly_financials
        
        if financials is None or financials.empty:
            return pd.Series(dtype=float), np.nan
        
        # Cerca ricavi
        revenue_row = None
        for idx in financials.index:
            idx_str = str(idx).lower()
            if any(term in idx_str for term in ["total revenue", "revenue", "net sales", "sales"]):
                revenue_row = idx
                break
        
        if revenue_row is None:
            return pd.Series(dtype=float), np.nan
        
        revenue_series = financials.loc[revenue_row]
        revenue_series = pd.to_numeric(revenue_series, errors="coerce").dropna()
        revenue_series = revenue_series.sort_index()
        
        # Shares outstanding
        info = ticker.info
        shares = info.get("sharesOutstanding", None)
        if shares is None:
            shares = info.get("impliedSharesOutstanding", None)
        if shares is None:
            shares = info.get("floatShares", None)
        
        shares = float(shares) if shares and np.isfinite(float(shares)) else np.nan
        
        return revenue_series, shares
        
    except Exception:
        return pd.Series(dtype=float), np.nan

def compute_ps_from_yf_data(price: pd.Series, revenue_series: pd.Series, shares_outstanding: float) -> pd.Series:
    if revenue_series.empty or not np.isfinite(shares_outstanding) or shares_outstanding <= 0:
        return pd.Series(index=price.index, dtype=float)
    
    # Gestisce sia quarterly che annual
    if len(revenue_series) >= 4:
        time_diffs = revenue_series.index.to_series().diff().dt.days.dropna()
        avg_diff = time_diffs.mean()
        
        if avg_diff < 150:  # quarterly
            revenue_use = revenue_series.rolling(window=4, min_periods=1).sum()
        else:  # annual
            revenue_use = revenue_series
    else:
        revenue_use = revenue_series
    
    revenue_use = revenue_use.dropna()
    if revenue_use.empty:
        return pd.Series(index=price.index, dtype=float)
    
    # Revenue per share
    revps = revenue_use / shares_outstanding
    revps_daily = revps.reindex(price.index, method="ffill")
    
    # P/S giornaliero
    ps_daily = price / revps_daily
    ps_daily = ps_daily.replace([np.inf, -np.inf], np.nan)
    ps_daily = ps_daily[(ps_daily > 0) & np.isfinite(ps_daily)]
    
    return ps_daily

def get_approximate_market_cap(ticker: str, price: float) -> float:
    """Stima approssimativa market cap"""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        shares = info.get("sharesOutstanding", None)
        if shares is None:
            shares = info.get("impliedSharesOutstanding", None)
        if shares is None:
            shares = info.get("floatShares", None)
            
        if shares and np.isfinite(float(shares)):
            return float(shares) * price / 1e9  # in billions
        else:
            return np.nan
    except:
        return np.nan

# ========================= Selection Strategies ========================

def select_all_available(all_price_data: Dict[str, pd.Series], 
                        period_start: pd.Timestamp, 
                        period_end: pd.Timestamp,
                        cfg: BacktestConfig) -> List[str]:
    """APPROACH A: Usa TUTTI i titoli disponibili"""
    
    candidates = []
    
    for ticker, price_series in all_price_data.items():
        period_prices = price_series.loc[
            (price_series.index >= period_start) & 
            (price_series.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= cfg.min_trading_days:
            candidates.append(ticker)
    
    print(f"  APPROACH A - Tutti disponibili: {len(candidates)} titoli")
    return candidates

def select_top_market_cap(all_price_data: Dict[str, pd.Series], 
                         period_start: pd.Timestamp, 
                         period_end: pd.Timestamp,
                         cfg: BacktestConfig,
                         top_n: int = 50) -> List[str]:
    """APPROACH B: Solo top N per market cap (simula S&P 500)"""
    
    candidates = []
    
    for ticker, price_series in all_price_data.items():
        period_prices = price_series.loc[
            (price_series.index >= period_start) & 
            (price_series.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= cfg.min_trading_days:
            # Usa prezzo medio del periodo per stima market cap
            avg_price = period_prices.mean()
            market_cap = get_approximate_market_cap(ticker, avg_price)
            
            if pd.notna(market_cap) and market_cap >= cfg.min_market_cap_billion:
                candidates.append({
                    'ticker': ticker,
                    'market_cap': market_cap,
                    'avg_price': avg_price
                })
    
    # Ordina per market cap e prendi top N
    candidates_df = pd.DataFrame(candidates)
    if candidates_df.empty:
        print(f"  APPROACH B - Nessun candidato valido")
        return []
    
    candidates_df = candidates_df.sort_values(['market_cap'], ascending=False)
    selected = candidates_df.head(top_n)['ticker'].tolist()
    
    print(f"  APPROACH B - Top {top_n} market cap: {len(selected)} titoli (da {len(candidates)} candidati)")
    return selected

# ========================= Backtest Logic ==============================

def compute_signals_from_ps(price: pd.Series, ps_series: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    df["ps"] = ps_series.reindex(df.index)
    df.loc[~np.isfinite(df["ps"]), "ps"] = np.nan

    # Rolling stats
    df["ps_ma"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).mean()
    df["ps_std"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).std()

    # Soglie
    df["entry_threshold"] = df["ps_ma"] + cfg.entry_k_std * df["ps_std"]
    df["exit_threshold"] = df["ps_ma"] + cfg.exit_k_std * df["ps_std"]

    # Segnali base
    df["entry_cond"] = (df["ps"] < df["entry_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    df["exit_cond"] = (df["ps"] > df["exit_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    
    return df

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    action: str
    price: float
    qty: float
    cash_after: float

def simple_backtest(signal_map: Dict[str, pd.DataFrame], cfg: BacktestConfig, strategy_name: str = "") -> Tuple[pd.DataFrame, List[Trade]]:
    """Backtest semplificato"""
    
    first_idx = min(df.index.min() for df in signal_map.values())
    last_idx = max(df.index.max() for df in signal_map.values())
    
    spy = fetch_prices_yf("SPY", start=first_idx.strftime("%Y-%m-%d"), end=last_idx.strftime("%Y-%m-%d"))
    calendar = spy.index

    cash = cfg.initial_capital
    holdings_qty: Dict[str, float] = {t: 0.0 for t in signal_map.keys()}
    trades: List[Trade] = []
    equity_curve = []

    price_map: Dict[str, pd.Series] = {t: df["price"].reindex(calendar) for t, df in signal_map.items()}
    np.random.seed(cfg.random_seed)

    def portfolio_value(day: pd.Timestamp) -> float:
        val = cash
        for t, qty in holdings_qty.items():
            if qty == 0:
                continue
            p = price_map[t].get(day, np.nan)
            if np.isfinite(p):
                val += qty * p
        return val

    for day in calendar:
        port_val = portfolio_value(day)
        
        # 1) SELL prima
        for t, df in signal_map.items():
            price = price_map[t].get(day, np.nan)
            if not np.isfinite(price):
                continue
            if holdings_qty[t] > 0 and bool(df["exit_cond"].reindex(calendar).get(day, False)):
                qty = holdings_qty[t]
                proceeds = qty * price
                cash += proceeds
                holdings_qty[t] = 0.0
                trades.append(Trade(day, t, "SELL", float(price), float(qty), float(cash)))

        # Aggiorna port_val dopo vendite
        port_val = portfolio_value(day)

        # 2) BUY 
        for t, df in signal_map.items():
            price = price_map[t].get(day, np.nan)
            if not np.isfinite(price):
                continue

            entry_signal = bool(df["entry_cond"].reindex(calendar).get(day, False))
            if not entry_signal:
                continue
            
            # Controlli allocazione ticker
            max_alloc_for_ticker = cfg.max_alloc_per_ticker_frac * port_val
            current_alloc = holdings_qty[t] * price if holdings_qty[t] > 0 else 0
            remaining_alloc = max(0.0, max_alloc_for_ticker - current_alloc)
            amount = min(cfg.per_trade_usd, cash, remaining_alloc)
            
            if amount <= 0:
                continue
                
            qty = amount / price if cfg.fractional_shares else math.floor(amount / price)
            if qty <= 0:
                continue
                
            cash -= qty * price
            holdings_qty[t] += qty
            trades.append(Trade(day, t, "BUY", float(price), float(qty), float(cash)))

        equity_curve.append({"date": day, "equity": portfolio_value(day), "cash": cash})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    
    if VERBOSE and strategy_name:
        eq = equity_df["equity"].dropna()
        if len(eq) > 1:
            total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
            print(f"    {strategy_name}: {total_return:+.1%} total return, {len(trades)} trades")
    
    return equity_df, trades

def buy_hold_benchmark(start_date: str, end_date: str, initial_capital: float = 10000) -> pd.DataFrame:
    """Buy & Hold SPY benchmark"""
    spy = fetch_prices_yf("SPY", start_date, end_date)
    if spy.empty:
        return pd.DataFrame()
    
    # Compra tutto SPY all'inizio
    initial_shares = initial_capital / spy.iloc[0]
    equity_curve = spy * initial_shares
    
    return pd.DataFrame({"equity": equity_curve})

# ========================= COMPREHENSIVE COMPARISON ====================

def run_comprehensive_comparison(cfg: BacktestConfig):
    """Confronto completo: Tutti vs Top50 vs Buy&Hold"""
    print("=== CONFRONTO COMPLETO: 3 APPROCCI ===")
    
    # 1. Scarica dati universo realistico
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    universe_tickers = get_realistic_universe()
    print(f"Scarico dati per {len(universe_tickers)} titoli dal {cfg.start_year}...")
    
    all_price_data = {}
    
    for sym in tqdm(universe_tickers):
        price = fetch_prices_yf(sym, start_date, end_date)
        if not price.empty and len(price) > 200:  # Almeno 200 giorni
            all_price_data[sym] = price
        time.sleep(YF_SLEEP)
    
    print(f"Dati prezzi raccolti per {len(all_price_data)} titoli")
    
    # 2. Crea periodi
    periods = []
    current_year = cfg.start_year
    while current_year < 2025:
        end_year = min(current_year + cfg.rebalance_frequency_years, 2025)
        periods.append((current_year, end_year))
        current_year = end_year
    
    print(f"Creati {len(periods)} periodi di {cfg.rebalance_frequency_years} anni")
    
    # 3. Per ogni periodo, testa 3 approcci
    all_results = {'approach_a': [], 'approach_b': [], 'approach_c': []}
    
    for period_idx, (start_year, end_year) in enumerate(periods):
        print(f"\n--- Periodo {period_idx+1}: {start_year} to {end_year} ---")
        
        period_start = pd.to_datetime(f"{start_year}-01-01")
        period_end = pd.to_datetime(f"{end_year}-01-01")
        
        # APPROACH A: Tutti i titoli disponibili
        universe_all = select_all_available(all_price_data, period_start, period_end, cfg)
        
        # APPROACH B: Top 50 market cap
        universe_top50 = select_top_market_cap(all_price_data, period_start, period_end, cfg, top_n=50)
        
        # Testa entrambi gli approcci
        for approach_name, universe in [("approach_a", universe_all), ("approach_b", universe_top50)]:
            if len(universe) < 5:
                print(f"  {approach_name.upper()}: troppi pochi titoli, skip")
                continue
            
            # Raccogli signal_map
            signal_map = {}
            
            for ticker in universe:
                if ticker not in all_price_data:
                    continue
                    
                price = all_price_data[ticker]
                revenue_series, shares = fetch_yf_fundamentals(ticker)
                
                if revenue_series.empty or not np.isfinite(shares):
                    continue
                    
                ps_series = compute_ps_from_yf_data(price, revenue_series, shares)
                if ps_series.empty:
                    continue
                    
                df = compute_signals_from_ps(price, ps_series, cfg)
                if df["ps_ma"].dropna().empty:
                    continue
                    
                signal_map[ticker] = df
                time.sleep(YF_SLEEP * 0.02)
            
            if len(signal_map) < 3:
                print(f"  {approach_name.upper()}: troppi pochi segnali, skip")
                continue
            
            # Backtest
            equity, trades = simple_backtest(signal_map, cfg, approach_name.upper())
            
            # Calcola performance
            eq = equity["equity"].dropna()
            if len(eq) > 1:
                period_return = eq.iloc[-1] / eq.iloc[0] - 1.0
                annualized_return = (eq.iloc[-1] / eq.iloc[0]) ** (1/cfg.rebalance_frequency_years) - 1.0
            else:
                period_return = 0.0
                annualized_return = 0.0
            
            all_results[approach_name].append({
                'period': f"{start_year}-{end_year}",
                'start_year': start_year,
                'universe_size': len(universe),
                'signals_count': len(signal_map),
                'period_return': period_return,
                'annualized_return': annualized_return,
                'trades_count': len(trades),
                'equity': equity
            })
        
        # APPROACH C: Buy & Hold SPY
        spy_equity = buy_hold_benchmark(period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"), cfg.initial_capital)
        if not spy_equity.empty:
            eq_spy = spy_equity["equity"].dropna()
            if len(eq_spy) > 1:
                spy_return = eq_spy.iloc[-1] / eq_spy.iloc[0] - 1.0
                spy_annualized = (eq_spy.iloc[-1] / eq_spy.iloc[0]) ** (1/cfg.rebalance_frequency_years) - 1.0
            else:
                spy_return = 0.0
                spy_annualized = 0.0
            
            all_results['approach_c'].append({
                'period': f"{start_year}-{end_year}",
                'start_year': start_year,
                'universe_size': 1,
                'signals_count': 1,
                'period_return': spy_return,
                'annualized_return': spy_annualized,
                'trades_count': 1,
                'equity': spy_equity
            })
            
            print(f"    APPROACH C (SPY): {spy_return:+.1%} total return")
    
    # 4. Analisi comparativa
    print(f"\n=== CONFRONTO FINALE ===")
    
    approaches = {
        'A - Tutti i titoli': all_results['approach_a'],
        'B - Top 50 market cap': all_results['approach_b'], 
        'C - Buy & Hold SPY': all_results['approach_c']
    }
    
    # Plot comparativo
    plot_comprehensive_comparison(approaches)
    
    # Statistiche finali
    for name, results in approaches.items():
        if not results:
            continue
            
        returns = [r['annualized_return'] * 100 for r in results]
        trades = [r['trades_count'] for r in results]
        
        print(f"\n{name}:")
        print(f"  Rendimento annualizzato medio: {np.mean(returns):.1f}%")
        print(f"  Rendimento mediano: {np.median(returns):.1f}%")
        print(f"  Volatilità: {np.std(returns):.1f}%")
        print(f"  Trades medi per periodo: {np.mean(trades):.0f}")
        print(f"  Periodi: {len(results)}")
    
    return all_results

def plot_comprehensive_comparison(approaches: Dict[str, List[dict]]):
    """Plot comparativo dei 3 approcci"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green']
    
    # 1. Rendimenti annualizzati per approccio
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        periods = [r['period'] for r in results]
        returns = [r['annualized_return'] * 100 for r in results]
        
        ax1.plot(range(len(periods)), returns, 'o-', color=colors[i], 
                label=name, linewidth=2, markersize=6)
    
    ax1.set_title('Rendimenti Annualizzati per Periodo')
    ax1.set_ylabel('Rendimento Annualizzato (%)')
    ax1.set_xlabel('Periodo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 2. Box plot dei rendimenti
    returns_data = []
    labels = []
    for name, results in approaches.items():
        if results:
            returns_data.append([r['annualized_return'] * 100 for r in results])
            labels.append(name.split(' - ')[0])  # Solo la lettera
    
    if returns_data:
        ax2.boxplot(returns_data, labels=labels)
        ax2.set_title('Distribuzione Rendimenti')
        ax2.set_ylabel('Rendimento Annualizzato (%)')
        ax2.grid(True, alpha=0.3)
    
    # 3. Numero trades
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        periods = [r['period'] for r in results]
        trades = [r['trades_count'] for r in results]
        
        ax3.bar([x + i*0.25 for x in range(len(periods))], trades, 
               width=0.25, alpha=0.7, color=colors[i], label=name)
    
    ax3.set_title('Numero Trades per Periodo')
    ax3.set_ylabel('Numero Trades')
    ax3.set_xlabel('Periodo')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Universo size
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        periods = [r['period'] for r in results]
        universe_sizes = [r['universe_size'] for r in results]
        
        ax4.plot(range(len(periods)), universe_sizes, 'o-', 
                color=colors[i], label=name, linewidth=2, markersize=6)
    
    ax4.set_title('Dimensione Universo')
    ax4.set_ylabel('Numero Titoli')
    ax4.set_xlabel('Periodo')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ====================== MAIN ==========================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        per_trade_usd=50.0,
        start_year=2010,                     # Dal 2010 per dati più affidabili
        rolling_window=60,
        min_periods=30,
        entry_k_std=0.0,
        exit_k_std=1.0,
        accumulate_daily=True,
        fractional_shares=True,
        max_alloc_per_ticker_frac=0.02,      # 2% per titolo se usiamo tutti
        rebalance_frequency_years=2,         # Ogni 2 anni
        min_trading_days=100,
        min_market_cap_billion=1.0,          # Min 1B market cap
    )
    
    print("Avvio confronto completo: Tutti vs Top50 vs SPY...")
    results = run_comprehensive_comparison(cfg)
    
    print(f"\nConfronto completato!")