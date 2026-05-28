# comprehensive_comparison.py - Confronto completo con Sharpe Ratio semplificato e Parallelizzazione
import os
import time
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================= CONFIG =================================

VERBOSE = True
YF_SLEEP = 0.02
MAX_WORKERS = min(8, mp.cpu_count())

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    per_trade_usd: float = 50.0
    start_year: int = 2010
    rolling_window: int = 60
    min_periods: int = 30
    entry_k_std: float = 0.0
    exit_k_std: float = 1.0
    accumulate_daily: bool = True
    fractional_shares: bool = True
    max_alloc_per_ticker_frac: float = 0.02
    
    # Parametri per confronto
    rebalance_frequency_years: int = 2
    min_trading_days: int = 100
    min_market_cap_billion: float = 1.0
    
    transaction_cost_per_trade: float = 0.0
    random_seed: int = 42

# ============================= UNIVERSE ===================

REALISTIC_LARGE_CAP = [
    # Tech
    "AAPL","MSFT","AMZN","GOOGL","GOOG","ORCL","CSCO","INTC","IBM","QCOM","AMAT","ADI","NVDA","AMD",
    
    # Finance
    "JPM","BAC","WFC","GS","MS","C","USB","PNC","COF","AXP","BLK","SCHW",
    
    # Healthcare
    "JNJ","PFE","MRK","ABBV","LLY","UNH","CVS","BMY","ABT","GILD","AMGN","MDT","TMO","DHR","BSX","ISRG",
    
    # Consumer & Retail
    "WMT","HD","TGT","COST","LOW","KO","PEP","PG","MCD","SBUX","NKE","TJX","DIS",
    
    # Energy
    "XOM","CVX","COP","SLB","HAL","OXY","DVN","EOG",
    
    # Industrials
    "GE","CAT","MMM","HON","BA","LMT","NOC","GD","UPS","FDX","DE","EMR","ITW",
    
    # Materials
    "DD","DOW","LYB","APD","PPG","SHW","ECL","FCX","NEM",
    
    # Utilities
    "D","SO","NEE","DUK","EXC","ED","AEP","XEL",
    
    # REITs
    "AMT","PLD","PSA","EQR","AVB","UDR","CPT","MAA","ESS","BXP","SPG"
]

def get_realistic_universe() -> List[str]:
    return REALISTIC_LARGE_CAP

# ============================= UTILS ==================================

def today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    """Calcola il drawdown della serie equity"""
    running_max = equity_series.expanding().max()
    drawdown = (equity_series / running_max - 1) * 100
    return drawdown

def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """Calcola il massimo drawdown"""
    drawdown = calculate_drawdown(equity_series)
    return drawdown.min()

def calculate_sharpe_ratio(equity_series: pd.Series) -> float:
    """Calcola lo Sharpe Ratio semplificato: Return / Volatility"""
    if len(equity_series) < 2:
        return 0.0
    
    # Calcola rendimenti giornalieri
    returns = equity_series.pct_change().dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    # Annualizza i rendimenti
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio semplificato = rendimento / volatilità
    sharpe = annual_return / annual_volatility
    
    return sharpe

# ========================= Data fetching (PARALLELIZZATO) ===============================

def fetch_prices_yf_single(args: Tuple[str, str, str]) -> Tuple[str, pd.Series]:
    """Versione per parallelizzazione - scarica prezzo singolo ticker"""
    symbol, start, end = args
    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        if data is None or len(data) == 0:
            return symbol, pd.Series(dtype=float)
        
        close = data["Close"] if "Close" in data.columns else data["Adj Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.name = "price"
        
        time.sleep(YF_SLEEP)
        return symbol, close
    except Exception:
        return symbol, pd.Series(dtype=float)

def fetch_prices_parallel(symbols: List[str], start: str, end: str) -> Dict[str, pd.Series]:
    """Download parallelo dei prezzi"""
    print(f"Download parallelo di {len(symbols)} simboli con {MAX_WORKERS} workers...")
    
    args_list = [(sym, start, end) for sym in symbols]
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(fetch_prices_yf_single, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Download prezzi"):
            symbol = future_to_symbol[future]
            try:
                _, price_series = future.result()
                if not price_series.empty and len(price_series) > 200:
                    results[symbol] = price_series
            except Exception as e:
                if VERBOSE:
                    print(f"Errore downloading {symbol}: {e}")
    
    return results

def fetch_yf_fundamentals_single(symbol: str) -> Tuple[str, pd.Series, float]:
    """Versione per parallelizzazione - scarica fondamentali singolo ticker"""
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
            return symbol, pd.Series(dtype=float), np.nan
        
        # Cerca ricavi
        revenue_row = None
        for idx in financials.index:
            idx_str = str(idx).lower()
            if any(term in idx_str for term in ["total revenue", "revenue", "net sales", "sales"]):
                revenue_row = idx
                break
        
        if revenue_row is None:
            return symbol, pd.Series(dtype=float), np.nan
        
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
        
        time.sleep(YF_SLEEP)
        return symbol, revenue_series, shares
        
    except Exception:
        return symbol, pd.Series(dtype=float), np.nan

def fetch_fundamentals_parallel(symbols: List[str]) -> Dict[str, Tuple[pd.Series, float]]:
    """Download parallelo dei dati fondamentali"""
    print(f"Download parallelo dati fondamentali per {len(symbols)} simboli...")
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(fetch_yf_fundamentals_single, sym): sym for sym in symbols}
        
        for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Download fondamentali"):
            symbol = future_to_symbol[future]
            try:
                _, revenue_series, shares = future.result()
                if not revenue_series.empty and np.isfinite(shares):
                    results[symbol] = (revenue_series, shares)
            except Exception as e:
                if VERBOSE:
                    print(f"Errore downloading fundamentals {symbol}: {e}")
    
    return results

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
    """APPROACH B: Solo top N per market cap"""
    
    candidates = []
    
    for ticker, price_series in all_price_data.items():
        period_prices = price_series.loc[
            (price_series.index >= period_start) & 
            (price_series.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= cfg.min_trading_days:
            avg_price = period_prices.mean()
            market_cap = get_approximate_market_cap(ticker, avg_price)
            
            if pd.notna(market_cap) and market_cap >= cfg.min_market_cap_billion:
                candidates.append({
                    'ticker': ticker,
                    'market_cap': market_cap,
                    'avg_price': avg_price
                })
    
    candidates_df = pd.DataFrame(candidates)
    if candidates_df.empty:
        print(f"  APPROACH B - Nessun candidato valido")
        return []
    
    candidates_df = candidates_df.sort_values(['market_cap'], ascending=False)
    selected = candidates_df.head(top_n)['ticker'].tolist()
    
    print(f"  APPROACH B - Top {top_n} market cap: {len(selected)} titoli")
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

    # Segnali
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
    
    spy = fetch_prices_yf_single(("SPY", first_idx.strftime("%Y-%m-%d"), last_idx.strftime("%Y-%m-%d")))[1]
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
            sharpe = calculate_sharpe_ratio(eq)
            max_dd = calculate_max_drawdown(eq)
            print(f"    {strategy_name}: {total_return:+.1%} total return, {len(trades)} trades, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.1f}%")
    
    return equity_df, trades

def buy_hold_benchmark(start_date: str, end_date: str, initial_capital: float = 10000) -> pd.DataFrame:
    """Buy & Hold SPY benchmark"""
    spy = fetch_prices_yf_single(("SPY", start_date, end_date))[1]
    if spy.empty:
        return pd.DataFrame()
    
    initial_shares = initial_capital / spy.iloc[0]
    equity_curve = spy * initial_shares
    
    return pd.DataFrame({"equity": equity_curve})

# ========================= SIGNAL GENERATION PARALLEL ====================

def process_signal_generation(args: Tuple[str, pd.Series, Dict, BacktestConfig]) -> Tuple[str, Optional[pd.DataFrame]]:
    """Processo singolo per generazione segnali"""
    ticker, price, fundamentals_data, cfg = args
    
    try:
        if ticker not in fundamentals_data:
            return ticker, None
            
        revenue_series, shares = fundamentals_data[ticker]
        
        if revenue_series.empty or not np.isfinite(shares):
            return ticker, None
            
        ps_series = compute_ps_from_yf_data(price, revenue_series, shares)
        if ps_series.empty:
            return ticker, None
            
        df = compute_signals_from_ps(price, ps_series, cfg)
        if df["ps_ma"].dropna().empty:
            return ticker, None
            
        return ticker, df
    except Exception:
        return ticker, None

def generate_signals_parallel(universe: List[str], all_price_data: Dict[str, pd.Series], 
                            fundamentals_data: Dict[str, Tuple[pd.Series, float]], 
                            cfg: BacktestConfig) -> Dict[str, pd.DataFrame]:
    """Generazione parallela dei segnali"""
    print(f"Generazione parallela segnali per {len(universe)} titoli...")
    
    args_list = []
    for ticker in universe:
        if ticker in all_price_data and ticker in fundamentals_data:
            args_list.append((ticker, all_price_data[ticker], fundamentals_data, cfg))
    
    signal_map = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_signal_generation, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(future_to_ticker), total=len(args_list), desc="Generazione segnali"):
            ticker = future_to_ticker[future]
            try:
                _, df = future.result()
                if df is not None:
                    signal_map[ticker] = df
            except Exception as e:
                if VERBOSE:
                    print(f"Errore generating signals {ticker}: {e}")
    
    return signal_map

# ========================= PLOTTING FUNCTIONS ==========================

def plot_portfolio_equity_and_drawdown(all_results: Dict[str, List[dict]]):
    """Plot completo dell'equity line e drawdown"""
    print("\nGenerazione plot equity curve e drawdown...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    
    approaches = {
        'A - Tutti i titoli': all_results['approach_a'],
        'B - Top 50 market cap': all_results['approach_b'], 
        'C - Buy & Hold SPY': all_results['approach_c']
    }
    
    colors = ['blue', 'red', 'green']
    
    # 1. Equity Curve Concatenata
    ax1 = fig.add_subplot(gs[0, :])
    
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
            
        concatenated_equity = pd.Series(dtype=float)
        
        for result in results:
            if 'equity' in result and not result['equity'].empty:
                equity = result['equity']['equity'].dropna()
                if not concatenated_equity.empty:
                    last_value = concatenated_equity.iloc[-1]
                    first_value = equity.iloc[0]
                    equity = equity * (last_value / first_value)
                
                concatenated_equity = pd.concat([concatenated_equity, equity])
        
        if not concatenated_equity.empty:
            ax1.plot(concatenated_equity.index, concatenated_equity.values, 
                    color=colors[i], label=name, linewidth=2, alpha=0.8)
    
    ax1.set_title('Equity Curve Completa (Tutti i Periodi)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Valore Portafoglio ($)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Drawdown Concatenato
    ax2 = fig.add_subplot(gs[1, :])
    
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
            
        concatenated_equity = pd.Series(dtype=float)
        
        for result in results:
            if 'equity' in result and not result['equity'].empty:
                equity = result['equity']['equity'].dropna()
                if not concatenated_equity.empty:
                    last_value = concatenated_equity.iloc[-1]
                    first_value = equity.iloc[0]
                    equity = equity * (last_value / first_value)
                
                concatenated_equity = pd.concat([concatenated_equity, equity])
        
        if not concatenated_equity.empty:
            drawdown = calculate_drawdown(concatenated_equity)
            ax2.fill_between(drawdown.index, drawdown.values, 0, 
                           color=colors[i], alpha=0.3, label=f'{name} DD')
            ax2.plot(drawdown.index, drawdown.values, color=colors[i], linewidth=1)
    
    ax2.set_title('Drawdown Storico (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Performance Cumulativa per Periodo
    ax3 = fig.add_subplot(gs[2, 0])
    
    periods_labels = []
    if approaches['A - Tutti i titoli']:
        periods_labels = [r['period'] for r in approaches['A - Tutti i titoli']]
    
    width = 0.25
    x_positions = np.arange(len(periods_labels))
    
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        
        cumulative_returns = []
        cumulative_return = 1.0
        
        for result in results:
            period_return = result.get('period_return', 0.0)
            cumulative_return *= (1 + period_return)
            cumulative_returns.append((cumulative_return - 1) * 100)
        
        ax3.bar(x_positions + i * width, cumulative_returns, width, 
               color=colors[i], alpha=0.7, label=name.split(' - ')[0])
    
    ax3.set_title('Performance Cumulativa per Periodo', fontsize=12)
    ax3.set_ylabel('Rendimento Cumulativo (%)', fontsize=10)
    ax3.set_xlabel('Periodo', fontsize=10)
    ax3.set_xticks(x_positions + width)
    ax3.set_xticklabels(periods_labels, rotation=45, fontsize=8)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk-Return Scatter Plot
    ax4 = fig.add_subplot(gs[2, 1])
    
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        
        returns = [r['annualized_return'] * 100 for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        
        if returns and sharpe_ratios:
            avg_return = np.mean(returns)
            avg_sharpe = np.mean(sharpe_ratios)
            
            ax4.scatter(avg_sharpe, avg_return, color=colors[i], 
                       s=100, alpha=0.7, label=name.split(' - ')[0])
            
            ax4.annotate(name.split(' - ')[0], 
                        (avg_sharpe, avg_return), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax4.set_title('Risk-Return Profile', fontsize=12)
    ax4.set_xlabel('Sharpe Ratio Medio', fontsize=10)
    ax4.set_ylabel('Rendimento Annuo Medio (%)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout(pad=2.0)
    plt.show()

def plot_comprehensive_comparison(approaches: Dict[str, List[dict]]):
    """Plot comparativo standard"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green']
    
    # 1. Rendimenti annualizzati
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
    
    # 2. Sharpe Ratio
    for i, (name, results) in enumerate(approaches.items()):
        if not results:
            continue
        periods = [r['period'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        
        ax2.plot(range(len(periods)), sharpe_ratios, 's-', color=colors[i], 
                label=name, linewidth=2, markersize=6)
    
    ax2.set_title('Sharpe Ratio per Periodo')
    ax2.set_ylabel('Sharpe Ratio (Return/Volatility)')
    ax2.set_xlabel('Periodo')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
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
    
    # 4. Box plot Sharpe ratio
    sharpe_data = []
    labels = []
    for name, results in approaches.items():
        if results:
            sharpe_data.append([r['sharpe_ratio'] for r in results])
            labels.append(name.split(' - ')[0])
    
    if sharpe_data:
        ax4.boxplot(sharpe_data, labels=labels)
        ax4.set_title('Distribuzione Sharpe Ratio')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# ========================= MAIN COMPARISON ====================

def run_comprehensive_comparison(cfg: BacktestConfig):
    """Confronto completo: Tutti vs Top50 vs Buy&Hold"""
    print("=== CONFRONTO COMPLETO: 3 APPROCCI (PARALLELIZZATO) ===")
    
    # 1. Scarica dati in parallelo
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    universe_tickers = get_realistic_universe()
    print(f"Scarico dati per {len(universe_tickers)} titoli dal {cfg.start_year}...")
    
    all_price_data = fetch_prices_parallel(universe_tickers, start_date, end_date)
    print(f"Dati prezzi raccolti per {len(all_price_data)} titoli")
    
    fundamentals_data = fetch_fundamentals_parallel(list(all_price_data.keys()))
    print(f"Dati fondamentali raccolti per {len(fundamentals_data)} titoli")
    
    # 2. Crea periodi
    periods = []
    current_year = cfg.start_year
    while current_year < 2025:
        end_year = min(current_year + cfg.rebalance_frequency_years, 2025)
        periods.append((current_year, end_year))
        current_year = end_year
    
    print(f"Creati {len(periods)} periodi di {cfg.rebalance_frequency_years} anni")
    
    # 3. Test dei 3 approcci
    all_results = {'approach_a': [], 'approach_b': [], 'approach_c': []}
    
    for period_idx, (start_year, end_year) in enumerate(periods):
        print(f"\n--- Periodo {period_idx+1}: {start_year} to {end_year} ---")
        
        period_start = pd.to_datetime(f"{start_year}-01-01")
        period_end = pd.to_datetime(f"{end_year}-01-01")
        
        # APPROACH A: Tutti i titoli
        universe_all = select_all_available(all_price_data, period_start, period_end, cfg)
        
        # APPROACH B: Top 50 market cap
        universe_top50 = select_top_market_cap(all_price_data, period_start, period_end, cfg, top_n=50)
        
        # Test entrambi
        for approach_name, universe in [("approach_a", universe_all), ("approach_b", universe_top50)]:
            if len(universe) < 5:
                print(f"  {approach_name.upper()}: troppi pochi titoli, skip")
                continue
            
            signal_map = generate_signals_parallel(universe, all_price_data, fundamentals_data, cfg)
            
            if len(signal_map) < 3:
                print(f"  {approach_name.upper()}: troppi pochi segnali, skip")
                continue
            
            equity, trades = simple_backtest(signal_map, cfg, approach_name.upper())
            
            eq = equity["equity"].dropna()
            if len(eq) > 1:
                period_return = eq.iloc[-1] / eq.iloc[0] - 1.0
                annualized_return = (eq.iloc[-1] / eq.iloc[0]) ** (1/cfg.rebalance_frequency_years) - 1.0
                sharpe_ratio = calculate_sharpe_ratio(eq)
                max_drawdown = calculate_max_drawdown(eq)
            else:
                period_return = 0.0
                annualized_return = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
            
            all_results[approach_name].append({
                'period': f"{start_year}-{end_year}",
                'start_year': start_year,
                'universe_size': len(universe),
                'signals_count': len(signal_map),
                'period_return': period_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
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
                spy_sharpe = calculate_sharpe_ratio(eq_spy)
                spy_max_dd = calculate_max_drawdown(eq_spy)
            else:
                spy_return = 0.0
                spy_annualized = 0.0
                spy_sharpe = 0.0
                spy_max_dd = 0.0
            
            all_results['approach_c'].append({
                'period': f"{start_year}-{end_year}",
                'start_year': start_year,
                'universe_size': 1,
                'signals_count': 1,
                'period_return': spy_return,
                'annualized_return': spy_annualized,
                'sharpe_ratio': spy_sharpe,
                'max_drawdown': spy_max_dd,
                'trades_count': 1,
                'equity': spy_equity
            })
            
            print(f"    APPROACH C (SPY): {spy_return:+.1%} total return, Sharpe: {spy_sharpe:.2f}, Max DD: {spy_max_dd:.1f}%")
    
    # 4. Analisi e plot
    print(f"\n=== CONFRONTO FINALE ===")
    
    approaches = {
        'A - Tutti i titoli': all_results['approach_a'],
        'B - Top 50 market cap': all_results['approach_b'], 
        'C - Buy & Hold SPY': all_results['approach_c']
    }
    
    plot_comprehensive_comparison(approaches)
    plot_portfolio_equity_and_drawdown(all_results)
    
    # Statistiche finali
    for name, results in approaches.items():
        if not results:
            continue
            
        returns = [r['annualized_return'] * 100 for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        trades = [r['trades_count'] for r in results]
        
        print(f"\n{name}:")
        print(f"  Rendimento annualizzato medio: {np.mean(returns):.1f}%")
        print(f"  Rendimento mediano: {np.median(returns):.1f}%")
        print(f"  Volatilità: {np.std(returns):.1f}%")
        print(f"  Sharpe Ratio medio: {np.mean(sharpe_ratios):.2f}")
        print(f"  Sharpe Ratio mediano: {np.median(sharpe_ratios):.2f}")
        print(f"  Max Drawdown medio: {np.mean(max_drawdowns):.1f}%")
        print(f"  Max Drawdown peggiore: {np.min(max_drawdowns):.1f}%")
        print(f"  Trades medi per periodo: {np.mean(trades):.0f}")
        print(f"  Periodi: {len(results)}")
    
    return all_results

# ====================== MAIN ==========================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        per_trade_usd=50.0,
        start_year=2010,
        rolling_window=60,
        min_periods=30,
        entry_k_std=0.0,
        exit_k_std=1.0,
        accumulate_daily=True,
        fractional_shares=True,
        max_alloc_per_ticker_frac=0.02,
        rebalance_frequency_years=2,
        min_trading_days=100,
        min_market_cap_billion=1.0,
    )
    
    print(f"Avvio confronto completo parallelizzato con {MAX_WORKERS} workers...")
    print("Sharpe Ratio: Return/Volatility (semplificato)")
    
    start_time = time.time()
    results = run_comprehensive_comparison(cfg)
    end_time = time.time()
    
    print(f"\nConfronto completato in {end_time - start_time:.1f} secondi!")