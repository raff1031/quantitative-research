# comprehensive_comparison.py - ENHANCED with Larry Connors RSI(2) Strategy (Clean Version)
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
    per_trade_usd: float = 200.0
    start_year: int = 2021
    rolling_window: int = 30
    min_periods: int = 15
    entry_k_std: float = -0.5
    exit_k_std: float = 0.5
    accumulate_daily: bool = True
    fractional_shares: bool = True
    max_alloc_per_ticker_frac: float = 0.05
    
    # Parametri per confronto
    rebalance_frequency_years: int = 2
    min_trading_days: int = 100
    min_market_cap_billion: float = 1.0
    
    # Larry Connors RSI(2) parameters
    rsi_period: int = 2                 # RSI period (classic Larry Connors uses 2)
    rsi_entry_threshold: float = 20     # Buy when RSI(2) < 20
    rsi_exit_threshold: float = 70      # Sell when RSI(2) > 70
    rsi_ma_period: int = 200            # Moving average filter period
    rsi_ma_type: str = "SMA"            # "SMA" or "EMA" for trend filter
    rsi_max_holding_days: int = 10      # Maximum days to hold position
    rsi_profit_target_pct: float = 5.0  # Optional profit target %
    
    transaction_cost_per_trade: float = 0.0
    random_seed: int = 42
    risk_free_rate: float = 0.02

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

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    """Calcola il drawdown della serie equity"""
    running_max = equity_series.expanding().max()
    drawdown = (equity_series / running_max - 1) * 100
    return drawdown

def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """Calcola il massimo drawdown"""
    drawdown = calculate_drawdown(equity_series)
    return drawdown.min()

def calculate_sharpe_ratio(equity_series: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calcola lo Sharpe Ratio STANDARD: (Ra - Rf) / σ"""
    if len(equity_series) < 2:
        return 0.0
    
    returns = equity_series.pct_change().dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    
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
        
        annual_financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
        
        financials = None
        
        if annual_financials is not None and not annual_financials.empty:
            financials = annual_financials
        elif quarterly_financials is not None and not quarterly_financials.empty:
            financials = quarterly_financials
        
        if financials is None or financials.empty:
            return symbol, pd.Series(dtype=float), np.nan
        
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
    
    revps = revenue_use / shares_outstanding
    revps_daily = revps.reindex(price.index, method="ffill")
    
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

def select_rsi2_candidates(all_price_data: Dict[str, pd.Series], 
                          period_start: pd.Timestamp, 
                          period_end: pd.Timestamp,
                          cfg: BacktestConfig) -> List[str]:
    """APPROACH D: RSI(2) strategy candidates"""
    
    candidates = []
    
    for ticker, price_series in all_price_data.items():
        period_prices = price_series.loc[
            (price_series.index >= period_start) & 
            (price_series.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= max(cfg.min_trading_days, cfg.rsi_ma_period + 50):
            candidates.append(ticker)
    
    print(f"  APPROACH D - RSI(2) candidates: {len(candidates)} titoli")
    return candidates

# ========================= Signal Generation ==============================

def compute_signals_from_ps(price: pd.Series, ps_series: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    df["ps"] = ps_series.reindex(df.index)
    df.loc[~np.isfinite(df["ps"]), "ps"] = np.nan

    df["ps_ma"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).mean()
    df["ps_std"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).std()

    df["entry_threshold"] = df["ps_ma"] + cfg.entry_k_std * df["ps_std"]
    df["exit_threshold"] = df["ps_ma"] + cfg.exit_k_std * df["ps_std"]

    df["entry_cond"] = (df["ps"] < df["entry_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    df["exit_cond"] = (df["ps"] > df["exit_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    
    return df

def compute_rsi2_signals(price: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """Larry Connors RSI(2) signals"""
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    
    # Calculate RSI(2)
    df["rsi2"] = calculate_rsi(price, cfg.rsi_period)
    
    # Calculate moving average filter (200-day SMA or EMA)
    if cfg.rsi_ma_type == "EMA":
        df["ma200"] = calculate_ema(price, cfg.rsi_ma_period)
    else:  # Default to SMA
        df["ma200"] = calculate_sma(price, cfg.rsi_ma_period)
    
    # Entry condition: RSI(2) < 20 AND price above 200 MA
    df["rsi_oversold"] = df["rsi2"] < cfg.rsi_entry_threshold
    df["above_ma"] = df["price"] > df["ma200"]
    
    df["entry_cond"] = (
        df["rsi_oversold"] & 
        df["above_ma"] & 
        df["rsi2"].notna() &
        df["ma200"].notna()
    )
    
    # Exit condition: RSI(2) > 70 (overbought)
    df["rsi_overbought"] = df["rsi2"] > cfg.rsi_exit_threshold
    df["exit_cond"] = (
        df["rsi_overbought"] & 
        df["rsi2"].notna()
    )
    
    return df

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    action: str
    price: float
    qty: float
    cash_after: float
    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None

def continuous_backtest(signal_map: Dict[str, pd.DataFrame], cfg: BacktestConfig, strategy_name: str = "", is_rsi2_strategy: bool = False) -> Tuple[pd.DataFrame, List[Trade]]:
    """Enhanced backtest with RSI(2) strategy support and debugging"""
    
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    # Get trading calendar
    try:
        spy = fetch_prices_yf_single(("SPY", start_date, end_date))[1]
        if spy.empty:
            raise Exception("SPY data empty")
    except:
        print(f"WARNING: Impossibile scaricare SPY, uso calendario basato sui dati disponibili")
        all_dates = []
        for df in signal_map.values():
            all_dates.extend(df.index.tolist())
        calendar = pd.to_datetime(sorted(set(all_dates)))
        calendar = calendar[(calendar >= pd.to_datetime(start_date)) & (calendar <= pd.to_datetime(end_date))]
    else:
        calendar = spy.index
    
    if len(calendar) == 0:
        print(f"ERROR: Calendario vuoto per {strategy_name}")
        return pd.DataFrame(), []
    
    cash = cfg.initial_capital
    holdings_qty: Dict[str, float] = {t: 0.0 for t in signal_map.keys()}
    trades: List[Trade] = []
    equity_curve = []
    
    # Track entry info for RSI(2) strategy
    entry_prices: Dict[str, float] = {}
    entry_dates: Dict[str, pd.Timestamp] = {}

    price_map: Dict[str, pd.Series] = {t: df["price"].reindex(calendar, method='ffill') for t, df in signal_map.items()}
    np.random.seed(cfg.random_seed)
    
    # DEBUG: RSI(2) signal tracking
    if is_rsi2_strategy:
        debug_stats = {
            'total_checks': 0,
            'rsi_oversold': 0,
            'above_ma': 0,
            'valid_signals': 0,
            'monthly_signals': {},
            'sample_signals': []  # Store first 10 signals for analysis
        }

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
        
        # 1) SELL logic
        for t, df in signal_map.items():
            price = price_map[t].get(day, np.nan)
            if not np.isfinite(price):
                continue
                
            if holdings_qty[t] > 0:
                should_sell = False
                
                if is_rsi2_strategy:
                    # RSI(2) exit logic
                    rsi_exit = bool(df["exit_cond"].reindex(calendar).get(day, False))
                    
                    # Check maximum holding period
                    max_days_exit = False
                    if t in entry_dates:
                        days_held = (day - entry_dates[t]).days
                        if days_held >= cfg.rsi_max_holding_days:
                            max_days_exit = True
                    
                    # Check profit target
                    profit_target_exit = False
                    if t in entry_prices:
                        return_pct = (price / entry_prices[t] - 1) * 100
                        if return_pct >= cfg.rsi_profit_target_pct:
                            profit_target_exit = True
                    
                    should_sell = rsi_exit or max_days_exit or profit_target_exit
                else:
                    # Original P/S strategy exit logic
                    should_sell = bool(df["exit_cond"].reindex(calendar).get(day, False))
                
                if should_sell:
                    qty = holdings_qty[t]
                    proceeds = qty * price
                    cash += proceeds
                    holdings_qty[t] = 0.0
                    
                    # Clean up tracking
                    entry_price = entry_prices.pop(t, None)
                    entry_date = entry_dates.pop(t, None)
                    
                    trades.append(Trade(day, t, "SELL", float(price), float(qty), float(cash), 
                                      entry_price, entry_date))

        # Update portfolio value after sales
        port_val = portfolio_value(day)

        # 2) BUY logic with DEBUG
        for t, df in signal_map.items():
            price = price_map[t].get(day, np.nan)
            if not np.isfinite(price):
                continue

            # DEBUG: Track RSI(2) signal conditions
            if is_rsi2_strategy:
                rsi_val = df["rsi2"].reindex(calendar).get(day, np.nan)
                ma_val = df["ma200"].reindex(calendar).get(day, np.nan)
                
                if np.isfinite(rsi_val) and np.isfinite(ma_val):
                    debug_stats['total_checks'] += 1
                    
                    month_key = f"{day.year}-{day.month:02d}"
                    if month_key not in debug_stats['monthly_signals']:
                        debug_stats['monthly_signals'][month_key] = 0
                    
                    rsi_oversold = rsi_val < cfg.rsi_entry_threshold
                    above_ma = price > ma_val
                    
                    if rsi_oversold:
                        debug_stats['rsi_oversold'] += 1
                    if above_ma:
                        debug_stats['above_ma'] += 1
                    
                    if rsi_oversold and above_ma:
                        debug_stats['valid_signals'] += 1
                        debug_stats['monthly_signals'][month_key] += 1
                        
                        # Store sample for analysis
                        if len(debug_stats['sample_signals']) < 10:
                            debug_stats['sample_signals'].append({
                                'date': day,
                                'ticker': t,
                                'price': price,
                                'rsi2': rsi_val,
                                'ma200': ma_val
                            })

            entry_signal = bool(df["entry_cond"].reindex(calendar).get(day, False))
            if not entry_signal:
                continue
            
            # For RSI(2) strategy, only allow one position per stock at a time
            if is_rsi2_strategy and holdings_qty[t] > 0:
                continue
            
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
            
            # Track entry info for RSI(2) strategy
            if is_rsi2_strategy:
                entry_prices[t] = price
                entry_dates[t] = day
            
            trades.append(Trade(day, t, "BUY", float(price), float(qty), float(cash), 
                              float(price), day))

        equity_curve.append({"date": day, "equity": portfolio_value(day), "cash": cash})

    # Handle empty equity curve
    if not equity_curve:
        print(f"WARNING: Equity curve vuoto per {strategy_name}")
        return pd.DataFrame({"equity": [cfg.initial_capital], "cash": [cfg.initial_capital]}, 
                          index=[calendar[0] if len(calendar) > 0 else pd.Timestamp.now()]), []

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    
    # DEBUG: Print detailed RSI(2) analysis
    if is_rsi2_strategy:
        print(f"\n=== DEBUG RSI(2) ANALYSIS ===")
        print(f"Total signal checks: {debug_stats['total_checks']:,}")
        print(f"RSI(2) < 20 occurrences: {debug_stats['rsi_oversold']:,}")
        print(f"Price > MA200 occurrences: {debug_stats['above_ma']:,}")
        print(f"Valid entry signals (both conditions): {debug_stats['valid_signals']:,}")
        
        print(f"\nSegnali validi per mese:")
        for month, count in sorted(debug_stats['monthly_signals'].items()):
            if count > 0:
                print(f"  {month}: {count} segnali")
        
        # Show early 2022 analysis
        early_2022_months = [f"2022-{i:02d}" for i in range(1, 7)]
        early_2022_total = sum(debug_stats['monthly_signals'].get(month, 0) for month in early_2022_months)
        print(f"\nPrimi 6 mesi 2022: {early_2022_total} segnali validi")
        
        # Show sample signals
        if debug_stats['sample_signals']:
            print(f"\nPrimi segnali trovati:")
            for i, signal in enumerate(debug_stats['sample_signals'][:5]):
                print(f"  {signal['date'].strftime('%Y-%m-%d')}: {signal['ticker']} - Price: ${signal['price']:.2f}, RSI(2): {signal['rsi2']:.1f}, MA200: ${signal['ma200']:.2f}")
    
    # Debug trade distribution
    if trades:
        trades_df = pd.DataFrame([{
            'date': t.date,
            'year': t.date.year,
            'month': t.date.month,
            'action': t.action,
            'ticker': t.ticker
        } for t in trades])
        
        trades_by_year = trades_df.groupby('year').size()
        print(f"DEBUG {strategy_name} - Distribuzione trades per anno:")
        for year, count in trades_by_year.items():
            print(f"  {year}: {count} trades")
        
        # Monthly breakdown for 2022
        if is_rsi2_strategy:
            trades_2022 = trades_df[trades_df['year'] == 2022]
            if not trades_2022.empty:
                trades_2022_monthly = trades_2022.groupby('month').size()
                print(f"  2022 breakdown by month:")
                for month, count in trades_2022_monthly.items():
                    print(f"    {month:02d}: {count} trades")
        
        first_trade = trades_df['date'].min()
        print(f"  Primo trade: {first_trade.strftime('%Y-%m-%d')}")
    else:
        print(f"DEBUG {strategy_name} - NESSUN TRADE!")
        if is_rsi2_strategy:
            print("POSSIBILI CAUSE:")
            print("- Mercato in forte uptrend nel 2022 (pochi titoli oversold)")
            print("- Parametri troppo restrittivi (RSI < 20 + price > MA200)")
            print("- Problemi con i dati storici")
    
    if VERBOSE and strategy_name:
        eq = equity_df["equity"].dropna()
        if len(eq) > 1:
            total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
            sharpe = calculate_sharpe_ratio(eq, cfg.risk_free_rate)
            max_dd = calculate_max_drawdown(eq)
            print(f"    {strategy_name}: {total_return:+.1%} TOTAL return, {len(trades)} trades, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.1f}%")
        else:
            print(f"    {strategy_name}: Nessun trade eseguito!")
    
    return equity_df, trades

def buy_hold_benchmark_continuous(start_date: str, end_date: str, initial_capital: float = 10000) -> pd.DataFrame:
    """Buy & Hold SPY CONTINUO dal 2021 al 2025"""
    try:
        spy = fetch_prices_yf_single(("SPY", start_date, end_date))[1]
        if spy.empty:
            return pd.DataFrame()
        
        initial_shares = initial_capital / spy.iloc[0]
        equity_curve = spy * initial_shares
        
        return pd.DataFrame({"equity": equity_curve})
    except:
        print("WARNING: Impossibile scaricare SPY per benchmark")
        return pd.DataFrame()

# ========================= SIGNAL GENERATION PARALLEL ====================

def process_signal_generation(args: Tuple[str, pd.Series, Dict, BacktestConfig, str]) -> Tuple[str, Optional[pd.DataFrame]]:
    """Enhanced signal generation with strategy type support"""
    ticker, price, fundamentals_data, cfg, strategy_type = args
    
    try:
        if strategy_type == "rsi2":
            # RSI(2) strategy (Larry Connors)
            df = compute_rsi2_signals(price, cfg)
            if df["ma200"].dropna().empty or df["rsi2"].dropna().empty:
                return ticker, None
            return ticker, df
        else:
            # Original P/S strategy
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
    except Exception as e:
        if VERBOSE:
            print(f"    ERROR {ticker}: {str(e)}")
        return ticker, None

def generate_signals_parallel(universe: List[str], all_price_data: Dict[str, pd.Series], 
                            fundamentals_data: Dict[str, Tuple[pd.Series, float]], 
                            cfg: BacktestConfig, strategy_type: str = "ps_ratio") -> Dict[str, pd.DataFrame]:
    """Enhanced parallel signal generation with strategy type support"""
    print(f"Generazione parallela segnali {strategy_type} per {len(universe)} titoli...")
    
    args_list = []
    for ticker in universe:
        if ticker in all_price_data:
            if strategy_type == "rsi2":
                # RSI(2) doesn't need fundamentals
                args_list.append((ticker, all_price_data[ticker], fundamentals_data, cfg, strategy_type))
            elif ticker in fundamentals_data:
                # P/S strategy needs fundamentals
                args_list.append((ticker, all_price_data[ticker], fundamentals_data, cfg, strategy_type))
    
    signal_map = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_signal_generation, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(future_to_ticker), total=len(args_list), desc=f"Generazione segnali {strategy_type}"):
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

def plot_portfolio_equity_and_drawdown(all_results: Dict[str, Dict]):
    """Enhanced plot with 4 strategies including RSI(2)"""
    print("\nGenerazione plot equity curve e drawdown...")
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    
    approaches = {
        'A - Tutti i titoli (P/S)': all_results['approach_a'],
        'B - Top 50 market cap (P/S)': all_results['approach_b'], 
        'C - Buy & Hold SPY': all_results['approach_c'],
        'D - RSI(2) Strategy (Connors)': all_results.get('approach_d')
    }
    
    colors = ['blue', 'red', 'green', 'purple']
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    
    for i, (name, result) in enumerate(approaches.items()):
        if result and 'equity' in result:
            equity = result['equity']['equity'].dropna()
            ax1.plot(equity.index, equity.values, 
                    color=colors[i], label=name, linewidth=2, alpha=0.8)
    
    ax1.set_title('Equity Curve Continua (2021-2025) - TUTTE LE STRATEGIE', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Valore Portafoglio ($)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    
    for i, (name, result) in enumerate(approaches.items()):
        if result and 'equity' in result:
            equity = result['equity']['equity'].dropna()
            drawdown = calculate_drawdown(equity)
            ax2.fill_between(drawdown.index, drawdown.values, 0, 
                           color=colors[i], alpha=0.3, label=f'{name.split(" - ")[0]} DD')
            ax2.plot(drawdown.index, drawdown.values, color=colors[i], linewidth=1)
    
    ax2.set_title('Drawdown Storico (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Performance Summary
    ax3 = fig.add_subplot(gs[2, 0])
    
    returns = []
    labels = []
    for name, result in approaches.items():
        if result and 'total_return' in result:
            returns.append(result['total_return'] * 100)
            labels.append(name.split(' - ')[0])
    
    if returns:
        bars = ax3.bar(labels, returns, color=colors[:len(returns)], alpha=0.7)
        ax3.set_title('Total Return (2021-2025)', fontsize=12)
        ax3.set_ylabel('Total Return (%)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.1f}%', ha='center', va='bottom')
    
    # 4. Risk-Return Profile
    ax4 = fig.add_subplot(gs[2, 1])
    
    for i, (name, result) in enumerate(approaches.items()):
        if result and 'sharpe_ratio' in result and 'annualized_return' in result:
            sharpe = result['sharpe_ratio']
            annual_ret = result['annualized_return'] * 100
            
            ax4.scatter(sharpe, annual_ret, color=colors[i], 
                       s=120, alpha=0.7, label=name.split(' - ')[0])
            
            ax4.annotate(name.split(' - ')[0], 
                        (sharpe, annual_ret), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_title('Risk-Return Profile', fontsize=12)
    ax4.set_xlabel('Sharpe Ratio', fontsize=10)
    ax4.set_ylabel('Rendimento Annualizzato (%)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout(pad=2.0)
    plt.show()

# ========================= MAIN COMPARISON ====================

def run_comprehensive_comparison(cfg: BacktestConfig):
    """Enhanced comparison with 4 strategies including RSI(2)"""
    print("=== CONFRONTO COMPLETO: 4 APPROCCI (2021-2025) ===")
    print("A: Tutti i titoli (P/S strategy)")
    print("B: Top 50 market cap (P/S strategy)")
    print("C: Buy & Hold SPY")
    print("D: RSI(2) Strategy (Larry Connors)")
    
    # 1. Download data
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    universe_tickers = get_realistic_universe()
    print(f"Scarico dati per {len(universe_tickers)} titoli dal {cfg.start_year}...")
    
    all_price_data = fetch_prices_parallel(universe_tickers, start_date, end_date)
    print(f"Dati prezzi raccolti per {len(all_price_data)} titoli")
    
    fundamentals_data = fetch_fundamentals_parallel(list(all_price_data.keys()))
    print(f"Dati fondamentali raccolti per {len(fundamentals_data)} titoli")
    
    # 2. Define universes
    full_period_start = pd.to_datetime(start_date)
    full_period_end = pd.to_datetime(end_date)
    
    universe_all = select_all_available(all_price_data, full_period_start, full_period_end, cfg)
    universe_top50 = select_top_market_cap(all_price_data, full_period_start, full_period_end, cfg, top_n=50)
    universe_rsi2 = select_rsi2_candidates(all_price_data, full_period_start, full_period_end, cfg)
    
    print(f"\n=== GENERAZIONE SEGNALI PER BACKTEST CONTINUI ===")
    
    # 3. Run backtests
    all_results = {}
    
    # Approach A & B: P/S strategies
    for approach_name, universe in [("approach_a", universe_all), ("approach_b", universe_top50)]:
        if len(universe) < 5:
            print(f"  {approach_name.upper()}: troppi pochi titoli, skip")
            all_results[approach_name] = None
            continue
        
        signal_map = generate_signals_parallel(universe, all_price_data, fundamentals_data, cfg, "ps_ratio")
        
        if len(signal_map) < 3:
            print(f"  {approach_name.upper()}: troppi pochi segnali, skip")
            all_results[approach_name] = None
            continue
        
        equity, trades = continuous_backtest(signal_map, cfg, approach_name.upper(), is_rsi2_strategy=False)
        
        eq = equity["equity"].dropna()
        if len(eq) > 1:
            total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
            years = (eq.index[-1] - eq.index[0]).days / 365.25
            annualized_return = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1.0
            sharpe_ratio = calculate_sharpe_ratio(eq, cfg.risk_free_rate)
            max_drawdown = calculate_max_drawdown(eq)
        else:
            total_return = annualized_return = sharpe_ratio = max_drawdown = 0.0
        
        all_results[approach_name] = {
            'universe_size': len(universe),
            'signals_count': len(signal_map),
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_count': len(trades),
            'equity': equity
        }
    
    # Approach D - RSI(2) Strategy
    print(f"\n=== RSI(2) STRATEGY (LARRY CONNORS) ===")
    if len(universe_rsi2) >= 5:
        signal_map_rsi2 = generate_signals_parallel(universe_rsi2, all_price_data, fundamentals_data, cfg, "rsi2")
        
        if len(signal_map_rsi2) >= 3:
            equity_rsi2, trades_rsi2 = continuous_backtest(signal_map_rsi2, cfg, "APPROACH_D", is_rsi2_strategy=True)
            
            eq_rsi2 = equity_rsi2["equity"].dropna()
            if len(eq_rsi2) > 1:
                total_return_rsi2 = eq_rsi2.iloc[-1] / eq_rsi2.iloc[0] - 1.0
                years_rsi2 = (eq_rsi2.index[-1] - eq_rsi2.index[0]).days / 365.25
                annualized_return_rsi2 = (eq_rsi2.iloc[-1] / eq_rsi2.iloc[0]) ** (1/years_rsi2) - 1.0
                sharpe_ratio_rsi2 = calculate_sharpe_ratio(eq_rsi2, cfg.risk_free_rate)
                max_drawdown_rsi2 = calculate_max_drawdown(eq_rsi2)
            else:
                total_return_rsi2 = annualized_return_rsi2 = sharpe_ratio_rsi2 = max_drawdown_rsi2 = 0.0
            
            all_results['approach_d'] = {
                'universe_size': len(universe_rsi2),
                'signals_count': len(signal_map_rsi2),
                'total_return': total_return_rsi2,
                'annualized_return': annualized_return_rsi2,
                'sharpe_ratio': sharpe_ratio_rsi2,
                'max_drawdown': max_drawdown_rsi2,
                'trades_count': len(trades_rsi2),
                'equity': equity_rsi2
            }
        else:
            print(f"  APPROACH_D: troppi pochi segnali RSI(2), skip")
            all_results['approach_d'] = None
    else:
        print(f"  APPROACH_D: troppi pochi titoli per RSI(2), skip")
        all_results['approach_d'] = None
    
    # Approach C: Buy & Hold SPY
    print(f"\n=== BUY & HOLD SPY CONTINUO ===")
    spy_equity = buy_hold_benchmark_continuous(start_date, end_date, cfg.initial_capital)
    if not spy_equity.empty:
        eq_spy = spy_equity["equity"].dropna()
        if len(eq_spy) > 1:
            spy_total_return = eq_spy.iloc[-1] / eq_spy.iloc[0] - 1.0
            years = (eq_spy.index[-1] - eq_spy.index[0]).days / 365.25
            spy_annualized = (eq_spy.iloc[-1] / eq_spy.iloc[0]) ** (1/years) - 1.0
            spy_sharpe = calculate_sharpe_ratio(eq_spy, cfg.risk_free_rate)
            spy_max_dd = calculate_max_drawdown(eq_spy)
        else:
            spy_total_return = spy_annualized = spy_sharpe = spy_max_dd = 0.0
        
        all_results['approach_c'] = {
            'universe_size': 1,
            'signals_count': 1,
            'total_return': spy_total_return,
            'annualized_return': spy_annualized,
            'sharpe_ratio': spy_sharpe,
            'max_drawdown': spy_max_dd,
            'trades_count': 1,
            'equity': spy_equity
        }
        
        print(f"    SPY BUY & HOLD: {spy_total_return:+.1%} TOTAL return, Sharpe: {spy_sharpe:.2f}, Max DD: {spy_max_dd:.1f}%")
    else:
        all_results['approach_c'] = None
    
    # 4. Analysis and plotting
    print(f"\n=== CONFRONTO FINALE (2021-2025) - 4 STRATEGIE ===")
    
    plot_portfolio_equity_and_drawdown(all_results)
    
    # Final statistics
    approaches = {
        'A - Tutti i titoli (P/S)': all_results.get('approach_a'),
        'B - Top 50 market cap (P/S)': all_results.get('approach_b'), 
        'C - Buy & Hold SPY': all_results.get('approach_c'),
        'D - RSI(2) Strategy (Connors)': all_results.get('approach_d')
    }
    
    for name, result in approaches.items():
        if not result:
            print(f"\n{name}: Dati non disponibili")
            continue
            
        print(f"\n{name}:")
        print(f"  Total Return (2021-2025):        {result['total_return']*100:+.1f}%")
        print(f"  Rendimento Annualizzato:         {result['annualized_return']*100:.1f}%")
        print(f"  Sharpe Ratio:                    {result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:                    {result['max_drawdown']:.1f}%")
        print(f"  Trades totali:                   {result['trades_count']}")
        print(f"  Universe size:                   {result['universe_size']}")
    
    return all_results

# ====================== MAIN ==========================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        per_trade_usd=600.0,
        start_year=2022,  # Try starting from 2022 to see if that helps with RSI signals
        rolling_window=60,
        min_periods=30,
        entry_k_std=-0.5,
        exit_k_std=0.5,
        accumulate_daily=True,
        fractional_shares=True,
        max_alloc_per_ticker_frac=0.08,
        rebalance_frequency_years=1,
        min_trading_days=100,
        min_market_cap_billion=1.0,
        risk_free_rate=0.02,
        
        # Larry Connors RSI(2) parameters (try more sensitive settings)
        rsi_period=2,                   # RSI period
        rsi_entry_threshold=20,         # Slightly higher threshold (was 20)
        rsi_exit_threshold=90,          # Keep exit at 70
        rsi_ma_period=200,              # 200-day moving average filter
        rsi_ma_type="SMA",              # Use Simple Moving Average
        rsi_max_holding_days=10,        # Max 10 days holding period
        rsi_profit_target_pct=10.0,      # 5% profit target
    )
    
    print(f"Avvio confronto completo con RSI(2) STRATEGY (LARRY CONNORS)!")
    print(f"RSI(2) Parameters (adjusted for more signals):")
    print(f"  - RSI Period: {cfg.rsi_period}")
    print(f"  - Entry when RSI(2) < {cfg.rsi_entry_threshold} (raised from 20)")
    print(f"  - Exit when RSI(2) > {cfg.rsi_exit_threshold}")
    print(f"  - MA Filter: {cfg.rsi_ma_period}-day {cfg.rsi_ma_type}")
    print(f"  - Max holding: {cfg.rsi_max_holding_days} days")
    print(f"  - Profit target: {cfg.rsi_profit_target_pct}%")
    print(f"Parallelizzazione: {MAX_WORKERS} workers...")
    print("PERIODO: 2022-2025 (started later to see market conditions)")
    
    start_time = time.time()
    results = run_comprehensive_comparison(cfg)
    end_time = time.time()
    
    print(f"\nConfronto completato in {end_time - start_time:.1f} secondi!")
    print("\nNOTE: Se RSI(2) strategy ha pochi/nessun trade, considera:")
    print("1. Mercato in uptrend = pochi titoli oversold")
    print("2. Filtro MA200 troppo restrittivo durante bear market")
    print("3. Prova RSI entry threshold = 30 invece di 25")
    print("4. Considera periodi diversi o universe diverso")