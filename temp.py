# comprehensive_comparison.py - INTEGRATO CON RSI BULL RANGE-MOMENTUM dal paper CMT (FIXED)
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
    
    # RSI BULL RANGE-MOMENTUM PARAMETERS (dal paper CMT)
    rsi_period: int = 14              # Standard Wilder RSI
    rsi_lookback_days: int = 100      # 100 giorni ottimali dal paper (75-125 range)
    rsi_bull_range_low: int = 40      # RSI non deve scendere sotto 40
    rsi_bull_momentum_high: int = 70  # RSI deve superare 70
    
    # Market Regime Filter (suggerito nel paper)
    use_market_regime_filter: bool = True
    market_ma_period: int = 200       # S&P 500 sopra/sotto 200 MA
    
    # Parametri originali
    rebalance_frequency_years: int = 2
    min_trading_days: int = 100
    min_market_cap_billion: float = 1.0
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

# ========================= RSI CALCULATIONS (dal paper CMT) ===============================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calcola RSI standard di Wilder"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's smoothing (EMA)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def check_rsi_bull_range(rsi: pd.Series, lookback_days: int, range_low: int = 40) -> pd.Series:
    """
    RSI Bull Range: RSI rimane tra 40 e 100 per N giorni
    True quando il minimo RSI degli ultimi N giorni è >= 40
    """
    rsi_min = rsi.rolling(lookback_days).min()
    return rsi_min >= range_low

def check_rsi_bull_momentum(rsi: pd.Series, lookback_days: int, momentum_high: int = 70) -> pd.Series:
    """
    RSI Bull Momentum: RSI supera 70 negli ultimi N giorni
    True quando il massimo RSI degli ultimi N giorni è > 70
    """
    rsi_max = rsi.rolling(lookback_days).max()
    return rsi_max > momentum_high

def check_rsi_bull_range_momentum(rsi: pd.Series, lookback_days: int, 
                                 range_low: int = 40, momentum_high: int = 70) -> pd.Series:
    """
    RSI Bull Range-Momentum: Combinazione di Range e Momentum
    True quando ENTRAMBE le condizioni sono soddisfatte
    """
    bull_range = check_rsi_bull_range(rsi, lookback_days, range_low)
    bull_momentum = check_rsi_bull_momentum(rsi, lookback_days, momentum_high)
    
    return bull_range & bull_momentum

# ========================= Data fetching (PARALLELIZZATO) ===============================

def fetch_ohlcv_yf_single(args: Tuple[str, str, str]) -> Tuple[str, pd.DataFrame]:
    """Scarica OHLCV per calcoli tecnici avanzati"""
    symbol, start, end = args
    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        if data is None or len(data) == 0:
            return symbol, pd.DataFrame()
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                return symbol, pd.DataFrame()
        
        data.index = pd.to_datetime(data.index).tz_localize(None)
        
        time.sleep(YF_SLEEP)
        return symbol, data
    except Exception:
        return symbol, pd.DataFrame()

def fetch_ohlcv_parallel(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download parallelo dei dati OHLCV"""
    print(f"Download parallelo OHLCV di {len(symbols)} simboli con {MAX_WORKERS} workers...")
    
    args_list = [(sym, start, end) for sym in symbols]
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(fetch_ohlcv_yf_single, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Download OHLCV"):
            symbol = future_to_symbol[future]
            try:
                _, ohlcv_data = future.result()
                if not ohlcv_data.empty and len(ohlcv_data) > 200:
                    results[symbol] = ohlcv_data
            except Exception as e:
                if VERBOSE:
                    print(f"Errore downloading OHLCV {symbol}: {e}")
    
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

def select_all_available(all_ohlcv_data: Dict[str, pd.DataFrame], 
                        period_start: pd.Timestamp, 
                        period_end: pd.Timestamp,
                        cfg: BacktestConfig) -> List[str]:
    """APPROACH A: Usa TUTTI i titoli disponibili"""
    
    candidates = []
    
    for ticker, ohlcv_data in all_ohlcv_data.items():
        period_prices = ohlcv_data['Close'].loc[
            (ohlcv_data.index >= period_start) & 
            (ohlcv_data.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= cfg.min_trading_days:
            candidates.append(ticker)
    
    print(f"  APPROACH A - Tutti disponibili: {len(candidates)} titoli")
    return candidates

def select_top_market_cap(all_ohlcv_data: Dict[str, pd.DataFrame], 
                         period_start: pd.Timestamp, 
                         period_end: pd.Timestamp,
                         cfg: BacktestConfig,
                         top_n: int = 50) -> List[str]:
    """APPROACH B: Solo top N per market cap - FIXED"""
    
    candidates = []
    
    for ticker, ohlcv_data in all_ohlcv_data.items():
        period_prices = ohlcv_data['Close'].loc[
            (ohlcv_data.index >= period_start) & 
            (ohlcv_data.index <= period_end)
        ].dropna()
        
        if len(period_prices) >= cfg.min_trading_days:
            avg_price = period_prices.mean()
            market_cap = get_approximate_market_cap(ticker, avg_price)
            
            # FIX: Assicurati che market_cap sia scalare
            if isinstance(market_cap, pd.Series):
                market_cap = market_cap.iloc[0] if not market_cap.empty else np.nan
            elif not isinstance(market_cap, (int, float)):
                market_cap = np.nan
            
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

# ========================= SIGNAL GENERATION CON RSI BULL RANGE-MOMENTUM ==============================

def compute_rsi_enhanced_signals(ohlcv: pd.DataFrame, ps_series: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """Calcola segnali combinando P/S value e RSI Bull Range-Momentum"""
    
    price = ohlcv['Close']
    
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    df["ps"] = ps_series.reindex(df.index)
    df.loc[~np.isfinite(df["ps"]), "ps"] = np.nan

    # ===== SEGNALI P/S VALUE =====
    df["ps_ma"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).mean()
    df["ps_std"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).std()

    df["entry_threshold"] = df["ps_ma"] + cfg.entry_k_std * df["ps_std"]
    df["exit_threshold"] = df["ps_ma"] + cfg.exit_k_std * df["ps_std"]

    df["ps_entry_cond"] = (df["ps"] < df["entry_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    df["ps_exit_cond"] = (df["ps"] > df["exit_threshold"]) & df["ps_ma"].notna() & df["ps_std"].notna()
    
    # ===== RSI BULL RANGE-MOMENTUM (dal paper CMT) =====
    df["rsi"] = calculate_rsi(price, cfg.rsi_period)
    
    # RSI Bull Range: RSI non scende sotto 40 negli ultimi N giorni
    df["rsi_bull_range"] = check_rsi_bull_range(df["rsi"], cfg.rsi_lookback_days, cfg.rsi_bull_range_low)
    
    # RSI Bull Momentum: RSI supera 70 negli ultimi N giorni  
    df["rsi_bull_momentum"] = check_rsi_bull_momentum(df["rsi"], cfg.rsi_lookback_days, cfg.rsi_bull_momentum_high)
    
    # RSI Bull Range-Momentum: ENTRAMBE le condizioni
    df["rsi_bull_range_momentum"] = df["rsi_bull_range"] & df["rsi_bull_momentum"]
    
    # ===== SEGNALI FINALI COMBINATI =====
    # Entry: P/S value signal E RSI Bull Range-Momentum
    df["entry_cond"] = df["ps_entry_cond"] & df["rsi_bull_range_momentum"]
    
    # Exit: P/S exit signal O perdita di RSI Bull Range-Momentum
    df["exit_cond"] = df["ps_exit_cond"] | ~df["rsi_bull_range_momentum"]
    
    return df

def get_market_regime(start_date: str, end_date: str, ma_period: int = 200) -> pd.Series:
    """Determina il regime di mercato basato su S&P 500 vs 200MA"""
    try:
        spy = fetch_ohlcv_yf_single(("SPY", start_date, end_date))[1]
        if spy.empty:
            return pd.Series()
        
        spy_close = spy['Close']
        spy_ma = spy_close.rolling(ma_period).mean()
        
        # True quando SPY è sopra 200MA
        market_regime = spy_close > spy_ma
        
        return market_regime
    except:
        print("WARNING: Impossibile determinare regime di mercato")
        return pd.Series()

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    action: str
    price: float
    qty: float
    cash_after: float

def continuous_backtest_rsi_enhanced(signal_map: Dict[str, pd.DataFrame], cfg: BacktestConfig, 
                                   market_regime: pd.Series, strategy_name: str = "") -> Tuple[pd.DataFrame, List[Trade]]:
    """Backtest con RSI Bull Range-Momentum e filtro regime di mercato"""
    
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    # Gestisci il calendario
    try:
        spy_ohlcv = fetch_ohlcv_yf_single(("SPY", start_date, end_date))[1]
        if spy_ohlcv.empty:
            raise Exception("SPY data empty")
        calendar = spy_ohlcv.index
    except:
        print(f"WARNING: Impossibile scaricare SPY, uso calendario basato sui dati disponibili")
        all_dates = []
        for df in signal_map.values():
            all_dates.extend(df.index.tolist())
        calendar = pd.to_datetime(sorted(set(all_dates)))
        calendar = calendar[(calendar >= pd.to_datetime(start_date)) & (calendar <= pd.to_datetime(end_date))]
    
    if len(calendar) == 0:
        print(f"ERROR: Calendario vuoto per {strategy_name}")
        return pd.DataFrame(), []
    
    cash = cfg.initial_capital
    holdings_qty: Dict[str, float] = {t: 0.0 for t in signal_map.keys()}
    trades: List[Trade] = []
    equity_curve = []

    price_map: Dict[str, pd.Series] = {t: df["price"].reindex(calendar, method='ffill') for t, df in signal_map.items()}
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

    # Contatori per analisi
    total_ps_signals = 0
    filtered_by_rsi = 0
    filtered_by_market = 0
    final_signals = 0

    for day in calendar:
        port_val = portfolio_value(day)
        
        # Controlla regime di mercato
        if cfg.use_market_regime_filter:
            market_favorable = market_regime.reindex(calendar).get(day, True)
        else:
            market_favorable = True
        
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

        # 2) BUY solo se mercato favorevole
        if market_favorable:
            for t, df in signal_map.items():
                price = price_map[t].get(day, np.nan)
                if not np.isfinite(price):
                    continue

                # Analizza componenti del segnale per debug
                ps_signal = bool(df["ps_entry_cond"].reindex(calendar).get(day, False))
                if ps_signal:
                    total_ps_signals += 1
                    
                    rsi_signal = bool(df["rsi_bull_range_momentum"].reindex(calendar).get(day, False))
                    if not rsi_signal:
                        filtered_by_rsi += 1
                        continue

                entry_signal = bool(df["entry_cond"].reindex(calendar).get(day, False))
                if not entry_signal:
                    continue
                    
                final_signals += 1
                
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
        else:
            filtered_by_market += 1  # Conta giorni filtrati

        equity_curve.append({"date": day, "equity": portfolio_value(day), "cash": cash})

    if not equity_curve:
        print(f"WARNING: Equity curve vuoto per {strategy_name}")
        return pd.DataFrame({"equity": [cfg.initial_capital], "cash": [cfg.initial_capital]}, 
                          index=[calendar[0] if len(calendar) > 0 else pd.Timestamp.now()]), []

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    
    # DEBUG: Analisi filtri
    if VERBOSE:
        print(f"\n=== ANALISI RSI BULL RANGE-MOMENTUM {strategy_name} ===")
        print(f"Segnali P/S totali: {total_ps_signals}")
        if total_ps_signals > 0:
            print(f"Filtrati da RSI Bull Range-Momentum: {filtered_by_rsi} ({filtered_by_rsi/total_ps_signals*100:.1f}%)")
            print(f"Giorni con mercato sfavorevole: {filtered_by_market}")
            print(f"Segnali finali: {final_signals}")
    
    # DEBUG: Analizza distribuzione trades nel tempo
    if trades:
        trades_df = pd.DataFrame([{
            'date': t.date,
            'year': t.date.year,
            'action': t.action,
            'ticker': t.ticker
        } for t in trades])
        
        trades_by_year = trades_df.groupby('year').size()
        print(f"DEBUG {strategy_name} - Distribuzione trades per anno:")
        for year, count in trades_by_year.items():
            print(f"  {year}: {count} trades")
        
        first_trade = trades_df['date'].min()
        print(f"  Primo trade: {first_trade.strftime('%Y-%m-%d')}")
    else:
        print(f"DEBUG {strategy_name} - NESSUN TRADE!")
    
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
    """Buy & Hold SPY CONTINUO"""
    try:
        spy_ohlcv = fetch_ohlcv_yf_single(("SPY", start_date, end_date))[1]
        if spy_ohlcv.empty:
            return pd.DataFrame()
        
        spy_price = spy_ohlcv['Close']
        initial_shares = initial_capital / spy_price.iloc[0]
        equity_curve = spy_price * initial_shares
        
        return pd.DataFrame({"equity": equity_curve})
    except:
        print("WARNING: Impossibile scaricare SPY per benchmark")
        return pd.DataFrame()

# ========================= SIGNAL GENERATION PARALLEL ====================

def process_rsi_enhanced_signal_generation(args: Tuple[str, pd.DataFrame, Dict, BacktestConfig]) -> Tuple[str, Optional[pd.DataFrame]]:
    """Processo singolo per generazione segnali RSI Enhanced"""
    ticker, ohlcv, fundamentals_data, cfg = args
    
    try:
        if ticker not in fundamentals_data:
            return ticker, None
            
        revenue_series, shares = fundamentals_data[ticker]
        
        if revenue_series.empty or not np.isfinite(shares):
            return ticker, None
            
        ps_series = compute_ps_from_yf_data(ohlcv['Close'], revenue_series, shares)
        if ps_series.empty:
            return ticker, None
            
        df = compute_rsi_enhanced_signals(ohlcv, ps_series, cfg)
        if df["ps_ma"].dropna().empty:
            return ticker, None
            
        return ticker, df
    except Exception as e:
        if VERBOSE:
            print(f"    ERROR {ticker}: {str(e)}")
        return ticker, None

def generate_rsi_enhanced_signals_parallel(universe: List[str], all_ohlcv_data: Dict[str, pd.DataFrame], 
                                         fundamentals_data: Dict[str, Tuple[pd.Series, float]], 
                                         cfg: BacktestConfig) -> Dict[str, pd.DataFrame]:
    """Generazione parallela dei segnali RSI Enhanced"""
    print(f"Generazione parallela segnali RSI Enhanced per {len(universe)} titoli...")
    
    args_list = []
    for ticker in universe:
        if ticker in all_ohlcv_data and ticker in fundamentals_data:
            args_list.append((ticker, all_ohlcv_data[ticker], fundamentals_data, cfg))
    
    signal_map = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_rsi_enhanced_signal_generation, args): args[0] for args in args_list}
        
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

def plot_rsi_enhanced_results(all_results: Dict[str, Dict]):
    """Plot completo per strategia RSI Enhanced"""
    print("\nGenerazione plot RSI enhanced results...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 1, 1, 1])
    
    approaches = {
        'A - Tutti i titoli (P/S + RSI Bull Range-Momentum)': all_results['approach_a'],
        'B - Top 50 (P/S + RSI Bull Range-Momentum)': all_results['approach_b'], 
        'C - Buy & Hold SPY': all_results['approach_c']
    }
    
    colors = ['blue', 'red', 'green']
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    
    for i, (name, result) in enumerate(approaches.items()):
        if result and 'equity' in result:
            equity = result['equity']['equity'].dropna()
            ax1.plot(equity.index, equity.values, 
                    color=colors[i], label=name, linewidth=2, alpha=0.8)
    
    ax1.set_title('P/S Value + RSI Bull Range-Momentum Strategy (CMT Paper)', 
                 fontsize=16, fontweight='bold')
    ax1.set_ylabel('Valore Portafoglio ($)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    
    for i, (name, result) in enumerate(approaches.items()):
        if result and 'equity' in result:
            equity = result['equity']['equity'].dropna()
            drawdown = calculate_drawdown(equity)
            ax2.fill_between(drawdown.index, drawdown.values, 0, 
                           color=colors[i], alpha=0.3, label=f'{name} DD')
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
        ax3.set_title('Total Return (RSI Bull Range-Momentum)', fontsize=12)
        ax3.set_ylabel('Total Return (%)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
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
                       s=100, alpha=0.7, label=name.split(' - ')[0])
            
            ax4.annotate(name.split(' - ')[0], 
                        (sharpe, annual_ret), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax4.set_title('Risk-Return Profile (RSI Enhanced)', fontsize=12)
    ax4.set_xlabel('Sharpe Ratio', fontsize=10)
    ax4.set_ylabel('Rendimento Annualizzato (%)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # 5. Success Rate Comparison
    ax5 = fig.add_subplot(gs[3, 0])
    
    success_rates = []
    profit_loss_ratios = []
    labels_short = []
    
    for name, result in approaches.items():
        if result and 'success_rate' in result:
            success_rates.append(result['success_rate'] * 100)
            profit_loss_ratios.append(result.get('profit_loss_ratio', 0))
            labels_short.append(name.split(' - ')[0])
    
    if success_rates:
        x = np.arange(len(labels_short))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, success_rates, width, label='Success Rate (%)', alpha=0.7)
        ax5_twin = ax5.twinx()
        bars2 = ax5_twin.bar(x + width/2, profit_loss_ratios, width, label='P/L Ratio', alpha=0.7, color='orange')
        
        ax5.set_xlabel('Strategia')
        ax5.set_ylabel('Success Rate (%)', color='blue')
        ax5_twin.set_ylabel('Profit/Loss Ratio', color='orange')
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels_short)
        ax5.axhline(y=50, color='blue', linestyle='--', alpha=0.5)
        ax5_twin.axhline(y=2, color='orange', linestyle='--', alpha=0.5)
        ax5.set_title('Success Rate vs P/L Ratio', fontsize=12)
    
    # 6. Trade Distribution
    ax6 = fig.add_subplot(gs[3, 1])
    
    trades_counts = []
    labels = []
    for name, result in approaches.items():
        if result and 'trades_count' in result:
            trades_counts.append(result['trades_count'])
            labels.append(name.split(' - ')[0])
    
    if trades_counts:
        bars = ax6.bar(labels, trades_counts, color=colors[:len(trades_counts)], alpha=0.7)
        ax6.set_title('Numero Trades (RSI Filtered)', fontsize=12)
        ax6.set_ylabel('Numero Trades', fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        for bar, count in zip(bars, trades_counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
    
    plt.tight_layout(pad=2.0)
    plt.show()

# ========================= MAIN COMPARISON ====================

def run_rsi_enhanced_comparison(cfg: BacktestConfig):
    """Confronto con strategia RSI Bull Range-Momentum dal paper CMT"""
    print("=== STRATEGIA RSI BULL RANGE-MOMENTUM (CMT PAPER) ===")
    print(f"Parametri RSI:")
    print(f"  - RSI Period: {cfg.rsi_period} giorni")
    print(f"  - Lookback: {cfg.rsi_lookback_days} giorni") 
    print(f"  - Bull Range: RSI >= {cfg.rsi_bull_range_low}")
    print(f"  - Bull Momentum: RSI > {cfg.rsi_bull_momentum_high}")
    print(f"  - Market Regime Filter: {cfg.use_market_regime_filter}")
    
    # 1. Scarica dati OHLCV in parallelo
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    universe_tickers = get_realistic_universe()
    print(f"Scarico dati OHLCV per {len(universe_tickers)} titoli dal {cfg.start_year}...")
    
    all_ohlcv_data = fetch_ohlcv_parallel(universe_tickers, start_date, end_date)
    print(f"Dati OHLCV raccolti per {len(all_ohlcv_data)} titoli")
    
    fundamentals_data = fetch_fundamentals_parallel(list(all_ohlcv_data.keys()))
    print(f"Dati fondamentali raccolti per {len(fundamentals_data)} titoli")
    
    # 2. Determina regime di mercato
    if cfg.use_market_regime_filter:
        print("Calcolo regime di mercato basato su S&P 500 200MA...")
        market_regime = get_market_regime(start_date, end_date, cfg.market_ma_period)
    else:
        market_regime = pd.Series()
    
    # 3. Determina universo
    full_period_start = pd.to_datetime(start_date)
    full_period_end = pd.to_datetime(end_date)
    
    universe_all = select_all_available(all_ohlcv_data, full_period_start, full_period_end, cfg)
    universe_top50 = select_top_market_cap(all_ohlcv_data, full_period_start, full_period_end, cfg, top_n=50)
    
    print(f"\n=== GENERAZIONE SEGNALI RSI ENHANCED ===")
    
    # 4. Genera segnali RSI Enhanced e esegui backtest
    all_results = {}
    
    for approach_name, universe in [("approach_a", universe_all), ("approach_b", universe_top50)]:
        if len(universe) < 5:
            print(f"  {approach_name.upper()}: troppi pochi titoli, skip")
            all_results[approach_name] = None
            continue
        
        signal_map = generate_rsi_enhanced_signals_parallel(universe, all_ohlcv_data, fundamentals_data, cfg)
        
        if len(signal_map) < 3:
            print(f"  {approach_name.upper()}: troppi pochi segnali, skip")
            all_results[approach_name] = None
            continue
        
        equity, trades = continuous_backtest_rsi_enhanced(signal_map, cfg, market_regime, approach_name.upper())
        
        eq = equity["equity"].dropna()
        if len(eq) > 1:
            total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
            years = (eq.index[-1] - eq.index[0]).days / 365.25
            annualized_return = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1.0
            sharpe_ratio = calculate_sharpe_ratio(eq, cfg.risk_free_rate)
            max_drawdown = calculate_max_drawdown(eq)
            
            # Calcola success rate e P/L ratio (approssimati)
            if trades:
                profitable_trades = sum(1 for i, t in enumerate(trades) if i > 0 and 
                                      t.action == "SELL" and trades[i-1].action == "BUY" and
                                      t.price > trades[i-1].price)
                total_round_trips = sum(1 for t in trades if t.action == "SELL") 
                success_rate = profitable_trades / total_round_trips if total_round_trips > 0 else 0
                
                # Approssima P/L ratio dal Sharpe
                profit_loss_ratio = max(1.0, 1 + sharpe_ratio)
            else:
                success_rate = 0.0
                profit_loss_ratio = 0.0
        else:
            total_return = 0.0
            annualized_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            success_rate = 0.0
            profit_loss_ratio = 0.0
        
        all_results[approach_name] = {
            'universe_size': len(universe),
            'signals_count': len(signal_map),
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_count': len(trades),
            'success_rate': success_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'equity': equity
        }
    
    # APPROACH C: Buy & Hold SPY
    print(f"\n=== BUY & HOLD SPY BENCHMARK ===")
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
            spy_total_return = 0.0
            spy_annualized = 0.0
            spy_sharpe = 0.0
            spy_max_dd = 0.0
        
        all_results['approach_c'] = {
            'universe_size': 1,
            'signals_count': 1,
            'total_return': spy_total_return,
            'annualized_return': spy_annualized,
            'sharpe_ratio': spy_sharpe,
            'max_drawdown': spy_max_dd,
            'trades_count': 1,
            'success_rate': 1.0,
            'profit_loss_ratio': 1.0,
            'equity': spy_equity
        }
        
        print(f"    SPY BUY & HOLD: {spy_total_return:+.1%} TOTAL return, Sharpe: {spy_sharpe:.2f}, Max DD: {spy_max_dd:.1f}%")
    else:
        all_results['approach_c'] = None
    
    # 5. Analisi e plot RSI enhanced
    print(f"\n=== CONFRONTO FINALE RSI BULL RANGE-MOMENTUM ===")
    
    plot_rsi_enhanced_results(all_results)
    
    # Statistiche finali
    approaches = {
        'A - Tutti i titoli (P/S + RSI Bull Range-Momentum)': all_results.get('approach_a'),
        'B - Top 50 market cap (P/S + RSI Bull Range-Momentum)': all_results.get('approach_b'), 
        'C - Buy & Hold SPY': all_results.get('approach_c')
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
        print(f"  Success Rate (approx):           {result['success_rate']*100:.1f}%")
        print(f"  P/L Ratio (approx):              {result['profit_loss_ratio']:.2f}")
        print(f"  Trades totali:                   {result['trades_count']}")
        print(f"  Universe size:                   {result['universe_size']}")
    
    return all_results

# ====================== MAIN ==========================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        per_trade_usd=200.0,
        start_year=2021,
        rolling_window=30,
        min_periods=15,
        entry_k_std=-0.5,
        exit_k_std=0.5,
        accumulate_daily=True,
        fractional_shares=True,
        max_alloc_per_ticker_frac=0.05,
        
        # RSI Bull Range-Momentum parameters (CMT paper optimal)
        rsi_period=14,               # Standard Wilder
        rsi_lookback_days=100,       # 100 giorni ottimale (tra 75-125)
        rsi_bull_range_low=40,       # Range minimo
        rsi_bull_momentum_high=70,   # Momentum threshold
        
        # Market regime filter
        use_market_regime_filter=True,
        market_ma_period=200,
        
        rebalance_frequency_years=2,
        min_trading_days=100,
        min_market_cap_billion=1.0,
        risk_free_rate=0.02,
    )
    
    print(f"Avvio strategia RSI BULL RANGE-MOMENTUM con {MAX_WORKERS} workers...")
    print("STRATEGIA: P/S Value + RSI Bull Range-Momentum (CMT Paper)")
    print("Solo segnali BULLISH (bearish non funzionano secondo il paper)")
    
    start_time = time.time()
    results = run_rsi_enhanced_comparison(cfg)
    end_time = time.time()
    
    print(f"\nAnalisi RSI Bull Range-Momentum completata in {end_time - start_time:.1f} secondi!")