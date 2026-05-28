# cointegration_only_strategy.py - Solo cointegrazione e momentum pairs
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
    start_year: int = 2010
    rolling_window: int = 60
    min_periods: int = 30
    entry_k_std: float = 0.0
    exit_k_std: float = 1.0
    accumulate_daily: bool = True
    fractional_shares: bool = True
    max_alloc_per_ticker_frac: float = 0.02
    
    # SOLO COINTEGRAZIONE (la tua idea originale)
    max_cluster_exposure_frac: float = 0.15    # Max 15% per cluster cointegrato
    momentum_pair_weight: float = 0.5          # Peso ridotto per momentum pairs
    correlation_threshold: float = 0.6         # Soglia per clustering
    cointegration_lookback: int = 252          # 1 anno per analisi correlazione
    
    rebalance_frequency_years: int = 2
    min_trading_days: int = 100
    transaction_cost_per_trade: float = 0.0
    random_seed: int = 42

# ============================= UNIVERSE ===============================

LARGE_CAP_UNIVERSE = [
    # Tech
    "AAPL","MSFT","AMZN","GOOGL","GOOG","ORCL","CSCO","INTC","IBM","QCOM","AMAT","ADI","NVDA","AMD","CRM",
    
    # Finance
    "JPM","BAC","WFC","GS","MS","C","USB","PNC","COF","AXP","BLK","SCHW","KEY","CMA","FITB","RF",
    
    # Healthcare
    "JNJ","PFE","MRK","ABBV","LLY","UNH","CVS","BMY","ABT","GILD","AMGN","MDT","TMO","DHR","BSX","ISRG",
    
    # Consumer
    "WMT","HD","TGT","COST","LOW","KO","PEP","PG","MCD","SBUX","NKE","TJX","DIS",
    
    # Energy
    "XOM","CVX","COP","SLB","HAL","OXY","APA","DVN","EOG",
    
    # Industrials
    "GE","CAT","MMM","HON","BA","LMT","NOC","GD","UPS","FDX","DE","EMR","ITW",
    
    # Materials
    "DD","DOW","LYB","APD","PPG","SHW","ECL","FCX","NEM",
    
    # Utilities
    "D","SO","NEE","DUK","EXC","PCG","ED","AEP","XEL",
    
    # REITs
    "AMT","PLD","PSA","EQR","AVB","UDR","CPT","MAA","ESS","BXP","SPG"
]

def get_universe() -> List[str]:
    return LARGE_CAP_UNIVERSE

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
        
        annual_financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
        
        financials = None
        
        if annual_financials is not None and not annual_financials.empty:
            financials = annual_financials
        elif quarterly_financials is not None and not quarterly_financials.empty:
            financials = quarterly_financials
        
        if financials is None or financials.empty:
            return pd.Series(dtype=float), np.nan
        
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

# ========================= COINTEGRAZIONE ==============================

def analyze_correlations(price_data: Dict[str, pd.Series], lookback_days: int = 252):
    """
    Analisi correlazione per trovare clusters cointegrati
    """
    print(f"=== ANALISI CORRELAZIONE ({len(price_data)} titoli) ===")
    
    # Allinea le serie
    df_prices = pd.DataFrame(price_data)
    df_prices = df_prices.dropna().tail(lookback_days)
    
    if len(df_prices) < 100:
        print("Dati insufficienti per correlazione")
        return {}, {}
    
    # Calcola correlazioni sui rendimenti
    returns = df_prices.pct_change().dropna()
    correlation_matrix = returns.corr()
    
    tickers = correlation_matrix.index.tolist()
    
    # Identifica coppie altamente correlate
    cointegrated_pairs = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.6:
                cointegrated_pairs.append({
                    'stock1': tickers[i],
                    'stock2': tickers[j],
                    'correlation': corr_val
                })
    
    print(f"Trovate {len(cointegrated_pairs)} coppie altamente correlate")
    
    # Crea clusters
    clusters = create_correlation_clusters(correlation_matrix)
    
    return cointegrated_pairs, clusters

def create_correlation_clusters(correlation_matrix, threshold=0.6):
    """Clustering basato su correlazione"""
    tickers = correlation_matrix.index.tolist()
    clusters = {}
    assigned = set()
    cluster_id = 0
    
    for i, ticker1 in enumerate(tickers):
        if ticker1 in assigned:
            continue
            
        # Inizia nuovo cluster
        cluster_members = [ticker1]
        assigned.add(ticker1)
        
        # Cerca altri ticker correlati
        for j, ticker2 in enumerate(tickers):
            if j <= i or ticker2 in assigned:
                continue
                
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                cluster_members.append(ticker2)
                assigned.add(ticker2)
        
        # Salva cluster se ha almeno 2 membri
        if len(cluster_members) >= 2:
            clusters[cluster_id] = cluster_members
            cluster_id += 1
    
    print(f"Identificati {len(clusters)} clusters:")
    for cluster_id, members in clusters.items():
        print(f"  Cluster {cluster_id}: {members}")
    
    return clusters

# ========================= SIGNALS CON MOMENTUM PAIRS =================

def compute_signals_from_ps(price: pd.Series, ps_series: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """Segnali P/S base (senza filtri aggiuntivi)"""
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
    
    # Inizializza colonna momentum pairs
    df["momentum_pair_signal"] = False
    
    return df

def enhanced_trading_signals_with_momentum_pairs(signal_map: Dict[str, pd.DataFrame], 
                                               cointegrated_pairs: List[dict], 
                                               clusters: Dict) -> Dict[str, pd.DataFrame]:
    """
    LA TUA IDEA: Momentum pairs logic
    Quando un titolo ha buy signal, cerca titoli correlati più cari e compra anche quelli
    """
    print("=== MOMENTUM PAIRS LOGIC ===")
    
    enhanced_signals = signal_map.copy()
    
    # Crea mappa cluster per ticker
    ticker_to_cluster = {}
    for cluster_id, members in clusters.items():
        for ticker in members:
            ticker_to_cluster[ticker] = cluster_id
    
    # Crea mappa delle coppie correlate
    pairs_map = {}
    for pair in cointegrated_pairs:
        s1, s2 = pair['stock1'], pair['stock2']
        if s1 not in pairs_map:
            pairs_map[s1] = []
        if s2 not in pairs_map:
            pairs_map[s2] = []
        pairs_map[s1].append((s2, pair['correlation']))
        pairs_map[s2].append((s1, pair['correlation']))
    
    # Calendario comune
    calendar = pd.date_range(
        start=min(df.index.min() for df in signal_map.values()),
        end=max(df.index.max() for df in signal_map.values()),
        freq='D'
    )
    
    momentum_signals_count = 0
    
    for date in calendar:
        # Identifica chi ha segnali di acquisto oggi
        buy_signals = {}
        for ticker, df in signal_map.items():
            if date in df.index and df.loc[date, 'entry_cond']:
                buy_signals[ticker] = df.loc[date, 'ps']
        
        if not buy_signals:
            continue
            
        # Per ogni acquisto, cerca momentum pairs
        for buying_ticker in buy_signals:
            # Metodo 1: Pairs correlate
            if buying_ticker in pairs_map:
                for other_ticker, corr_val in pairs_map[buying_ticker]:
                    if (other_ticker in signal_map and 
                        date in signal_map[other_ticker].index and
                        other_ticker not in buy_signals and  # Non ha già segnale normale
                        corr_val > 0.4):  # Correlazione positiva
                        
                        other_ps = signal_map[other_ticker].loc[date, 'ps']
                        buying_ps = buy_signals[buying_ticker]
                        
                        # Se l'altro è più caro ma correlato (momentum)
                        if (pd.notna(other_ps) and pd.notna(buying_ps) and 
                            other_ps > buying_ps * 1.05):  # almeno 5% più caro
                            
                            enhanced_signals[other_ticker].loc[date, 'momentum_pair_signal'] = True
                            momentum_signals_count += 1
            
            # Metodo 2: Membri dello stesso cluster
            if buying_ticker in ticker_to_cluster:
                cluster_id = ticker_to_cluster[buying_ticker]
                cluster_members = clusters[cluster_id]
                
                for other_ticker in cluster_members:
                    if (other_ticker != buying_ticker and 
                        other_ticker in signal_map and
                        date in signal_map[other_ticker].index and
                        other_ticker not in buy_signals):
                        
                        other_ps = signal_map[other_ticker].loc[date, 'ps']
                        buying_ps = buy_signals[buying_ticker]
                        
                        if (pd.notna(other_ps) and pd.notna(buying_ps) and 
                            other_ps > buying_ps * 1.03):  # almeno 3% più caro per cluster
                            
                            enhanced_signals[other_ticker].loc[date, 'momentum_pair_signal'] = True
                            momentum_signals_count += 1
    
    print(f"Aggiunti {momentum_signals_count} segnali momentum pairs")
    return enhanced_signals

# ========================= BACKTEST CON COINTEGRAZIONE ================

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    action: str
    price: float
    qty: float
    cash_after: float

def backtest_portfolio_with_cointegration(signal_map: Dict[str, pd.DataFrame], 
                                        clusters: Dict,
                                        cfg: BacktestConfig) -> Tuple[pd.DataFrame, List[Trade]]:
    """
    Backtest con controllo cluster exposure e momentum pairs
    """
    print("=== BACKTEST CON COINTEGRAZIONE ===")
    
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
    
    # Mappa ticker->cluster
    ticker_to_cluster = {}
    for cluster_id, members in clusters.items():
        for ticker in members:
            ticker_to_cluster[ticker] = cluster_id

    def portfolio_value(day: pd.Timestamp) -> float:
        val = cash
        for t, qty in holdings_qty.items():
            if qty == 0:
                continue
            p = price_map[t].get(day, np.nan)
            if np.isfinite(p):
                val += qty * p
        return val
    
    def cluster_exposure(cluster_id: int, day: pd.Timestamp) -> float:
        """Calcola esposizione corrente a un cluster"""
        exposure = 0.0
        cluster_members = clusters.get(cluster_id, [])
        for ticker in cluster_members:
            if ticker in holdings_qty and holdings_qty[ticker] > 0:
                p = price_map[ticker].get(day, np.nan)
                if np.isfinite(p):
                    exposure += holdings_qty[ticker] * p
        return exposure

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

        # 2) BUY con controllo cluster
        for t, df in signal_map.items():
            price = price_map[t].get(day, np.nan)
            if not np.isfinite(price):
                continue

            # Check segnali
            entry_signal = bool(df["entry_cond"].reindex(calendar).get(day, False))
            momentum_signal = bool(df["momentum_pair_signal"].reindex(calendar).get(day, False))
            
            if not (entry_signal or momentum_signal):
                continue
            
            # Controllo esposizione cluster
            cluster_id = ticker_to_cluster.get(t)
            if cluster_id is not None:
                current_cluster_exposure = cluster_exposure(cluster_id, day)
                max_cluster_value = cfg.max_cluster_exposure_frac * port_val
                
                if current_cluster_exposure >= max_cluster_value:
                    continue  # Skip, cluster già troppo esposto
            
            # Determina size ordine
            if momentum_signal and not entry_signal:
                amount = cfg.per_trade_usd * cfg.momentum_pair_weight
            else:
                amount = cfg.per_trade_usd
            
            # Controlli allocazione ticker
            max_alloc_for_ticker = cfg.max_alloc_per_ticker_frac * port_val
            current_alloc = holdings_qty[t] * price if holdings_qty[t] > 0 else 0
            remaining_alloc = max(0.0, max_alloc_for_ticker - current_alloc)
            amount = min(amount, cash, remaining_alloc)
            
            if amount <= 0:
                continue
                
            qty = amount / price if cfg.fractional_shares else math.floor(amount / price)
            if qty <= 0:
                continue
                
            cash -= qty * price
            holdings_qty[t] += qty
            action = "BUY_MOMENTUM" if momentum_signal and not entry_signal else "BUY"
            trades.append(Trade(day, t, action, float(price), float(qty), float(cash)))

        equity_curve.append({"date": day, "equity": portfolio_value(day), "cash": cash})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return equity_df, trades

def buy_hold_benchmark(start_date: str, end_date: str, initial_capital: float = 10000) -> pd.DataFrame:
    """Buy & Hold SPY benchmark"""
    spy = fetch_prices_yf("SPY", start_date, end_date)
    if spy.empty:
        return pd.DataFrame()
    
    initial_shares = initial_capital / spy.iloc[0]
    equity_curve = spy * initial_shares
    
    return pd.DataFrame({"equity": equity_curve})

# ========================= ANALYSIS & METRICS =========================

def calculate_metrics(equity: pd.DataFrame, benchmark_equity: pd.DataFrame, trades: List[Trade]) -> Dict:
    """Calcola metriche di performance"""
    
    eq = equity["equity"].dropna()
    bench = benchmark_equity["equity"].dropna() if not benchmark_equity.empty else pd.Series()
    
    if len(eq) < 2:
        return {}
    
    # Allinea serie temporali
    common_dates = eq.index.intersection(bench.index) if not bench.empty else eq.index
    eq_aligned = eq.reindex(common_dates)
    bench_aligned = bench.reindex(common_dates) if not bench.empty else None
    
    # Rendimenti
    returns = eq_aligned.pct_change().fillna(0.0)
    bench_returns = bench_aligned.pct_change().fillna(0.0) if bench_aligned is not None else pd.Series()
    
    # Basic metrics
    total_return = eq_aligned.iloc[-1] / eq_aligned.iloc[0] - 1.0
    days = (eq_aligned.index[-1] - eq_aligned.index[0]).days
    annualized_return = (eq_aligned.iloc[-1] / eq_aligned.iloc[0]) ** (365.25/days) - 1.0
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown
    running_max = eq_aligned.cummax()
    drawdown = (eq_aligned - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Benchmark comparison
    benchmark_annual = 0
    alpha = 0
    
    if not bench_returns.empty and len(bench_returns) > 10:
        benchmark_total = bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1.0
        benchmark_annual = (bench_aligned.iloc[-1] / bench_aligned.iloc[0]) ** (365.25/days) - 1.0
        alpha = annualized_return - benchmark_annual
    
    # Trade analysis
    normal_trades = [t for t in trades if t.action == "BUY"]
    momentum_trades = [t for t in trades if t.action == "BUY_MOMENTUM"]
    sell_trades = [t for t in trades if t.action == "SELL"]
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'benchmark_return': benchmark_annual,
        'alpha': alpha,
        'num_trades': len(trades),
        'normal_buys': len(normal_trades),
        'momentum_buys': len(momentum_trades),
        'sells': len(sell_trades)
    }
    
    return metrics

# ========================= MAIN COINTEGRATION STRATEGY ================

def run_cointegration_strategy(cfg: BacktestConfig):
    """
    Strategia basata su P/S con cointegrazione e momentum pairs
    """
    print("=== STRATEGIA COINTEGRAZIONE ===")
    
    # Setup
    start_date = f"{cfg.start_year}-01-01"
    end_date = today_str()
    
    universe_tickers = get_universe()
    print(f"Universo: {len(universe_tickers)} titoli")
    
    # Scarica dati
    print("Scarico dati...")
    all_price_data = {}
    
    for sym in tqdm(universe_tickers):
        price = fetch_prices_yf(sym, start_date, end_date)
        if not price.empty and len(price) > 200:
            all_price_data[sym] = price
        time.sleep(YF_SLEEP)
    
    print(f"Dati prezzi per {len(all_price_data)} titoli")
    
    # Periodi
    periods = []
    current_year = cfg.start_year
    while current_year < 2025:
        end_year = min(current_year + cfg.rebalance_frequency_years, 2025)
        periods.append((current_year, end_year))
        current_year = end_year
    
    print(f"Testando su {len(periods)} periodi...")
    
    # Backtest per ogni periodo
    all_results = []
    
    for period_idx, (start_year, end_year) in enumerate(periods):
        print(f"\n--- Periodo {period_idx+1}: {start_year} to {end_year} ---")
        
        period_start = pd.to_datetime(f"{start_year}-01-01")
        period_end = pd.to_datetime(f"{end_year}-01-01")
        
        # Filtra universo per periodo
        universe_period = []
        for ticker, price_series in all_price_data.items():
            period_prices = price_series.loc[
                (price_series.index >= period_start) & 
                (price_series.index <= period_end)
            ].dropna()
            
            if len(period_prices) >= cfg.min_trading_days:
                universe_period.append(ticker)
        
        print(f"Universo periodo: {len(universe_period)} titoli")
        
        # Analisi cointegrazione per questo periodo
        period_price_data = {}
        for ticker in universe_period:
            period_prices = all_price_data[ticker].loc[
                (all_price_data[ticker].index >= period_start) & 
                (all_price_data[ticker].index <= period_end)
            ]
            period_price_data[ticker] = period_prices
        
        cointegrated_pairs, clusters = analyze_correlations(period_price_data, cfg.cointegration_lookback)
        
        # Genera segnali base
        signal_map = {}
        
        for ticker in universe_period:
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
            time.sleep(YF_SLEEP * 0.01)
        
        if len(signal_map) < 5:
            print(f"Troppi pochi segnali nel periodo {start_year}-{end_year}")
            continue
        
        print(f"Segnali generati per {len(signal_map)} titoli")
        
        # Momentum pairs
        enhanced_signals = enhanced_trading_signals_with_momentum_pairs(signal_map, cointegrated_pairs, clusters)
        
        # Backtest
        equity, trades = backtest_portfolio_with_cointegration(enhanced_signals, clusters, cfg)
        
        # Benchmark
        benchmark_equity = buy_hold_benchmark(
            period_start.strftime("%Y-%m-%d"), 
            period_end.strftime("%Y-%m-%d"), 
            cfg.initial_capital
        )
        
        # Metriche
        metrics = calculate_metrics(equity, benchmark_equity, trades)
        metrics['period'] = f"{start_year}-{end_year}"
        
        all_results.append({
            'period': f"{start_year}-{end_year}",
            'equity': equity,
            'trades': trades,
            'clusters': clusters,
            'metrics': metrics
        })
        
        print(f"Risultati: {metrics['annualized_return']:.1%} return, {metrics['alpha']:.1%} alpha, "
              f"{metrics['sharpe_ratio']:.2f} Sharpe, "
              f"{metrics['normal_buys']} buys, {metrics['momentum_buys']} momentum pairs")
    
    # Summary
    print_summary(all_results)
    
    # Plot
    if all_results:
        plot_results(all_results)
    
    return all_results

def print_summary(results: List[Dict]):
    """Stampa sommario dei risultati"""
    print(f"\n{'='*60}")
    print("SUMMARY COINTEGRATION STRATEGY")
    print(f"{'='*60}")
    
    if not results:
        print("Nessun risultato disponibile.")
        return
    
    # Metriche aggregate
    avg_return = np.mean([r['metrics'].get('annualized_return', 0) for r in results]) * 100
    avg_benchmark = np.mean([r['metrics'].get('benchmark_return', 0) for r in results]) * 100
    avg_alpha = np.mean([r['metrics'].get('alpha', 0) for r in results]) * 100
    avg_sharpe = np.mean([r['metrics'].get('sharpe_ratio', 0) for r in results])
    avg_drawdown = np.mean([r['metrics'].get('max_drawdown', 0) for r in results]) * 100
    
    total_normal_buys = sum([r['metrics'].get('normal_buys', 0) for r in results])
    total_momentum_buys = sum([r['metrics'].get('momentum_buys', 0) for r in results])
    
    print(f"Periodi analizzati: {len(results)}")
    print(f"")
    print(f"PERFORMANCE:")
    print(f"  Rendimento annualizzato medio: {avg_return:.1f}%")
    print(f"  SPY benchmark medio:           {avg_benchmark:.1f}%")
    print(f"  Alpha medio:                   {avg_alpha:.1f}%")
    print(f"  Sharpe Ratio medio:            {avg_sharpe:.2f}")
    print(f"  Max Drawdown medio:            {avg_drawdown:.1f}%")
    print(f"")
    print(f"TRADING:")
    print(f"  Buy normali totali:            {total_normal_buys}")
    print(f"  Momentum pairs totali:         {total_momentum_buys}")
    print(f"  % Momentum pairs:              {total_momentum_buys/(total_normal_buys+total_momentum_buys)*100:.1f}%")

def plot_results(results: List[Dict]):
    """Plot risultati di tutti i periodi"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Equity Curve cumulativa
    ax1 = axes[0, 0]
    cumulative_equity = [10000]
    benchmark_equity = [10000]
    years = [int(results[0]['period'].split('-')[0])]
    
    for result in results:
        metrics = result['metrics']
        strategy_mult = 1 + metrics.get('total_return', 0)
        benchmark_mult = 1 + metrics.get('benchmark_return', 0)
        
        cumulative_equity.append(cumulative_equity[-1] * strategy_mult)
        benchmark_equity.append(benchmark_equity[-1] * benchmark_mult)
        years.append(int(result['period'].split('-')[1]))
    
    ax1.plot(years, cumulative_equity, 'b-', linewidth=3, label='Cointegration Strategy')
    ax1.plot(years, benchmark_equity, 'r--', linewidth=2, label='SPY')
    ax1.set_title('Equity Curve Cumulativa')
    ax1.set_ylabel('Valore Portfolio ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Alpha per periodo
    ax2 = axes[0, 1]
    alphas = [r['metrics'].get('alpha', 0) * 100 for r in results]
    periods = [r['period'] for r in results]
    
    bars = ax2.bar(range(len(alphas)), alphas, 
                 color=['green' if x > 0 else 'red' for x in alphas])
    ax2.set_title('Alpha vs SPY per Periodo')
    ax2.set_ylabel('Alpha (%)')
    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels(periods, rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Momentum Pairs %
    ax3 = axes[1, 0]
    normal_buys = [r['metrics'].get('normal_buys', 0) for r in results]
    momentum_buys = [r['metrics'].get('momentum_buys', 0) for r in results]
    
    x = range(len(periods))
    width = 0.35
    ax3.bar(x, normal_buys, width, label='Normal Buys')
    ax3.bar(x, momentum_buys, width, bottom=normal_buys, label='Momentum Pairs')
    ax3.set_title('Normal vs Momentum Buys')
    ax3.set_ylabel('Numero Trades')
    ax3.set_xticks(x)
    ax3.set_xticklabels(periods, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Sharpe Ratio
    ax4 = axes[1, 1]
    sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in results]
    
    ax4.bar(range(len(sharpes)), sharpes, color='blue', alpha=0.7)
    ax4.set_title('Sharpe Ratio per Periodo')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_xticks(range(len(periods)))
    ax4.set_xticklabels(periods, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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
        
        # SOLO COINTEGRAZIONE
        max_cluster_exposure_frac=0.15,
        momentum_pair_weight=0.5,
        correlation_threshold=0.6,
        cointegration_lookback=252,
        
        rebalance_frequency_years=2,
        min_trading_days=100
    )
    
    print("Avvio Cointegration Strategy...")
    results = run_cointegration_strategy(cfg)
    
    print("\nBacktest completato!")