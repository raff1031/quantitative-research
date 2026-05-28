# optimized_ps_strategy.py - Strategia P/S con gestione portafoglio ottimizzata
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================= CONFIG =================================

# Flag per attivare le stampe di debug in caso di problemi
VERBOSE = False
# Pausa tra le richieste a yfinance per evitare di essere bloccati
YF_SLEEP = 0.05

@dataclass
class BacktestConfig:
    """Configurazione completa per il backtest."""
    initial_capital: float = 10_000.0
    start_year: int = 2010
    
    # Parametri della strategia
    rolling_window: int = 60
    min_periods: int = 30
    entry_k_std: float = 0.0  # Compra quando P/S è sotto la sua media mobile
    exit_k_std: float = 1.0   # Vendi quando P/S è sopra la media + 1 dev. std.
    
    # Gestione del portafoglio
    equal_position_sizing: bool = True
    position_size_pct: float = 0.04  # Aumentato leggermente per usare più capitale (4%)
    max_positions: int = 25          # Massimo 25 posizioni contemporaneamente
    rebalance_frequency: int = 21    # Frequenza di ribilanciamento (in giorni di calendario)
    
    # Gestione del rischio
    stop_loss_atr_mult: float = 2.0  # Stop loss basato sull'ATR
    take_profit_pct: float = 0.15    # Take profit fisso al 15%
    
    # Parametri tecnici
    transaction_cost_pct: float = 0.0001 # Commissioni dello 0.01%
    atr_period: int = 14
    random_seed: int = 42

# ============================= UNIVERSE ===============================

LARGE_CAP_UNIVERSE = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","ORCL","CSCO","INTC","IBM","QCOM","AMAT","ADI","NVDA","AMD","CRM",
    "JPM","BAC","WFC","GS","MS","C","USB","PNC","COF","AXP","BLK","SCHW",
    "JNJ","PFE","MRK","ABBV","LLY","UNH","CVS","BMY","ABT","GILD","AMGN","MDT","TMO","DHR","BSX","ISRG",
    "WMT","HD","TGT","COST","LOW","KO","PEP","PG","MCD","SBUX","NKE","TJX","DIS",
    "XOM","CVX","COP","SLB","HAL","OXY","DVN","EOG",
    "GE","CAT","MMM","HON","BA","LMT","NOC","GD","UPS","FDX","DE","EMR","ITW",
    "DD","DOW","LYB","APD","PPG","SHW","ECL","FCX","NEM",
    "D","SO","NEE","DUK","EXC","ED","AEP","XEL",
    "AMT","PLD","PSA","EQR","AVB","UDR","CPT","MAA","ESS","BXP","SPG"
]

def get_universe() -> List[str]:
    return LARGE_CAP_UNIVERSE

def today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

# ========================= Data fetching (VERSIONE ROBUSTA) ===============================

def fetch_ohlcv_yf(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        time.sleep(YF_SLEEP)  # Aggiunta pausa
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        if data.empty: return pd.DataFrame()
        data.index = pd.to_datetime(data.index).tz_localize(None)
        return data
    except Exception:
        return pd.DataFrame()

def fetch_yf_fundamentals_robust(symbol: str) -> Tuple[pd.Series, pd.Series]:
    """Versione robusta che scarica le azioni in circolazione come serie storica."""
    try:
        time.sleep(YF_SLEEP)  # Aggiunta pausa
        ticker = yf.Ticker(symbol)
        
        # 1. Fatturato (Revenue)
        financials = ticker.financials
        if financials.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        revenue_row = next((idx for idx in financials.index if 'revenue' in str(idx).lower()), None)
        if revenue_row is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        revenue_series = financials.loc[revenue_row].dropna()
        revenue_series.index = pd.to_datetime(revenue_series.index)
        
        # 2. Azioni in circolazione (Shares Outstanding) come serie storica
        shares_series = ticker.get_shares_full(start="2000-01-01")
        if shares_series is None or shares_series.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        shares_series.index = pd.to_datetime(shares_series.index).tz_localize(None)
        return revenue_series, shares_series
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float)

def compute_ps_from_yf_data(price: pd.Series, revenue_series: pd.Series, shares_series: pd.Series) -> pd.Series:
    """Calcola il P/S giornaliero usando dati fondamentali storici."""
    if revenue_series.empty or shares_series.empty:
        return pd.Series(dtype=float)
    
    # CORREZIONE: Rimuovi i duplicati nell'indice prima del reindex
    revenue_series = revenue_series.groupby(level=0).last()  # Mantieni l'ultimo valore per date duplicate
    shares_series = shares_series.groupby(level=0).last()    # Mantieni l'ultimo valore per date duplicate
    
    revenue_daily = revenue_series.reindex(price.index, method="ffill")
    shares_daily = shares_series.reindex(price.index, method="ffill")
    
    rev_ps_daily = revenue_daily / shares_daily
    ps_daily = (price / rev_ps_daily).replace([np.inf, -np.inf], np.nan).dropna()
    
    return ps_daily[ps_daily > 0]

def calculate_atr(ohlc: pd.DataFrame, period: int) -> pd.Series:
    """Calcola l'Average True Range (ATR)."""
    high_low = ohlc['High'] - ohlc['Low']
    high_close = np.abs(ohlc['High'] - ohlc['Close'].shift())
    low_close = np.abs(ohlc['Low'] - ohlc['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ========================= Signal Generation =========================

def compute_signals_from_ps(price: pd.Series, ps_series: pd.Series, atr: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    df["ps"] = ps_series
    df["atr"] = atr
    
    df["ps_ma"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).mean()
    df["ps_std"] = df["ps"].rolling(cfg.rolling_window, min_periods=cfg.min_periods).std()
    
    df["entry_threshold"] = df["ps_ma"] + cfg.entry_k_std * df["ps_std"]
    df["exit_threshold"] = df["ps_ma"] + cfg.exit_k_std * df["ps_std"]
    
    df["entry_cond"] = (df["ps"] < df["entry_threshold"]) & df["ps_ma"].notna()
    df["exit_cond"] = (df["ps"] > df["exit_threshold"]) & df["ps_ma"].notna()
    
    return df

# ========================= Backtesting Engine ======================

@dataclass
class Position:
    ticker: str; entry_date: pd.Timestamp; entry_price: float; qty: float
    stop_loss: float; take_profit: float; last_rebalanced: pd.Timestamp

@dataclass
class Trade:
    date: pd.Timestamp; ticker: str; action: str; price: float; qty: float
    cash_after: float; reason: str = "SIGNAL"

def optimized_portfolio_backtest(signal_map: Dict[str, pd.DataFrame], cfg: BacktestConfig) -> Tuple[pd.DataFrame, List[Trade]]:
    print("=== Esecuzione Backtest Ottimizzato ===")
    
    # Trova il calendario comune di trading
    first_idx = min(df.index.min() for df in signal_map.values())
    last_idx = max(df.index.max() for df in signal_map.values())
    spy = fetch_ohlcv_yf("SPY", start=first_idx.strftime("%Y-%m-%d"), end=last_idx.strftime("%Y-%m-%d"))
    calendar = spy.index
    
    cash = cfg.initial_capital
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_curve = []
    
    price_map: Dict[str, pd.Series] = {t: df["price"].reindex(calendar, method='ffill') for t, df in signal_map.items()}
    
    last_rebalance_day = calendar[0]

    for day in tqdm(calendar, desc="Simulazione giornaliera"):
        portfolio_val = sum(pos.qty * price_map[ticker].get(day, 0) for ticker, pos in positions.items()) + cash
        
        # --- Blocco di VENDITA ---
        for ticker, pos in list(positions.items()):
            price = price_map[ticker].get(day)
            if pd.isna(price): continue

            exit_signal = bool(signal_map[ticker]["exit_cond"].get(day, False))
            
            reason = None
            if price <= pos.stop_loss: reason = "STOP"
            elif price >= pos.take_profit: reason = "PROFIT"
            elif exit_signal: reason = "SIGNAL"
            
            if reason:
                proceeds = pos.qty * price * (1 - cfg.transaction_cost_pct)
                cash += proceeds
                trades.append(Trade(day, ticker, "SELL", price, pos.qty, cash, reason))
                del positions[ticker]

        # --- Blocco di RIBILANCIAMENTO (e vendita per eccesso di peso) ---
        rebalance_needed = (day - last_rebalance_day).days >= cfg.rebalance_frequency
        if rebalance_needed:
            portfolio_val = sum(pos.qty * price_map[ticker].get(day, 0) for ticker, pos in positions.items()) + cash
            target_pos_val = portfolio_val * cfg.position_size_pct
            
            for ticker, pos in list(positions.items()):
                price = price_map[ticker].get(day)
                if pd.isna(price): continue
                
                current_val = pos.qty * price
                if current_val > target_pos_val * 1.2: # Tolleranza del 20%
                    excess_qty = (current_val - target_pos_val) / price
                    proceeds = excess_qty * price * (1 - cfg.transaction_cost_pct)
                    cash += proceeds
                    pos.qty -= excess_qty
                    trades.append(Trade(day, ticker, "SELL", price, excess_qty, cash, "REBALANCE"))

        # --- Blocco di ACQUISTO ---
        if len(positions) < cfg.max_positions:
            portfolio_val = sum(pos.qty * price_map[ticker].get(day, 0) for ticker, pos in positions.items()) + cash
            
            buy_candidates = []
            for ticker, df in signal_map.items():
                if ticker in positions: continue
                if bool(df["entry_cond"].get(day, False)):
                    price = price_map[ticker].get(day)
                    atr = df["atr"].get(day)
                    if pd.notna(price) and pd.notna(atr) and atr > 0:
                        buy_candidates.append({'ticker': ticker, 'price': price, 'atr': atr, 'ps': df["ps"].get(day, 999)})
            
            if buy_candidates:
                buy_candidates.sort(key=lambda x: x['ps']) # Compra prima i P/S più bassi
                
                slots_available = cfg.max_positions - len(positions)
                for cand in buy_candidates[:slots_available]:
                    pos_size_usd = portfolio_val * cfg.position_size_pct
                    if cash < pos_size_usd: continue
                    
                    qty = pos_size_usd / cand['price']
                    cost = qty * cand['price'] * (1 + cfg.transaction_cost_pct)
                    cash -= cost
                    
                    sl = cand['price'] - cand['atr'] * cfg.stop_loss_atr_mult
                    tp = cand['price'] * (1 + cfg.take_profit_pct)
                    
                    positions[cand['ticker']] = Position(cand['ticker'], day, cand['price'], qty, sl, tp, day)
                    trades.append(Trade(day, cand['ticker'], "BUY", cand['price'], qty, cash, "SIGNAL"))
        
        if rebalance_needed:
            last_rebalance_day = day
            
        equity_curve.append({"date": day, "equity": portfolio_val, "cash": cash, "positions": len(positions)})

    return pd.DataFrame(equity_curve).set_index("date"), trades

# ========================= Benchmark Function (AGGIUNTA) =========================

def buy_hold_benchmark(start_date: str, end_date: str, initial_capital: float) -> pd.DataFrame:
    """Crea un benchmark buy-and-hold di SPY."""
    spy_data = fetch_ohlcv_yf("SPY", start_date, end_date)
    if spy_data.empty:
        return pd.DataFrame()
    
    spy_price = spy_data['Close'].fillna(method='ffill')
    initial_price = spy_price.iloc[0]
    shares = initial_capital / initial_price
    
    equity_values = spy_price * shares
    
    return pd.DataFrame({
        'equity': equity_values,
        'cash': 0,
        'positions': 1
    }, index=spy_price.index)

# ========================= Analysis & Metrics (con SHARPE RATIO) =========================

def calculate_metrics(equity: pd.DataFrame, benchmark_equity: pd.DataFrame, trades: List[Trade]) -> Dict:
    """Calcola le metriche di performance, incluso lo Sharpe Ratio."""
    eq = equity["equity"].dropna()
    bench = benchmark_equity["equity"].dropna()
    if len(eq) < 2: return {}
    
    common_idx = eq.index.intersection(bench.index)
    eq, bench = eq.loc[common_idx], bench.loc[common_idx]
    
    # Calcolo rendimenti giornalieri
    returns = eq.pct_change().fillna(0.0)
    
    # Calcolo metriche
    days = (eq.index[-1] - eq.index[0]).days
    if days == 0: return {}

    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (365.25 / days) - 1
    
    benchmark_total_return = (bench.iloc[-1] / bench.iloc[0]) - 1
    benchmark_annual_return = (1 + benchmark_total_return) ** (365.25 / days) - 1
    
    # Volatilità annualizzata
    volatility = returns.std() * np.sqrt(252)
    
    # **** INTEGRAZIONE SHARPE RATIO ****
    # Si assume un risk-free rate dello 0% per semplicità
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    max_drawdown = (eq / eq.cummax() - 1).min()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'benchmark_annual_return': benchmark_annual_return,
        'alpha': annualized_return - benchmark_annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio, # Aggiunto qui
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
    }

def print_detailed_summary(metrics: Dict, equity: pd.DataFrame):
    print(f"\n{'='*60}\nRIASSUNTO DELLA STRATEGIA\n{'='*60}")
    print(f"PERFORMANCE:")
    print(f"  Rendimento Annualizzato:         {metrics.get('annualized_return', 0)*100:.2f}%")
    print(f"  Benchmark SPY (Ann.):            {metrics.get('benchmark_annual_return', 0)*100:.2f}%")
    print(f"  Alpha vs SPY:                    {metrics.get('alpha', 0)*100:.2f}%")
    
    print(f"\nRISCHIO:")
    print(f"  Volatilità Annualizzata:         {metrics.get('volatility', 0)*100:.2f}%")
    print(f"  Max Drawdown:                    {metrics.get('max_drawdown', 0)*100:.2f}%")
    
    print(f"\nMETRICHE CHIAVE:")
    print(f"  Sharpe Ratio:                    {metrics.get('sharpe_ratio', 0):.2f}") # Aggiunto qui
    
    print(f"\nSTATISTICHE TRADING:")
    print(f"  Numero totale di trade:          {metrics.get('num_trades', 0)}")
    print(f"  Numero medio di posizioni:       {equity['positions'].mean():.1f}")
    
    print(f"{'='*60}")

def plot_detailed_results(equity: pd.DataFrame, benchmark: pd.DataFrame, trades: List[Trade], metrics: Dict):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, :])
    eq, ben = equity["equity"], benchmark["equity"]
    ax1.plot(eq, 'b-', label=f'Strategia P/S ({metrics["annualized_return"]*100:.1f}%)')
    ax1.plot(ben, 'r--', label=f'SPY Buy & Hold ({metrics["benchmark_annual_return"]*100:.1f}%)')
    ax1.set_title('Equity Curve (Scala Logaritmica)', fontsize=14)
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')
    
    ax2 = fig.add_subplot(gs[1, 0])
    drawdown = (eq / eq.cummax() - 1) * 100
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown Storico (%)'); ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(equity.index, equity['positions'], color='purple')
    ax3.set_title('Numero di Posizioni Aperte'); ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, :])
    reasons = pd.Series([t.reason for t in trades]).value_counts().sort_values()
    reasons.plot(kind='barh', ax=ax4, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax4.set_title('Conteggio Trade per Motivo di Chiusura/Apertura'); ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.show()
    
# ========================= Main Execution ===================

def run_strategy(cfg: BacktestConfig):
    """Funzione principale che orchestra l'intero processo."""
    start_date, end_date = f"{cfg.start_year}-01-01", today_str()
    universe = get_universe()
    print(f"Universo: {len(universe)} titoli, dal {start_date} al {end_date}")
    
    print("Fase 1: Download dati OHLCV...")
    all_ohlcv_data = {
        sym: fetch_ohlcv_yf(sym, start_date, end_date)
        for sym in tqdm(universe, desc="Download OHLCV")
    }
    all_ohlcv_data = {k: v for k, v in all_ohlcv_data.items() if len(v) > 200}
    
    print(f"Dati validi per {len(all_ohlcv_data)} titoli. Fase 2: Generazione segnali...")
    
    signal_map = {}
    for ticker, ohlcv in tqdm(all_ohlcv_data.items(), desc="Generazione Segnali"):
        price = ohlcv['Close']
        revenue, shares = fetch_yf_fundamentals_robust(ticker)
        if revenue.empty or shares.empty:
            if VERBOSE: print(f"DEBUG: Scartato {ticker} - Dati fondamentali mancanti.")
            continue
            
        ps = compute_ps_from_yf_data(price, revenue, shares)
        if len(ps.dropna()) < cfg.min_periods:
            if VERBOSE: print(f"DEBUG: Scartato {ticker} - Serie P/S con dati insufficienti.")
            continue
        
        atr = calculate_atr(ohlcv, cfg.atr_period)
        signals = compute_signals_from_ps(price, ps, atr, cfg)
        
        if signals["ps_ma"].dropna().empty:
            if VERBOSE: print(f"DEBUG: Scartato {ticker} - Impossibile calcolare la media mobile del P/S.")
            continue
            
        signal_map[ticker] = signals
    
    if len(signal_map) < 5:
        print(f"\nERRORE: Generati segnali solo per {len(signal_map)} titoli. Impossibile continuare.")
        return
    
    print(f"Segnali generati per {len(signal_map)} titoli. Fase 3: Esecuzione backtest...")
    
    equity, trades = optimized_portfolio_backtest(signal_map, cfg)
    if equity.empty:
        print("Il backtest non ha prodotto risultati.")
        return
        
    benchmark = buy_hold_benchmark(equity.index[0].strftime("%Y-%m-%d"), equity.index[-1].strftime("%Y-%m-%d"), cfg.initial_capital)
    
    metrics = calculate_metrics(equity, benchmark, trades)
    
    print_detailed_summary(metrics, equity)
    plot_detailed_results(equity, benchmark, trades, metrics)
    
    return equity, trades, metrics

if __name__ == "__main__":
    config = BacktestConfig()
    run_strategy(config)
    print("\nProcesso completato.")