#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  PAIRS TRADING V2 — Enhanced Cointegration Strategy
  Coppia: KO (Coca-Cola) vs PEP (PepsiCo)
  Dati: Orari (1h) — ultimi 2 anni
================================================================================

  MIGLIORAMENTI RISPETTO A V1:

  1. KALMAN FILTER Hedge Ratio   — Sostituisce la OLS rolling con un filtro
     di Kalman online, più reattivo e stabile (niente warm-up parziale).

  2. HALF-LIFE DEL MEAN REVERSION — Calibra automaticamente la finestra dello
     Z-Score usando l'half-life dell'Ornstein-Uhlenbeck, garantendo una
     finestra statisticamente ottimale.

  3. STOP-LOSS DINAMICO          — Chiude forzatamente le posizioni quando
     Z-Score > STOPLOSS_ZSCORE (divergenza, breakdown della cointegrazione).

  4. ROLLING COINTEGRATION CHECK — Verifica la cointegrazione su finestra
     mobile: si opera SOLO quando p-value < soglia negli ultimi N periodi.

  5. REGIME FILTER (Volatilita') — Sospende il trading quando la volatilita'
     dello spread e' anomalmente alta (regime di stress).

  6. TRANSACTION COSTS           — Modella slippage + commissioni realistiche.

  7. PARAMETER GRID SEARCH       — Testa combinazioni di parametri e mostra
     la heatmap Sharpe Ratio.

  Autore: Quant Research Desk
  Data  : Aprile 2026
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS & CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from datetime import datetime, timedelta
from itertools import product

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (18, 10),
    "font.family": "monospace",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "--",
    "lines.linewidth": 1.2,
    "figure.dpi": 120,
})

# ── Parametri della Strategia V2 ─────────────────────────────────────────────
TICKER_A         = "KO"
TICKER_B         = "PEP"
INTERVAL         = "1h"
LOOKBACK_YEARS   = 2
HOURS_PER_YEAR   = 252 * 6.5

# Kalman Filter parameters
KALMAN_DELTA     = 1e-4     # Rumore di transizione del Kalman Filter
KALMAN_VE        = 1e-3     # Varianza dell'errore di osservazione iniziale

# Z-Score & Segnali (verranno ottimizzati dal grid search)
ENTRY_ZSCORE     = 1.5      # Soglia di entrata (abbassata da 2.0 per piu' segnali)
EXIT_ZSCORE      = 0.5      # Exit NON a zero ma a 0.5 (take profit parziale)
STOPLOSS_ZSCORE  = 4.0      # Stop-loss: chiude se divergenza > 4 sigma

# Rolling cointegration
COINT_WINDOW     = 250      # Finestra per il test di cointegrazione rolling
COINT_PVALUE     = 0.10     # P-value per validare (10% — piu' permissivo per rolling)

# Regime filter
VOL_WINDOW       = 60       # Finestra per volatilita' dello spread
VOL_QUANTILE     = 0.85     # Sopra l'85° percentile = regime di stress

# Transaction costs (round-trip: entry + exit)
TC_BPS           = 5        # Costo totale in basis points (0.05% round-trip)

print("=" * 80)
print("  PAIRS TRADING V2 — Enhanced Cointegration Strategy")
print(f"  Coppia: {TICKER_A} / {TICKER_B}  |  Intervallo: {INTERVAL}")
print(f"  Entry: +/-{ENTRY_ZSCORE}s  |  Exit: {EXIT_ZSCORE}s  |  StopLoss: {STOPLOSS_ZSCORE}s")
print(f"  Kalman Filter HR  |  Half-Life Z-Window  |  Rolling Coint Check")
print(f"  Transaction Costs: {TC_BPS} bps round-trip")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/9] Download dati orari da Yahoo Finance...")

end_date   = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

data_a = yf.download(TICKER_A, start=start_date, end=end_date,
                     interval=INTERVAL, progress=False, auto_adjust=True)
data_b = yf.download(TICKER_B, start=start_date, end=end_date,
                     interval=INTERVAL, progress=False, auto_adjust=True)

prices = pd.DataFrame({
    TICKER_A: data_a["Close"].squeeze(),
    TICKER_B: data_b["Close"].squeeze(),
}).dropna()

print(f"  Barre orarie allineate: {len(prices):,}")
print(f"  Periodo: {prices.index[0].strftime('%Y-%m-%d %H:%M')} -> "
      f"{prices.index[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"  {TICKER_A}: ${prices[TICKER_A].iloc[-1]:.2f}  |  "
      f"{TICKER_B}: ${prices[TICKER_B].iloc[-1]:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEST DI COINTEGRAZIONE STATICO (Full-Sample)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/9] Test di cointegrazione Engle-Granger (full sample)...")

coint_stat, p_value, crit_values = coint(prices[TICKER_A], prices[TICKER_B])

print(f"  Statistica del test : {coint_stat:.4f}")
print(f"  P-value             : {p_value:.6f}")
print(f"  Valori critici      : 1%={crit_values[0]:.4f}  "
      f"5%={crit_values[1]:.4f}  10%={crit_values[2]:.4f}")

if p_value < 0.05:
    print(f"  >> COINTEGRAZIONE CONFERMATA al 5% (p={p_value:.6f})")
elif p_value < 0.10:
    print(f"  >> Cointegrazione marginale al 10% (p={p_value:.6f})")
else:
    print(f"  >> Cointegrazione NON confermata (p={p_value:.6f})")
    print(f"     Attenzione: il rolling check potrebbe trovare finestre valide.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. KALMAN FILTER HEDGE RATIO
# ══════════════════════════════════════════════════════════════════════════════
#
#  Il Kalman Filter stima online il coefficiente β (hedge ratio) senza
#  bisogno di una finestra fissa. Vantaggi rispetto a OLS rolling:
#
#    - Adattamento continuo senza "salti" quando entra/esce un dato
#    - Nessun parametro di finestra da ottimizzare
#    - Incorpora naturalmente l'incertezza sulla stima (P matrix)
#
#  Modello di stato:
#    θ_t = θ_{t-1} + w_t   (random walk su [α, β])
#    y_t = X_t · θ_t + v_t  (osservazione: prezzo_A)
#
#  dove θ = [intercetta, hedge_ratio]
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n[3/9] Kalman Filter per Hedge Ratio dinamico...")

n = len(prices)
price_a = prices[TICKER_A].values
price_b = prices[TICKER_B].values

# Stato: θ = [intercetta, β]
theta = np.zeros((n, 2))  # [alpha, beta] nel tempo
P = np.zeros((n, 2, 2))   # Matrice di covarianza
e = np.zeros(n)            # Errore di previsione (innovation)
Q_var = np.zeros(n)        # Varianza dell'innovazione

# Inizializzazione
theta[0] = [0.0, 0.0]
P[0] = np.eye(2) * 1.0     # Incertezza iniziale alta
R = KALMAN_VE               # Varianza dell'osservazione
Q = np.eye(2) * KALMAN_DELTA  # Matrice di covarianza del rumore di processo

for t in range(1, n):
    # ── Prediction step ──
    theta_pred = theta[t-1]       # Random walk: θ_t|t-1 = θ_t-1
    P_pred = P[t-1] + Q           # P_t|t-1 = P_t-1 + Q

    # ── Observation ──
    x_t = np.array([1.0, price_b[t]])  # [1, prezzo_B]
    y_t = price_a[t]

    # ── Innovation ──
    y_pred = x_t @ theta_pred
    e[t] = y_t - y_pred
    S = x_t @ P_pred @ x_t + R    # Varianza dell'innovazione (scalare)
    Q_var[t] = S

    # ── Kalman Gain ──
    K = P_pred @ x_t / S

    # ── Update step ──
    theta[t] = theta_pred + K * e[t]
    P[t] = P_pred - np.outer(K, x_t) @ P_pred

# Estrai hedge ratio dal Kalman Filter
hedge_ratio = pd.Series(theta[:, 1], index=prices.index, name="HedgeRatio")
kalman_intercept = pd.Series(theta[:, 0], index=prices.index, name="Intercept")

# Warm-up: saltiamo le prime 60 barre per la convergenza del filtro
WARMUP = 60
prices      = prices.iloc[WARMUP:]
hedge_ratio = hedge_ratio.iloc[WARMUP:]
kalman_intercept = kalman_intercept.iloc[WARMUP:]

print(f"  Hedge Ratio medio     : {hedge_ratio.mean():.4f}")
print(f"  Hedge Ratio std       : {hedge_ratio.std():.4f}")
print(f"  Range                 : [{hedge_ratio.min():.4f}, {hedge_ratio.max():.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HALF-LIFE DEL MEAN REVERSION (Ornstein-Uhlenbeck)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Lo spread segue (idealmente) un processo di Ornstein-Uhlenbeck:
#    dS = κ(μ - S)dt + σdW
#
#  L'half-life è il tempo impiegato dallo spread per ridurre del 50%
#  la deviazione dalla media:
#    half_life = -ln(2) / κ
#
#  Usiamo l'half-life per calibrare la FINESTRA dello Z-Score.
#  Una finestra pari all'half-life cattura esattamente il regime di
#  mean-reversion dello spread.
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n[4/9] Calcolo Half-Life dello spread (Ornstein-Uhlenbeck)...")

spread_raw = prices[TICKER_A] - hedge_ratio * prices[TICKER_B]

# Regressione: ΔS_t = κ·S_{t-1} + c + ε
spread_lag = spread_raw.shift(1).dropna()
spread_diff = spread_raw.diff().dropna()

# Allinea indici
common_idx = spread_lag.index.intersection(spread_diff.index)
spread_lag = spread_lag.loc[common_idx]
spread_diff = spread_diff.loc[common_idx]

X_hl = add_constant(spread_lag.values)
y_hl = spread_diff.values
model_hl = OLS(y_hl, X_hl).fit()

kappa = model_hl.params[1]  # Coefficiente di mean-reversion

if kappa < 0:
    half_life = int(round(-np.log(2) / kappa))
    half_life = max(20, min(half_life, 200))  # Clamp tra 20 e 200
    print(f"  Kappa (velocita' MR) : {kappa:.6f}")
    print(f"  Half-Life calcolato  : {half_life} periodi (ore)")
    print(f"  >> La finestra Z-Score sara' calibrata a {half_life} periodi")
else:
    half_life = 60  # Fallback
    print(f"  ATTENZIONE: kappa positivo ({kappa:.6f}) — lo spread NON mean-reverts!")
    print(f"  >> Uso finestra di default: {half_life} periodi")

ZSCORE_WINDOW = half_life


# ══════════════════════════════════════════════════════════════════════════════
# 5. SPREAD, Z-SCORE e FILTRI DI REGIME
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/9] Calcolo Spread, Z-Score e filtri di regime...")

# Spread dinamico (con Kalman hedge ratio)
spread = prices[TICKER_A] - hedge_ratio * prices[TICKER_B]

# Media mobile e deviazione standard dello spread
spread_mean = spread.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
spread_std  = spread.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std()

# Z-Score
zscore = (spread - spread_mean) / spread_std

# ── Filtro di Volatilita' (Regime Filter) ────────────────────────────────────
# Calcoliamo la volatilita' rolling dello spread. Quando e' troppo alta
# (sopra il quantile VOL_QUANTILE), il regime e' di stress e non operiamo.
spread_vol = spread.rolling(window=VOL_WINDOW).std()
vol_threshold = spread_vol.quantile(VOL_QUANTILE)
regime_ok = (spread_vol <= vol_threshold).astype(float)

# ── Rolling Cointegration Check ──────────────────────────────────────────────
# Verifichiamo la cointegrazione su finestra mobile: operiamo solo quando
# la coppia e' cointegrata RECENTEMENTE (non solo storicamente).
rolling_coint_pval = pd.Series(np.nan, index=prices.index)

for i in range(COINT_WINDOW, len(prices)):
    window_a = prices[TICKER_A].iloc[i - COINT_WINDOW:i]
    window_b = prices[TICKER_B].iloc[i - COINT_WINDOW:i]
    try:
        _, pval, _ = coint(window_a, window_b)
        rolling_coint_pval.iloc[i] = pval
    except:
        rolling_coint_pval.iloc[i] = 1.0

coint_ok = (rolling_coint_pval <= COINT_PVALUE).astype(float)

# ── Rimuoviamo warm-up ───────────────────────────────────────────────────────
valid_idx = zscore.dropna().index
# Intersezione con rolling coint validity
valid_coint = rolling_coint_pval.dropna().index
valid_idx = valid_idx.intersection(valid_coint)

prices          = prices.loc[valid_idx]
hedge_ratio     = hedge_ratio.loc[valid_idx]
spread          = spread.loc[valid_idx]
zscore          = zscore.loc[valid_idx]
regime_ok       = regime_ok.loc[valid_idx]
coint_ok        = coint_ok.loc[valid_idx]
rolling_coint_pval = rolling_coint_pval.loc[valid_idx]

# Filtro combinato: operiamo SOLO quando regime OK E cointegrazione OK
can_trade = (regime_ok * coint_ok).astype(float)

pct_tradeable = can_trade.mean() * 100
pct_coint     = coint_ok.mean() * 100
pct_regime    = regime_ok.mean() * 100

print(f"  Barre valide         : {len(zscore):,}")
print(f"  Z-Score Window       : {ZSCORE_WINDOW} (calibrato da half-life)")
print(f"  Z-Score range        : [{zscore.min():.2f}, {zscore.max():.2f}]")
print(f"  Barre cointegrate    : {pct_coint:.1f}%")
print(f"  Barre regime OK      : {pct_regime:.1f}%")
print(f"  Barre TRADEABLE      : {pct_tradeable:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 6. GENERAZIONE DEI SEGNALI DI TRADING V2
# ══════════════════════════════════════════════════════════════════════════════
#
#  Logica migliorata:
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  CONDIZIONI DI ENTRATA (solo se can_trade == 1):                        │
#   │    Z > +ENTRY   →  SHORT SPREAD  (Vendi A, Compra beta*B)              │
#   │    Z < -ENTRY   →  LONG SPREAD   (Compra A, Vendi beta*B)              │
#   │                                                                          │
#   │  CONDIZIONI DI USCITA:                                                   │
#   │    |Z| < EXIT   →  TAKE PROFIT (mean reversion raggiunta)               │
#   │    |Z| > STOP   →  STOP LOSS   (divergenza, cointegrazione rotta)       │
#   │    can_trade==0  →  REGIME EXIT (volatilita' o cointegrazione persa)    │
#   └──────────────────────────────────────────────────────────────────────────┘
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n[6/9] Generazione segnali di trading V2...")

position = pd.Series(0.0, index=prices.index)
trade_type = pd.Series("", index=prices.index)  # Per tracking
current_pos = 0.0

for i in range(1, len(zscore)):
    z = zscore.iloc[i]
    tradeable = can_trade.iloc[i]

    if current_pos == 0:
        # ── Nessuna posizione: cerchiamo entrata ──
        if tradeable == 1.0:
            if z > ENTRY_ZSCORE:
                current_pos = -1.0
                trade_type.iloc[i] = "SHORT_ENTRY"
            elif z < -ENTRY_ZSCORE:
                current_pos = 1.0
                trade_type.iloc[i] = "LONG_ENTRY"
    else:
        # ── Posizione aperta: cerchiamo uscita ──

        # 1) Take Profit: Z-Score rientrato nella banda di uscita
        if current_pos == 1.0 and z >= -EXIT_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "TP_LONG"
        elif current_pos == -1.0 and z <= EXIT_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "TP_SHORT"

        # 2) Stop Loss: Z-Score diverge troppo (cointegrazione rotta)
        elif current_pos == 1.0 and z < -STOPLOSS_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "SL_LONG"
        elif current_pos == -1.0 and z > STOPLOSS_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "SL_SHORT"

        # 3) Regime Exit: condizioni di mercato non favorevoli
        elif tradeable == 0.0:
            current_pos = 0.0
            trade_type.iloc[i] = "REGIME_EXIT"

    position.iloc[i] = current_pos

# Statistiche
n_long  = (position == 1).sum()
n_short = (position == -1).sum()
n_flat  = (position == 0).sum()
position_diff = position.diff().fillna(0)
n_entries = (position_diff != 0).sum()

# Conta trades per tipo
n_tp = trade_type.str.startswith("TP_").sum()
n_sl = trade_type.str.startswith("SL_").sum()
n_re = (trade_type == "REGIME_EXIT").sum()

print(f"  Barre LONG           : {n_long:,}  ({100*n_long/len(position):.1f}%)")
print(f"  Barre SHORT          : {n_short:,}  ({100*n_short/len(position):.1f}%)")
print(f"  Barre FLAT           : {n_flat:,}  ({100*n_flat/len(position):.1f}%)")
print(f"  Transizioni totali   : {n_entries}")
print(f"  Take Profit exits    : {n_tp}")
print(f"  Stop Loss exits      : {n_sl}")
print(f"  Regime exits         : {n_re}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. BACKTEST VETTORIALE CON TRANSACTION COSTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[7/9] Calcolo P&L con transaction costs ({TC_BPS} bps)...")

# Rendimenti logaritmici
returns_a = np.log(prices[TICKER_A] / prices[TICKER_A].shift(1))
returns_b = np.log(prices[TICKER_B] / prices[TICKER_B].shift(1))

# Rendimento dello spread
spread_returns = returns_a - hedge_ratio * returns_b

# Rendimento della strategia (position laggata di 1)
strategy_returns_gross = position.shift(1) * spread_returns
strategy_returns_gross = strategy_returns_gross.fillna(0)

# Costi di transazione: applicati ad ogni cambio di posizione
tc_per_trade = TC_BPS / 10000  # Converti bps in decimale
position_changes = position.diff().abs().fillna(0)
transaction_costs = position_changes * tc_per_trade

# Rendimenti netti
strategy_returns = strategy_returns_gross - transaction_costs

# Rendimenti cumulati
cumulative_returns = (1 + strategy_returns).cumprod()
cumulative_gross   = (1 + strategy_returns_gross).cumprod()
cumulative_bh_a    = (1 + returns_a.fillna(0)).cumprod()
cumulative_bh_b    = (1 + returns_b.fillna(0)).cumprod()

print(f"  Rend. cumulato NETTO   : {(cumulative_returns.iloc[-1] - 1)*100:+.2f}%")
print(f"  Rend. cumulato LORDO   : {(cumulative_gross.iloc[-1] - 1)*100:+.2f}%")
print(f"  Costi totali           : {transaction_costs.sum()*100:.3f}%")
print(f"  Rend. {TICKER_A} (B&H)        : {(cumulative_bh_a.iloc[-1] - 1)*100:+.2f}%")
print(f"  Rend. {TICKER_B} (B&H)       : {(cumulative_bh_b.iloc[-1] - 1)*100:+.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 8. METRICHE DI PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[8/9] Calcolo metriche di performance...")

# Sharpe Ratio Annualizzato
mean_return = strategy_returns.mean()
std_return  = strategy_returns.std()
sharpe_ratio = (mean_return / std_return) * np.sqrt(HOURS_PER_YEAR) if std_return > 0 else 0.0

# Maximum Drawdown
cum_max   = cumulative_returns.cummax()
drawdown  = (cumulative_returns - cum_max) / cum_max
max_drawdown = drawdown.min()

# Win Rate (per-bar)
winning_bars = (strategy_returns > 0).sum()
losing_bars  = (strategy_returns < 0).sum()
total_active = winning_bars + losing_bars
win_rate = winning_bars / total_active if total_active > 0 else 0.0

# Profit Factor
gross_profit = strategy_returns[strategy_returns > 0].sum()
gross_loss   = abs(strategy_returns[strategy_returns < 0].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

# Calmar Ratio
annual_return = mean_return * HOURS_PER_YEAR
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

# Sortino Ratio
downside_returns = strategy_returns[strategy_returns < 0]
downside_std = downside_returns.std() if len(downside_returns) > 0 else 1.0
sortino_ratio = (mean_return / downside_std) * np.sqrt(HOURS_PER_YEAR) if downside_std > 0 else 0.0

# Per-Trade Analysis
trades = []
entry_idx = None
entry_pos = 0
for i in range(1, len(position)):
    if position.iloc[i] != 0 and position.iloc[i-1] == 0:
        entry_idx = i
        entry_pos = position.iloc[i]
    elif position.iloc[i] == 0 and position.iloc[i-1] != 0 and entry_idx is not None:
        trade_ret = strategy_returns.iloc[entry_idx:i+1].sum()
        trades.append(trade_ret)
        entry_idx = None

trades = pd.Series(trades) if trades else pd.Series([0.0])
avg_win  = trades[trades > 0].mean() if (trades > 0).any() else 0
avg_loss = trades[trades < 0].mean() if (trades < 0).any() else 0
trade_win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0

print("\n" + "=" * 80)
print("  RIEPILOGO PERFORMANCE V2")
print("=" * 80)
print(f"  | Rendimento Totale (netto)  : {(cumulative_returns.iloc[-1]-1)*100:>+10.2f} %")
print(f"  | Rendimento Annualizzato    : {annual_return*100:>+10.2f} %")
print(f"  | Sharpe Ratio (ann.)        : {sharpe_ratio:>10.3f}")
print(f"  | Sortino Ratio (ann.)       : {sortino_ratio:>10.3f}")
print(f"  | Calmar Ratio               : {calmar_ratio:>10.3f}")
print(f"  | Maximum Drawdown           : {max_drawdown*100:>10.2f} %")
print(f"  | Win Rate (per-bar)         : {win_rate*100:>10.1f} %")
print(f"  | Win Rate (per-trade)       : {trade_win_rate*100:>10.1f} %")
print(f"  | Profit Factor              : {profit_factor:>10.3f}")
print(f"  | Avg Winning Trade          : {avg_win*100:>+10.3f} %")
print(f"  | Avg Losing Trade           : {avg_loss*100:>+10.3f} %")
print(f"  | Payoff Ratio (W/L)         : {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.3f}")
print(f"  | Numero Trades              : {len(trades):>10}")
print(f"  | Take Profits               : {n_tp:>10}")
print(f"  | Stop Losses                : {n_sl:>10}")
print(f"  | Regime Exits               : {n_re:>10}")
print(f"  | Transaction Costs Totali   : {transaction_costs.sum()*100:>10.3f} %")
print(f"  | Barre Totali               : {len(prices):>10,}")
print(f"  | Half-Life (Z-Window)       : {ZSCORE_WINDOW:>10} ore")
print(f"  | % Tempo Tradeable          : {pct_tradeable:>10.1f} %")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# 9. VISUALIZZAZIONE COMPLETA
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[9/9] Generazione grafici...")

fig, axes = plt.subplots(5, 1, figsize=(22, 22),
                         gridspec_kw={"height_ratios": [2, 1.5, 3, 2, 1.5]})
fig.suptitle(
    f"PAIRS TRADING V2  |  {TICKER_A} / {TICKER_B}  |  "
    f"Sharpe: {sharpe_ratio:.2f}  |  MaxDD: {max_drawdown*100:.1f}%  |  "
    f"Return: {(cumulative_returns.iloc[-1]-1)*100:+.1f}%",
    fontsize=16, fontweight="bold", color="#E0E0E0", y=0.98
)

# ── Pannello 1: Prezzi ──
ax1 = axes[0]
ax1.plot(prices.index, prices[TICKER_A], color="#00D4AA", alpha=0.9,
         label=f"{TICKER_A}", linewidth=1.0)
ax1_twin = ax1.twinx()
ax1_twin.plot(prices.index, prices[TICKER_B], color="#FF6B6B", alpha=0.9,
              label=f"{TICKER_B}", linewidth=1.0)
ax1.set_ylabel(f"{TICKER_A} ($)", color="#00D4AA")
ax1_twin.set_ylabel(f"{TICKER_B} ($)", color="#FF6B6B")
ax1.set_title("Prezzi delle Azioni", fontweight="bold", color="#AAAAAA")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
           framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 2: Kalman Hedge Ratio ──
ax2 = axes[1]
ax2.plot(hedge_ratio.index, hedge_ratio, color="#FFA726", alpha=0.85,
         linewidth=1.0, label="Kalman Hedge Ratio")
ax2.axhline(y=hedge_ratio.mean(), color="#FFA726", linestyle="--", alpha=0.4,
            label=f"Media: {hedge_ratio.mean():.4f}")
ax2.fill_between(hedge_ratio.index,
                 hedge_ratio.mean() - hedge_ratio.std(),
                 hedge_ratio.mean() + hedge_ratio.std(),
                 alpha=0.08, color="#FFA726")
ax2.set_ylabel("Hedge Ratio")
ax2.set_title("Kalman Filter Hedge Ratio (Online)", fontweight="bold", color="#AAAAAA")
ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 3: Z-Score con segnali ──
ax3 = axes[2]

# Colora sfondo per posizioni
for i in range(1, len(zscore)):
    if position.iloc[i] == 1:
        ax3.axvspan(zscore.index[i-1], zscore.index[i],
                    alpha=0.08, color="#00D4AA", linewidth=0)
    elif position.iloc[i] == -1:
        ax3.axvspan(zscore.index[i-1], zscore.index[i],
                    alpha=0.08, color="#FF6B6B", linewidth=0)

# Z-Score
ax3.plot(zscore.index, zscore, color="#64B5F6", alpha=0.9, linewidth=0.8,
         label="Z-Score")

# Livelli
ax3.axhline(y=ENTRY_ZSCORE, color="#FF5252", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Short Entry (+{ENTRY_ZSCORE})")
ax3.axhline(y=-ENTRY_ZSCORE, color="#69F0AE", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Long Entry (-{ENTRY_ZSCORE})")
ax3.axhline(y=EXIT_ZSCORE, color="#FFD740", linestyle=":", linewidth=1.0,
            alpha=0.5, label=f"Exit (+{EXIT_ZSCORE})")
ax3.axhline(y=-EXIT_ZSCORE, color="#FFD740", linestyle=":", linewidth=1.0,
            alpha=0.5, label=f"Exit (-{EXIT_ZSCORE})")
ax3.axhline(y=STOPLOSS_ZSCORE, color="#FF1744", linestyle="-.", linewidth=1.0,
            alpha=0.5, label=f"StopLoss (+/-{STOPLOSS_ZSCORE})")
ax3.axhline(y=-STOPLOSS_ZSCORE, color="#FF1744", linestyle="-.", linewidth=1.0,
            alpha=0.5)

# Zone
ax3.fill_between(zscore.index, ENTRY_ZSCORE, STOPLOSS_ZSCORE,
                 alpha=0.03, color="#FF5252")
ax3.fill_between(zscore.index, -ENTRY_ZSCORE, -STOPLOSS_ZSCORE,
                 alpha=0.03, color="#69F0AE")

# Non-tradeable zones (regime filter)
non_trade_mask = (can_trade == 0)
if non_trade_mask.any():
    for i in range(1, len(can_trade)):
        if can_trade.iloc[i] == 0:
            ax3.axvspan(can_trade.index[i-1], can_trade.index[i],
                        alpha=0.06, color="#9E9E9E", linewidth=0)

ax3.set_ylabel("Z-Score")
ax3.set_title("Z-Score con Segnali, Stop-Loss e Regime Filter",
              fontweight="bold", color="#AAAAAA")
ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e", ncol=3,
           fontsize=9)
ax3.set_ylim(max(zscore.min() - 0.5, -6), min(zscore.max() + 0.5, 6))

# ── Pannello 4: Equity Curve ──
ax4 = axes[3]
ax4.plot(cumulative_returns.index, cumulative_returns, color="#00E5FF",
         linewidth=1.5, alpha=0.95, label="V2 Strategy (Net)")
ax4.plot(cumulative_gross.index, cumulative_gross, color="#00E5FF",
         linewidth=0.8, alpha=0.3, linestyle="--", label="V2 Strategy (Gross)")
ax4.fill_between(cumulative_returns.index, 1, cumulative_returns,
                 where=(cumulative_returns >= 1), alpha=0.1, color="#00E5FF")
ax4.fill_between(cumulative_returns.index, 1, cumulative_returns,
                 where=(cumulative_returns < 1), alpha=0.1, color="#FF5252")
ax4.axhline(y=1.0, color="white", linestyle=":", alpha=0.2)
ax4.set_ylabel("Cumulative ($1)")
ax4.set_title("Equity Curve", fontweight="bold", color="#AAAAAA")
ax4.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 5: Rolling Cointegration P-Value ──
ax5 = axes[4]
ax5.plot(rolling_coint_pval.index, rolling_coint_pval, color="#AB47BC",
         alpha=0.7, linewidth=0.8, label="Rolling Coint P-Value")
ax5.axhline(y=COINT_PVALUE, color="#FF5252", linestyle="--", linewidth=1.2,
            alpha=0.8, label=f"Soglia ({COINT_PVALUE})")
ax5.fill_between(rolling_coint_pval.index, 0, rolling_coint_pval,
                 where=(rolling_coint_pval <= COINT_PVALUE),
                 alpha=0.15, color="#69F0AE", label="Cointegrata")
ax5.fill_between(rolling_coint_pval.index, rolling_coint_pval, 1,
                 where=(rolling_coint_pval > COINT_PVALUE),
                 alpha=0.05, color="#FF5252")
ax5.set_ylabel("P-Value")
ax5.set_xlabel("Data")
ax5.set_title("Rolling Cointegration Test (Validita' Coppia nel Tempo)",
              fontweight="bold", color="#AAAAAA")
ax5.legend(loc="upper right", framealpha=0.3, facecolor="#1a1a2e")
ax5.set_ylim(-0.02, 1.02)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("pairs_trading_v2.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Grafico principale salvato: pairs_trading_v2.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax_dd = plt.subplots(figsize=(20, 4))
ax_dd.fill_between(drawdown.index, drawdown * 100, 0,
                   alpha=0.5, color="#FF5252", label="Drawdown")
ax_dd.plot(drawdown.index, drawdown * 100, color="#FF8A80", linewidth=0.8)
ax_dd.axhline(y=max_drawdown * 100, color="#FF1744", linestyle="--",
              alpha=0.6, label=f"Max DD: {max_drawdown*100:.2f}%")
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_xlabel("Data")
ax_dd.set_title("Underwater Equity (Drawdown)", fontweight="bold", color="#AAAAAA")
ax_dd.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax_dd.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("pairs_trading_v2_drawdown.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 11. PARAMETER GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════
#
#  Testiamo combinazioni di ENTRY e EXIT per trovare i parametri ottimali.
#  ATTENZIONE: questo e' un in-sample optimization, quindi i risultati
#  migliori sono soggetti a overfitting. Usare con cautela.
# ──────────────────────────────────────────────────────────────────────────────

print("\n[BONUS] Parameter Grid Search...")

entry_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
exit_range  = [0.0, 0.25, 0.5, 0.75, 1.0]

results_grid = pd.DataFrame(index=entry_range, columns=exit_range, dtype=float)

for entry_z, exit_z in product(entry_range, exit_range):
    if exit_z >= entry_z:
        results_grid.loc[entry_z, exit_z] = np.nan
        continue

    # Quick backtest con questi parametri
    pos = pd.Series(0.0, index=prices.index)
    cur = 0.0

    for i in range(1, len(zscore)):
        z = zscore.iloc[i]
        t = can_trade.iloc[i]

        if cur == 0:
            if t == 1.0:
                if z > entry_z:
                    cur = -1.0
                elif z < -entry_z:
                    cur = 1.0
        else:
            if cur == 1.0 and z >= -exit_z:
                cur = 0.0
            elif cur == -1.0 and z <= exit_z:
                cur = 0.0
            elif abs(z) > STOPLOSS_ZSCORE:
                cur = 0.0
            elif t == 0.0:
                cur = 0.0
        pos.iloc[i] = cur

    ret = pos.shift(1) * spread_returns
    ret = ret.fillna(0)
    tc = pos.diff().abs().fillna(0) * tc_per_trade
    ret_net = ret - tc

    sr = (ret_net.mean() / ret_net.std()) * np.sqrt(HOURS_PER_YEAR) if ret_net.std() > 0 else 0
    results_grid.loc[entry_z, exit_z] = sr

results_grid = results_grid.astype(float)

print("\n  Sharpe Ratio Grid (Entry x Exit):")
print(results_grid.round(3).to_string())

# Trova il miglior set di parametri
best_val = results_grid.max().max()
best_entry = results_grid.max(axis=1).idxmax()
best_exit  = results_grid.loc[best_entry].idxmax()

print(f"\n  >> MIGLIOR COMBINAZIONE: Entry={best_entry}, Exit={best_exit}")
print(f"  >> Sharpe Ratio: {best_val:.3f}")

# Heatmap
fig3, ax_hm = plt.subplots(figsize=(10, 7))

# Custom colormap: rosso -> nero -> verde
colors_cmap = ["#FF1744", "#1a1a2e", "#00E676"]
cmap = LinearSegmentedColormap.from_list("custom", colors_cmap, N=256)

# Plot con imshow
data_grid = results_grid.values.astype(float)
im = ax_hm.imshow(data_grid, cmap=cmap, aspect="auto",
                   vmin=-1.0, vmax=max(1.0, np.nanmax(data_grid)))

# Labels
ax_hm.set_xticks(range(len(exit_range)))
ax_hm.set_xticklabels([f"{x:.2f}" for x in exit_range])
ax_hm.set_yticks(range(len(entry_range)))
ax_hm.set_yticklabels([f"{x:.2f}" for x in entry_range])
ax_hm.set_xlabel("Exit Z-Score")
ax_hm.set_ylabel("Entry Z-Score")
ax_hm.set_title("Sharpe Ratio Heatmap (Entry x Exit Z-Score)",
                 fontweight="bold", color="#AAAAAA")

# Annota valori
for i in range(len(entry_range)):
    for j in range(len(exit_range)):
        val = data_grid[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 0.3 else "#AAAAAA"
            ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color=color, fontsize=9, fontweight="bold")

plt.colorbar(im, label="Sharpe Ratio")
plt.tight_layout()
plt.savefig("pairs_trading_v2_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=fig3.get_facecolor())
plt.show()
print("Heatmap salvata: pairs_trading_v2_heatmap.png")

print("\n" + "=" * 80)
print("  BACKTEST V2 COMPLETATO")
print("=" * 80)
