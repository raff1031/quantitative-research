#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  PAIRS TRADING BACKTEST — Engle-Granger Cointegration Strategy
  Coppia: KO (Coca-Cola) vs PEP (PepsiCo)
  Dati: Orari (1h) — ultimi 2 anni
================================================================================

  Questo script implementa un backtest vettoriale completo di una strategia
  di Pairs Trading basata sulla cointegrazione statistica.

  Pipeline:
    1. Download dati orari via yfinance
    2. Test di cointegrazione di Engle-Granger
    3. Hedge Ratio dinamico (OLS rolling a 60 periodi)
    4. Calcolo Spread e Z-Score
    5. Generazione segnali (entry ±2σ, exit a mean-reversion)
    6. Backtest vettoriale con calcolo P&L
    7. Metriche: Sharpe Ratio annualizzato, Maximum Drawdown
    8. Visualizzazione Z-Score con livelli di entrata

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
from matplotlib.patches import FancyBboxPatch
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from datetime import datetime, timedelta

# Stile grafico professionale
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

# ── Parametri della Strategia ─────────────────────────────────────────────────
TICKER_A        = "KO"          # Leg A della coppia
TICKER_B        = "PEP"         # Leg B della coppia
INTERVAL        = "1h"          # Timeframe orario
LOOKBACK_YEARS  = 2             # Finestra storica (anni)
OLS_WINDOW      = 60            # Finestra rolling per Hedge Ratio (periodi)
ZSCORE_WINDOW   = 60            # Finestra rolling per Z-Score (periodi)
ENTRY_ZSCORE    = 2.0           # Soglia di entrata (±2 deviazioni standard)
EXIT_ZSCORE     = 0.0           # Soglia di uscita (mean reversion → zero)
COINT_PVALUE    = 0.05          # P-value massimo per validare la cointegrazione
HOURS_PER_YEAR  = 252 * 6.5     # Ore di trading annuali (252 giorni × 6.5h)

print("=" * 80)
print("  PAIRS TRADING BACKTEST — Cointegrazione Engle-Granger")
print(f"  Coppia: {TICKER_A} / {TICKER_B}  |  Intervallo: {INTERVAL}")
print(f"  Periodo: ultimi {LOOKBACK_YEARS} anni  |  OLS Window: {OLS_WINDOW}")
print(f"  Entry Z-Score: ±{ENTRY_ZSCORE}  |  Exit Z-Score: {EXIT_ZSCORE}")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
#
#  yfinance limita i dati orari a ~730 giorni. Scarichiamo il massimo
#  disponibile fissando la data di inizio a 2 anni fa.
# ──────────────────────────────────────────────────────────────────────────────

print("\n▸ [1/7] Download dati orari da Yahoo Finance...")

end_date   = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

data_a = yf.download(TICKER_A, start=start_date, end=end_date,
                     interval=INTERVAL, progress=False, auto_adjust=True)
data_b = yf.download(TICKER_B, start=start_date, end=end_date,
                     interval=INTERVAL, progress=False, auto_adjust=True)

# Allineamento temporale: manteniamo solo le barre in comune
prices = pd.DataFrame({
    TICKER_A: data_a["Close"].squeeze(),
    TICKER_B: data_b["Close"].squeeze(),
}).dropna()

print(f"  ✓ Barre orarie allineate: {len(prices):,}")
print(f"  ✓ Periodo: {prices.index[0].strftime('%Y-%m-%d %H:%M')} → "
      f"{prices.index[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"  ✓ {TICKER_A} ultimo prezzo: ${prices[TICKER_A].iloc[-1]:.2f}")
print(f"  ✓ {TICKER_B} ultimo prezzo: ${prices[TICKER_B].iloc[-1]:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEST DI COINTEGRAZIONE DI ENGLE-GRANGER
# ══════════════════════════════════════════════════════════════════════════════
#
#  Il test di Engle-Granger verifica l'ipotesi nulla H₀: "le due serie
#  NON sono cointegrate". Se il p-value risultante è inferiore alla soglia
#  scelta (tipicamente 0.05), rifiutiamo H₀ e concludiamo che esiste una
#  relazione di lungo periodo stabile tra le due serie.
#
#  Un p-value < 0.05 indica con il 95% di confidenza che lo spread tende
#  a ritornare verso la sua media (mean-reverting), condizione necessaria
#  per la strategia di Pairs Trading.
# ──────────────────────────────────────────────────────────────────────────────

print("\n▸ [2/7] Test di cointegrazione Engle-Granger...")

coint_stat, p_value, crit_values = coint(prices[TICKER_A], prices[TICKER_B])

print(f"  ✓ Statistica del test : {coint_stat:.4f}")
print(f"  ✓ P-value             : {p_value:.6f}")
print(f"  ✓ Valori critici      : 1%={crit_values[0]:.4f}  "
      f"5%={crit_values[1]:.4f}  10%={crit_values[2]:.4f}")

if p_value < COINT_PVALUE:
    print(f"  ✓ RISULTATO: Cointegrazione CONFERMATA (p={p_value:.6f} < {COINT_PVALUE})")
    print(f"    → La coppia {TICKER_A}/{TICKER_B} è statisticamente valida per il Pairs Trading.")
else:
    print(f"  ⚠ RISULTATO: Cointegrazione NON confermata (p={p_value:.6f} ≥ {COINT_PVALUE})")
    print(f"    → Attenzione: la coppia potrebbe non essere ideale. "
          f"Il backtest procede comunque a scopo didattico.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROLLING HEDGE RATIO (OLS Dinamico)
# ══════════════════════════════════════════════════════════════════════════════
#
#  L'Hedge Ratio NON è fisso: viene ricalcolato ad ogni barra usando una
#  regressione OLS su una finestra mobile di OLS_WINDOW periodi.
#
#  Modello:  Price_A = β * Price_B + α + ε
#  Il coefficiente β rappresenta l'Hedge Ratio dinamico.
#
#  Questo approccio cattura le variazioni strutturali nella relazione
#  tra le due azioni nel tempo.
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n▸ [3/7] Calcolo Rolling Hedge Ratio (finestra: {OLS_WINDOW} periodi)...")

hedge_ratio = pd.Series(index=prices.index, dtype=float)

for i in range(OLS_WINDOW, len(prices)):
    window_a = prices[TICKER_A].iloc[i - OLS_WINDOW:i]
    window_b = prices[TICKER_B].iloc[i - OLS_WINDOW:i]

    # Regressione OLS:  A = β·B + α + ε
    X = add_constant(window_b.values)
    y = window_a.values
    model = OLS(y, X).fit()

    # β è il coefficiente di B (indice 1; indice 0 è l'intercetta α)
    hedge_ratio.iloc[i] = model.params[1]

# Rimuoviamo le righe senza Hedge Ratio (warm-up period)
prices = prices.iloc[OLS_WINDOW:]
hedge_ratio = hedge_ratio.iloc[OLS_WINDOW:]

print(f"  ✓ Hedge Ratio medio : {hedge_ratio.mean():.4f}")
print(f"  ✓ Hedge Ratio std   : {hedge_ratio.std():.4f}")
print(f"  ✓ Range             : [{hedge_ratio.min():.4f}, {hedge_ratio.max():.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CALCOLO SPREAD E Z-SCORE
# ══════════════════════════════════════════════════════════════════════════════
#
#  Spread = Prezzo_A - HedgeRatio × Prezzo_B
#
#  Lo Z-Score normalizza lo spread usando media e deviazione standard
#  calcolate su una finestra mobile, permettendo di identificare le
#  deviazioni statisticamente significative dalla relazione di equilibrio.
#
#  Z-Score = (Spread - Media_Mobile_Spread) / DevStd_Mobile_Spread
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n▸ [4/7] Calcolo Spread e Z-Score (finestra: {ZSCORE_WINDOW} periodi)...")

# Spread dinamico
spread = prices[TICKER_A] - hedge_ratio * prices[TICKER_B]

# Media mobile e deviazione standard dello spread
spread_mean = spread.rolling(window=ZSCORE_WINDOW).mean()
spread_std  = spread.rolling(window=ZSCORE_WINDOW).std()

# Z-Score
zscore = (spread - spread_mean) / spread_std

# Rimuoviamo il warm-up period dello Z-Score
valid_idx = zscore.dropna().index
prices     = prices.loc[valid_idx]
hedge_ratio = hedge_ratio.loc[valid_idx]
spread     = spread.loc[valid_idx]
zscore     = zscore.loc[valid_idx]

print(f"  ✓ Barre valide per il trading: {len(zscore):,}")
print(f"  ✓ Z-Score medio : {zscore.mean():.4f}")
print(f"  ✓ Z-Score std   : {zscore.std():.4f}")
print(f"  ✓ Z-Score range : [{zscore.min():.2f}, {zscore.max():.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 5. GENERAZIONE DEI SEGNALI DI TRADING
# ══════════════════════════════════════════════════════════════════════════════
#
#  Logica dei segnali:
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  Z-Score > +2.0  →  SHORT SPREAD  (Vendi A, Compra β×B)          │
#   │  Z-Score < -2.0  →  LONG SPREAD   (Compra A, Vendi β×B)          │
#   │  Z-Score incrocia 0  →  CHIUDI POSIZIONI (Mean Reversion)         │
#   └─────────────────────────────────────────────────────────────────────┘
#
#  La posizione viene mantenuta fino al mean reversion (Z → 0).
#  Non si aprono nuove posizioni finché la precedente non è stata chiusa.
# ──────────────────────────────────────────────────────────────────────────────

print("\n▸ [5/7] Generazione segnali di trading...")

# Inizializzazione del vettore posizione
# +1 = Long Spread (long A, short B)
# -1 = Short Spread (short A, long B)
#  0 = Flat (nessuna posizione)
position = pd.Series(0.0, index=prices.index)

current_pos = 0.0

for i in range(1, len(zscore)):
    z = zscore.iloc[i]
    z_prev = zscore.iloc[i - 1]

    if current_pos == 0:
        # ── Nessuna posizione aperta: cerchiamo entrata ──
        if z > ENTRY_ZSCORE:
            # Z-Score alto → spread sopravvalutato → SHORT spread
            current_pos = -1.0
        elif z < -ENTRY_ZSCORE:
            # Z-Score basso → spread sottovalutato → LONG spread
            current_pos = 1.0
    else:
        # ── Posizione aperta: cerchiamo uscita (mean reversion) ──
        # Uscita quando Z-Score incrocia lo zero
        if current_pos == 1.0 and z >= EXIT_ZSCORE and z_prev < EXIT_ZSCORE:
            current_pos = 0.0
        elif current_pos == 1.0 and z <= EXIT_ZSCORE and z_prev > EXIT_ZSCORE:
            current_pos = 0.0
        elif current_pos == -1.0 and z >= EXIT_ZSCORE and z_prev < EXIT_ZSCORE:
            current_pos = 0.0
        elif current_pos == -1.0 and z <= EXIT_ZSCORE and z_prev > EXIT_ZSCORE:
            current_pos = 0.0

    position.iloc[i] = current_pos

# Statistiche sui segnali
n_long  = (position == 1).sum()
n_short = (position == -1).sum()
n_flat  = (position == 0).sum()

# Conta i trades (transizioni)
position_diff = position.diff().fillna(0)
n_entries = (position_diff != 0).sum()

print(f"  ✓ Barre in posizione LONG spread  : {n_long:,}  "
      f"({100*n_long/len(position):.1f}%)")
print(f"  ✓ Barre in posizione SHORT spread : {n_short:,}  "
      f"({100*n_short/len(position):.1f}%)")
print(f"  ✓ Barre FLAT (nessuna posizione)  : {n_flat:,}  "
      f"({100*n_flat/len(position):.1f}%)")
print(f"  ✓ Numero di transizioni (trades)  : {n_entries}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. BACKTEST VETTORIALE — CALCOLO P&L
# ══════════════════════════════════════════════════════════════════════════════
#
#  Il rendimento dello spread ad ogni barra è:
#    r_spread = r_A - HedgeRatio × r_B
#
#  Il rendimento della strategia è:
#    r_strategy = position[t-1] × r_spread[t]
#
#  Usiamo la posizione del periodo precedente per evitare look-ahead bias.
# ──────────────────────────────────────────────────────────────────────────────

print("\n▸ [6/7] Calcolo rendimenti e P&L della strategia...")

# Rendimenti logaritmici
returns_a = np.log(prices[TICKER_A] / prices[TICKER_A].shift(1))
returns_b = np.log(prices[TICKER_B] / prices[TICKER_B].shift(1))

# Rendimento dello spread (dollar-neutral con hedge ratio)
spread_returns = returns_a - hedge_ratio * returns_b

# Rendimento della strategia (position laggata di 1 per evitare look-ahead bias)
strategy_returns = position.shift(1) * spread_returns
strategy_returns = strategy_returns.fillna(0)

# Rendimenti cumulati
cumulative_returns = (1 + strategy_returns).cumprod()
cumulative_bh_a    = (1 + returns_a.fillna(0)).cumprod()
cumulative_bh_b    = (1 + returns_b.fillna(0)).cumprod()

print(f"  ✓ Rendimento cumulato strategia : {(cumulative_returns.iloc[-1] - 1)*100:+.2f}%")
print(f"  ✓ Rendimento cumulato {TICKER_A} (B&H)  : {(cumulative_bh_a.iloc[-1] - 1)*100:+.2f}%")
print(f"  ✓ Rendimento cumulato {TICKER_B} (B&H) : {(cumulative_bh_b.iloc[-1] - 1)*100:+.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 7. METRICHE DI PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n▸ [7/7] Calcolo metriche di performance...")

# ── Sharpe Ratio Annualizzato ────────────────────────────────────────────────
# Annualizzazione: √(ore_per_anno) dato che usiamo dati orari
mean_return = strategy_returns.mean()
std_return  = strategy_returns.std()
sharpe_ratio = (mean_return / std_return) * np.sqrt(HOURS_PER_YEAR) if std_return > 0 else 0.0

# ── Maximum Drawdown ─────────────────────────────────────────────────────────
cum_max   = cumulative_returns.cummax()
drawdown  = (cumulative_returns - cum_max) / cum_max
max_drawdown = drawdown.min()

# ── Win Rate ─────────────────────────────────────────────────────────────────
winning_bars = (strategy_returns > 0).sum()
losing_bars  = (strategy_returns < 0).sum()
total_active = winning_bars + losing_bars
win_rate = winning_bars / total_active if total_active > 0 else 0.0

# ── Profit Factor ────────────────────────────────────────────────────────────
gross_profit = strategy_returns[strategy_returns > 0].sum()
gross_loss   = abs(strategy_returns[strategy_returns < 0].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

# ── Calmar Ratio ─────────────────────────────────────────────────────────────
annual_return = mean_return * HOURS_PER_YEAR
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

print("\n" + "═" * 80)
print("  📊  RIEPILOGO PERFORMANCE")
print("═" * 80)
print(f"  │ Rendimento Totale        : {(cumulative_returns.iloc[-1]-1)*100:>+10.2f} %")
print(f"  │ Rendimento Annualizzato  : {annual_return*100:>+10.2f} %")
print(f"  │ Sharpe Ratio (ann.)      : {sharpe_ratio:>10.3f}")
print(f"  │ Calmar Ratio             : {calmar_ratio:>10.3f}")
print(f"  │ Maximum Drawdown         : {max_drawdown*100:>10.2f} %")
print(f"  │ Win Rate                 : {win_rate*100:>10.1f} %")
print(f"  │ Profit Factor            : {profit_factor:>10.3f}")
print(f"  │ Numero Trades            : {n_entries:>10}")
print(f"  │ Barre Totali             : {len(prices):>10,}")
print("═" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# 8. VISUALIZZAZIONE — Z-SCORE CON LIVELLI DI ENTRATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n▸ Generazione grafici...")

fig, axes = plt.subplots(4, 1, figsize=(20, 16), gridspec_kw={"height_ratios": [2, 2, 3, 2]})
fig.suptitle(
    f"PAIRS TRADING BACKTEST  —  {TICKER_A} / {TICKER_B}  |  "
    f"Sharpe: {sharpe_ratio:.2f}  |  MaxDD: {max_drawdown*100:.1f}%",
    fontsize=16, fontweight="bold", color="#E0E0E0", y=0.98
)

# ── Pannello 1: Prezzi delle due azioni ──────────────────────────────────────
ax1 = axes[0]
ax1.plot(prices.index, prices[TICKER_A], color="#00D4AA", alpha=0.9,
         label=f"{TICKER_A} (Coca-Cola)", linewidth=1.0)
ax1_twin = ax1.twinx()
ax1_twin.plot(prices.index, prices[TICKER_B], color="#FF6B6B", alpha=0.9,
              label=f"{TICKER_B} (PepsiCo)", linewidth=1.0)
ax1.set_ylabel(f"{TICKER_A} Price ($)", color="#00D4AA")
ax1_twin.set_ylabel(f"{TICKER_B} Price ($)", color="#FF6B6B")
ax1.set_title("Prezzi delle Azioni", fontweight="bold", color="#AAAAAA")

# Legenda combinata
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
           framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 2: Hedge Ratio Dinamico ────────────────────────────────────────
ax2 = axes[1]
ax2.plot(hedge_ratio.index, hedge_ratio, color="#FFA726", alpha=0.85,
         linewidth=1.0, label="Rolling Hedge Ratio (β)")
ax2.axhline(y=hedge_ratio.mean(), color="#FFA726", linestyle="--", alpha=0.4,
            label=f"Media: {hedge_ratio.mean():.4f}")
ax2.fill_between(hedge_ratio.index,
                 hedge_ratio.mean() - hedge_ratio.std(),
                 hedge_ratio.mean() + hedge_ratio.std(),
                 alpha=0.08, color="#FFA726")
ax2.set_ylabel("Hedge Ratio (β)")
ax2.set_title("Rolling Hedge Ratio (OLS)", fontweight="bold", color="#AAAAAA")
ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 3: Z-Score con livelli di entrata ──────────────────────────────
ax3 = axes[2]

# Colorazione dello sfondo in base alla posizione
for i in range(1, len(zscore)):
    if position.iloc[i] == 1:
        ax3.axvspan(zscore.index[i-1], zscore.index[i],
                    alpha=0.08, color="#00D4AA", linewidth=0)
    elif position.iloc[i] == -1:
        ax3.axvspan(zscore.index[i-1], zscore.index[i],
                    alpha=0.08, color="#FF6B6B", linewidth=0)

# Z-Score line
ax3.plot(zscore.index, zscore, color="#64B5F6", alpha=0.9, linewidth=0.8,
         label="Z-Score")

# Livelli di entrata e uscita
ax3.axhline(y=ENTRY_ZSCORE, color="#FF5252", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Short Entry (+{ENTRY_ZSCORE}σ)")
ax3.axhline(y=-ENTRY_ZSCORE, color="#69F0AE", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Long Entry (−{ENTRY_ZSCORE}σ)")
ax3.axhline(y=EXIT_ZSCORE, color="#FFD740", linestyle="-", linewidth=1.0,
            alpha=0.6, label="Exit (Mean Reversion)")

# Aree di pericolo
ax3.fill_between(zscore.index, ENTRY_ZSCORE, zscore.max() + 0.5,
                 alpha=0.04, color="#FF5252")
ax3.fill_between(zscore.index, -ENTRY_ZSCORE, zscore.min() - 0.5,
                 alpha=0.04, color="#69F0AE")

ax3.set_ylabel("Z-Score (σ)")
ax3.set_title("Z-Score dello Spread con Livelli di Entrata / Uscita",
              fontweight="bold", color="#AAAAAA")
ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e", ncol=2)
ax3.set_ylim(zscore.min() - 0.5, zscore.max() + 0.5)

# ── Pannello 4: Equity Curve ────────────────────────────────────────────────
ax4 = axes[3]
ax4.plot(cumulative_returns.index, cumulative_returns, color="#00E5FF",
         linewidth=1.5, alpha=0.95, label="Strategia Pairs Trading")
ax4.fill_between(cumulative_returns.index, 1, cumulative_returns,
                 where=(cumulative_returns >= 1), alpha=0.1, color="#00E5FF")
ax4.fill_between(cumulative_returns.index, 1, cumulative_returns,
                 where=(cumulative_returns < 1), alpha=0.1, color="#FF5252")
ax4.axhline(y=1.0, color="white", linestyle=":", alpha=0.2)
ax4.set_ylabel("Valore Cumulato ($1)")
ax4.set_xlabel("Data")
ax4.set_title("Equity Curve della Strategia", fontweight="bold", color="#AAAAAA")
ax4.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# Formattazione degli assi temporali
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("pairs_trading_backtest.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

print("\n✓ Grafico salvato: pairs_trading_backtest.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. GRAFICO SUPPLEMENTARE — DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax_dd = plt.subplots(figsize=(20, 4))
ax_dd.fill_between(drawdown.index, drawdown * 100, 0,
                   alpha=0.5, color="#FF5252", label="Drawdown")
ax_dd.plot(drawdown.index, drawdown * 100, color="#FF8A80", linewidth=0.8)
ax_dd.axhline(y=max_drawdown * 100, color="#FF1744", linestyle="--",
              alpha=0.6, label=f"Max Drawdown: {max_drawdown*100:.2f}%")
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_xlabel("Data")
ax_dd.set_title("Underwater Equity (Drawdown)", fontweight="bold", color="#AAAAAA")
ax_dd.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax_dd.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("pairs_trading_drawdown.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()

print("✓ Grafico drawdown salvato: pairs_trading_drawdown.png")
print("\n" + "═" * 80)
print("  ✅  BACKTEST COMPLETATO")
print("═" * 80)
