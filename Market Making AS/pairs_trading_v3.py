#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  PAIRS SCANNER + BACKTEST V3 — Automatic Cointegration Pair Discovery
================================================================================

  Questo script:
    1. Scansiona un universo di azioni dello stesso settore
    2. Trova coppie REALMENTE cointegrate nel periodo corrente
    3. Esegue il backtest sulla migliore coppia trovata
    4. Usa tutti i miglioramenti V2 (Kalman, Half-Life, Stop-Loss, etc.)

  Il problema della V1/V2: KO e PEP NON sono cointegrate (p=0.53).
  Forzare una strategia mean-reverting su uno spread che diverge = perdite.

  Autore: Quant Research Desk — Aprile 2026
================================================================================
"""

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
from itertools import combinations

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

INTERVAL       = "1h"
LOOKBACK_YEARS = 2
HOURS_PER_YEAR = 252 * 6.5

# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PAIRS SCANNER — Trova coppie cointegrate
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  FASE 1: PAIRS SCANNER — Ricerca coppie cointegrate")
print("=" * 80)

# Universo di azioni per settore (alta probabilita' di cointegrazione)
SECTORS = {
    "Consumer Staples": ["KO", "PEP", "PG", "CL", "KHC", "MDLZ", "GIS", "K", "SJM"],
    "Financials":       ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
    "Energy":           ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "VLO"],
    "Tech Hardware":    ["MSFT", "AAPL", "GOOG", "META", "AMZN", "NVDA"],
    "Utilities":        ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "REITs":            ["PLD", "AMT", "CCI", "SPG", "O", "WELL", "DLR"],
    "Telecom":          ["T", "VZ", "TMUS"],
}

print("\n[1/3] Download dati per tutti i ticker...")

end_date   = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

all_tickers = []
for sector, tickers in SECTORS.items():
    all_tickers.extend(tickers)
all_tickers = list(set(all_tickers))

# Download tutti i dati
all_data = {}
failed = []
for ticker in sorted(all_tickers):
    try:
        data = yf.download(ticker, start=start_date, end=end_date,
                          interval=INTERVAL, progress=False, auto_adjust=True)
        if len(data) > 500:
            all_data[ticker] = data["Close"].squeeze()
            print(f"  OK  {ticker:6s} — {len(data):,} barre")
        else:
            failed.append(ticker)
    except:
        failed.append(ticker)

if failed:
    print(f"\n  FALLITI: {', '.join(failed)}")

print(f"\n  Ticker scaricati con successo: {len(all_data)}")

# ── Allinea tutti i dati ──
prices_all = pd.DataFrame(all_data).dropna()
print(f"  Barre comuni (allineate): {len(prices_all):,}")

# ── Test di cointegrazione per tutte le coppie ──
print(f"\n[2/3] Test di cointegrazione su tutte le coppie...")

valid_tickers = list(prices_all.columns)
pair_results = []

total_pairs = len(list(combinations(valid_tickers, 2)))
tested = 0

for t1, t2 in combinations(valid_tickers, 2):
    tested += 1
    try:
        stat, pval, crit = coint(prices_all[t1], prices_all[t2])

        # Calcola anche la correlazione
        corr = prices_all[t1].corr(prices_all[t2])

        # Half-life check
        ols_model = OLS(
            prices_all[t1].values,
            add_constant(prices_all[t2].values)
        ).fit()
        hr = ols_model.params[1]
        sp = prices_all[t1] - hr * prices_all[t2]

        sp_lag = sp.shift(1).dropna()
        sp_diff = sp.diff().dropna()
        common = sp_lag.index.intersection(sp_diff.index)
        hl_model = OLS(sp_diff.loc[common].values,
                       add_constant(sp_lag.loc[common].values)).fit()
        kappa = hl_model.params[1]
        hl = -np.log(2) / kappa if kappa < 0 else 999

        pair_results.append({
            "Pair": f"{t1}/{t2}",
            "T1": t1, "T2": t2,
            "P-Value": pval,
            "Coint Stat": stat,
            "Correlation": corr,
            "HedgeRatio": hr,
            "Kappa": kappa,
            "HalfLife": hl,
            "MeanReverting": kappa < 0,
        })
    except:
        pass

    if tested % 100 == 0:
        print(f"  ... {tested}/{total_pairs} coppie testate")

print(f"  Totale coppie testate: {tested}")

# ── Risultati ──
results_df = pd.DataFrame(pair_results)
results_df = results_df.sort_values("P-Value")

# Filtro: p-value < 0.05 AND mean-reverting (kappa < 0)
cointegrated = results_df[
    (results_df["P-Value"] < 0.05) &
    (results_df["MeanReverting"] == True) &
    (results_df["HalfLife"] < 200) &
    (results_df["HalfLife"] > 10)
].copy()

print(f"\n  Coppie con p-value < 0.05 E mean-reverting: {len(cointegrated)}")

if len(cointegrated) > 0:
    print("\n  TOP 15 COPPIE COINTEGRATE:")
    print("  " + "-" * 78)
    print(f"  {'Coppia':12s} {'P-Value':>10s} {'Corr':>8s} {'HedgeRatio':>10s} "
          f"{'HalfLife':>10s} {'Kappa':>10s}")
    print("  " + "-" * 78)
    for _, row in cointegrated.head(15).iterrows():
        print(f"  {row['Pair']:12s} {row['P-Value']:>10.6f} {row['Correlation']:>8.3f} "
              f"{row['HedgeRatio']:>10.4f} {row['HalfLife']:>10.1f} {row['Kappa']:>10.6f}")
    print("  " + "-" * 78)
else:
    # Se nessuna coppia e' cointegrata al 5%, mostra le migliori al 10%
    cointegrated = results_df[
        (results_df["P-Value"] < 0.10) &
        (results_df["MeanReverting"] == True)
    ].copy()
    print(f"\n  Nessuna coppia al 5%. Coppie al 10%: {len(cointegrated)}")
    if len(cointegrated) > 0:
        print("\n  TOP 10 (marginal cointegration):")
        for _, row in cointegrated.head(10).iterrows():
            print(f"  {row['Pair']:12s}  p={row['P-Value']:.6f}  "
                  f"HL={row['HalfLife']:.0f}h  corr={row['Correlation']:.3f}")

# Mostra anche KO/PEP per confronto
ko_pep = results_df[results_df["Pair"] == "KO/PEP"]
if len(ko_pep) > 0:
    row = ko_pep.iloc[0]
    print(f"\n  [CONFRONTO] KO/PEP: p={row['P-Value']:.6f}  "
          f"kappa={row['Kappa']:.6f}  MR={row['MeanReverting']}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: BACKTEST SULLA MIGLIORE COPPIA
# ══════════════════════════════════════════════════════════════════════════════

if len(cointegrated) == 0:
    print("\n  ATTENZIONE: Nessuna coppia cointegrata trovata.")
    print("  Il backtest procedera' con la coppia meno peggiore.")
    cointegrated = results_df[results_df["MeanReverting"] == True].head(5)
    if len(cointegrated) == 0:
        cointegrated = results_df.head(1)

best_pair = cointegrated.iloc[0]
TICKER_A = best_pair["T1"]
TICKER_B = best_pair["T2"]

print("\n" + "=" * 80)
print(f"  FASE 2: BACKTEST SULLA COPPIA MIGLIORE: {TICKER_A} / {TICKER_B}")
print(f"  P-Value: {best_pair['P-Value']:.6f}  |  "
      f"Half-Life: {best_pair['HalfLife']:.0f}h  |  "
      f"Corr: {best_pair['Correlation']:.3f}")
print("=" * 80)

# ── Prepara dati della coppia ──
prices = pd.DataFrame({
    TICKER_A: prices_all[TICKER_A],
    TICKER_B: prices_all[TICKER_B],
}).dropna()

print(f"\n  Barre: {len(prices):,}")


# ── Kalman Filter Hedge Ratio ──
print(f"\n  Kalman Filter Hedge Ratio...")

n = len(prices)
price_a = prices[TICKER_A].values
price_b = prices[TICKER_B].values

KALMAN_DELTA = 1e-4
KALMAN_VE    = 1e-3

theta = np.zeros((n, 2))
P_mat = np.zeros((n, 2, 2))

theta[0] = [0.0, 0.0]
P_mat[0] = np.eye(2) * 1.0
R = KALMAN_VE
Q = np.eye(2) * KALMAN_DELTA

for t in range(1, n):
    theta_pred = theta[t-1]
    P_pred = P_mat[t-1] + Q
    x_t = np.array([1.0, price_b[t]])
    y_t = price_a[t]
    y_pred = x_t @ theta_pred
    e = y_t - y_pred
    S = x_t @ P_pred @ x_t + R
    K = P_pred @ x_t / S
    theta[t] = theta_pred + K * e
    P_mat[t] = P_pred - np.outer(K, x_t) @ P_pred

hedge_ratio = pd.Series(theta[:, 1], index=prices.index)

# Warm-up
WARMUP = 60
prices      = prices.iloc[WARMUP:]
hedge_ratio = hedge_ratio.iloc[WARMUP:]

print(f"  HR medio: {hedge_ratio.mean():.4f}  |  "
      f"Range: [{hedge_ratio.min():.4f}, {hedge_ratio.max():.4f}]")


# ── Half-Life calibration ──
spread_raw = prices[TICKER_A] - hedge_ratio * prices[TICKER_B]

sp_lag  = spread_raw.shift(1).dropna()
sp_diff = spread_raw.diff().dropna()
common  = sp_lag.index.intersection(sp_diff.index)
hl_model = OLS(sp_diff.loc[common].values,
               add_constant(sp_lag.loc[common].values)).fit()
kappa = hl_model.params[1]

if kappa < 0:
    ZSCORE_WINDOW = int(round(-np.log(2) / kappa))
    ZSCORE_WINDOW = max(20, min(ZSCORE_WINDOW, 200))
    print(f"  Half-Life: {ZSCORE_WINDOW} ore (kappa={kappa:.6f})")
else:
    ZSCORE_WINDOW = 60
    print(f"  ATTENZIONE: kappa positivo, fallback a {ZSCORE_WINDOW}")


# ── Spread & Z-Score ──
spread      = prices[TICKER_A] - hedge_ratio * prices[TICKER_B]
spread_mean = spread.rolling(window=ZSCORE_WINDOW).mean()
spread_std  = spread.rolling(window=ZSCORE_WINDOW).std()
zscore      = (spread - spread_mean) / spread_std

# ── Regime Filter ──
VOL_WINDOW   = 60
VOL_QUANTILE = 0.85
spread_vol    = spread.rolling(window=VOL_WINDOW).std()
vol_threshold = spread_vol.quantile(VOL_QUANTILE)
regime_ok     = (spread_vol <= vol_threshold).astype(float)

# ── Rolling Cointegration ──
COINT_WINDOW = 250
COINT_PVALUE_ROLL = 0.10
rolling_coint_pval = pd.Series(np.nan, index=prices.index)

for i in range(COINT_WINDOW, len(prices)):
    try:
        _, pval, _ = coint(
            prices[TICKER_A].iloc[i - COINT_WINDOW:i],
            prices[TICKER_B].iloc[i - COINT_WINDOW:i]
        )
        rolling_coint_pval.iloc[i] = pval
    except:
        rolling_coint_pval.iloc[i] = 1.0

coint_ok = (rolling_coint_pval <= COINT_PVALUE_ROLL).astype(float)

# Trim to valid
valid_idx = zscore.dropna().index.intersection(rolling_coint_pval.dropna().index)
prices          = prices.loc[valid_idx]
hedge_ratio     = hedge_ratio.loc[valid_idx]
spread          = spread.loc[valid_idx]
zscore          = zscore.loc[valid_idx]
regime_ok       = regime_ok.loc[valid_idx]
coint_ok        = coint_ok.loc[valid_idx]
rolling_coint_pval = rolling_coint_pval.loc[valid_idx]
can_trade       = (regime_ok * coint_ok).astype(float)

pct_tradeable = can_trade.mean() * 100
print(f"  Barre valide: {len(zscore):,}  |  Tradeable: {pct_tradeable:.1f}%")


# ── Grid Search per parametri ottimali ──
print(f"\n  Grid Search parametri...")

entry_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
exit_range  = [0.0, 0.25, 0.5, 0.75, 1.0]
STOPLOSS_ZSCORE = 4.0
TC_BPS = 5
tc_per_trade = TC_BPS / 10000

returns_a = np.log(prices[TICKER_A] / prices[TICKER_A].shift(1))
returns_b = np.log(prices[TICKER_B] / prices[TICKER_B].shift(1))
spread_returns = returns_a - hedge_ratio * returns_b

results_grid = pd.DataFrame(index=entry_range, columns=exit_range, dtype=float)
ret_grid     = pd.DataFrame(index=entry_range, columns=exit_range, dtype=float)

best_sharpe = -999
best_entry  = 1.5
best_exit   = 0.5

for entry_z in entry_range:
    for exit_z in exit_range:
        if exit_z >= entry_z:
            results_grid.loc[entry_z, exit_z] = np.nan
            ret_grid.loc[entry_z, exit_z] = np.nan
            continue

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
        total_ret = (1 + ret_net).cumprod().iloc[-1] - 1

        results_grid.loc[entry_z, exit_z] = sr
        ret_grid.loc[entry_z, exit_z] = total_ret * 100

        if sr > best_sharpe:
            best_sharpe = sr
            best_entry  = entry_z
            best_exit   = exit_z

print(f"  Miglior parametri: Entry={best_entry}, Exit={best_exit}, Sharpe={best_sharpe:.3f}")

ENTRY_ZSCORE = best_entry
EXIT_ZSCORE  = best_exit

# ── Backtest finale con parametri ottimali ──
print(f"\n  Backtest finale: Entry={ENTRY_ZSCORE}, Exit={EXIT_ZSCORE}...")

position = pd.Series(0.0, index=prices.index)
trade_type = pd.Series("", index=prices.index)
current_pos = 0.0

for i in range(1, len(zscore)):
    z = zscore.iloc[i]
    tradeable = can_trade.iloc[i]

    if current_pos == 0:
        if tradeable == 1.0:
            if z > ENTRY_ZSCORE:
                current_pos = -1.0
                trade_type.iloc[i] = "SHORT_ENTRY"
            elif z < -ENTRY_ZSCORE:
                current_pos = 1.0
                trade_type.iloc[i] = "LONG_ENTRY"
    else:
        if current_pos == 1.0 and z >= -EXIT_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "TP_LONG"
        elif current_pos == -1.0 and z <= EXIT_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "TP_SHORT"
        elif current_pos == 1.0 and z < -STOPLOSS_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "SL_LONG"
        elif current_pos == -1.0 and z > STOPLOSS_ZSCORE:
            current_pos = 0.0
            trade_type.iloc[i] = "SL_SHORT"
        elif tradeable == 0.0:
            current_pos = 0.0
            trade_type.iloc[i] = "REGIME_EXIT"

    position.iloc[i] = current_pos

# P&L
strategy_returns_gross = position.shift(1) * spread_returns
strategy_returns_gross = strategy_returns_gross.fillna(0)
position_changes = position.diff().abs().fillna(0)
transaction_costs = position_changes * tc_per_trade
strategy_returns_net = strategy_returns_gross - transaction_costs

cumulative_returns = (1 + strategy_returns_net).cumprod()
cumulative_gross   = (1 + strategy_returns_gross).cumprod()
cumulative_bh_a    = (1 + returns_a.fillna(0)).cumprod()
cumulative_bh_b    = (1 + returns_b.fillna(0)).cumprod()

# Metriche
mean_return = strategy_returns_net.mean()
std_return  = strategy_returns_net.std()
sharpe_ratio = (mean_return / std_return) * np.sqrt(HOURS_PER_YEAR) if std_return > 0 else 0.0

cum_max   = cumulative_returns.cummax()
drawdown  = (cumulative_returns - cum_max) / cum_max
max_drawdown = drawdown.min()

winning_bars = (strategy_returns_net > 0).sum()
losing_bars  = (strategy_returns_net < 0).sum()
total_active = winning_bars + losing_bars
win_rate = winning_bars / total_active if total_active > 0 else 0.0

gross_profit = strategy_returns_net[strategy_returns_net > 0].sum()
gross_loss_val = abs(strategy_returns_net[strategy_returns_net < 0].sum())
profit_factor = gross_profit / gross_loss_val if gross_loss_val > 0 else np.inf

annual_return = mean_return * HOURS_PER_YEAR
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

downside_returns = strategy_returns_net[strategy_returns_net < 0]
downside_std = downside_returns.std() if len(downside_returns) > 0 else 1.0
sortino_ratio = (mean_return / downside_std) * np.sqrt(HOURS_PER_YEAR) if downside_std > 0 else 0.0

n_long  = (position == 1).sum()
n_short = (position == -1).sum()
n_flat  = (position == 0).sum()
n_entries = (position.diff().fillna(0) != 0).sum()
n_tp = trade_type.str.startswith("TP_").sum()
n_sl = trade_type.str.startswith("SL_").sum()
n_re = (trade_type == "REGIME_EXIT").sum()

# Per-trade stats
trades = []
entry_idx = None
for i in range(1, len(position)):
    if position.iloc[i] != 0 and position.iloc[i-1] == 0:
        entry_idx = i
    elif position.iloc[i] == 0 and position.iloc[i-1] != 0 and entry_idx is not None:
        trade_ret = strategy_returns_net.iloc[entry_idx:i+1].sum()
        trades.append(trade_ret)
        entry_idx = None

trades_s = pd.Series(trades) if trades else pd.Series([0.0])
avg_win = trades_s[trades_s > 0].mean() if (trades_s > 0).any() else 0
avg_loss = trades_s[trades_s < 0].mean() if (trades_s < 0).any() else 0
trade_wr = (trades_s > 0).sum() / len(trades_s) if len(trades_s) > 0 else 0

print("\n" + "=" * 80)
print(f"  RIEPILOGO PERFORMANCE V3 — {TICKER_A} / {TICKER_B}")
print("=" * 80)
print(f"  | Coppia Selezionata        : {TICKER_A} / {TICKER_B}")
print(f"  | P-Value Cointegrazione    : {best_pair['P-Value']:.6f}")
print(f"  | Entry / Exit / StopLoss   : {ENTRY_ZSCORE} / {EXIT_ZSCORE} / {STOPLOSS_ZSCORE}")
print(f"  |")
print(f"  | Rendimento Totale (netto)  : {(cumulative_returns.iloc[-1]-1)*100:>+10.2f} %")
print(f"  | Rendimento Annualizzato    : {annual_return*100:>+10.2f} %")
print(f"  | Sharpe Ratio (ann.)        : {sharpe_ratio:>10.3f}")
print(f"  | Sortino Ratio (ann.)       : {sortino_ratio:>10.3f}")
print(f"  | Calmar Ratio               : {calmar_ratio:>10.3f}")
print(f"  | Maximum Drawdown           : {max_drawdown*100:>10.2f} %")
print(f"  | Win Rate (per-bar)         : {win_rate*100:>10.1f} %")
print(f"  | Win Rate (per-trade)       : {trade_wr*100:>10.1f} %")
print(f"  | Profit Factor              : {profit_factor:>10.3f}")
print(f"  | Avg Winning Trade          : {avg_win*100:>+10.3f} %")
print(f"  | Avg Losing Trade           : {avg_loss*100:>+10.3f} %")
print(f"  | Payoff Ratio               : {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.3f}")
print(f"  | Numero Trades              : {len(trades_s):>10}")
print(f"  | TP / SL / Regime           : {n_tp} / {n_sl} / {n_re}")
print(f"  | Transaction Costs          : {transaction_costs.sum()*100:>10.3f} %")
print(f"  | Barre Totali               : {len(prices):>10,}")
print(f"  | Half-Life (Z-Window)       : {ZSCORE_WINDOW:>10} ore")
print(f"  | % Tempo Tradeable          : {pct_tradeable:>10.1f} %")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(5, 1, figsize=(22, 22),
                         gridspec_kw={"height_ratios": [2, 1.5, 3, 2, 1.5]})
fig.suptitle(
    f"PAIRS TRADING V3  |  {TICKER_A} / {TICKER_B}  |  "
    f"Sharpe: {sharpe_ratio:.2f}  |  MaxDD: {max_drawdown*100:.1f}%  |  "
    f"Return: {(cumulative_returns.iloc[-1]-1)*100:+.1f}%",
    fontsize=16, fontweight="bold", color="#E0E0E0", y=0.98
)

# Pannello 1: Prezzi
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

# Pannello 2: Kalman Hedge Ratio
ax2 = axes[1]
ax2.plot(hedge_ratio.index, hedge_ratio, color="#FFA726", alpha=0.85,
         linewidth=1.0, label="Kalman Hedge Ratio")
ax2.axhline(y=hedge_ratio.mean(), color="#FFA726", linestyle="--", alpha=0.4,
            label=f"Media: {hedge_ratio.mean():.4f}")
ax2.set_ylabel("Hedge Ratio")
ax2.set_title("Kalman Filter Hedge Ratio", fontweight="bold", color="#AAAAAA")
ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# Pannello 3: Z-Score
ax3 = axes[2]
for i in range(1, len(zscore)):
    if position.iloc[i] == 1:
        ax3.axvspan(zscore.index[i-1], zscore.index[i],
                    alpha=0.08, color="#00D4AA", linewidth=0)
    elif position.iloc[i] == -1:
        ax3.axvspan(zscore.index[i-1], zscore.index[i],
                    alpha=0.08, color="#FF6B6B", linewidth=0)

ax3.plot(zscore.index, zscore, color="#64B5F6", alpha=0.9, linewidth=0.8,
         label="Z-Score")
ax3.axhline(y=ENTRY_ZSCORE, color="#FF5252", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Short Entry (+{ENTRY_ZSCORE})")
ax3.axhline(y=-ENTRY_ZSCORE, color="#69F0AE", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Long Entry (-{ENTRY_ZSCORE})")
ax3.axhline(y=EXIT_ZSCORE, color="#FFD740", linestyle=":", linewidth=1.0,
            alpha=0.5, label=f"Exit ({EXIT_ZSCORE})")
ax3.axhline(y=-EXIT_ZSCORE, color="#FFD740", linestyle=":", linewidth=1.0,
            alpha=0.5)
ax3.axhline(y=STOPLOSS_ZSCORE, color="#FF1744", linestyle="-.", linewidth=1.0,
            alpha=0.5, label=f"StopLoss ({STOPLOSS_ZSCORE})")
ax3.axhline(y=-STOPLOSS_ZSCORE, color="#FF1744", linestyle="-.", linewidth=1.0,
            alpha=0.5)
ax3.set_ylabel("Z-Score")
ax3.set_title("Z-Score con Segnali e Regime Filter", fontweight="bold", color="#AAAAAA")
ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e", ncol=3, fontsize=9)
ax3.set_ylim(max(zscore.min() - 0.5, -6), min(zscore.max() + 0.5, 6))

# Pannello 4: Equity Curve
ax4 = axes[3]
ax4.plot(cumulative_returns.index, cumulative_returns, color="#00E5FF",
         linewidth=1.5, alpha=0.95, label="V3 Strategy (Net)")
ax4.plot(cumulative_gross.index, cumulative_gross, color="#00E5FF",
         linewidth=0.8, alpha=0.3, linestyle="--", label="Gross")
ax4.fill_between(cumulative_returns.index, 1, cumulative_returns,
                 where=(cumulative_returns >= 1), alpha=0.1, color="#00E5FF")
ax4.fill_between(cumulative_returns.index, 1, cumulative_returns,
                 where=(cumulative_returns < 1), alpha=0.1, color="#FF5252")
ax4.axhline(y=1.0, color="white", linestyle=":", alpha=0.2)
ax4.set_ylabel("Cumulative ($1)")
ax4.set_title("Equity Curve", fontweight="bold", color="#AAAAAA")
ax4.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# Pannello 5: Rolling Cointegration
ax5 = axes[4]
ax5.plot(rolling_coint_pval.index, rolling_coint_pval, color="#AB47BC",
         alpha=0.7, linewidth=0.8, label="Rolling Coint P-Value")
ax5.axhline(y=COINT_PVALUE_ROLL, color="#FF5252", linestyle="--", linewidth=1.2,
            alpha=0.8, label=f"Soglia ({COINT_PVALUE_ROLL})")
ax5.fill_between(rolling_coint_pval.index, 0, rolling_coint_pval,
                 where=(rolling_coint_pval <= COINT_PVALUE_ROLL),
                 alpha=0.15, color="#69F0AE", label="Cointegrata")
ax5.set_ylabel("P-Value")
ax5.set_xlabel("Data")
ax5.set_title("Rolling Cointegration Test", fontweight="bold", color="#AAAAAA")
ax5.legend(loc="upper right", framealpha=0.3, facecolor="#1a1a2e")
ax5.set_ylim(-0.02, 1.02)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("pairs_trading_v3.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

# Heatmap
fig3, ax_hm = plt.subplots(figsize=(10, 7))
colors_cmap = ["#FF1744", "#1a1a2e", "#00E676"]
cmap = LinearSegmentedColormap.from_list("custom", colors_cmap, N=256)
data_grid = results_grid.values.astype(float)
vmax = max(1.0, np.nanmax(data_grid))
im = ax_hm.imshow(data_grid, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax)
ax_hm.set_xticks(range(len(exit_range)))
ax_hm.set_xticklabels([f"{x:.2f}" for x in exit_range])
ax_hm.set_yticks(range(len(entry_range)))
ax_hm.set_yticklabels([f"{x:.2f}" for x in entry_range])
ax_hm.set_xlabel("Exit Z-Score")
ax_hm.set_ylabel("Entry Z-Score")
ax_hm.set_title(f"Sharpe Ratio Heatmap — {TICKER_A}/{TICKER_B}",
                 fontweight="bold", color="#AAAAAA")

for i in range(len(entry_range)):
    for j in range(len(exit_range)):
        val = data_grid[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 0.3 else "#AAAAAA"
            ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color=color, fontsize=9, fontweight="bold")

plt.colorbar(im, label="Sharpe Ratio")
plt.tight_layout()
plt.savefig("pairs_trading_v3_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=fig3.get_facecolor())
plt.show()

# Drawdown
fig4, ax_dd = plt.subplots(figsize=(20, 4))
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
plt.savefig("pairs_trading_v3_drawdown.png", dpi=150, bbox_inches="tight",
            facecolor=fig4.get_facecolor())
plt.show()

print("\n" + "=" * 80)
print("  BACKTEST V3 COMPLETATO")
print("=" * 80)
