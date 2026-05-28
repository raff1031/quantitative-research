#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  PAIRS TRADING V5 — Walk-Forward Intra-Sector Strategy
================================================================================

  Problemi risolti:
    V1: KO/PEP non cointegrate → perdite
    V3: Solo 2 trades → no significativita'
    V4: Cross-sector pairs senza logica economica → 609 trades ma -12.5%

  V5 FIX:
    1. SOLO coppie INTRA-SETTORIALI (stessa industria = relazione economica)
    2. TRAIN/TEST SPLIT per evitare overfitting (60/40)
    3. Half-life reale (non capped arbitrariamente)
    4. Z-Score window = half-life (calibrato su train, applicato su test)
    5. Parametri calibrati su TRAINING, testati su OUT-OF-SAMPLE
    6. Operare su LOG PRICES (spread in log e' piu' stazionario)

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
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from datetime import datetime, timedelta
from itertools import combinations

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (18, 10), "font.family": "monospace", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12, "axes.grid": True,
    "grid.alpha": 0.15, "grid.linestyle": "--", "lines.linewidth": 1.2,
    "figure.dpi": 120,
})

INTERVAL       = "1h"
LOOKBACK_YEARS = 2
HOURS_PER_YEAR = 252 * 6.5
TRAIN_RATIO    = 0.60        # 60% training, 40% test

# Signal parameters
ENTRY_ZSCORE    = 1.5
EXIT_ZSCORE     = 0.0        # Exit a zero per massimo mean-reversion profit
STOPLOSS_ZSCORE = 3.5
TC_BPS          = 5

# Kalman
KALMAN_DELTA    = 1e-5       # Piu' lento = piu' stabile (era 1e-4)
KALMAN_VE       = 1e-3

TOP_N_PAIRS     = 8

print("=" * 80)
print("  PAIRS TRADING V5 — Walk-Forward Intra-Sector")
print(f"  Train/Test: {TRAIN_RATIO*100:.0f}/{(1-TRAIN_RATIO)*100:.0f}  |  "
      f"Entry: +/-{ENTRY_ZSCORE}  |  Exit: {EXIT_ZSCORE}  |  SL: {STOPLOSS_ZSCORE}")
print(f"  Solo coppie INTRA-settoriali  |  Log-price spread")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1: DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/6] Download dati...")

end_date   = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

# Settori con coppie economicamente sensate
SECTORS = {
    "Consumer_Staples": ["KO", "PEP", "PG", "CL", "KHC", "MDLZ", "GIS"],
    "Financials_Banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC"],
    "Financials_IB":    ["GS", "MS"],
    "Energy_Majors":    ["XOM", "CVX", "COP", "EOG"],
    "Energy_Services":  ["SLB", "OXY", "PSX", "VLO"],
    "Utilities":        ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "REITs":            ["PLD", "AMT", "CCI", "SPG", "O", "WELL", "DLR"],
    "Telecom":          ["T", "VZ", "TMUS"],
}

all_tickers = set()
for tickers in SECTORS.values():
    all_tickers.update(tickers)

all_data = {}
for ticker in sorted(all_tickers):
    try:
        data = yf.download(ticker, start=start_date, end=end_date,
                          interval=INTERVAL, progress=False, auto_adjust=True)
        if len(data) > 500:
            all_data[ticker] = data["Close"].squeeze()
    except:
        pass

prices_all = pd.DataFrame(all_data).dropna()
print(f"  Ticker OK: {len(prices_all.columns)}  |  Barre: {len(prices_all):,}")

# Train/Test split
split_idx = int(len(prices_all) * TRAIN_RATIO)
train_prices = prices_all.iloc[:split_idx]
test_prices  = prices_all.iloc[split_idx:]

print(f"  TRAIN: {len(train_prices):,} barre ({train_prices.index[0].strftime('%Y-%m-%d')} -> "
      f"{train_prices.index[-1].strftime('%Y-%m-%d')})")
print(f"  TEST:  {len(test_prices):,} barre ({test_prices.index[0].strftime('%Y-%m-%d')} -> "
      f"{test_prices.index[-1].strftime('%Y-%m-%d')})")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2: PAIR SCANNING su TRAINING DATA (solo intra-settoriali)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/6] Scanning coppie intra-settoriali su TRAINING data...")

pair_results = []

for sector_name, tickers in SECTORS.items():
    available = [t for t in tickers if t in train_prices.columns]
    if len(available) < 2:
        continue

    for t1, t2 in combinations(available, 2):
        try:
            # Log prices per spread piu' stazionario
            log_a = np.log(train_prices[t1])
            log_b = np.log(train_prices[t2])

            stat, pval, _ = coint(log_a, log_b)

            if pval > 0.10:
                continue

            # Hedge ratio su log prices
            ols_m = OLS(log_a.values, add_constant(log_b.values)).fit()
            hr = ols_m.params[1]
            sp = log_a - hr * log_b

            # Half-life
            sl = sp.shift(1).dropna()
            sd = sp.diff().dropna()
            ci = sl.index.intersection(sd.index)
            hl_m = OLS(sd.loc[ci].values, add_constant(sl.loc[ci].values)).fit()
            kappa = hl_m.params[1]

            if kappa >= 0:
                continue

            hl = -np.log(2) / kappa
            if hl < 5 or hl > 300:
                continue

            corr = train_prices[t1].corr(train_prices[t2])

            pair_results.append({
                "Sector": sector_name, "Pair": f"{t1}/{t2}",
                "T1": t1, "T2": t2, "PValue": pval,
                "HedgeRatio_train": hr, "Kappa": kappa,
                "HalfLife": hl, "Correlation": corr,
            })
        except:
            pass

results_df = pd.DataFrame(pair_results)
if len(results_df) > 0:
    results_df = results_df.sort_values("PValue")

print(f"\n  Coppie intra-settoriali trovate: {len(results_df)}")
if len(results_df) > 0:
    print(f"\n  TOP COPPIE COINTEGRATE (TRAINING):")
    print("  " + "-" * 80)
    print(f"  {'Settore':20s} {'Coppia':10s} {'P-Value':>10s} {'HL':>6s} "
          f"{'Corr':>8s} {'HR':>8s}")
    print("  " + "-" * 80)
    for _, row in results_df.head(20).iterrows():
        marker = " ***" if row["PValue"] < 0.05 else ""
        print(f"  {row['Sector']:20s} {row['Pair']:10s} {row['PValue']:>10.6f} "
              f"{row['HalfLife']:>6.0f} {row['Correlation']:>8.3f} "
              f"{row['HedgeRatio_train']:>8.4f}{marker}")
    print("  " + "-" * 80)
    print("  *** = p < 0.05")

# Seleziona top N per il backtest
portfolio = results_df[results_df["PValue"] < 0.10].head(TOP_N_PAIRS)

if len(portfolio) == 0:
    print("\n  ATTENZIONE: Nessuna coppia intra-settoriale cointegrata!")
    print("  Usando tutte le coppie disponibili...")
    portfolio = results_df.head(TOP_N_PAIRS)

print(f"\n  Coppie selezionate per il backtest: {len(portfolio)}")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 3: BACKTEST OUT-OF-SAMPLE (TEST SET)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Calibrazione su TRAINING:
#    - Coppia selezionata
#    - Half-life calcolato → Z-Score window
#
#  Backtest su TEST:
#    - Kalman Filter ricalcolato in-tempo su tutto il dataset (no lookahead)
#    - Z-Score con window calibrata dal training
#    - Segnali generati solo nel periodo di test
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n[3/6] Backtest OUT-OF-SAMPLE per ogni coppia...")

tc_per_trade = TC_BPS / 10000
pair_equity = {}
pair_metrics = []
pair_zscores_test = {}
pair_positions_test = {}

for _, row in portfolio.iterrows():
    t1, t2 = row["T1"], row["T2"]
    pair_name = row["Pair"]
    sector = row["Sector"]

    # Z-Score window dal training half-life
    zw = int(round(row["HalfLife"]))
    zw = max(15, min(zw, 150))

    # Usa TUTTO il dataset per Kalman (non lookahead: il Kalman e' online)
    pr = pd.DataFrame({t1: prices_all[t1], t2: prices_all[t2]}).dropna()
    n = len(pr)

    # ── Kalman su LOG PRICES ──
    log_a = np.log(pr[t1].values)
    log_b = np.log(pr[t2].values)

    theta = np.zeros((n, 2))
    P_mat = np.zeros((n, 2, 2))
    theta[0] = [0.0, row["HedgeRatio_train"]]  # Inizializza con HR del training
    P_mat[0] = np.eye(2) * 0.1
    R_k = KALMAN_VE
    Q_k = np.eye(2) * KALMAN_DELTA

    for t in range(1, n):
        theta_pred = theta[t-1]
        P_pred = P_mat[t-1] + Q_k
        x_t = np.array([1.0, log_b[t]])
        y_t = log_a[t]
        e_t = y_t - x_t @ theta_pred
        S_t = x_t @ P_pred @ x_t + R_k
        K_t = P_pred @ x_t / S_t
        theta[t] = theta_pred + K_t * e_t
        P_mat[t] = P_pred - np.outer(K_t, x_t) @ P_pred

    hr = pd.Series(theta[:, 1], index=pr.index)

    # ── Spread su log prices ──
    log_price_a = np.log(pr[t1])
    log_price_b = np.log(pr[t2])
    spread = log_price_a - hr * log_price_b

    sm = spread.rolling(window=zw).mean()
    ss = spread.rolling(window=zw).std()
    zs = (spread - sm) / ss

    # ── Isola solo il TEST period ──
    test_start = test_prices.index[0]
    test_mask = pr.index >= test_start

    pr_test   = pr.loc[test_mask]
    hr_test   = hr.loc[test_mask]
    zs_test   = zs.loc[test_mask].dropna()
    spread_test = spread.loc[test_mask]

    if len(zs_test) < 100:
        print(f"  {pair_name:12s}  SKIPPED (troppo poche barre test: {len(zs_test)})")
        continue

    # ── Segnali sul test set ──
    position = pd.Series(0.0, index=zs_test.index)
    cur = 0.0

    for i in range(1, len(zs_test)):
        z = zs_test.iloc[i]

        if cur == 0:
            if z > ENTRY_ZSCORE:
                cur = -1.0
            elif z < -ENTRY_ZSCORE:
                cur = 1.0
        else:
            # Take profit
            if cur == 1.0 and z >= EXIT_ZSCORE:
                cur = 0.0
            elif cur == -1.0 and z <= EXIT_ZSCORE:
                cur = 0.0
            # Stop loss
            elif cur == 1.0 and z < -STOPLOSS_ZSCORE:
                cur = 0.0
            elif cur == -1.0 and z > STOPLOSS_ZSCORE:
                cur = 0.0

        position.iloc[i] = cur

    # ── P&L su log-return ──
    # Rendimento dello spread in log-prices:
    # Se long spread: guadagno = delta(log_A) - HR * delta(log_B)
    # che e' approssimativamente il rendimento del portafoglio hedged
    ret_a = log_price_a.diff().loc[zs_test.index]
    ret_b = log_price_b.diff().loc[zs_test.index]
    hr_aligned = hr.loc[zs_test.index]
    sret = ret_a - hr_aligned * ret_b

    strat_ret = position.shift(1) * sret
    strat_ret = strat_ret.fillna(0)
    tc = position.diff().abs().fillna(0) * tc_per_trade
    net_ret = strat_ret - tc

    cum_ret = (1 + net_ret).cumprod()

    # Metriche
    n_trades = (position.diff().fillna(0) != 0).sum()

    sr = 0
    if net_ret.std() > 0:
        sr = (net_ret.mean() / net_ret.std()) * np.sqrt(HOURS_PER_YEAR)

    total_ret = (cum_ret.iloc[-1] - 1) * 100
    cum_max_dd = cum_ret.cummax()
    dd = (cum_ret - cum_max_dd) / cum_max_dd
    max_dd = dd.min() * 100

    # Per-trade analysis
    trades_list = []
    entry_i = None
    for i in range(1, len(position)):
        if position.iloc[i] != 0 and position.iloc[i-1] == 0:
            entry_i = i
        elif position.iloc[i] == 0 and position.iloc[i-1] != 0 and entry_i is not None:
            tr = net_ret.iloc[entry_i:i+1].sum()
            trades_list.append(tr)
            entry_i = None

    trade_wr = (pd.Series(trades_list) > 0).mean() * 100 if trades_list else 0

    pair_metrics.append({
        "Sector": sector, "Pair": pair_name, "T1": t1, "T2": t2,
        "Return": total_ret, "Sharpe": sr, "MaxDD": max_dd,
        "Trades": n_trades, "NumTrades": len(trades_list),
        "TradeWR": trade_wr, "HL": zw,
    })
    pair_equity[pair_name] = cum_ret
    pair_zscores_test[pair_name] = zs_test
    pair_positions_test[pair_name] = position

    flag = " <-- BEST" if sr > 0.5 else ""
    print(f"  {sector:20s} {pair_name:10s}  Ret={total_ret:>+7.2f}%  "
          f"SR={sr:>+6.2f}  DD={max_dd:>6.2f}%  "
          f"Trades={len(trades_list):>3}  WR={trade_wr:>5.1f}%  HL={zw:>3}h{flag}")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 4: PORTAFOGLIO AGGREGATO
# ══════════════════════════════════════════════════════════════════════════════

if len(pair_equity) == 0:
    print("\n  ERRORE: Nessuna coppia ha prodotto risultati!")
    exit(1)

print(f"\n[4/6] Aggregazione portafoglio ({len(pair_equity)} coppie)...")

metrics_df = pd.DataFrame(pair_metrics)

all_eq = pd.DataFrame(pair_equity)
all_eq = all_eq.dropna(how="all").fillna(method="ffill").fillna(1.0)

# Portfolio: equal weight
port_ret = all_eq.pct_change().fillna(0).mean(axis=1)
port_cum = (1 + port_ret).cumprod()

# Portfolio metrics
pm = port_ret.mean()
ps = port_ret.std()
port_sharpe = (pm / ps) * np.sqrt(HOURS_PER_YEAR) if ps > 0 else 0
port_cm = port_cum.cummax()
port_dd = (port_cum - port_cm) / port_cm
port_mdd = port_dd.min()
port_total = (port_cum.iloc[-1] - 1) * 100
port_annual = pm * HOURS_PER_YEAR * 100

gp = port_ret[port_ret > 0].sum()
gl = abs(port_ret[port_ret < 0].sum())
port_pf = gp / gl if gl > 0 else np.inf

ds = port_ret[port_ret < 0].std()
port_sortino = (pm / ds) * np.sqrt(HOURS_PER_YEAR) if ds > 0 else 0
port_calmar = (pm * HOURS_PER_YEAR) / abs(port_mdd) if port_mdd != 0 else 0

total_trades = int(metrics_df["NumTrades"].sum())

print("\n" + "=" * 80)
print(f"  RIEPILOGO V5 — Walk-Forward OOS su {len(pair_equity)} Coppie Intra-Settoriali")
print("=" * 80)
print(f"  |")
print(f"  | === PORTAFOGLIO OUT-OF-SAMPLE ===")
print(f"  | Rendimento Totale (netto)  : {port_total:>+10.2f} %")
print(f"  | Rendimento Annualizzato    : {port_annual:>+10.2f} %")
print(f"  | Sharpe Ratio (ann.)        : {port_sharpe:>10.3f}")
print(f"  | Sortino Ratio (ann.)       : {port_sortino:>10.3f}")
print(f"  | Calmar Ratio               : {port_calmar:>10.3f}")
print(f"  | Maximum Drawdown           : {port_mdd*100:>10.2f} %")
print(f"  | Profit Factor              : {port_pf:>10.3f}")
print(f"  | Trades Totali              : {total_trades:>10}")
print(f"  | Trades Medi / Coppia       : {total_trades/len(pair_equity):>10.1f}")
print(f"  |")
print(f"  | === DETTAGLIO PER COPPIA (OOS) ===")
print(f"  | {'Settore':20s} {'Coppia':10s} {'Ret%':>8s} {'SR':>8s} "
      f"{'MaxDD%':>8s} {'#Trd':>6s} {'WR%':>6s} {'HL':>4s}")
print(f"  | {'-'*70}")

for _, m in metrics_df.sort_values("Sharpe", ascending=False).iterrows():
    star = " ***" if m["Sharpe"] > 0 else ""
    print(f"  | {m['Sector']:20s} {m['Pair']:10s} {m['Return']:>+8.2f} "
          f"{m['Sharpe']:>+8.2f} {m['MaxDD']:>8.2f} {m['NumTrades']:>6.0f} "
          f"{m['TradeWR']:>6.1f} {m['HL']:>4.0f}{star}")

print(f"  | {'-'*70}")
avg_sr = metrics_df['Sharpe'].mean()
avg_ret = metrics_df['Return'].mean()
avg_trd = metrics_df['NumTrades'].mean()
print(f"  | {'MEDIA':20s} {'':10s} {avg_ret:>+8.2f} {avg_sr:>+8.2f} "
      f"{'':>8s} {avg_trd:>6.0f}")
print(f"  |")

# Coppie profittevoli
profitable = (metrics_df["Return"] > 0).sum()
print(f"  | Coppie profittevoli: {profitable}/{len(metrics_df)} "
      f"({profitable/len(metrics_df)*100:.0f}%)")

print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 5: GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Generazione grafici...")

colors = ["#00E5FF", "#FF6B6B", "#69F0AE", "#FFA726", "#AB47BC",
          "#FF5252", "#64B5F6", "#FFD740", "#E040FB", "#00BFA5",
          "#F48FB1", "#80CBC4", "#FFAB91", "#CE93D8"]

fig, axes = plt.subplots(3, 1, figsize=(22, 18),
                         gridspec_kw={"height_ratios": [3, 2, 2]})

fig.suptitle(
    f"PAIRS TRADING V5 — Walk-Forward OOS  |  "
    f"{len(pair_equity)} Coppie Intra-Settoriali  |  "
    f"Sharpe: {port_sharpe:.2f}  |  Return: {port_total:+.1f}%  |  "
    f"Trades: {total_trades}",
    fontsize=14, fontweight="bold", color="#E0E0E0", y=0.98
)

# Pannello 1: Equity curves
ax1 = axes[0]
for i, (pname, eq) in enumerate(pair_equity.items()):
    ax1.plot(eq.index, eq, color=colors[i % len(colors)],
             alpha=0.4, linewidth=0.8, label=pname)
ax1.plot(port_cum.index, port_cum, color="white",
         linewidth=2.5, alpha=0.95, label=f"PORTFOLIO (SR={port_sharpe:.2f})")
ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

# Train/Test separator
ax1.axvline(x=test_prices.index[0], color="#FFD740", linestyle="--",
            linewidth=1.5, alpha=0.6, label="Train|Test Split")

ax1.fill_between(port_cum.index, 1, port_cum,
                 where=(port_cum >= 1), alpha=0.08, color="#00E5FF")
ax1.fill_between(port_cum.index, 1, port_cum,
                 where=(port_cum < 1), alpha=0.08, color="#FF5252")
ax1.set_ylabel("Cumulative ($1)")
ax1.set_title("Equity Curves — Out-of-Sample Period", fontweight="bold", color="#AAAAAA")
ax1.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e", fontsize=8, ncol=3)

# Pannello 2: Z-Scores
ax2 = axes[1]
for i, (pname, zs) in enumerate(pair_zscores_test.items()):
    ax2.plot(zs.index, zs, color=colors[i % len(colors)],
             alpha=0.35, linewidth=0.6, label=pname)
ax2.axhline(y=ENTRY_ZSCORE, color="#FF5252", linestyle="--", linewidth=1.5,
            alpha=0.7, label=f"Entry +/-{ENTRY_ZSCORE}")
ax2.axhline(y=-ENTRY_ZSCORE, color="#69F0AE", linestyle="--", linewidth=1.5, alpha=0.7)
ax2.axhline(y=0, color="#FFD740", linestyle="-", linewidth=0.8, alpha=0.4)
ax2.set_ylabel("Z-Score")
ax2.set_title("Z-Scores Out-of-Sample", fontweight="bold", color="#AAAAAA")
ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e", fontsize=8, ncol=4)
ax2.set_ylim(-5, 5)

# Pannello 3: Drawdown
ax3 = axes[2]
ax3.fill_between(port_dd.index, port_dd * 100, 0,
                 alpha=0.5, color="#FF5252", label="Portfolio Drawdown")
ax3.plot(port_dd.index, port_dd * 100, color="#FF8A80", linewidth=0.8)
ax3.axhline(y=port_mdd * 100, color="#FF1744", linestyle="--",
            alpha=0.6, label=f"Max DD: {port_mdd*100:.2f}%")
ax3.set_ylabel("Drawdown (%)")
ax3.set_xlabel("Data")
ax3.set_title("Portfolio Drawdown (OOS)", fontweight="bold", color="#AAAAAA")
ax3.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("pairs_trading_v5.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("  Salvato: pairs_trading_v5.png")


# ── Performance Breakdown ──
fig2, (ax_ret, ax_sr) = plt.subplots(1, 2, figsize=(18, 7))

sorted_df = metrics_df.sort_values("Sharpe", ascending=True)
pair_names = sorted_df["Pair"].values

# Returns
rets = sorted_df["Return"].values
bcols = ["#69F0AE" if r > 0 else "#FF5252" for r in rets]
ax_ret.barh(pair_names, rets, color=bcols, alpha=0.8, edgecolor="white", linewidth=0.5)
ax_ret.axvline(x=0, color="white", linestyle="-", alpha=0.3)
ax_ret.set_xlabel("Return (%)")
ax_ret.set_title("Return per Coppia (OOS)", fontweight="bold", color="#AAAAAA")
for i, r in enumerate(rets):
    ax_ret.text(r + (0.3 if r >= 0 else -0.3), i, f"{r:+.1f}%", va="center",
                ha="left" if r >= 0 else "right", fontsize=10, color="white")

# Sharpe
srs = sorted_df["Sharpe"].values
scols = ["#69F0AE" if s > 0 else "#FF5252" for s in srs]
ax_sr.barh(pair_names, srs, color=scols, alpha=0.8, edgecolor="white", linewidth=0.5)
ax_sr.axvline(x=0, color="white", linestyle="-", alpha=0.3)
ax_sr.set_xlabel("Sharpe Ratio")
ax_sr.set_title("Sharpe Ratio per Coppia (OOS)", fontweight="bold", color="#AAAAAA")
for i, s in enumerate(srs):
    ax_sr.text(s + (0.05 if s >= 0 else -0.05), i, f"{s:+.2f}", va="center",
               ha="left" if s >= 0 else "right", fontsize=10, color="white")

plt.tight_layout()
plt.savefig("pairs_trading_v5_breakdown.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()
print("  Salvato: pairs_trading_v5_breakdown.png")


# ── Sector performance ──
print("\n[6/6] Analisi per settore...")
if len(metrics_df) > 0:
    sector_perf = metrics_df.groupby("Sector").agg({
        "Return": "mean", "Sharpe": "mean", "NumTrades": "sum"
    }).sort_values("Sharpe", ascending=False)

    print(f"\n  Performance media per settore:")
    print(f"  {'Settore':20s} {'Ret%':>8s} {'SR':>8s} {'Trades':>8s}")
    print(f"  {'-'*46}")
    for sect, row in sector_perf.iterrows():
        print(f"  {sect:20s} {row['Return']:>+8.2f} {row['Sharpe']:>+8.2f} "
              f"{row['NumTrades']:>8.0f}")

print("\n" + "=" * 80)
print("  BACKTEST V5 COMPLETATO")
print(f"  Trades totali OOS: {total_trades}  |  Portfolio Sharpe: {port_sharpe:.3f}")
print("=" * 80)
