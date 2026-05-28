#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  PAIRS TRADING V4 — Multi-Pair Portfolio Strategy
================================================================================

  Soluzione al problema della V3: troppo poche operazioni (2 trades).

  APPROCCIO V4:
    1. Scanner: trova tutte le coppie cointegrate
    2. Filtra per Half-Life CORTA (< 80h) → mean reversion veloce → piu' trades
    3. Opera su un PORTAFOGLIO di N coppie in parallelo
    4. Aumenta la frequenza con filtri meno restrittivi ma piu' intelligenti
    5. Aggrega i rendimenti di tutte le coppie

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

# ── Parametri V4 ──
TOP_N_PAIRS     = 8          # Numero di coppie da tradare in parallelo
MAX_HALF_LIFE   = 80         # Solo coppie con HL < 80h (mean reversion veloce)
MIN_HALF_LIFE   = 10         # Almeno 10h (troppo corto = rumore)
COINT_PVALUE_MAX = 0.05      # P-value massimo per selezione coppia
ENTRY_ZSCORE    = 1.5        # Entry moderato per piu' segnali
EXIT_ZSCORE     = 0.25       # Exit vicino a zero ma non esatto
STOPLOSS_ZSCORE = 3.5        # Stop loss
TC_BPS          = 5          # Transaction costs
KALMAN_DELTA    = 1e-4
KALMAN_VE       = 1e-3

# Filtri di regime piu' permissivi
VOL_QUANTILE    = 0.90       # Solo top 10% volatilita' bloccato (era 85%)
COINT_ROLLING_W = 150        # Finestra rolling coint piu' corta (era 250)
COINT_ROLLING_P = 0.15       # P-value rolling piu' permissivo (era 0.10)

print("=" * 80)
print("  PAIRS TRADING V4 — Multi-Pair Portfolio")
print(f"  Top {TOP_N_PAIRS} coppie  |  HL range: [{MIN_HALF_LIFE}, {MAX_HALF_LIFE}]h")
print(f"  Entry: +/-{ENTRY_ZSCORE}s  |  Exit: {EXIT_ZSCORE}s  |  SL: {STOPLOSS_ZSCORE}s")
print(f"  Rolling Coint: window={COINT_ROLLING_W}, p<{COINT_ROLLING_P}")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1: DOWNLOAD & PAIR SCANNING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Download dati...")

end_date   = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

UNIVERSE = [
    # Consumer Staples
    "KO", "PEP", "PG", "CL", "KHC", "MDLZ", "GIS", "SJM",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "VLO",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    # REITs
    "PLD", "AMT", "CCI", "SPG", "O", "WELL", "DLR",
    # Telecom
    "T", "VZ", "TMUS",
    # Tech
    "MSFT", "AAPL", "GOOG", "META", "AMZN", "NVDA",
]

all_data = {}
for ticker in sorted(set(UNIVERSE)):
    try:
        data = yf.download(ticker, start=start_date, end=end_date,
                          interval=INTERVAL, progress=False, auto_adjust=True)
        if len(data) > 500:
            all_data[ticker] = data["Close"].squeeze()
    except:
        pass

prices_all = pd.DataFrame(all_data).dropna()
valid_tickers = list(prices_all.columns)
print(f"  Ticker OK: {len(valid_tickers)}  |  Barre comuni: {len(prices_all):,}")


# ── Cointegration scan ──
print("\n[2/5] Scanning coppie cointegrate...")

pair_results = []
for t1, t2 in combinations(valid_tickers, 2):
    try:
        stat, pval, _ = coint(prices_all[t1], prices_all[t2])
        if pval > 0.10:  # Skip rapido le coppie chiaramente non cointegrate
            continue

        # Hedge ratio e half-life
        ols_m = OLS(prices_all[t1].values,
                    add_constant(prices_all[t2].values)).fit()
        hr = ols_m.params[1]
        sp = prices_all[t1] - hr * prices_all[t2]

        sp_lag = sp.shift(1).dropna()
        sp_diff = sp.diff().dropna()
        common = sp_lag.index.intersection(sp_diff.index)
        hl_m = OLS(sp_diff.loc[common].values,
                   add_constant(sp_lag.loc[common].values)).fit()
        kappa = hl_m.params[1]

        if kappa >= 0:
            continue  # Non mean-reverting

        hl = -np.log(2) / kappa
        corr = prices_all[t1].corr(prices_all[t2])

        pair_results.append({
            "Pair": f"{t1}/{t2}", "T1": t1, "T2": t2,
            "PValue": pval, "HedgeRatio": hr, "Kappa": kappa,
            "HalfLife": hl, "Correlation": corr,
        })
    except:
        pass

results_df = pd.DataFrame(pair_results)

# Filtra: p < 0.05, half-life nel range
selected = results_df[
    (results_df["PValue"] < COINT_PVALUE_MAX) &
    (results_df["HalfLife"] >= MIN_HALF_LIFE) &
    (results_df["HalfLife"] <= MAX_HALF_LIFE)
].sort_values("PValue")

print(f"\n  Coppie totali p<0.10: {len(results_df)}")
print(f"  Coppie con p<{COINT_PVALUE_MAX} e HL in [{MIN_HALF_LIFE},{MAX_HALF_LIFE}]: {len(selected)}")

# Prendi le top N
portfolio_pairs = selected.head(TOP_N_PAIRS)

print(f"\n  PORTAFOGLIO SELEZIONATO ({len(portfolio_pairs)} coppie):")
print("  " + "-" * 72)
print(f"  {'Coppia':12s} {'P-Value':>10s} {'HL (ore)':>10s} {'Corr':>8s} {'HR':>8s}")
print("  " + "-" * 72)
for _, row in portfolio_pairs.iterrows():
    print(f"  {row['Pair']:12s} {row['PValue']:>10.6f} {row['HalfLife']:>10.1f} "
          f"{row['Correlation']:>8.3f} {row['HedgeRatio']:>8.4f}")
print("  " + "-" * 72)


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2: BACKTEST PER OGNI COPPIA
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/5] Backtest individuale per ogni coppia...")

tc_per_trade = TC_BPS / 10000
pair_equity_curves = {}
pair_metrics = []
all_pair_positions = {}
all_pair_zscores = {}

for idx, row in portfolio_pairs.iterrows():
    t1, t2 = row["T1"], row["T2"]
    pair_name = row["Pair"]

    # ── Prepara dati ──
    pr = pd.DataFrame({t1: prices_all[t1], t2: prices_all[t2]}).dropna()
    n = len(pr)

    # ── Kalman Filter Hedge Ratio ──
    pa = pr[t1].values
    pb = pr[t2].values

    theta = np.zeros((n, 2))
    P_mat = np.zeros((n, 2, 2))
    theta[0] = [0.0, 0.0]
    P_mat[0] = np.eye(2) * 1.0
    R_k = KALMAN_VE
    Q_k = np.eye(2) * KALMAN_DELTA

    for t in range(1, n):
        theta_pred = theta[t-1]
        P_pred = P_mat[t-1] + Q_k
        x_t = np.array([1.0, pb[t]])
        y_t = pa[t]
        e_t = y_t - x_t @ theta_pred
        S_t = x_t @ P_pred @ x_t + R_k
        K_t = P_pred @ x_t / S_t
        theta[t] = theta_pred + K_t * e_t
        P_mat[t] = P_pred - np.outer(K_t, x_t) @ P_pred

    hr = pd.Series(theta[:, 1], index=pr.index)

    # Warm-up
    WARMUP = 30
    pr = pr.iloc[WARMUP:]
    hr = hr.iloc[WARMUP:]

    # ── Half-Life per Z-Score window ──
    spread_raw = pr[t1] - hr * pr[t2]
    sl = spread_raw.shift(1).dropna()
    sd = spread_raw.diff().dropna()
    ci = sl.index.intersection(sd.index)
    hl_mod = OLS(sd.loc[ci].values, add_constant(sl.loc[ci].values)).fit()
    kp = hl_mod.params[1]

    if kp < 0:
        zw = int(round(-np.log(2) / kp))
        zw = max(15, min(zw, 100))  # Clamp 15-100 per avere piu' trades
    else:
        zw = 40

    # ── Spread & Z-Score ──
    spread = pr[t1] - hr * pr[t2]
    sm = spread.rolling(window=zw).mean()
    ss = spread.rolling(window=zw).std()
    zs = (spread - sm) / ss

    # ── Regime Filter (permissivo) ──
    svol = spread.rolling(window=40).std()
    vt = svol.quantile(VOL_QUANTILE)
    rok = (svol <= vt).astype(float)

    # ── Rolling Cointegration (piu' corto e permissivo) ──
    rc_pval = pd.Series(np.nan, index=pr.index)
    for i in range(COINT_ROLLING_W, len(pr)):
        try:
            _, pv, _ = coint(pr[t1].iloc[i-COINT_ROLLING_W:i],
                             pr[t2].iloc[i-COINT_ROLLING_W:i])
            rc_pval.iloc[i] = pv
        except:
            rc_pval.iloc[i] = 1.0

    cok = (rc_pval <= COINT_ROLLING_P).astype(float)

    # Trim valid
    vi = zs.dropna().index.intersection(rc_pval.dropna().index)
    pr = pr.loc[vi]
    hr = hr.loc[vi]
    spread = spread.loc[vi]
    zs = zs.loc[vi]
    rok = rok.loc[vi]
    cok = cok.loc[vi]
    can_trade = (rok * cok).astype(float)

    # ── Segnali ──
    position = pd.Series(0.0, index=pr.index)
    cur = 0.0

    for i in range(1, len(zs)):
        z = zs.iloc[i]
        tr = can_trade.iloc[i]

        if cur == 0:
            if tr == 1.0:
                if z > ENTRY_ZSCORE:
                    cur = -1.0
                elif z < -ENTRY_ZSCORE:
                    cur = 1.0
        else:
            # Take profit
            if cur == 1.0 and z >= -EXIT_ZSCORE:
                cur = 0.0
            elif cur == -1.0 and z <= EXIT_ZSCORE:
                cur = 0.0
            # Stop loss
            elif abs(z) > STOPLOSS_ZSCORE:
                cur = 0.0
            # Regime exit
            elif tr == 0.0:
                cur = 0.0

        position.iloc[i] = cur

    # ── P&L ──
    ret_a = np.log(pr[t1] / pr[t1].shift(1))
    ret_b = np.log(pr[t2] / pr[t2].shift(1))
    sret = ret_a - hr * ret_b

    strat_ret = position.shift(1) * sret
    strat_ret = strat_ret.fillna(0)
    tc = position.diff().abs().fillna(0) * tc_per_trade
    net_ret = strat_ret - tc

    cum_ret = (1 + net_ret).cumprod()
    cum_max = cum_ret.cummax()
    dd = (cum_ret - cum_max) / cum_max

    # Trades count
    n_trades = (position.diff().fillna(0) != 0).sum()

    # Sharpe
    sr = (net_ret.mean() / net_ret.std()) * np.sqrt(HOURS_PER_YEAR) if net_ret.std() > 0 else 0
    total_ret = (cum_ret.iloc[-1] - 1) * 100
    max_dd = dd.min() * 100
    pct_trade = can_trade.mean() * 100

    # Per-trade win rate
    trades_list = []
    entry_i = None
    for i in range(1, len(position)):
        if position.iloc[i] != 0 and position.iloc[i-1] == 0:
            entry_i = i
        elif position.iloc[i] == 0 and position.iloc[i-1] != 0 and entry_i is not None:
            tr_ret = net_ret.iloc[entry_i:i+1].sum()
            trades_list.append(tr_ret)
            entry_i = None

    trade_wr = (pd.Series(trades_list) > 0).mean() * 100 if trades_list else 0

    pair_metrics.append({
        "Pair": pair_name, "T1": t1, "T2": t2,
        "Return": total_ret, "Sharpe": sr, "MaxDD": max_dd,
        "Trades": n_trades, "TradeWR": trade_wr,
        "HL": zw, "Tradeable": pct_trade,
    })
    pair_equity_curves[pair_name] = cum_ret
    all_pair_positions[pair_name] = position
    all_pair_zscores[pair_name] = zs

    print(f"  {pair_name:12s}  Ret={total_ret:>+7.2f}%  SR={sr:>+6.2f}  "
          f"DD={max_dd:>6.2f}%  Trades={n_trades:>3}  WR={trade_wr:>5.1f}%  "
          f"HL={zw:>3}h  Trad={pct_trade:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 3: PORTAFOGLIO AGGREGATO (Equal Weight)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/5] Aggregazione portafoglio (equal weight)...")

# Allinea tutte le equity curves su un indice comune
all_eq = pd.DataFrame(pair_equity_curves)
all_eq = all_eq.dropna(how="all").fillna(method="ffill").fillna(1.0)

# Rendimento del portafoglio: media dei rendimenti di ogni coppia
pair_returns = all_eq.pct_change().fillna(0)
portfolio_returns = pair_returns.mean(axis=1)
portfolio_cum = (1 + portfolio_returns).cumprod()

# Portfolio metrics
port_mean = portfolio_returns.mean()
port_std = portfolio_returns.std()
port_sharpe = (port_mean / port_std) * np.sqrt(HOURS_PER_YEAR) if port_std > 0 else 0
port_cum_max = portfolio_cum.cummax()
port_dd = (portfolio_cum - port_cum_max) / port_cum_max
port_max_dd = port_dd.min()
port_total_ret = (portfolio_cum.iloc[-1] - 1) * 100
port_annual_ret = port_mean * HOURS_PER_YEAR * 100

# Win rate del portafoglio
port_wr = (portfolio_returns > 0).sum() / ((portfolio_returns != 0).sum()) * 100

# Profit factor
gp = portfolio_returns[portfolio_returns > 0].sum()
gl = abs(portfolio_returns[portfolio_returns < 0].sum())
port_pf = gp / gl if gl > 0 else np.inf

# Total trades
total_trades = sum(m["Trades"] for m in pair_metrics)

# Calmar
port_calmar = (port_mean * HOURS_PER_YEAR) / abs(port_max_dd) if port_max_dd != 0 else 0

# Sortino
ds = portfolio_returns[portfolio_returns < 0].std()
port_sortino = (port_mean / ds) * np.sqrt(HOURS_PER_YEAR) if ds > 0 else 0

metrics_df = pd.DataFrame(pair_metrics)

print("\n" + "=" * 80)
print(f"  RIEPILOGO PERFORMANCE V4 — Portafoglio {len(portfolio_pairs)} Coppie")
print("=" * 80)
print(f"  |")
print(f"  | === PORTAFOGLIO AGGREGATO ===")
print(f"  | Rendimento Totale (netto)  : {port_total_ret:>+10.2f} %")
print(f"  | Rendimento Annualizzato    : {port_annual_ret:>+10.2f} %")
print(f"  | Sharpe Ratio (ann.)        : {port_sharpe:>10.3f}")
print(f"  | Sortino Ratio              : {port_sortino:>10.3f}")
print(f"  | Calmar Ratio               : {port_calmar:>10.3f}")
print(f"  | Maximum Drawdown           : {port_max_dd*100:>10.2f} %")
print(f"  | Win Rate (per-bar)         : {port_wr:>10.1f} %")
print(f"  | Profit Factor              : {port_pf:>10.3f}")
print(f"  | Trades Totali (tutte)      : {total_trades:>10}")
print(f"  | Trades Medi per Coppia     : {total_trades/len(portfolio_pairs):>10.1f}")
print(f"  |")
print(f"  | === DETTAGLIO PER COPPIA ===")
print(f"  | {'Coppia':12s} {'Ret%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} "
      f"{'Trades':>8s} {'WR%':>8s} {'HL':>5s}")
print(f"  | {'-'*60}")
for _, m in metrics_df.iterrows():
    flag = " <<<" if m["Sharpe"] > 0.5 else ""
    print(f"  | {m['Pair']:12s} {m['Return']:>+8.2f} {m['Sharpe']:>+8.2f} "
          f"{m['MaxDD']:>8.2f} {m['Trades']:>8.0f} {m['TradeWR']:>8.1f} "
          f"{m['HL']:>5.0f}{flag}")
print(f"  | {'-'*60}")
print(f"  | MEDIA        {metrics_df['Return'].mean():>+8.2f} "
      f"{metrics_df['Sharpe'].mean():>+8.2f} "
      f"{metrics_df['MaxDD'].mean():>8.2f} "
      f"{metrics_df['Trades'].mean():>8.0f} "
      f"{metrics_df['TradeWR'].mean():>8.1f}")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 4: GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/5] Generazione grafici...")

# ── Grafico 1: Portfolio Equity + Singole coppie ──
fig, axes = plt.subplots(3, 1, figsize=(22, 18),
                         gridspec_kw={"height_ratios": [3, 2, 2]})
fig.suptitle(
    f"PAIRS TRADING V4 — Multi-Pair Portfolio ({len(portfolio_pairs)} coppie)  |  "
    f"Sharpe: {port_sharpe:.2f}  |  Return: {port_total_ret:+.1f}%  |  "
    f"MaxDD: {port_max_dd*100:.1f}%  |  Trades: {total_trades}",
    fontsize=15, fontweight="bold", color="#E0E0E0", y=0.98
)

# Pannello 1: Equity curves
ax1 = axes[0]
colors = ["#00E5FF", "#FF6B6B", "#69F0AE", "#FFA726", "#AB47BC",
          "#FF5252", "#64B5F6", "#FFD740", "#E040FB", "#00BFA5"]

for i, (pname, eq) in enumerate(pair_equity_curves.items()):
    ax1.plot(eq.index, eq, color=colors[i % len(colors)],
             alpha=0.4, linewidth=0.8, label=pname)

ax1.plot(portfolio_cum.index, portfolio_cum, color="white",
         linewidth=2.5, alpha=0.95, label=f"PORTFOLIO (SR={port_sharpe:.2f})")
ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
ax1.fill_between(portfolio_cum.index, 1, portfolio_cum,
                 where=(portfolio_cum >= 1), alpha=0.08, color="#00E5FF")
ax1.fill_between(portfolio_cum.index, 1, portfolio_cum,
                 where=(portfolio_cum < 1), alpha=0.08, color="#FF5252")
ax1.set_ylabel("Valore Cumulato ($1)")
ax1.set_title("Equity Curves — Singole Coppie + Portafoglio Aggregato",
              fontweight="bold", color="#AAAAAA")
ax1.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e",
           fontsize=9, ncol=3)

# Pannello 2: Z-Scores di tutte le coppie (overlay)
ax2 = axes[1]
for i, (pname, zs) in enumerate(all_pair_zscores.items()):
    ax2.plot(zs.index, zs, color=colors[i % len(colors)],
             alpha=0.35, linewidth=0.6, label=pname)

ax2.axhline(y=ENTRY_ZSCORE, color="#FF5252", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Entry (+/-{ENTRY_ZSCORE})")
ax2.axhline(y=-ENTRY_ZSCORE, color="#69F0AE", linestyle="--", linewidth=1.5,
            alpha=0.8)
ax2.axhline(y=0, color="#FFD740", linestyle="-", linewidth=0.8, alpha=0.4)
ax2.axhline(y=STOPLOSS_ZSCORE, color="#FF1744", linestyle="-.", linewidth=1.0,
            alpha=0.4, label=f"SL (+/-{STOPLOSS_ZSCORE})")
ax2.axhline(y=-STOPLOSS_ZSCORE, color="#FF1744", linestyle="-.", linewidth=1.0,
            alpha=0.4)
ax2.set_ylabel("Z-Score")
ax2.set_title("Z-Scores di Tutte le Coppie", fontweight="bold", color="#AAAAAA")
ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e",
           fontsize=8, ncol=4)
ax2.set_ylim(-5, 5)

# Pannello 3: Drawdown
ax3 = axes[2]
ax3.fill_between(port_dd.index, port_dd * 100, 0,
                 alpha=0.5, color="#FF5252", label="Portfolio Drawdown")
ax3.plot(port_dd.index, port_dd * 100, color="#FF8A80", linewidth=0.8)
ax3.axhline(y=port_max_dd * 100, color="#FF1744", linestyle="--",
            alpha=0.6, label=f"Max DD: {port_max_dd*100:.2f}%")
ax3.set_ylabel("Drawdown (%)")
ax3.set_xlabel("Data")
ax3.set_title("Portfolio Drawdown", fontweight="bold", color="#AAAAAA")
ax3.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("pairs_trading_v4_portfolio.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("  Salvato: pairs_trading_v4_portfolio.png")


# ── Grafico 2: Performance breakdown per coppia ──
fig2, (ax_ret, ax_sr) = plt.subplots(1, 2, figsize=(18, 7))

# Returns barplot
pair_names = metrics_df["Pair"].values
returns = metrics_df["Return"].values
bar_colors = ["#69F0AE" if r > 0 else "#FF5252" for r in returns]

ax_ret.barh(pair_names, returns, color=bar_colors, alpha=0.8, edgecolor="white",
            linewidth=0.5)
ax_ret.axvline(x=0, color="white", linestyle="-", alpha=0.3)
ax_ret.set_xlabel("Rendimento Totale (%)")
ax_ret.set_title("Rendimento per Coppia", fontweight="bold", color="#AAAAAA")

for i, (r, n) in enumerate(zip(returns, pair_names)):
    ax_ret.text(r + (0.3 if r >= 0 else -0.3), i, f"{r:+.1f}%",
                va="center", ha="left" if r >= 0 else "right",
                fontsize=10, color="white", fontweight="bold")

# Sharpe barplot
sharpes = metrics_df["Sharpe"].values
sr_colors = ["#69F0AE" if s > 0 else "#FF5252" for s in sharpes]

ax_sr.barh(pair_names, sharpes, color=sr_colors, alpha=0.8, edgecolor="white",
           linewidth=0.5)
ax_sr.axvline(x=0, color="white", linestyle="-", alpha=0.3)
ax_sr.set_xlabel("Sharpe Ratio")
ax_sr.set_title("Sharpe Ratio per Coppia", fontweight="bold", color="#AAAAAA")

for i, (s, n) in enumerate(zip(sharpes, pair_names)):
    ax_sr.text(s + (0.05 if s >= 0 else -0.05), i, f"{s:+.2f}",
               va="center", ha="left" if s >= 0 else "right",
               fontsize=10, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("pairs_trading_v4_breakdown.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()
print("  Salvato: pairs_trading_v4_breakdown.png")


# ── Grafico 3: Trades heatmap ──
fig3, ax_hm = plt.subplots(figsize=(14, 5))

metrics_cols = ["Return", "Sharpe", "MaxDD", "Trades", "TradeWR"]
hm_data = metrics_df[metrics_cols].values.astype(float)

# Normalize per colonna
hm_norm = np.zeros_like(hm_data)
for j in range(hm_data.shape[1]):
    col = hm_data[:, j]
    rng = col.max() - col.min()
    if rng > 0:
        hm_norm[:, j] = (col - col.min()) / rng
    else:
        hm_norm[:, j] = 0.5

colors_cmap = ["#FF1744", "#1a1a2e", "#00E676"]
cmap = LinearSegmentedColormap.from_list("custom", colors_cmap, N=256)

im = ax_hm.imshow(hm_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
ax_hm.set_xticks(range(len(metrics_cols)))
ax_hm.set_xticklabels(["Return%", "Sharpe", "MaxDD%", "Trades", "WinRate%"])
ax_hm.set_yticks(range(len(pair_names)))
ax_hm.set_yticklabels(pair_names)
ax_hm.set_title("Performance Matrix — Multi-Pair Portfolio",
                 fontweight="bold", color="#AAAAAA")

for i in range(len(pair_names)):
    for j in range(len(metrics_cols)):
        val = hm_data[i, j]
        fmt = f"{val:+.1f}" if j < 3 else (f"{val:.0f}" if j == 3 else f"{val:.0f}%")
        ax_hm.text(j, i, fmt, ha="center", va="center",
                   color="white", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("pairs_trading_v4_matrix.png", dpi=150, bbox_inches="tight",
            facecolor=fig3.get_facecolor())
plt.show()
print("  Salvato: pairs_trading_v4_matrix.png")


print("\n" + "=" * 80)
print("  BACKTEST V4 COMPLETATO")
print(f"  Trades totali: {total_trades}  |  SR portfolio: {port_sharpe:.3f}")
print("=" * 80)
