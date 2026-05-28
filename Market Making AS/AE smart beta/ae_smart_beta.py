#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  AUTOENCODER SMART BETA — Long-Only Overlay
================================================================================

  Evoluzione dell'AE Stat-Arb: invece di long/short market-neutral,
  usiamo i residui dell'autoencoder per PESARE un portafoglio long-only.

  La V2 ci ha dimostrato che:
    - Long leg:   +62% (batte B&H +48% di 14pp)
    - Short leg:  -32% (perde soldi in bull market)
    - Combinato:  -3%  (lo short distrugge il profitto)

  SOLUZIONE: rimaniamo long-only ma usiamo l'AE per decidere QUANTO
  comprare di ogni azione. Catturiamo il beta settoriale + l'alpha
  dell'AE stock picking.

  Meccanica:
    1. Autoencoder apprende la struttura cross-sezionale
    2. Residuo negativo = sottovalutata → SOVRAppeso
    3. Residuo positivo = sopravvalutata → SOTTOpeso (ma mai short!)
    4. Pesi normalizzati a somma 1 (fully invested)

  3 Portafogli a confronto:
    A) AE Smart Beta: pesi dinamici basati su AE residuo
    B) AE Top-K:      solo le K azioni piu' sottovalutate (concentrated)
    C) Benchmark:     equal-weight buy & hold

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
from datetime import datetime, timedelta
from scipy.stats import rankdata
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (18, 10), "font.family": "monospace", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12, "axes.grid": True,
    "grid.alpha": 0.15, "grid.linestyle": "--", "lines.linewidth": 1.2,
    "figure.dpi": 120,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════

UTILITIES = [
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC",
    "ES", "AWK", "ETR", "PPL", "FE", "AEE", "CMS", "CNP", "DTE", "EVRG",
    "ATO", "LNT", "NI", "PNW",
]

LOOKBACK_YEARS = 3

# Walk-forward
TRAIN_WINDOW   = 252
RETRAIN_EVERY  = 21

# Autoencoder
LATENT_DIM     = 5
HIDDEN_1       = 32
HIDDEN_2       = 16
DROPOUT        = 0.10
LR             = 5e-4
EPOCHS         = 100
BATCH_SIZE     = 32

# Portfolio construction
SIGNAL_SMOOTH  = 5          # Smoothing del segnale
REBALANCE_DAYS = 5          # Rebalance settimanale
TOP_K          = 5          # Per la versione concentrated
TILT_STRENGTH  = 2.0        # Forza del tilt (softmax temperature inverse)
MIN_WEIGHT     = 0.01       # Peso minimo per azione (1%)
MAX_WEIGHT     = 0.15       # Peso massimo per azione (15%)

TC_BPS         = 5
DAYS_PER_YEAR  = 252

print("=" * 80)
print("  AUTOENCODER SMART BETA — Long-Only Overlay")
print(f"  Universo: {len(UTILITIES)} Utilities  |  Latent: {LATENT_DIM}")
print(f"  Top-K: {TOP_K}  |  Tilt: {TILT_STRENGTH}  |  Rebalance: {REBALANCE_DAYS}d")
print(f"  Weight range: [{MIN_WEIGHT*100:.0f}%, {MAX_WEIGHT*100:.0f}%]  |  TC: {TC_BPS} bps")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/7] Download dati giornalieri...")

end_date   = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

all_data = {}
for ticker in UTILITIES:
    try:
        data = yf.download(ticker, start=start_date, end=end_date,
                          interval="1d", progress=False, auto_adjust=True)
        if len(data) > 200:
            all_data[ticker] = data["Close"].squeeze()
    except:
        pass

prices = pd.DataFrame(all_data).dropna()
tickers = list(prices.columns)
N = len(tickers)

print(f"  Titoli: {N}  |  Giorni: {len(prices):,}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/7] Preprocessing...")

log_returns = np.log(prices / prices.shift(1)).dropna()
market_ret  = log_returns.mean(axis=1)
excess_ret  = log_returns.sub(market_ret, axis=0)

ts_mean = excess_ret.rolling(60).mean()
ts_std  = excess_ret.rolling(60).std().replace(0, 1e-8)
normalized = ((excess_ret - ts_mean) / ts_std).dropna()

common_idx = normalized.index
log_returns = log_returns.loc[common_idx]
prices = prices.loc[common_idx]

print(f"  Shape: {normalized.shape}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AE(nn.Module):
    def __init__(self, n, h1, h2, lat, drop=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, h1), nn.BatchNorm1d(h1), nn.LeakyReLU(0.1), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.LeakyReLU(0.1), nn.Dropout(drop),
            nn.Linear(h2, lat),
        )
        self.dec = nn.Sequential(
            nn.Linear(lat, h2), nn.BatchNorm1d(h2), nn.LeakyReLU(0.1), nn.Dropout(drop),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.LeakyReLU(0.1),
            nn.Linear(h1, n),
        )
    def forward(self, x):
        return self.dec(self.enc(x))
    def encode(self, x):
        return self.enc(x)


def train_ae(model, data, epochs=100, lr=5e-4, bs=32):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    dl = DataLoader(TensorDataset(data), batch_size=bs, shuffle=True, drop_last=True)

    for _ in range(epochs):
        for (b,) in dl:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(model(b), b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        full_loss = crit(model(data.to(device)), data.to(device)).item()
    return full_loss


# ══════════════════════════════════════════════════════════════════════════════
#  4. WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/7] Walk-Forward Backtest...")

norm_arr = normalized.values
raw_ret  = log_returns.values
dates    = normalized.index

start_idx = TRAIN_WINDOW

# Allocazione per 3 strategie
weights_smart = np.zeros_like(raw_ret)    # AE Smart Beta (tutti, pesati)
weights_topk  = np.zeros_like(raw_ret)    # AE Top-K (concentrated)
weights_ew    = np.ones_like(raw_ret) / N # Equal weight (benchmark)

model = None
residual_buffer = []
train_losses = []
retrain_dates = []

current_w_smart = np.ones(N) / N  # Start equal weight
current_w_topk  = np.ones(N) / N
last_rebalance  = -REBALANCE_DAYS
n_retrains = 0

for t in range(start_idx, len(norm_arr)):
    # ── Retrain ──
    if (t - start_idx) % RETRAIN_EVERY == 0:
        ts = max(0, t - TRAIN_WINDOW)
        train_t = torch.FloatTensor(norm_arr[ts:t])
        model = AE(N, HIDDEN_1, HIDDEN_2, LATENT_DIM, DROPOUT).to(device)
        loss = train_ae(model, train_t, EPOCHS, LR, BATCH_SIZE)
        train_losses.append(loss)
        retrain_dates.append(dates[t])
        n_retrains += 1
        if n_retrains % 5 == 0:
            print(f"    Retrain #{n_retrains} @ {dates[t].strftime('%Y-%m-%d')} "
                  f" Loss: {loss:.6f}")

    # ── Residui ──
    if model is not None:
        with torch.no_grad():
            x = torch.FloatTensor(norm_arr[t:t+1]).to(device)
            x_hat = model(x)
            residual = (x - x_hat).cpu().numpy().flatten()

        residual_buffer.append(residual)
        if len(residual_buffer) > SIGNAL_SMOOTH:
            residual_buffer.pop(0)

        # ── Rebalance ──
        if t - last_rebalance >= REBALANCE_DAYS and len(residual_buffer) >= SIGNAL_SMOOTH:
            smooth_res = np.mean(residual_buffer, axis=0)

            # ════════════════════════════════════════════════════════════
            #  STRATEGIA A: AE SMART BETA (Tilted Weights)
            # ════════════════════════════════════════════════════════════
            #
            #  Idea: residuo negativo → sottovalutata → piu' peso
            #        residuo positivo → sopravvalutata → meno peso
            #
            #  Usiamo -residuo * tilt_strength come logit per softmax:
            #    w_i = softmax(-residual_i * TILT_STRENGTH)
            #
            #  Softmax garantisce che i pesi siano positivi e sommino a 1.
            #  TILT_STRENGTH controlla quanto aggressivo e' il tilt.
            # ──────────────────────────────────────────────────────────────

            logits = -smooth_res * TILT_STRENGTH
            w = softmax(logits)

            # Clamp pesi
            w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
            w = w / w.sum()  # Rinormalizza dopo clamp

            current_w_smart = w

            # ════════════════════════════════════════════════════════════
            #  STRATEGIA B: AE TOP-K (Concentrated)
            # ════════════════════════════════════════════════════════════
            #
            #  Seleziona le TOP_K azioni con residuo piu' negativo
            #  (le piu' sottovalutate) e investi equal-weight solo su quelle.
            # ──────────────────────────────────────────────────────────────

            top_k_idx = np.argsort(smooth_res)[:TOP_K]
            w_topk = np.zeros(N)
            w_topk[top_k_idx] = 1.0 / TOP_K

            current_w_topk = w_topk

            last_rebalance = t

    weights_smart[t] = current_w_smart
    weights_topk[t]  = current_w_topk

print(f"    Retrains totali: {n_retrains}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. P&L PER OGNI STRATEGIA
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/7] Calcolo P&L per 3 strategie...")

tc_rate = TC_BPS / 10000

def calc_pnl(weights, raw_returns, start, tc_rate):
    """Calcola P&L di un portafoglio pesato con TC."""
    n_t = len(raw_returns)
    ret = np.zeros(n_t)
    tc_total = 0
    for t in range(start + 1, n_t):
        daily = np.sum(weights[t-1] * raw_returns[t])
        tc = np.sum(np.abs(weights[t] - weights[t-1])) * tc_rate
        ret[t] = daily - tc
        tc_total += tc
    return ret, tc_total

pnl_smart, tc_smart = calc_pnl(weights_smart, raw_ret, start_idx, tc_rate)
pnl_topk,  tc_topk  = calc_pnl(weights_topk,  raw_ret, start_idx, tc_rate)
pnl_ew,    tc_ew    = calc_pnl(weights_ew,    raw_ret, start_idx, 0)  # B&H no TC

# Series
ret_smart = pd.Series(pnl_smart, index=dates).iloc[start_idx:]
ret_topk  = pd.Series(pnl_topk,  index=dates).iloc[start_idx:]
ret_ew    = pd.Series(pnl_ew,    index=dates).iloc[start_idx:]

cum_smart = (1 + ret_smart).cumprod()
cum_topk  = (1 + ret_topk).cumprod()
cum_ew    = (1 + ret_ew).cumprod()


# ══════════════════════════════════════════════════════════════════════════════
#  6. METRICHE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/7] Metriche...")

def calc_metrics(ret_series, cum_series, name):
    """Calcola tutte le metriche per una strategia."""
    m = ret_series.mean()
    s = ret_series.std()
    sharpe = (m / s) * np.sqrt(DAYS_PER_YEAR) if s > 0 else 0
    cm = cum_series.cummax()
    dd = (cum_series - cm) / cm
    mdd = dd.min()
    tot = (cum_series.iloc[-1] - 1) * 100
    ann = m * DAYS_PER_YEAR * 100

    gp = ret_series[ret_series > 0].sum()
    gl = abs(ret_series[ret_series < 0].sum())
    pf = gp / gl if gl > 0 else np.inf

    wr = (ret_series > 0).sum() / (ret_series != 0).sum() * 100

    dsr = ret_series[ret_series < 0].std()
    sortino = (m / dsr) * np.sqrt(DAYS_PER_YEAR) if dsr > 0 else 0
    calmar = (m * DAYS_PER_YEAR) / abs(mdd) if mdd != 0 else 0

    return {
        "Name": name, "TotalRet": tot, "AnnRet": ann, "Sharpe": sharpe,
        "Sortino": sortino, "Calmar": calmar, "MaxDD": mdd * 100,
        "WinRate": wr, "ProfitFactor": pf, "Drawdown": dd,
    }

m_smart = calc_metrics(ret_smart, cum_smart, "AE Smart Beta")
m_topk  = calc_metrics(ret_topk,  cum_topk,  f"AE Top-{TOP_K}")
m_ew    = calc_metrics(ret_ew,    cum_ew,    "Equal Weight B&H")

# Active return (alpha)
alpha_smart = m_smart["TotalRet"] - m_ew["TotalRet"]
alpha_topk  = m_topk["TotalRet"]  - m_ew["TotalRet"]

# Information Ratio
track_err_smart = (ret_smart - ret_ew).std() * np.sqrt(DAYS_PER_YEAR)
ir_smart = (m_smart["AnnRet"] - m_ew["AnnRet"]) / (track_err_smart * 100) if track_err_smart > 0 else 0
track_err_topk = (ret_topk - ret_ew).std() * np.sqrt(DAYS_PER_YEAR)
ir_topk = (m_topk["AnnRet"] - m_ew["AnnRet"]) / (track_err_topk * 100) if track_err_topk > 0 else 0

# Turnover
to_smart = np.abs(np.diff(weights_smart[start_idx:], axis=0)).sum(axis=1).mean()
to_topk  = np.abs(np.diff(weights_topk[start_idx:], axis=0)).sum(axis=1).mean()

# Rebalance count
reb_count = (np.abs(np.diff(weights_smart[start_idx:], axis=0)).sum(axis=1) > 0.01).sum()

print("\n" + "=" * 80)
print("  CONFRONTO STRATEGIE — Long-Only Overlay")
print("=" * 80)
print(f"  |")
print(f"  | {'Metrica':30s} {'AE Smart Beta':>15s} {'AE Top-'+str(TOP_K):>15s} {'EW B&H':>15s}")
print(f"  | {'-'*75}")
print(f"  | {'Return Totale':30s} {m_smart['TotalRet']:>+14.2f}% {m_topk['TotalRet']:>+14.2f}% {m_ew['TotalRet']:>+14.2f}%")
print(f"  | {'Return Annualizzato':30s} {m_smart['AnnRet']:>+14.2f}% {m_topk['AnnRet']:>+14.2f}% {m_ew['AnnRet']:>+14.2f}%")
print(f"  | {'Sharpe Ratio':30s} {m_smart['Sharpe']:>15.3f} {m_topk['Sharpe']:>15.3f} {m_ew['Sharpe']:>15.3f}")
print(f"  | {'Sortino Ratio':30s} {m_smart['Sortino']:>15.3f} {m_topk['Sortino']:>15.3f} {m_ew['Sortino']:>15.3f}")
print(f"  | {'Calmar Ratio':30s} {m_smart['Calmar']:>15.3f} {m_topk['Calmar']:>15.3f} {m_ew['Calmar']:>15.3f}")
print(f"  | {'Maximum Drawdown':30s} {m_smart['MaxDD']:>14.2f}% {m_topk['MaxDD']:>14.2f}% {m_ew['MaxDD']:>14.2f}%")
print(f"  | {'Win Rate':30s} {m_smart['WinRate']:>14.1f}% {m_topk['WinRate']:>14.1f}% {m_ew['WinRate']:>14.1f}%")
print(f"  | {'Profit Factor':30s} {m_smart['ProfitFactor']:>15.3f} {m_topk['ProfitFactor']:>15.3f} {m_ew['ProfitFactor']:>15.3f}")
print(f"  | {'-'*75}")
print(f"  | {'Alpha vs B&H':30s} {alpha_smart:>+14.2f}% {alpha_topk:>+14.2f}%")
print(f"  | {'Information Ratio':30s} {ir_smart:>15.3f} {ir_topk:>15.3f}")
print(f"  | {'Turnover medio/d':30s} {to_smart:>15.3f} {to_topk:>15.3f}")
print(f"  | {'TC totali':30s} {tc_smart*100:>14.3f}% {tc_topk*100:>14.3f}%")
print(f"  | {'Ribilanciamenti':30s} {reb_count:>15}")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  7. GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/7] Grafici...")

fig, axes = plt.subplots(4, 1, figsize=(22, 24),
                         gridspec_kw={"height_ratios": [3, 2, 2, 2]})

fig.suptitle(
    f"AUTOENCODER SMART BETA  |  {N} Utilities  |  "
    f"Smart Beta: {m_smart['TotalRet']:+.1f}%  |  "
    f"Top-{TOP_K}: {m_topk['TotalRet']:+.1f}%  |  "
    f"B&H: {m_ew['TotalRet']:+.1f}%",
    fontsize=14, fontweight="bold", color="#E0E0E0", y=0.98
)

# ── Pannello 1: Equity Curves ──
ax1 = axes[0]
ax1.plot(cum_smart.index, cum_smart, color="#00E5FF", linewidth=2.5,
         alpha=0.95, label=f"AE Smart Beta (SR={m_smart['Sharpe']:.2f})")
ax1.plot(cum_topk.index, cum_topk, color="#69F0AE", linewidth=2.0,
         alpha=0.85, label=f"AE Top-{TOP_K} (SR={m_topk['Sharpe']:.2f})")
ax1.plot(cum_ew.index, cum_ew, color="#FF6B6B", linewidth=1.5,
         alpha=0.7, linestyle="--", label=f"EW B&H (SR={m_ew['Sharpe']:.2f})")

ax1.fill_between(cum_smart.index, cum_ew, cum_smart,
                 where=(cum_smart > cum_ew), alpha=0.08, color="#00E5FF",
                 label="Alpha (Smart Beta)")
ax1.fill_between(cum_smart.index, cum_ew, cum_smart,
                 where=(cum_smart < cum_ew), alpha=0.08, color="#FF5252")

ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
ax1.set_ylabel("Cumulative ($1)")
ax1.set_title("Equity Curves — AE Smart Beta vs Benchmark",
              fontweight="bold", color="#AAAAAA")
ax1.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e", fontsize=10)

# ── Pannello 2: Active Return (Rolling) ──
ax2 = axes[1]
rolling_alpha_smart = (ret_smart - ret_ew).rolling(63).sum() * 100
rolling_alpha_topk  = (ret_topk - ret_ew).rolling(63).sum() * 100

ax2.plot(rolling_alpha_smart.index, rolling_alpha_smart, color="#00E5FF",
         linewidth=1.2, alpha=0.8, label="Smart Beta alpha (3m rolling)")
ax2.plot(rolling_alpha_topk.index, rolling_alpha_topk, color="#69F0AE",
         linewidth=1.2, alpha=0.8, label=f"Top-{TOP_K} alpha (3m rolling)")
ax2.axhline(y=0, color="#FFD740", linestyle="-", alpha=0.5)

ax2.fill_between(rolling_alpha_smart.index, 0, rolling_alpha_smart,
                 where=(rolling_alpha_smart > 0), alpha=0.1, color="#00E5FF")
ax2.fill_between(rolling_alpha_smart.index, 0, rolling_alpha_smart,
                 where=(rolling_alpha_smart < 0), alpha=0.1, color="#FF5252")
ax2.set_ylabel("Alpha (%)")
ax2.set_title("Active Return vs Benchmark (Rolling 3 Mesi)",
              fontweight="bold", color="#AAAAAA")
ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 3: Peso Herfindahl (concentrazione) ──
ax3 = axes[2]
w_smart_df = pd.DataFrame(weights_smart[start_idx:], index=dates[start_idx:],
                            columns=tickers)
# Top 5 azioni per peso medio
top_5_avg = w_smart_df.mean().nlargest(5).index
colors_w = ["#00E5FF", "#69F0AE", "#FFA726", "#AB47BC", "#FF6B6B"]

for i, tk in enumerate(top_5_avg):
    ax3.fill_between(w_smart_df.index, 0, w_smart_df[tk] * 100,
                     alpha=0.3, color=colors_w[i], label=tk)
    ax3.plot(w_smart_df.index, w_smart_df[tk] * 100,
             color=colors_w[i], linewidth=0.8, alpha=0.7)

ew_line = 100 / N
ax3.axhline(y=ew_line, color="#FFD740", linestyle="--", alpha=0.5,
            label=f"Equal Weight ({ew_line:.1f}%)")
ax3.set_ylabel("Peso (%)")
ax3.set_title("Pesi del Portafoglio Smart Beta (Top 5 holding)",
              fontweight="bold", color="#AAAAAA")
ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e",
           fontsize=9, ncol=3)

# ── Pannello 4: Drawdown comparativo ──
ax4 = axes[3]
dd_smart = m_smart["Drawdown"]
dd_topk  = m_topk["Drawdown"]
dd_ew    = m_ew["Drawdown"]

ax4.fill_between(dd_smart.index, dd_smart * 100, 0, alpha=0.3,
                 color="#00E5FF", label=f"Smart Beta (MaxDD={m_smart['MaxDD']:.1f}%)")
ax4.fill_between(dd_topk.index, dd_topk * 100, 0, alpha=0.3,
                 color="#69F0AE", label=f"Top-{TOP_K} (MaxDD={m_topk['MaxDD']:.1f}%)")
ax4.plot(dd_ew.index, dd_ew * 100, color="#FF6B6B", linewidth=1.0,
         alpha=0.7, linestyle="--", label=f"B&H (MaxDD={m_ew['MaxDD']:.1f}%)")
ax4.set_ylabel("Drawdown (%)")
ax4.set_xlabel("Data")
ax4.set_title("Drawdown Comparativo", fontweight="bold", color="#AAAAAA")
ax4.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("ae_smart_beta.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("  Salvato: ae_smart_beta.png")


# ── Stock-Level Analysis ──
print(f"\n[7/7] Analisi per azione...")

fig2, (ax_w, ax_r) = plt.subplots(1, 2, figsize=(20, 8))

# Average weights
avg_w = w_smart_df.mean().sort_values(ascending=True) * 100
ew_w = 100 / N

colors_aw = ["#00E5FF" if w > ew_w else "#FF6B6B" for w in avg_w.values]
ax_w.barh(avg_w.index, avg_w.values, color=colors_aw, alpha=0.8,
          edgecolor="white", linewidth=0.5)
ax_w.axvline(x=ew_w, color="#FFD740", linestyle="--", linewidth=1.5,
             alpha=0.7, label=f"Equal Weight ({ew_w:.1f}%)")
ax_w.set_xlabel("Peso Medio (%)")
ax_w.set_title("Peso Medio Smart Beta vs Equal Weight",
               fontweight="bold", color="#AAAAAA")
ax_w.legend(framealpha=0.3, facecolor="#1a1a2e")

for i, (nm, v) in enumerate(avg_w.items()):
    ax_w.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9, color="white")

# Cumulative return di ogni azione
stock_cum = (1 + log_returns.iloc[start_idx:]).cumprod()
stock_total = (stock_cum.iloc[-1] - 1).sort_values() * 100
colors_sr = ["#69F0AE" if r > stock_total.median() else "#FF5252"
             for r in stock_total.values]
ax_r.barh(stock_total.index, stock_total.values, color=colors_sr,
          alpha=0.8, edgecolor="white", linewidth=0.5)
ax_r.axvline(x=stock_total.median(), color="#FFD740", linestyle="--",
             alpha=0.7, label=f"Mediana ({stock_total.median():.1f}%)")
ax_r.set_xlabel("Return Totale (%)")
ax_r.set_title("Return per Azione nel Periodo OOS",
               fontweight="bold", color="#AAAAAA")
ax_r.legend(framealpha=0.3, facecolor="#1a1a2e")

for i, (nm, v) in enumerate(stock_total.items()):
    ax_r.text(v + 0.5, i, f"{v:+.0f}%", va="center", fontsize=9, color="white")

plt.tight_layout()
plt.savefig("ae_smart_beta_stocks.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()
print("  Salvato: ae_smart_beta_stocks.png")

# ── Correlation della strategia con il benchmark ──
corr_smart = ret_smart.corr(ret_ew)
corr_topk  = ret_topk.corr(ret_ew)
beta_smart = ret_smart.cov(ret_ew) / ret_ew.var()
beta_topk  = ret_topk.cov(ret_ew) / ret_ew.var()

print(f"\n  Correlazione con benchmark:")
print(f"    Smart Beta: {corr_smart:.3f}  |  Beta: {beta_smart:.3f}")
print(f"    Top-{TOP_K}:      {corr_topk:.3f}  |  Beta: {beta_topk:.3f}")

print("\n" + "=" * 80)
print("  AUTOENCODER SMART BETA COMPLETATO")
print("=" * 80)
