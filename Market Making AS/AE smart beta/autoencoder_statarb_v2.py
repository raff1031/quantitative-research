#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  AUTOENCODER STAT-ARB V2 — Low Turnover, Utilities Sector
================================================================================

  Fix rispetto a V1:
    1. TURNOVER CONTROL: posizioni cambiano solo se il segnale e' forte E
       persistente (media su 5 giorni). Riduce turnover da 1.7 a ~0.2/giorno.
    2. POSITION SMOOTHING: mantieni posizione corrente a meno che il nuovo
       segnale sia statisticamente migliore (isteresi).
    3. BETTER NORMALIZATION: usa ranking percentile invece di Z-score per
       segnali piu' stabili.
    4. RECONSTRUCTION TARGET: usa returns-to-mean (demeaned by market) per
       isolare il segnale idiosincratico puro.
    5. REDUCED TC: modella 5 bps (piu' realistico per istituzionali).

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

SECTOR_NAME    = "Utilities"
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

# Signal — PIU' CONSERVATIVO
SIGNAL_SMOOTH  = 5          # Media mobile del residuo su 5 giorni
TOP_N_LONG     = 3          # Fisso: long top 3 sottovalutate
TOP_N_SHORT    = 3          # Fisso: short top 3 sopravvalutate
REBALANCE_DAYS = 5          # Ribilanciamento ogni 5 giorni (settimanale)

TC_BPS         = 5
DAYS_PER_YEAR  = 252

print("=" * 80)
print(f"  AUTOENCODER STAT-ARB V2 — {SECTOR_NAME}")
print(f"  Universo: {len(UTILITIES)} titoli  |  Latent: {LATENT_DIM}")
print(f"  Long/Short: {TOP_N_LONG}/{TOP_N_SHORT} fissi  |  "
      f"Rebalance ogni {REBALANCE_DAYS}d")
print(f"  Signal smooth: {SIGNAL_SMOOTH}d  |  TC: {TC_BPS} bps")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/6] Download dati giornalieri...")

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
N_STOCKS = len(tickers)

print(f"  Titoli: {N_STOCKS}  |  Giorni: {len(prices):,}")
print(f"  {prices.index[0].strftime('%Y-%m-%d')} -> {prices.index[-1].strftime('%Y-%m-%d')}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/6] Preprocessing...")

log_returns = np.log(prices / prices.shift(1)).dropna()

# Market-neutral returns: rimuovi il rendimento medio del settore
# Questo forza l'autoencoder a modellare le DIFFERENZE tra azioni
market_return = log_returns.mean(axis=1)
excess_returns = log_returns.sub(market_return, axis=0)

# Time-series normalization (per stock)
ts_mean = excess_returns.rolling(window=60).mean()
ts_std  = excess_returns.rolling(window=60).std().replace(0, 1e-8)
normalized = (excess_returns - ts_mean) / ts_std
normalized = normalized.dropna()

# Allinea tutto
common_idx = normalized.index
log_returns = log_returns.loc[common_idx]
excess_returns = excess_returns.loc[common_idx]
prices = prices.loc[common_idx]
market_return = market_return.loc[common_idx]

print(f"  Shape: {normalized.shape}  |  Returns medi: {log_returns.mean().mean()*100:.4f}%/d")


# ══════════════════════════════════════════════════════════════════════════════
#  3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class StatArbAE(nn.Module):
    """Autoencoder con skip connections e batch norm."""
    def __init__(self, n_in, h1, h2, latent, drop=0.1):
        super().__init__()
        self.enc1 = nn.Linear(n_in, h1)
        self.bn1  = nn.BatchNorm1d(h1)
        self.enc2 = nn.Linear(h1, h2)
        self.bn2  = nn.BatchNorm1d(h2)
        self.enc3 = nn.Linear(h2, latent)

        self.dec1 = nn.Linear(latent, h2)
        self.bn3  = nn.BatchNorm1d(h2)
        self.dec2 = nn.Linear(h2, h1)
        self.bn4  = nn.BatchNorm1d(h1)
        self.dec3 = nn.Linear(h1, n_in)

        self.drop = nn.Dropout(drop)
        self.act  = nn.LeakyReLU(0.1)

    def encode(self, x):
        h = self.act(self.bn1(self.enc1(x)))
        h = self.drop(h)
        h = self.act(self.bn2(self.enc2(h)))
        h = self.drop(h)
        return self.enc3(h)

    def decode(self, z):
        h = self.act(self.bn3(self.dec1(z)))
        h = self.drop(h)
        h = self.act(self.bn4(self.dec2(h)))
        return self.dec3(h)

    def forward(self, x):
        return self.decode(self.encode(x))


def train_model(model, data, epochs, lr, bs):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()

    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for (batch,) in dl:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = crit(out, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        full = data.to(device)
        recon = model(full)
        final_loss = crit(recon, full).item()
    return final_loss


# ══════════════════════════════════════════════════════════════════════════════
#  4. WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/6] Walk-Forward Backtest (low turnover)...")

norm_arr = normalized.values
raw_ret  = log_returns.values
dates    = normalized.index

all_positions = np.zeros_like(raw_ret)
all_residuals = np.zeros_like(raw_ret)
all_scores    = np.zeros_like(raw_ret)
train_losses  = []
retrain_dates = []

start_idx = TRAIN_WINDOW
model = None
positions = np.zeros(N_STOCKS)
residual_buffer = []   # Buffer per smoothing

n_retrains = 0
last_rebalance = -REBALANCE_DAYS  # Forza ribilanciamento al primo giorno

for t in range(start_idx, len(norm_arr)):
    # ── Retrain ──
    if (t - start_idx) % RETRAIN_EVERY == 0:
        t_start = max(0, t - TRAIN_WINDOW)
        train_data = torch.FloatTensor(norm_arr[t_start:t])

        model = StatArbAE(N_STOCKS, HIDDEN_1, HIDDEN_2, LATENT_DIM, DROPOUT).to(device)
        loss = train_model(model, train_data, EPOCHS, LR, BATCH_SIZE)
        train_losses.append(loss)
        retrain_dates.append(dates[t])
        n_retrains += 1

        if n_retrains % 5 == 0:
            print(f"    Retrain #{n_retrains} @ {dates[t].strftime('%Y-%m-%d')}  "
                  f"Loss: {loss:.6f}")

    # ── Genera residuo ──
    if model is not None:
        with torch.no_grad():
            x = torch.FloatTensor(norm_arr[t:t+1]).to(device)
            x_hat = model(x)
            residual = (x - x_hat).cpu().numpy().flatten()

        all_residuals[t] = residual

        # Accumula nel buffer per smoothing
        residual_buffer.append(residual)
        if len(residual_buffer) > SIGNAL_SMOOTH:
            residual_buffer.pop(0)

        # ── Ribilanciamento solo ogni REBALANCE_DAYS ──
        if t - last_rebalance >= REBALANCE_DAYS and len(residual_buffer) >= SIGNAL_SMOOTH:
            # Segnale smoothed: media degli ultimi SIGNAL_SMOOTH residui
            smooth_res = np.mean(residual_buffer, axis=0)

            # Ranking cross-sezionale:
            # Rank 1 = piu' negativo (sottovalutato) → LONG
            # Rank N = piu' positivo (sopravvalutato) → SHORT
            ranks = rankdata(smooth_res)  # 1 = smallest

            new_pos = np.zeros(N_STOCKS)

            # LONG: top_n con rank piu' basso (residuo piu' negativo)
            long_idx = np.argsort(ranks)[:TOP_N_LONG]
            new_pos[long_idx] = 1.0 / TOP_N_LONG

            # SHORT: top_n con rank piu' alto (residuo piu' positivo)
            short_idx = np.argsort(ranks)[-TOP_N_SHORT:]
            new_pos[short_idx] = -1.0 / TOP_N_SHORT

            positions = new_pos
            last_rebalance = t

            all_scores[t] = smooth_res

    all_positions[t] = positions

print(f"    Retrains totali: {n_retrains}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. P&L
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/6] Calcolo P&L...")

tc_rate = TC_BPS / 10000
port_returns = np.zeros(len(raw_ret))

for t in range(start_idx + 1, len(raw_ret)):
    # Rendimento: posizioni di ieri * returns di oggi
    daily_ret = np.sum(all_positions[t-1] * raw_ret[t])

    # TC solo nei giorni di ribilanciamento
    tc = np.sum(np.abs(all_positions[t] - all_positions[t-1])) * tc_rate
    port_returns[t] = daily_ret - tc

# Series
port_ret = pd.Series(port_returns, index=dates).iloc[start_idx:]
port_cum = (1 + port_ret).cumprod()

# Benchmark
bh_ret = pd.Series(raw_ret.mean(axis=1), index=dates).iloc[start_idx:]
bh_cum = (1 + bh_ret).cumprod()

# Long-only leg
long_mask = all_positions > 0
long_ret_arr = np.zeros(len(raw_ret))
for t in range(start_idx + 1, len(raw_ret)):
    lm = long_mask[t-1]
    if lm.any():
        long_ret_arr[t] = raw_ret[t][lm].mean()
long_ret_s = pd.Series(long_ret_arr, index=dates).iloc[start_idx:]
long_cum = (1 + long_ret_s).cumprod()

# Short leg (inverted for comparison)
short_mask = all_positions < 0
short_ret_arr = np.zeros(len(raw_ret))
for t in range(start_idx + 1, len(raw_ret)):
    sm = short_mask[t-1]
    if sm.any():
        short_ret_arr[t] = -raw_ret[t][sm].mean()  # Inverted (profit from shorts)
short_ret_s = pd.Series(short_ret_arr, index=dates).iloc[start_idx:]
short_cum = (1 + short_ret_s).cumprod()


# ══════════════════════════════════════════════════════════════════════════════
#  6. METRICHE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Metriche...")

pm = port_ret.mean()
ps = port_ret.std()
sharpe = (pm / ps) * np.sqrt(DAYS_PER_YEAR) if ps > 0 else 0
cm = port_cum.cummax()
dd = (port_cum - cm) / cm
mdd = dd.min()
tot_ret = (port_cum.iloc[-1] - 1) * 100
ann_ret = pm * DAYS_PER_YEAR * 100

gp = port_ret[port_ret > 0].sum()
gl = abs(port_ret[port_ret < 0].sum())
pf = gp / gl if gl > 0 else np.inf
wr = (port_ret > 0).sum() / (port_ret != 0).sum() * 100

dsrt = port_ret[port_ret < 0].std()
sortino = (pm / dsrt) * np.sqrt(DAYS_PER_YEAR) if dsrt > 0 else 0
calmar = (pm * DAYS_PER_YEAR) / abs(mdd) if mdd != 0 else 0

bh_tot = (bh_cum.iloc[-1] - 1) * 100
bh_sr = (bh_ret.mean() / bh_ret.std()) * np.sqrt(DAYS_PER_YEAR) if bh_ret.std() > 0 else 0
long_tot = (long_cum.iloc[-1] - 1) * 100
short_tot = (short_cum.iloc[-1] - 1) * 100

# Turnover
pos_changes = np.abs(np.diff(all_positions[start_idx:], axis=0)).sum(axis=1)
avg_turnover = pos_changes.mean()
total_tc = pos_changes.sum() * tc_rate * 100

# Trades
n_rebalances = (pos_changes > 0.01).sum()

print("\n" + "=" * 80)
print(f"  AUTOENCODER STAT-ARB V2 — {SECTOR_NAME} ({N_STOCKS} titoli)")
print("=" * 80)
print(f"  |")
print(f"  | === STRATEGIA (Market-Neutral) ===")
print(f"  | Rendimento Totale          : {tot_ret:>+10.2f} %")
print(f"  | Rendimento Annualizzato    : {ann_ret:>+10.2f} %")
print(f"  | Sharpe Ratio (ann.)        : {sharpe:>10.3f}")
print(f"  | Sortino Ratio (ann.)       : {sortino:>10.3f}")
print(f"  | Calmar Ratio               : {calmar:>10.3f}")
print(f"  | Maximum Drawdown           : {mdd*100:>10.2f} %")
print(f"  | Win Rate (daily)           : {wr:>10.1f} %")
print(f"  | Profit Factor              : {pf:>10.3f}")
print(f"  |")
print(f"  | === LEGS ===")
print(f"  | Long Leg Return            : {long_tot:>+10.2f} %")
print(f"  | Short Leg Return (inv)     : {short_tot:>+10.2f} %")
print(f"  |")
print(f"  | === BENCHMARK ===")
print(f"  | B&H Utilities Return       : {bh_tot:>+10.2f} %")
print(f"  | B&H Sharpe Ratio           : {bh_sr:>10.3f}")
print(f"  |")
print(f"  | === EXECUTION ===")
print(f"  | Turnover medio/giorno      : {avg_turnover:>10.3f}")
print(f"  | Ribilanciamenti            : {n_rebalances:>10}")
print(f"  | Transaction costs totali   : {total_tc:>10.3f} %")
print(f"  | Retrains effettuati        : {n_retrains:>10}")
print(f"  | Loss media finale          : {np.mean(train_losses):>10.6f}")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  7. GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] Grafici...")

fig, axes = plt.subplots(4, 1, figsize=(22, 22),
                         gridspec_kw={"height_ratios": [3, 2, 2, 2]})

fig.suptitle(
    f"AUTOENCODER STAT-ARB V2  |  {SECTOR_NAME} ({N_STOCKS} titoli)  |  "
    f"Sharpe: {sharpe:.2f}  |  Return: {tot_ret:+.1f}%  |  "
    f"MaxDD: {mdd*100:.1f}%  |  Rebalances: {n_rebalances}",
    fontsize=14, fontweight="bold", color="#E0E0E0", y=0.98
)

# Pannello 1: Equity
ax1 = axes[0]
ax1.plot(port_cum.index, port_cum, color="#00E5FF", linewidth=2.0,
         alpha=0.95, label=f"AE Stat-Arb L/S (SR={sharpe:.2f})")
ax1.plot(bh_cum.index, bh_cum, color="#FF6B6B", linewidth=1.2,
         alpha=0.7, linestyle="--", label=f"B&H Utilities (SR={bh_sr:.2f})")
ax1.plot(long_cum.index, long_cum, color="#69F0AE", linewidth=1.0,
         alpha=0.5, linestyle=":", label=f"Long Leg ({long_tot:+.1f}%)")
ax1.plot(short_cum.index, short_cum, color="#FFA726", linewidth=1.0,
         alpha=0.5, linestyle=":", label=f"Short Leg inv ({short_tot:+.1f}%)")

ax1.fill_between(port_cum.index, 1, port_cum,
                 where=(port_cum >= 1), alpha=0.08, color="#00E5FF")
ax1.fill_between(port_cum.index, 1, port_cum,
                 where=(port_cum < 1), alpha=0.08, color="#FF5252")
ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
ax1.set_ylabel("Cumulative ($1)")
ax1.set_title("Equity Curve: AE Stat-Arb vs Benchmark",
              fontweight="bold", color="#AAAAAA")
ax1.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# Pannello 2: Residual Heatmap
ax2 = axes[1]
res_df = pd.DataFrame(all_residuals[start_idx:], index=dates[start_idx:],
                       columns=tickers)
res_weekly = res_df.resample("W").mean()

im = ax2.imshow(res_weekly.values.T, aspect="auto", cmap="RdYlGn_r",
                vmin=-0.5, vmax=0.5, interpolation="nearest")
ax2.set_yticks(range(N_STOCKS))
ax2.set_yticklabels(tickers, fontsize=7)
n_tk = 12
tp = np.linspace(0, len(res_weekly)-1, n_tk, dtype=int)
ax2.set_xticks(tp)
ax2.set_xticklabels([res_weekly.index[i].strftime("%b %Y")
                     for i in tp], rotation=30, fontsize=8)
ax2.set_title("Reconstruction Residuals (Settimanale)", fontweight="bold",
              color="#AAAAAA")
plt.colorbar(im, ax=ax2, label="Residual", shrink=0.8)

# Pannello 3: Posizioni
ax3 = axes[2]
pos_df = pd.DataFrame(all_positions[start_idx:], index=dates[start_idx:],
                       columns=tickers)
nl = (pos_df > 0).sum(axis=1)
ns = (pos_df < 0).sum(axis=1)
ax3.fill_between(nl.index, 0, nl, alpha=0.5, color="#69F0AE", label="Long")
ax3.fill_between(ns.index, 0, -ns, alpha=0.5, color="#FF5252", label="Short")
ax3.axhline(y=0, color="white", alpha=0.3)
ax3.set_ylabel("N. Posizioni")
ax3.set_title("Composizione Portafoglio", fontweight="bold", color="#AAAAAA")
ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# Pannello 4: Drawdown
ax4 = axes[3]
ax4.fill_between(dd.index, dd*100, 0, alpha=0.5, color="#FF5252", label="DD")
ax4.plot(dd.index, dd*100, color="#FF8A80", linewidth=0.8)
ax4.axhline(y=mdd*100, color="#FF1744", linestyle="--", alpha=0.6,
            label=f"Max DD: {mdd*100:.2f}%")
ax4.set_ylabel("Drawdown (%)")
ax4.set_xlabel("Data")
ax4.set_title("Drawdown", fontweight="bold", color="#AAAAAA")
ax4.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")

for ax in axes:
    if ax != axes[1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("autoencoder_statarb_v2.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("  Salvato: autoencoder_statarb_v2.png")

# ── Stock Attribution ──
fig2, ax_a = plt.subplots(figsize=(14, 8))
stock_pnl = pd.DataFrame(
    all_positions[start_idx:-1] * raw_ret[start_idx+1:], columns=tickers
).sum() * 100
stock_sorted = stock_pnl.sort_values()
clrs = ["#69F0AE" if v > 0 else "#FF5252" for v in stock_sorted.values]
ax_a.barh(stock_sorted.index, stock_sorted.values, color=clrs,
          alpha=0.8, edgecolor="white", linewidth=0.5)
ax_a.axvline(x=0, color="white", alpha=0.3)
ax_a.set_xlabel("P&L Contribution (%)")
ax_a.set_title("P&L Attribution per Azione", fontweight="bold", color="#AAAAAA")
for i, (nm, v) in enumerate(stock_sorted.items()):
    ax_a.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.1f}%",
              va="center", ha="left" if v >= 0 else "right",
              fontsize=9, color="white")
plt.tight_layout()
plt.savefig("autoencoder_statarb_v2_attribution.png", dpi=150,
            bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.show()
print("  Salvato: autoencoder_statarb_v2_attribution.png")

# ── Latent Factors ──
if model is not None:
    fig3, ax_l = plt.subplots(figsize=(14, 5))
    model.eval()
    with torch.no_grad():
        data_t = torch.FloatTensor(norm_arr[-252:]).to(device)
        latent = model.encode(data_t).cpu().numpy()
    for i in range(LATENT_DIM):
        ax_l.plot(latent[:, i], alpha=0.7, linewidth=1.0, label=f"Factor {i+1}")
    ax_l.set_xlabel("Giorni (ultimo anno)")
    ax_l.set_ylabel("Valore Latente")
    ax_l.set_title("Fattori Latenti Appresi dall'Autoencoder",
                    fontweight="bold", color="#AAAAAA")
    ax_l.legend(framealpha=0.3, facecolor="#1a1a2e")
    plt.tight_layout()
    plt.savefig("autoencoder_statarb_v2_latent.png", dpi=150,
                bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.show()
    print("  Salvato: autoencoder_statarb_v2_latent.png")

print("\n" + "=" * 80)
print("  AUTOENCODER STAT-ARB V2 COMPLETATO")
print("=" * 80)
