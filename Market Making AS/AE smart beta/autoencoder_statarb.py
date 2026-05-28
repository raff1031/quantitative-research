#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  AUTOENCODER STATISTICAL ARBITRAGE — Utilities Sector
================================================================================

  Strategia:
    Un autoencoder apprende le relazioni cross-sezionali non-lineari tra
    ~25 azioni del settore Utilities. Per ogni azione ad ogni timestep:

      residual = actual_return - reconstructed_return

    Se il residuo e' molto negativo, l'azione e' "sottovalutata" rispetto
    al gruppo → LONG. Se molto positivo → SHORT. Il portafoglio e'
    market-neutral per costruzione.

  Perche' e' meglio del Pairs Trading:
    - Non limitato a 2 azioni, cattura relazioni su N titoli
    - Non-lineare (vs OLS lineare)
    - Il bottleneck impara i "fattori latenti" del settore (simile a PCA
      non-lineare: tassi, regolamentazione, domanda energia, ecc.)

  Pipeline:
    1. Download dati giornalieri per ~25 utilities
    2. Calcolo log-returns cross-sezionali
    3. Walk-forward: training rolling + predizione OOS
    4. Segnali dal residuo normalizzato (Z-score cross-sezionale)
    5. Portafoglio long/short market-neutral
    6. Metriche e visualizzazione

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
from datetime import datetime, timedelta

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

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════

# Universo: Utilities US ampio
UTILITIES = [
    "NEE",   # NextEra Energy
    "DUK",   # Duke Energy
    "SO",    # Southern Company
    "D",     # Dominion Energy
    "AEP",   # American Electric Power
    "EXC",   # Exelon
    "SRE",   # Sempra
    "XEL",   # Xcel Energy
    "ED",    # Consolidated Edison
    "WEC",   # WEC Energy
    "ES",    # Eversource Energy
    "AWK",   # American Water Works
    "ETR",   # Entergy
    "PPL",   # PPL Corporation
    "FE",    # FirstEnergy
    "AEE",   # Ameren
    "CMS",   # CMS Energy
    "CNP",   # CenterPoint Energy
    "DTE",   # DTE Energy
    "EVRG",  # Evergy
    "ATO",   # Atmos Energy
    "LNT",   # Alliant Energy
    "NI",    # NiSource
    "PNW",   # Pinnacle West
]

SECTOR_NAME = "Utilities"

# Walk-forward parameters
TRAIN_WINDOW   = 252       # 1 anno di trading days per training
RETRAIN_EVERY  = 21        # Ritrain ogni mese
LOOKBACK_YEARS = 3         # Dati storici (giornalieri → piu' storia)

# Autoencoder architecture
LATENT_DIM     = 5         # Dimensione del bottleneck (fattori latenti)
HIDDEN_1       = 32        # Primo layer nascosto
HIDDEN_2       = 16        # Secondo layer nascosto
DROPOUT        = 0.15      # Dropout per regolarizzazione
LR             = 1e-3      # Learning rate
EPOCHS         = 80        # Epoche per training
BATCH_SIZE     = 32

# Signal parameters
ENTRY_Z        = 1.0       # Entry quando |residual_z| > 1.0
EXIT_Z         = 0.25      # Exit quando |residual_z| < 0.25
MAX_POSITIONS  = 5         # Max azioni long + max short
TC_BPS         = 10        # Transaction costs (10 bps round-trip)

DAYS_PER_YEAR  = 252

print("=" * 80)
print(f"  AUTOENCODER STAT-ARB — {SECTOR_NAME} Sector")
print(f"  Universo: {len(UTILITIES)} titoli  |  Latent Dim: {LATENT_DIM}")
print(f"  Walk-Forward: Train={TRAIN_WINDOW}d, Retrain ogni {RETRAIN_EVERY}d")
print(f"  Entry Z: {ENTRY_Z}  |  Exit Z: {EXIT_Z}  |  TC: {TC_BPS} bps")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA INGESTION
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
            print(f"  OK  {ticker:6s} {len(data):,} giorni  "
                  f"${data['Close'].squeeze().iloc[-1]:.2f}")
    except:
        print(f"  FAIL {ticker}")

prices = pd.DataFrame(all_data).dropna()
tickers = list(prices.columns)
N_STOCKS = len(tickers)

print(f"\n  Titoli validi: {N_STOCKS}  |  Giorni: {len(prices):,}")
print(f"  Periodo: {prices.index[0].strftime('%Y-%m-%d')} -> "
      f"{prices.index[-1].strftime('%Y-%m-%d')}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/6] Preprocessing...")

# Log returns
log_returns = np.log(prices / prices.shift(1)).dropna()

# Cross-sectional standardization: per ogni giorno, z-score across stocks
# Questo normalizza l'input per l'autoencoder
cs_mean = log_returns.mean(axis=1)   # media cross-sezionale per giorno
cs_std  = log_returns.std(axis=1)    # std cross-sezionale per giorno

# Normalizzazione: rendo ogni feature (stock) a media 0, std ~1 nel tempo
time_mean = log_returns.mean(axis=0)
time_std  = log_returns.std(axis=0)
returns_normalized = (log_returns - time_mean) / time_std

print(f"  Shape returns: {log_returns.shape}")
print(f"  Returns medi: {log_returns.mean().mean()*100:.4f}% al giorno")
print(f"  Volatilita' media: {log_returns.std().mean()*100:.3f}% al giorno")


# ══════════════════════════════════════════════════════════════════════════════
#  3. AUTOENCODER MODEL
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/6] Definizione Autoencoder ({N_STOCKS} -> {HIDDEN_1} -> "
      f"{HIDDEN_2} -> {LATENT_DIM} -> {HIDDEN_2} -> {HIDDEN_1} -> {N_STOCKS})...")


class StatArbAutoencoder(nn.Module):
    """
    Autoencoder per Statistical Arbitrage.

    Input:  Vettore di N returns standardizzati (un return per stock)
    Output: Ricostruzione del vettore → il "fair value" cross-sezionale

    Il bottleneck (latent_dim) cattura i fattori latenti del settore:
    - Tassi di interesse
    - Domanda energetica
    - Regolamentazione
    - Sentiment settoriale
    - ecc.

    Il residuo (input - output) rappresenta la componente IDIOSINCRATICA
    di ogni azione, ovvero il segnale di trading.
    """

    def __init__(self, n_stocks, hidden1, hidden2, latent_dim, dropout=0.1):
        super().__init__()

        # Encoder: comprime N stocks → latent factors
        self.encoder = nn.Sequential(
            nn.Linear(n_stocks, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim),
        )

        # Decoder: ricostruisce latent factors → N stocks
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden1, n_stocks),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_latent(self, x):
        """Estrae la rappresentazione latente (i fattori)."""
        return self.encoder(x)


def train_autoencoder(model, train_data, epochs=80, lr=1e-3, batch_size=32):
    """Allena l'autoencoder su una finestra di training."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    dataset = TensorDataset(train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        losses.append(epoch_loss / len(loader))

    return losses


# ══════════════════════════════════════════════════════════════════════════════
#  4. WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
#
#  Walk-forward loop:
#    - Ogni RETRAIN_EVERY giorni, riallenamo l'autoencoder sugli ultimi
#      TRAIN_WINDOW giorni
#    - Per i prossimi RETRAIN_EVERY giorni, generiamo segnali OOS:
#      * Passiamo il vettore di returns reali attraverso l'autoencoder
#      * residual = actual - reconstructed
#      * Normalizziamo i residui cross-sezionalmente → Z-score
#      * Long le azioni con Z < -ENTRY_Z (sottovalutate vs peers)
#      * Short le azioni con Z > +ENTRY_Z (sopravvalutate vs peers)
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n[4/6] Walk-Forward Backtest...")

returns_array = returns_normalized.values  # (T, N_STOCKS)
raw_returns = log_returns.values           # (T, N_STOCKS) — per P&L reale
dates = log_returns.index

# Output containers
all_positions = np.zeros_like(raw_returns)   # (T, N_STOCKS)
all_residuals = np.zeros_like(raw_returns)   # (T, N_STOCKS)
all_residual_z = np.zeros_like(raw_returns)  # (T, N_STOCKS)
train_losses_history = []
retrain_dates = []

start_idx = TRAIN_WINDOW
n_retrains = 0

print(f"  Walk-forward da giorno {start_idx} a {len(returns_array)}")
print(f"  Periodi di retrain previsti: ~{(len(returns_array) - start_idx) // RETRAIN_EVERY}")

current_model = None
positions = np.zeros(N_STOCKS)  # Posizioni correnti

for t in range(start_idx, len(returns_array)):
    # ── Retrain check ──
    if (t - start_idx) % RETRAIN_EVERY == 0:
        # Training window
        train_start = max(0, t - TRAIN_WINDOW)
        train_data = returns_array[train_start:t]

        # Converti a tensore
        train_tensor = torch.FloatTensor(train_data)

        # Nuovo modello (re-inizializzato per evitare bias da modelli precedenti)
        current_model = StatArbAutoencoder(
            N_STOCKS, HIDDEN_1, HIDDEN_2, LATENT_DIM, DROPOUT
        ).to(device)

        losses = train_autoencoder(
            current_model, train_tensor, EPOCHS, LR, BATCH_SIZE
        )
        train_losses_history.append(losses[-1])
        retrain_dates.append(dates[t])
        n_retrains += 1

        if n_retrains % 5 == 0:
            print(f"    Retrain #{n_retrains} @ {dates[t].strftime('%Y-%m-%d')}  "
                  f"Loss: {losses[-1]:.6f}")

    # ── Generate signal ──
    if current_model is not None:
        current_model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(returns_array[t:t+1]).to(device)
            x_hat = current_model(x)

            # Residual: actual - reconstructed
            residual = (x - x_hat).cpu().numpy().flatten()

        # Cross-sectional Z-score del residuo
        # Questo ci dice quali azioni sono anomale RISPETTO AL GRUPPO
        res_mean = residual.mean()
        res_std = residual.std()
        if res_std > 1e-8:
            residual_z = (residual - res_mean) / res_std
        else:
            residual_z = np.zeros(N_STOCKS)

        all_residuals[t] = residual
        all_residual_z[t] = residual_z

        # ── Position management ──
        new_positions = np.zeros(N_STOCKS)

        # Ranking: ordina per residual Z
        rankings = np.argsort(residual_z)

        # LONG: azioni con Z piu' negativo (sottovalutate vs peers)
        long_candidates = rankings[:MAX_POSITIONS]
        long_mask = residual_z[long_candidates] < -ENTRY_Z

        # SHORT: azioni con Z piu' positivo (sopravvalutate vs peers)
        short_candidates = rankings[-MAX_POSITIONS:]
        short_mask = residual_z[short_candidates] > ENTRY_Z

        # Assegna posizioni
        for idx in long_candidates[long_mask]:
            new_positions[idx] = 1.0

        for idx in short_candidates[short_mask]:
            new_positions[idx] = -1.0

        # Exit: chiudi posizioni dove |Z| < EXIT_Z
        for i in range(N_STOCKS):
            if positions[i] != 0 and abs(residual_z[i]) < EXIT_Z:
                new_positions[i] = 0

            # Mantieni posizioni aperte se non c'e' exit signal
            # e non c'e' un nuovo segnale contrario
            if positions[i] != 0 and new_positions[i] == 0 and abs(residual_z[i]) >= EXIT_Z:
                new_positions[i] = positions[i]

        # Normalizza: equal dollar long e short
        n_long = (new_positions > 0).sum()
        n_short = (new_positions < 0).sum()

        if n_long > 0:
            new_positions[new_positions > 0] = 1.0 / n_long
        if n_short > 0:
            new_positions[new_positions < 0] = -1.0 / n_short

        positions = new_positions

    all_positions[t] = positions


# ══════════════════════════════════════════════════════════════════════════════
#  5. CALCOLO P&L
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Calcolo P&L...")

# Rendimenti del portafoglio: somma pesata dei rendimenti di ogni azione
# position[t-1] * return[t] (lag per evitare lookahead)
portfolio_returns = np.zeros(len(raw_returns))
tc_per_trade = TC_BPS / 10000

for t in range(start_idx + 1, len(raw_returns)):
    # Rendimento: posizione di ieri * rendimento di oggi
    daily_ret = np.sum(all_positions[t-1] * raw_returns[t])

    # Transaction costs
    position_change = np.sum(np.abs(all_positions[t] - all_positions[t-1]))
    tc = position_change * tc_per_trade

    portfolio_returns[t] = daily_ret - tc

# Converti a Series
port_ret = pd.Series(portfolio_returns, index=dates)
port_ret = port_ret.iloc[start_idx:]
port_cum = (1 + port_ret).cumprod()

# Benchmark: equal-weight buy-and-hold del settore
bh_returns = raw_returns.mean(axis=1)  # Media semplice di tutti i titoli
bh_ret = pd.Series(bh_returns, index=dates).iloc[start_idx:]
bh_cum = (1 + bh_ret).cumprod()

# Long-only: media dei titoli con posizione long
long_mask = all_positions > 0
long_ret = np.zeros(len(raw_returns))
for t in range(start_idx + 1, len(raw_returns)):
    lm = long_mask[t-1]
    if lm.any():
        long_ret[t] = raw_returns[t][lm].mean()
long_ret_s = pd.Series(long_ret, index=dates).iloc[start_idx:]
long_cum = (1 + long_ret_s).cumprod()


# ══════════════════════════════════════════════════════════════════════════════
#  6. METRICHE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] Calcolo metriche...")

# Portfolio metrics
pm = port_ret.mean()
ps = port_ret.std()
sharpe = (pm / ps) * np.sqrt(DAYS_PER_YEAR) if ps > 0 else 0
cum_max = port_cum.cummax()
dd = (port_cum - cum_max) / cum_max
max_dd = dd.min()
total_ret = (port_cum.iloc[-1] - 1) * 100
annual_ret = pm * DAYS_PER_YEAR * 100

gp = port_ret[port_ret > 0].sum()
gl = abs(port_ret[port_ret < 0].sum())
pf = gp / gl if gl > 0 else np.inf

wr = (port_ret > 0).sum() / (port_ret != 0).sum() * 100

ds = port_ret[port_ret < 0].std()
sortino = (pm / ds) * np.sqrt(DAYS_PER_YEAR) if ds > 0 else 0
calmar = (pm * DAYS_PER_YEAR) / abs(max_dd) if max_dd != 0 else 0

# Benchmark metrics
bh_total = (bh_cum.iloc[-1] - 1) * 100
bh_sharpe = (bh_ret.mean() / bh_ret.std()) * np.sqrt(DAYS_PER_YEAR) if bh_ret.std() > 0 else 0

# Position stats
avg_long = (all_positions[start_idx:] > 0).sum(axis=1).mean()
avg_short = (all_positions[start_idx:] < 0).sum(axis=1).mean()
avg_total = avg_long + avg_short
pct_invested = (all_positions[start_idx:] != 0).any(axis=1).mean() * 100

# Turnover
position_changes = np.abs(np.diff(all_positions[start_idx:], axis=0)).sum(axis=1)
avg_turnover = position_changes.mean()
total_tc = (position_changes * tc_per_trade).sum() * 100

# Number of distinct trade signals
position_entries = np.diff((all_positions[start_idx:] != 0).astype(int), axis=0)
n_entries = (position_entries > 0).sum()

print("\n" + "=" * 80)
print(f"  AUTOENCODER STAT-ARB — {SECTOR_NAME} ({N_STOCKS} titoli)")
print("=" * 80)
print(f"  |")
print(f"  | === STRATEGIA ===")
print(f"  | Rendimento Totale          : {total_ret:>+10.2f} %")
print(f"  | Rendimento Annualizzato    : {annual_ret:>+10.2f} %")
print(f"  | Sharpe Ratio (ann.)        : {sharpe:>10.3f}")
print(f"  | Sortino Ratio (ann.)       : {sortino:>10.3f}")
print(f"  | Calmar Ratio               : {calmar:>10.3f}")
print(f"  | Maximum Drawdown           : {max_dd*100:>10.2f} %")
print(f"  | Win Rate (daily)           : {wr:>10.1f} %")
print(f"  | Profit Factor              : {pf:>10.3f}")
print(f"  |")
print(f"  | === BENCHMARK (Equal-Weight B&H) ===")
print(f"  | B&H Rendimento Totale      : {bh_total:>+10.2f} %")
print(f"  | B&H Sharpe Ratio           : {bh_sharpe:>10.3f}")
print(f"  |")
print(f"  | === POSIZIONI ===")
print(f"  | Avg posizioni long/giorno  : {avg_long:>10.1f}")
print(f"  | Avg posizioni short/giorno : {avg_short:>10.1f}")
print(f"  | % tempo investito          : {pct_invested:>10.1f} %")
print(f"  | Turnover medio/giorno      : {avg_turnover:>10.3f}")
print(f"  | Entry signals totali       : {n_entries:>10}")
print(f"  | Transaction costs totali   : {total_tc:>10.3f} %")
print(f"  |")
print(f"  | === MODELLO ===")
print(f"  | Architettura               : {N_STOCKS}->{HIDDEN_1}->{HIDDEN_2}"
      f"->{LATENT_DIM}->{HIDDEN_2}->{HIDDEN_1}->{N_STOCKS}")
print(f"  | Retrains effettuati        : {n_retrains:>10}")
print(f"  | Loss media finale          : {np.mean(train_losses_history):>10.6f}")
print(f"  | Giorni OOS testati         : {len(port_ret):>10}")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  7. GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerazione grafici...")

fig, axes = plt.subplots(4, 1, figsize=(22, 22),
                         gridspec_kw={"height_ratios": [3, 2, 2, 2]})

fig.suptitle(
    f"AUTOENCODER STAT-ARB  |  {SECTOR_NAME} ({N_STOCKS} titoli)  |  "
    f"Sharpe: {sharpe:.2f}  |  Return: {total_ret:+.1f}%  |  "
    f"MaxDD: {max_dd*100:.1f}%  |  Entries: {n_entries}",
    fontsize=14, fontweight="bold", color="#E0E0E0", y=0.98
)

# ── Pannello 1: Equity Curve ──
ax1 = axes[0]
ax1.plot(port_cum.index, port_cum, color="#00E5FF", linewidth=2.0,
         alpha=0.95, label=f"AE Stat-Arb (SR={sharpe:.2f})")
ax1.plot(bh_cum.index, bh_cum, color="#FF6B6B", linewidth=1.2,
         alpha=0.7, linestyle="--", label=f"B&H Utilities (SR={bh_sharpe:.2f})")
ax1.plot(long_cum.index, long_cum, color="#69F0AE", linewidth=1.0,
         alpha=0.5, linestyle=":", label="Long-Only Leg")

ax1.fill_between(port_cum.index, 1, port_cum,
                 where=(port_cum >= 1), alpha=0.08, color="#00E5FF")
ax1.fill_between(port_cum.index, 1, port_cum,
                 where=(port_cum < 1), alpha=0.08, color="#FF5252")
ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

# Retrain markers
for rd in retrain_dates[::3]:
    ax1.axvline(x=rd, color="#AB47BC", alpha=0.1, linewidth=0.5)

ax1.set_ylabel("Cumulative ($1)")
ax1.set_title("Equity Curve: AE Stat-Arb vs Buy & Hold",
              fontweight="bold", color="#AAAAAA")
ax1.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 2: Residual Z-Score Heatmap ──
ax2 = axes[1]
residual_z_df = pd.DataFrame(all_residual_z[start_idx:], index=dates[start_idx:],
                              columns=tickers)
# Downsample per visualizzazione (media settimanale)
residual_weekly = residual_z_df.resample("W").mean()

im = ax2.imshow(residual_weekly.values.T, aspect="auto", cmap="RdYlGn_r",
                vmin=-2, vmax=2, interpolation="nearest")
ax2.set_yticks(range(len(tickers)))
ax2.set_yticklabels(tickers, fontsize=8)

# X-axis: date labels
n_ticks = 12
tick_positions = np.linspace(0, len(residual_weekly) - 1, n_ticks, dtype=int)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels([residual_weekly.index[i].strftime("%b %Y")
                     for i in tick_positions], rotation=30, fontsize=8)

ax2.set_title("Residual Z-Score Heatmap (Settimanale)", fontweight="bold",
              color="#AAAAAA")
plt.colorbar(im, ax=ax2, label="Z-Score", shrink=0.8)

# ── Pannello 3: N. Posizioni nel tempo ──
ax3 = axes[2]
positions_df = pd.DataFrame(all_positions[start_idx:], index=dates[start_idx:],
                             columns=tickers)
n_long_ts = (positions_df > 0).sum(axis=1)
n_short_ts = (positions_df < 0).sum(axis=1)

ax3.fill_between(n_long_ts.index, 0, n_long_ts, alpha=0.5, color="#69F0AE",
                 label="Long Positions")
ax3.fill_between(n_short_ts.index, 0, -n_short_ts, alpha=0.5, color="#FF5252",
                 label="Short Positions")
ax3.axhline(y=0, color="white", linestyle="-", alpha=0.3)
ax3.set_ylabel("N. Posizioni")
ax3.set_title("Composizione del Portafoglio nel Tempo",
              fontweight="bold", color="#AAAAAA")
ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# ── Pannello 4: Drawdown ──
ax4 = axes[3]
ax4.fill_between(dd.index, dd * 100, 0, alpha=0.5, color="#FF5252",
                 label="Drawdown")
ax4.plot(dd.index, dd * 100, color="#FF8A80", linewidth=0.8)
ax4.axhline(y=max_dd * 100, color="#FF1744", linestyle="--", alpha=0.6,
            label=f"Max DD: {max_dd*100:.2f}%")
ax4.set_ylabel("Drawdown (%)")
ax4.set_xlabel("Data")
ax4.set_title("Drawdown", fontweight="bold", color="#AAAAAA")
ax4.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")

for ax in axes:
    if ax != axes[1]:  # Skip heatmap
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("autoencoder_statarb.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("  Salvato: autoencoder_statarb.png")


# ── Grafico 2: Training Loss nel Tempo ──
fig2, (ax_loss, ax_latent) = plt.subplots(1, 2, figsize=(18, 6))

ax_loss.plot(range(len(train_losses_history)), train_losses_history,
             color="#FFA726", linewidth=1.5, marker="o", markersize=3)
ax_loss.set_xlabel("Retrain #")
ax_loss.set_ylabel("Final MSE Loss")
ax_loss.set_title("Training Loss per Retrain", fontweight="bold", color="#AAAAAA")

# Latent factors visualization (ultimo modello)
if current_model is not None:
    current_model.eval()
    with torch.no_grad():
        last_data = torch.FloatTensor(returns_array[-TRAIN_WINDOW:]).to(device)
        latent = current_model.get_latent(last_data).cpu().numpy()

    for i in range(min(LATENT_DIM, 5)):
        ax_latent.plot(latent[:, i], alpha=0.7, linewidth=0.8,
                       label=f"Factor {i+1}")
    ax_latent.set_xlabel("Giorni (ultimi 252)")
    ax_latent.set_ylabel("Valore Latente")
    ax_latent.set_title("Fattori Latenti (Ultimo Training)",
                        fontweight="bold", color="#AAAAAA")
    ax_latent.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

plt.tight_layout()
plt.savefig("autoencoder_latent.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()
print("  Salvato: autoencoder_latent.png")


# ── Grafico 3: Per-stock Attribution ──
fig3, ax_attr = plt.subplots(figsize=(14, 8))

# Performance per stock
stock_returns = pd.DataFrame(
    all_positions[start_idx:-1] * raw_returns[start_idx + 1:],
    columns=tickers
)
stock_total = stock_returns.sum() * 100
stock_total_sorted = stock_total.sort_values()

colors_bar = ["#69F0AE" if v > 0 else "#FF5252" for v in stock_total_sorted.values]
ax_attr.barh(stock_total_sorted.index, stock_total_sorted.values,
             color=colors_bar, alpha=0.8, edgecolor="white", linewidth=0.5)
ax_attr.axvline(x=0, color="white", linestyle="-", alpha=0.3)
ax_attr.set_xlabel("Contributo al P&L (%)")
ax_attr.set_title("Contributo di Ogni Azione al P&L Totale",
                   fontweight="bold", color="#AAAAAA")

for i, (name, val) in enumerate(stock_total_sorted.items()):
    ax_attr.text(val + (0.1 if val >= 0 else -0.1), i, f"{val:+.1f}%",
                 va="center", ha="left" if val >= 0 else "right",
                 fontsize=9, color="white")

plt.tight_layout()
plt.savefig("autoencoder_attribution.png", dpi=150, bbox_inches="tight",
            facecolor=fig3.get_facecolor())
plt.show()
print("  Salvato: autoencoder_attribution.png")


print("\n" + "=" * 80)
print("  AUTOENCODER STAT-ARB COMPLETATO")
print("=" * 80)
