#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  AE SMART BETA — PRODUCTION GRADE
================================================================================

  Fix implementati:
    [1] SURVIVORSHIP BIAS:  universo ampio + ETF settoriale come sanity check
    [2] MULTI-SETTORE:      5 settori GICS indipendenti
    [3] PIU' DATI:          8+ anni giornalieri (2017-2026) — include COVID,
                            2022 bear, recovery, bull corrente
    [4] ENSEMBLE:           3 AE per settore con seed diversi, mediana residui
    [5] FEATURE RICHE:      multi-horizon returns + realized vol + rel volume
    [6] STATISTICAL TESTS:  bootstrap Sharpe CI, t-stat alpha, regime breakdown
    [7] SUB-PERIOD REPORT:  performance in bull, crash, bear, recovery separati

  Autore: Quant Research Desk — Aprile 2026
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.special import softmax
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

# Universo multi-settoriale — ampio per minimizzare survivorship bias
SECTORS = {
    "Utilities": {
        "etf": "XLU",
        "stocks": [
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC",
            "ES", "AWK", "ETR", "PPL", "FE", "AEE", "CMS", "CNP", "DTE", "EVRG",
            "ATO", "LNT", "NI", "PNW",
        ],
    },
    "Staples": {
        "etf": "XLP",
        "stocks": [
            "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS",
            "KHC", "SYY", "HSY", "KMB", "CHD", "MKC", "SJM", "CAG", "CPB", "HRL",
        ],
    },
    "Financials": {
        "etf": "XLF",
        "stocks": [
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "FITB",
            "KEY", "RF", "CFG", "HBAN", "MTB", "ZION", "CMA", "SIVB",
        ],
    },
    "Energy": {
        "etf": "XLE",
        "stocks": [
            "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY",
            "PXD", "DVN", "HES", "FANG", "BKR", "HAL",
        ],
    },
    "Healthcare": {
        "etf": "XLV",
        "stocks": [
            "JNJ", "UNH", "PFE", "ABT", "TMO", "MRK", "ABBV", "LLY", "BMY",
            "AMGN", "GILD", "MDT", "SYK", "BDX", "ZBH", "BAX", "BSX", "EW",
        ],
    },
}

LOOKBACK_YEARS  = 8         # 2017-2026
TRAIN_WINDOW    = 504       # 2 anni di trading
RETRAIN_EVERY   = 21        # Mensile
N_ENSEMBLE      = 2         # AE per settore

# AE architecture
LATENT_DIM  = 6
HIDDEN_1    = 48
HIDDEN_2    = 24
DROPOUT     = 0.12
LR          = 1e-3
EPOCHS      = 30
BATCH_SIZE  = 64

# Portfolio
REBALANCE_DAYS = 5
TILT_STRENGTH  = 2.0
MIN_W, MAX_W   = 0.01, 0.12
TC_BPS         = 7          # Piu' realistico: 7 bps (slippage incluso)

# Feature windows
RET_WINDOWS = [1, 5, 21, 63]       # 1d, 1w, 1m, 3m returns
VOL_WINDOW  = 21                    # Realized vol window
NORM_WINDOW = 63                    # Normalization lookback

DAYS_PER_YEAR = 252

# Regime periodi (per sub-period analysis)
REGIMES = {
    "Pre-COVID Bull":  ("2018-01-01", "2020-02-19"),
    "COVID Crash":     ("2020-02-20", "2020-03-23"),
    "Recovery/Bubble": ("2020-03-24", "2021-12-31"),
    "2022 Bear":       ("2022-01-01", "2022-12-31"),
    "2023-24 Recovery": ("2023-01-01", "2024-12-31"),
    "2025-Current":    ("2025-01-01", "2026-12-31"),
}

print("=" * 80)
print("  AE SMART BETA — PRODUCTION GRADE")
print(f"  Settori: {len(SECTORS)}  |  Dati: {LOOKBACK_YEARS}Y giornalieri")
print(f"  Ensemble: {N_ENSEMBLE} AE/settore  |  Train: {TRAIN_WINDOW}d")
print(f"  Features: returns {RET_WINDOWS} + vol + volume  |  TC: {TC_BPS} bps")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/8] Download dati (8 anni, multi-settore)...")

end_date   = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365 + 90)  # Buffer extra

all_sector_data = {}
sector_etf_data = {}

for sector_name, sector_info in SECTORS.items():
    print(f"\n  -- {sector_name} --")

    # Download ETF (benchmark survivorship-free)
    try:
        etf = yf.download(sector_info["etf"], start=start_date, end=end_date,
                          interval="1d", progress=False, auto_adjust=True)
        if len(etf) > 200:
            sector_etf_data[sector_name] = etf["Close"].squeeze()
            print(f"  ETF {sector_info['etf']}: {len(etf):,} giorni OK")
    except:
        print(f"  ETF {sector_info['etf']}: FAIL")

    # Download stocks
    stock_prices = {}
    stock_volumes = {}
    for ticker in sector_info["stocks"]:
        try:
            data = yf.download(ticker, start=start_date, end=end_date,
                              interval="1d", progress=False, auto_adjust=True)
            if len(data) > 500:
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                stock_prices[ticker] = close
                vol = data["Volume"]
                if isinstance(vol, pd.DataFrame):
                    vol = vol.iloc[:, 0]
                stock_volumes[ticker] = vol
        except:
            pass

    if len(stock_prices) >= 8:
        all_sector_data[sector_name] = {
            "prices": stock_prices,
            "volumes": stock_volumes,
        }
        print(f"  Titoli OK: {len(stock_prices)}/{len(sector_info['stocks'])}")
    else:
        print(f"  SKIP: solo {len(stock_prices)} titoli")

print(f"\n  Settori attivi: {len(all_sector_data)}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/8] Feature Engineering per settore...")


def build_features(prices_df, volume_df):
    """
    Costruisce feature multi-dimensionali per l'autoencoder.

    Per ogni stock, genera:
      - Log returns a multipli orizzonti (1d, 5d, 21d, 63d)
      - Realized volatility (21d)
      - Volume relativo (vs media 21d)

    Tutte normalizzate con rolling z-score.
    """
    features = {}
    n_stocks = prices_df.shape[1]
    tickers = list(prices_df.columns)

    for window in RET_WINDOWS:
        ret = np.log(prices_df / prices_df.shift(window))
        # Rolling z-score cross-sezionale (per giorno)
        cs_mean = ret.mean(axis=1)
        cs_std  = ret.std(axis=1).replace(0, 1e-8)
        ret_norm = ret.sub(cs_mean, axis=0).div(cs_std, axis=0)
        features[f"ret_{window}d"] = ret_norm

    # Realized vol (rolling std dei daily returns)
    daily_ret = np.log(prices_df / prices_df.shift(1))
    rvol = daily_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    cs_mean_v = rvol.mean(axis=1)
    cs_std_v  = rvol.std(axis=1).replace(0, 1e-8)
    features["rvol"] = rvol.sub(cs_mean_v, axis=0).div(cs_std_v, axis=0)

    # Volume relativo
    if volume_df is not None and len(volume_df) > 0:
        vol_ratio = volume_df / volume_df.rolling(VOL_WINDOW).mean()
        cs_mean_vr = vol_ratio.mean(axis=1)
        cs_std_vr  = vol_ratio.std(axis=1).replace(0, 1e-8)
        features["vol_ratio"] = vol_ratio.sub(cs_mean_vr, axis=0).div(cs_std_vr, axis=0)

    # Stack in un singolo array: (T, N_stocks * N_features)
    all_feats = []
    feat_names = sorted(features.keys())
    for fn in feat_names:
        all_feats.append(features[fn])

    combined = pd.concat(all_feats, axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna()

    # Clip outliers
    combined = combined.clip(-4, 4)

    return combined, feat_names, tickers


sector_features = {}
sector_prices = {}
sector_tickers = {}

for sector_name, stock_data in all_sector_data.items():
    prices_df = pd.DataFrame(stock_data["prices"]).dropna()
    volume_df = pd.DataFrame(stock_data["volumes"]).reindex(prices_df.index)

    # Daily returns per P&L
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()

    feats, feat_names, tickers = build_features(prices_df, volume_df)

    # Allinea tutto
    common_idx = feats.index.intersection(log_ret.index)
    feats = feats.loc[common_idx]
    log_ret = log_ret.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    sector_features[sector_name] = feats
    sector_prices[sector_name] = {
        "prices": prices_df,
        "log_returns": log_ret,
        "tickers": tickers,
        "n_stocks": len(tickers),
        "n_features": feats.shape[1],
    }
    sector_tickers[sector_name] = tickers

    print(f"  {sector_name:15s}: {len(tickers):2d} stocks, "
          f"{feats.shape[1]} features, {len(feats):,} days")


# ══════════════════════════════════════════════════════════════════════════════
#  3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AE(nn.Module):
    def __init__(self, n_in, h1, h2, lat, drop=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h2, lat),
        )
        self.dec = nn.Sequential(
            nn.Linear(lat, h2), nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.GELU(),
            nn.Linear(h1, n_in),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(model, data, epochs=80, lr=5e-4, bs=64):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss(delta=1.0)  # Robusta a outlier
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
        fl = crit(model(data.to(device)), data.to(device)).item()
    return fl


# ══════════════════════════════════════════════════════════════════════════════
#  4. WALK-FORWARD (PER SETTORE)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/8] Walk-Forward Backtest (multi-settore, ensemble)...")

tc_rate = TC_BPS / 10000
sector_results = {}

for sector_name in sector_features:
    feats = sector_features[sector_name]
    info = sector_prices[sector_name]
    raw_ret = info["log_returns"].values
    prices_df = info["prices"]
    tickers = info["tickers"]
    n_st = info["n_stocks"]
    dates = info["log_returns"].index
    feat_arr = feats.values

    n_feat = feat_arr.shape[1]

    print(f"\n  === {sector_name} ({n_st} stocks, {n_feat} features) ===")

    if len(feat_arr) < TRAIN_WINDOW + 100:
        print(f"    SKIP: troppo pochi dati ({len(feat_arr)} < {TRAIN_WINDOW + 100})")
        continue

    # Output containers
    weights_smart = np.ones((len(feat_arr), n_st)) / n_st  # Start equal weight
    residual_buffer = []
    models = []
    last_rebalance = -REBALANCE_DAYS
    n_retrains = 0
    current_w = np.ones(n_st) / n_st

    for t in range(TRAIN_WINDOW, len(feat_arr)):
        # ── Retrain ensemble ──
        if (t - TRAIN_WINDOW) % RETRAIN_EVERY == 0:
            train_d = torch.FloatTensor(feat_arr[max(0, t-TRAIN_WINDOW):t])

            models = []
            for seed in range(N_ENSEMBLE):
                torch.manual_seed(seed * 1000 + t)
                np.random.seed(seed * 1000 + t)
                m = AE(n_feat, HIDDEN_1, HIDDEN_2, LATENT_DIM, DROPOUT).to(device)
                train_ae(m, train_d, EPOCHS, LR, BATCH_SIZE)
                models.append(m)

            n_retrains += 1
            if n_retrains % 5 == 0:
                print(f"    Retrain #{n_retrains} @ {dates[t].strftime('%Y-%m-%d')}"); sys.stdout.flush()

        # ── Ensemble residual ──
        if models:
            all_res = []
            for m in models:
                with torch.no_grad():
                    x = torch.FloatTensor(feat_arr[t:t+1]).to(device)
                    res = (x - m(x)).cpu().numpy().flatten()
                all_res.append(res)

            # MEDIANA dell'ensemble (robusta a outlier di un singolo modello)
            ensemble_res = np.median(all_res, axis=0)

            # Il residuo ha n_feat dimensioni. Riduci a n_stocks:
            # media dei residui attraverso le feature per ogni stock
            n_feat_per_stock = n_feat // n_st
            stock_residual = np.zeros(n_st)
            for i in range(n_st):
                start_f = i * n_feat_per_stock
                end_f = start_f + n_feat_per_stock
                if end_f <= len(ensemble_res):
                    stock_residual[i] = ensemble_res[start_f:end_f].mean()

            residual_buffer.append(stock_residual)
            if len(residual_buffer) > 5:
                residual_buffer.pop(0)

            # ── Rebalance ──
            if t - last_rebalance >= REBALANCE_DAYS and len(residual_buffer) >= 3:
                smooth_res = np.median(residual_buffer, axis=0)
                logits = -smooth_res * TILT_STRENGTH
                w = softmax(logits)
                w = np.clip(w, MIN_W, MAX_W)
                w = w / w.sum()
                current_w = w
                last_rebalance = t

        weights_smart[t] = current_w

    print(f"    Retrains: {n_retrains}")

    # ── P&L ──
    port_ret = np.zeros(len(raw_ret))
    for t in range(TRAIN_WINDOW + 1, len(raw_ret)):
        daily = np.sum(weights_smart[t-1] * raw_ret[t])
        tc = np.sum(np.abs(weights_smart[t] - weights_smart[t-1])) * tc_rate
        port_ret[t] = daily - tc

    # EW benchmark
    ew_ret = raw_ret.mean(axis=1)

    # ETF benchmark
    etf_ret = None
    if sector_name in sector_etf_data:
        etf_prices = sector_etf_data[sector_name]
        etf_lr = np.log(etf_prices / etf_prices.shift(1)).dropna()
        etf_ret = etf_lr.reindex(dates).fillna(0)

    sector_results[sector_name] = {
        "port_ret": pd.Series(port_ret, index=dates).iloc[TRAIN_WINDOW:],
        "ew_ret": pd.Series(ew_ret, index=dates).iloc[TRAIN_WINDOW:],
        "etf_ret": etf_ret.iloc[TRAIN_WINDOW:] if etf_ret is not None else None,
        "weights": weights_smart[TRAIN_WINDOW:],
        "tickers": tickers,
        "dates": dates[TRAIN_WINDOW:],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. AGGREGATE PORTFOLIO (Cross-Sector)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/8] Aggregazione cross-settore...")

# Allinea tutti i settori
all_port_rets = {}
all_ew_rets = {}
all_etf_rets = {}

for sn, sr in sector_results.items():
    all_port_rets[sn] = sr["port_ret"]
    all_ew_rets[sn] = sr["ew_ret"]
    if sr["etf_ret"] is not None:
        all_etf_rets[sn] = sr["etf_ret"]

port_df = pd.DataFrame(all_port_rets).dropna()
ew_df   = pd.DataFrame(all_ew_rets).dropna()
etf_df  = pd.DataFrame(all_etf_rets).dropna() if all_etf_rets else None

# Portfolio aggregato: equal weight across sectors
agg_port_ret = port_df.mean(axis=1)
agg_ew_ret   = ew_df.mean(axis=1)
agg_etf_ret  = etf_df.mean(axis=1) if etf_df is not None else None

agg_port_cum = (1 + agg_port_ret).cumprod()
agg_ew_cum   = (1 + agg_ew_ret).cumprod()
agg_etf_cum  = (1 + agg_etf_ret).cumprod() if agg_etf_ret is not None else None


# ══════════════════════════════════════════════════════════════════════════════
#  6. METRICS & STATISTICAL TESTING
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/8] Metriche e test statistici...")


def full_metrics(ret, name=""):
    m = ret.mean()
    s = ret.std()
    sr = (m / s) * np.sqrt(DAYS_PER_YEAR) if s > 0 else 0
    cum = (1 + ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    mdd = dd.min()
    tot = (cum.iloc[-1] - 1)
    ann = m * DAYS_PER_YEAR

    gp = ret[ret > 0].sum()
    gl = abs(ret[ret < 0].sum())
    pf = gp / gl if gl > 0 else np.inf

    ds = ret[ret < 0].std()
    sortino = (m / ds) * np.sqrt(DAYS_PER_YEAR) if ds > 0 else 0
    calmar = ann / abs(mdd) if mdd != 0 else 0

    return {
        "name": name, "total": tot, "annual": ann, "sharpe": sr,
        "sortino": sortino, "calmar": calmar, "mdd": mdd,
        "pf": pf, "wr": (ret > 0).mean(),
    }


def bootstrap_sharpe_ci(ret, n_boot=5000, ci=0.95):
    """Bootstrap confidence interval per lo Sharpe Ratio."""
    srs = []
    n = len(ret)
    ret_arr = ret.values
    for _ in range(n_boot):
        sample = np.random.choice(ret_arr, size=n, replace=True)
        m = sample.mean()
        s = sample.std()
        if s > 0:
            srs.append((m / s) * np.sqrt(DAYS_PER_YEAR))
    srs = np.sort(srs)
    lo = srs[int(n_boot * (1 - ci) / 2)]
    hi = srs[int(n_boot * (1 + ci) / 2)]
    return lo, hi


# ── Per-sector metrics ──
print(f"\n{'='*90}")
print(f"  RISULTATI PER SETTORE")
print(f"{'='*90}")
print(f"  {'Settore':15s} {'AE Ret%':>9s} {'EW Ret%':>9s} {'Alpha%':>9s} "
      f"{'AE SR':>7s} {'EW SR':>7s} {'AE MDD%':>9s} {'Days':>6s}")
print(f"  {'-'*80}")

sector_metrics = {}
for sn, sr in sector_results.items():
    m_ae = full_metrics(sr["port_ret"], f"AE-{sn}")
    m_ew = full_metrics(sr["ew_ret"], f"EW-{sn}")
    alpha = m_ae["total"] - m_ew["total"]

    sector_metrics[sn] = {"ae": m_ae, "ew": m_ew, "alpha": alpha}

    flag = " ***" if alpha > 0 else ""
    print(f"  {sn:15s} {m_ae['total']*100:>+9.2f} {m_ew['total']*100:>+9.2f} "
          f"{alpha*100:>+9.2f} {m_ae['sharpe']:>7.2f} {m_ew['sharpe']:>7.2f} "
          f"{m_ae['mdd']*100:>9.2f} {len(sr['port_ret']):>6}{flag}")

print(f"  {'-'*80}")

# Aggregate
m_agg_ae = full_metrics(agg_port_ret, "AE-Aggregate")
m_agg_ew = full_metrics(agg_ew_ret, "EW-Aggregate")
alpha_agg = m_agg_ae["total"] - m_agg_ew["total"]

print(f"  {'AGGREGATO':15s} {m_agg_ae['total']*100:>+9.2f} "
      f"{m_agg_ew['total']*100:>+9.2f} {alpha_agg*100:>+9.2f} "
      f"{m_agg_ae['sharpe']:>7.2f} {m_agg_ew['sharpe']:>7.2f} "
      f"{m_agg_ae['mdd']*100:>9.2f} {len(agg_port_ret):>6}")

if agg_etf_ret is not None:
    m_agg_etf = full_metrics(agg_etf_ret, "ETF-Aggregate")
    alpha_vs_etf = m_agg_ae["total"] - m_agg_etf["total"]
    print(f"  {'ETF Benchmark':15s} {m_agg_etf['total']*100:>+9.2f} {'':>9s} "
          f"{alpha_vs_etf*100:>+9.2f} {m_agg_etf['sharpe']:>7.2f}")

print(f"{'='*90}")


# ── Bootstrap Sharpe CI ──
print(f"\n  Bootstrap Sharpe Ratio (95% CI, 5000 samples):")
lo_ae, hi_ae = bootstrap_sharpe_ci(agg_port_ret)
lo_ew, hi_ew = bootstrap_sharpe_ci(agg_ew_ret)
print(f"    AE Smart Beta:  SR = {m_agg_ae['sharpe']:.3f}  "
      f"  CI = [{lo_ae:.3f}, {hi_ae:.3f}]")
print(f"    EW Benchmark:   SR = {m_agg_ew['sharpe']:.3f}  "
      f"  CI = [{lo_ew:.3f}, {hi_ew:.3f}]")

# ── T-stat dell'alpha ──
excess = agg_port_ret - agg_ew_ret
t_stat = excess.mean() / (excess.std() / np.sqrt(len(excess))) if excess.std() > 0 else 0
p_val_alpha = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(t_stat)))
print(f"\n  Alpha test:")
print(f"    Alpha medio/giorno : {excess.mean()*10000:>+6.2f} bps")
print(f"    Alpha annualizzato : {excess.mean()*DAYS_PER_YEAR*100:>+6.2f}%")
print(f"    T-statistic        : {t_stat:>6.3f}")
print(f"    P-value (2-sided)  : {p_val_alpha:>6.4f}")
sig = "SIGNIFICATIVO" if p_val_alpha < 0.05 else "NON significativo"
print(f"    Conclusione        : {sig} al 5%")


# ══════════════════════════════════════════════════════════════════════════════
#  7. SUB-PERIOD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/8] Analisi per regime di mercato...")

print(f"\n  {'Regime':22s} {'AE Ret%':>9s} {'EW Ret%':>9s} {'Alpha%':>9s} "
      f"{'AE SR':>7s} {'EW SR':>7s} {'Days':>6s}")
print(f"  {'-'*75}")

for regime_name, (d_start, d_end) in REGIMES.items():
    mask = (agg_port_ret.index >= d_start) & (agg_port_ret.index <= d_end)
    if mask.sum() < 10:
        continue

    r_ae = agg_port_ret[mask]
    r_ew = agg_ew_ret[mask]
    m_r_ae = full_metrics(r_ae, regime_name)
    m_r_ew = full_metrics(r_ew, regime_name)
    a = m_r_ae["total"] - m_r_ew["total"]

    flag = " **" if a > 0.01 else (" *" if a > 0 else "")
    print(f"  {regime_name:22s} {m_r_ae['total']*100:>+9.2f} "
          f"{m_r_ew['total']*100:>+9.2f} {a*100:>+9.2f} "
          f"{m_r_ae['sharpe']:>7.2f} {m_r_ew['sharpe']:>7.2f} "
          f"{mask.sum():>6}{flag}")

print(f"  {'-'*75}")


# ══════════════════════════════════════════════════════════════════════════════
#  8. SURVIVORSHIP CHECK: confronto con ETF
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[7/8] Survivorship bias check (AE stocks vs ETF)...")

for sn, sr in sector_results.items():
    if sr["etf_ret"] is not None:
        ew_total = (1 + sr["ew_ret"]).cumprod().iloc[-1] - 1
        # Allinea ETF alla finestra del backtest
        etf_aligned = sr["etf_ret"].reindex(sr["ew_ret"].index).fillna(0)
        etf_total = (1 + etf_aligned).cumprod().iloc[-1] - 1
        bias = (ew_total - etf_total) * 100
        print(f"  {sn:15s}: EW stocks={ew_total*100:+.1f}%  "
              f"ETF={etf_total*100:+.1f}%  "
              f"Survivorship bias={bias:+.1f}pp")


# ══════════════════════════════════════════════════════════════════════════════
#  9. GRAFICI
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[8/8] Grafici...")

fig, axes = plt.subplots(4, 1, figsize=(22, 24),
                         gridspec_kw={"height_ratios": [3, 2, 2, 2]})

fig.suptitle(
    f"AE SMART BETA — PRODUCTION  |  {len(sector_results)} Settori  |  "
    f"Sharpe: {m_agg_ae['sharpe']:.2f}  |  "
    f"Alpha: {alpha_agg*100:+.1f}%  |  "
    f"t-stat: {t_stat:.2f}  |  "
    f"Return: {m_agg_ae['total']*100:+.1f}%",
    fontsize=14, fontweight="bold", color="#E0E0E0", y=0.98
)

# Pannello 1: Equity aggregate
ax1 = axes[0]
ax1.plot(agg_port_cum.index, agg_port_cum, color="#00E5FF", linewidth=2.5,
         alpha=0.95, label=f"AE Smart Beta (SR={m_agg_ae['sharpe']:.2f})")
ax1.plot(agg_ew_cum.index, agg_ew_cum, color="#FF6B6B", linewidth=1.5,
         alpha=0.7, linestyle="--", label=f"EW B&H (SR={m_agg_ew['sharpe']:.2f})")
if agg_etf_cum is not None:
    ax1.plot(agg_etf_cum.index, agg_etf_cum, color="#FFA726", linewidth=1.2,
             alpha=0.6, linestyle=":", label=f"ETF Blend (survivorship-free)")

# Regime shading
colors_regime = ["#2196F3", "#FF1744", "#69F0AE", "#FF5252", "#AB47BC", "#FFD740"]
for i, (rn, (ds, de)) in enumerate(REGIMES.items()):
    try:
        ax1.axvspan(pd.Timestamp(ds), pd.Timestamp(de),
                    alpha=0.03, color=colors_regime[i % len(colors_regime)])
    except:
        pass

ax1.fill_between(agg_port_cum.index, agg_ew_cum, agg_port_cum,
                 where=(agg_port_cum > agg_ew_cum), alpha=0.08, color="#00E5FF")
ax1.fill_between(agg_port_cum.index, agg_ew_cum, agg_port_cum,
                 where=(agg_port_cum < agg_ew_cum), alpha=0.08, color="#FF5252")
ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
ax1.set_ylabel("Cumulative ($1)")
ax1.set_title("Portafoglio Multi-Settore Aggregato (vs EW & ETF)",
              fontweight="bold", color="#AAAAAA")
ax1.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# Pannello 2: Alpha cumulativo
ax2 = axes[1]
alpha_cum = ((1 + agg_port_ret).cumprod() / (1 + agg_ew_ret).cumprod() - 1) * 100
ax2.plot(alpha_cum.index, alpha_cum, color="#00E5FF", linewidth=1.5, alpha=0.9)
ax2.fill_between(alpha_cum.index, 0, alpha_cum,
                 where=(alpha_cum > 0), alpha=0.15, color="#00E5FF")
ax2.fill_between(alpha_cum.index, 0, alpha_cum,
                 where=(alpha_cum < 0), alpha=0.15, color="#FF5252")
ax2.axhline(y=0, color="#FFD740", linestyle="-", alpha=0.5)
ax2.set_ylabel("Alpha Cumulativo (%)")
ax2.set_title("Alpha vs Equal-Weight Benchmark",
              fontweight="bold", color="#AAAAAA")

# Pannello 3: Equity per settore
ax3 = axes[2]
colors_s = ["#00E5FF", "#69F0AE", "#FFA726", "#AB47BC", "#FF6B6B"]
for i, (sn, sr) in enumerate(sector_results.items()):
    cum = (1 + sr["port_ret"]).cumprod()
    ax3.plot(cum.index, cum, color=colors_s[i % len(colors_s)],
             alpha=0.7, linewidth=1.2, label=sn)
ax3.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
ax3.set_ylabel("Cumulative ($1)")
ax3.set_title("AE Smart Beta per Settore",
              fontweight="bold", color="#AAAAAA")
ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e")

# Pannello 4: Drawdown
ax4 = axes[3]
dd_agg = (agg_port_cum - agg_port_cum.cummax()) / agg_port_cum.cummax()
dd_ew = (agg_ew_cum - agg_ew_cum.cummax()) / agg_ew_cum.cummax()
ax4.fill_between(dd_agg.index, dd_agg * 100, 0, alpha=0.4,
                 color="#00E5FF", label=f"AE (MDD={m_agg_ae['mdd']*100:.1f}%)")
ax4.plot(dd_ew.index, dd_ew * 100, color="#FF6B6B", linewidth=1.0,
         alpha=0.7, linestyle="--", label=f"EW (MDD={m_agg_ew['mdd']*100:.1f}%)")
ax4.set_ylabel("Drawdown (%)")
ax4.set_xlabel("Data")
ax4.set_title("Drawdown Comparativo",
              fontweight="bold", color="#AAAAAA")
ax4.legend(loc="lower left", framealpha=0.3, facecolor="#1a1a2e")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("ae_production.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("  Salvato: ae_production.png")


# ── Sector breakdown chart ──
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7))

sectors_names = list(sector_metrics.keys())
ae_rets = [sector_metrics[s]["ae"]["total"] * 100 for s in sectors_names]
ew_rets = [sector_metrics[s]["ew"]["total"] * 100 for s in sectors_names]
alphas = [sector_metrics[s]["alpha"] * 100 for s in sectors_names]

x = np.arange(len(sectors_names))
width = 0.35

bars1 = ax_a.bar(x - width/2, ae_rets, width, color="#00E5FF", alpha=0.8,
                 label="AE Smart Beta", edgecolor="white", linewidth=0.5)
bars2 = ax_a.bar(x + width/2, ew_rets, width, color="#FF6B6B", alpha=0.8,
                 label="EW B&H", edgecolor="white", linewidth=0.5)
ax_a.set_xlabel("Settore")
ax_a.set_ylabel("Return Totale (%)")
ax_a.set_title("Return per Settore", fontweight="bold", color="#AAAAAA")
ax_a.set_xticks(x)
ax_a.set_xticklabels(sectors_names, rotation=30)
ax_a.legend(framealpha=0.3, facecolor="#1a1a2e")
ax_a.axhline(y=0, color="white", alpha=0.3)

alpha_colors = ["#69F0AE" if a > 0 else "#FF5252" for a in alphas]
ax_b.bar(sectors_names, alphas, color=alpha_colors, alpha=0.8,
         edgecolor="white", linewidth=0.5)
ax_b.axhline(y=0, color="white", alpha=0.3)
ax_b.set_ylabel("Alpha vs B&H (%)")
ax_b.set_title("Alpha per Settore", fontweight="bold", color="#AAAAAA")
ax_b.tick_params(axis="x", rotation=30)

for i, a in enumerate(alphas):
    ax_b.text(i, a + (0.3 if a >= 0 else -0.5), f"{a:+.1f}%",
              ha="center", fontsize=10, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("ae_production_sectors.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()
print("  Salvato: ae_production_sectors.png")


print("\n" + "=" * 80)
print("  AE SMART BETA PRODUCTION — COMPLETATO")
print(f"  Alpha aggregato: {alpha_agg*100:+.2f}%  |  t-stat: {t_stat:.3f}  |"
      f"  p-value: {p_val_alpha:.4f}")
print("=" * 80)
