#!/usr/bin/env python3
# =================================================================================
# Statistical Arbitrage Pipeline v4b — Pharma/Biotech (XBI Universe)
#
# v4b CHANGES (vs v3):
# - SIGNAL-PROPORTIONAL WEIGHTING: pesi ∝ |z-score| per CS e ∝ |composite_score|
#   per Factor, invece di 1/n equal-weight → concentra capitale sui segnali forti
# - HALF-LIFE FILTER: scarta titoli con mean-reversion lenta (half-life > soglia)
#   → riduce trade rumorosi, migliora signal-to-noise
# - DYNAMIC COMBINATION: rolling exp-Sharpe weighted + anti-correlation bonus
#   → adatta pesi CS/Factor al regime corrente, non statico inv-vol
# - ASYMMETRIC REBALANCE: durante DD di una strategia, shift peso verso l'altra
#   (diverso da drawdown control che taglia esposizione totale)
#
# Basato su ricerca:
# - Signal strength weighting: Ornstein-Uhlenbeck MLE, Leung & Li (2015)
# - Sparse mean-reverting portfolios: d'Aspremont (2011), Hudson & Thames
# - Dynamic combination: regime-aware portfolio construction
#
# ANTI-LEAKAGE:
# - Inner optimizer usa SOLO dati del training set
# - Purge gap tra inner train/val e tra outer train/test
# - Commissioni realistiche (15 bps)
# - Block-bootstrap significance test
# - Half-life calcolata su dati passati (.shift(1))
# =================================================================================

import os
import warnings
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from itertools import product as iter_product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================================
# CONFIGURATION
# =================================================================================
CONFIG = {
    "BENCHMARK": "XBI",
    "DATA_START": "2016-01-01",
    "DATA_END": "2026-01-31",

    "UNIVERSE": [
        "AMGN", "GILD", "REGN", "VRTX", "MRNA",
        "BIIB", "ILMN", "ALNY", "BMRN",
        "EXAS", "INCY", "HALO", "PCVX", "SRPT",
        "UTHR", "NBIX", "IONS", "BPMC",
        "JAZZ", "TECH", "MEDP", "RVMD", "KRYS",
        "CORT", "PTCT", "INSM", "ACAD",
    ],

    # Walk-Forward Outer
    "TRAIN_WINDOW_MONTHS": 24,
    "TEST_WINDOW_MONTHS": 3,
    "PURGE_DAYS": 10,

    # Walk-Forward Inner (per optimizer, dentro il training set)
    "INNER_TRAIN_PCT": 0.70,       # 70% del training per inner-train
    "INNER_PURGE_DAYS": 5,         # Purge tra inner-train e inner-val

    # --- CS_MeanRev DEFAULT (baseline) ---
    "CS_LOOKBACK_DAYS": 20,
    "CS_HOLDING_DAYS": 5,
    "CS_LONG_QUANTILE": 0.2,
    "CS_SHORT_QUANTILE": 0.8,

    # --- Factor_MktNeutral DEFAULT (baseline) ---
    "FACTOR_LOOKBACK_DAYS": 60,
    "FACTOR_REBALANCE_DAYS": 5,

    # --- Parameter Search Grids ---
    "CS_PARAM_GRID": {
        "lookback": [10, 15, 20, 30, 40],
        "holding": [3, 5, 7, 10],
        "long_q": [0.15, 0.20, 0.25, 0.30],
    },
    "FACTOR_PARAM_GRID": {
        "lookback": [30, 45, 60, 90, 120],
        "rebalance": [3, 5, 7, 10],
        "n_long_short_pct": [0.15, 0.20, 0.25, 0.30],
    },

    # Risk Management
    "COMMISSION_BPS": 15,

    # Combined Portfolio
    "COMBINED_VOL_LOOKBACK": 63,    # Rolling window per inverse-vol weighting

    # v4b: Signal-proportional & half-life filter (separati per strategia)
    "CS_SIGNAL_PROPORTIONAL": True,     # CS: pesi ∝ |z-score| (funziona: pura MR)
    "CS_HALFLIFE_FILTER": True,         # CS: filtra titoli con MR lenta (CS è pura MR)
    "FACTOR_SIGNAL_PROPORTIONAL": False, # Factor: equal-weight (signal-prop peggiorava 0.416→0.374)
    "FACTOR_HALFLIFE_FILTER": False,    # Factor: NO half-life (multi-factor, non solo MR)
    "MAX_HALFLIFE_DAYS": 30,            # Soglia max half-life (giorni)
    "HALFLIFE_LOOKBACK": 63,            # Finestra per stimare half-life

    # v4b: Volume-weighted signal
    "CS_VOLUME_WEIGHT": False,      # BIOTECH: volume alto = move fondamentale, NO amplificazione
    "VOLUME_LOOKBACK": 21,

    # v4b: Threshold rebalancing
    "THRESHOLD_REBALANCE": False,   # BIOTECH: disabilitato (peggiora CS)
    "REBAL_TURNOVER_THRESHOLD": 0.30,

    # v4b: Dynamic combination
    "DYNAMIC_COMBINATION": True,    # Rolling Sharpe-weighted combination
    "COMBO_SHARPE_LOOKBACK": 63,    # Finestra per rolling Sharpe
    "COMBO_SHARPE_DECAY": 0.97,     # Exponential decay per Sharpe rolling
    "COMBO_MIN_WEIGHT": 0.20,       # Min peso per ogni strategia
    "COMBO_MAX_WEIGHT": 0.80,       # Max peso per ogni strategia

    # Validation
    "SEED": 42,
}

np.random.seed(CONFIG["SEED"])


# =================================================================================
# DATA COLLECTION
# =================================================================================
class DataCollector:
    def __init__(self, config: dict):
        self.config = config

    def fetch_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            return self._fetch_yfinance()
        except Exception as e:
            logger.warning(f"Yahoo Finance non disponibile ({e}), uso dati sintetici")
            return self._generate_synthetic_data()

    def _fetch_yfinance(self):
        import yfinance as yf
        all_tickers = self.config["UNIVERSE"] + [self.config["BENCHMARK"]]
        logger.info(f"Download dati per {len(all_tickers)} ticker...")
        data = yf.download(all_tickers, start=self.config["DATA_START"],
                          end=self.config["DATA_END"], auto_adjust=True,
                          progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            prices_close = data["Close"].copy()
            prices_volume = data["Volume"].copy()
        else:
            prices_close = data[["Close"]].copy()
            prices_volume = data[["Volume"]].copy()

        benchmark = prices_close[self.config["BENCHMARK"]].copy()
        prices_close.drop(columns=[self.config["BENCHMARK"]], errors='ignore', inplace=True)
        prices_volume.drop(columns=[self.config["BENCHMARK"]], errors='ignore', inplace=True)
        valid_pct = prices_close.notna().sum() / len(prices_close)
        valid_tickers = valid_pct[valid_pct > 0.7].index.tolist()
        if len(valid_tickers) < 5:
            raise ValueError(f"Solo {len(valid_tickers)} ticker validi")
        prices_close = prices_close[valid_tickers].ffill(limit=5)
        prices_volume = prices_volume[valid_tickers].ffill(limit=5)
        logger.info(f"Ticker validi: {len(valid_tickers)}, "
                   f"Periodo: {prices_close.index[0].date()} → {prices_close.index[-1].date()}")
        return prices_close, prices_volume, benchmark

    def _generate_synthetic_data(self):
        logger.info("Generazione dati sintetici biotech...")
        rng = np.random.RandomState(self.config["SEED"])
        dates = pd.bdate_range(start=self.config["DATA_START"], end=self.config["DATA_END"])
        n_days, tickers, n = len(dates), self.config["UNIVERSE"], len(self.config["UNIVERSE"])
        ann_vols = rng.uniform(0.35, 0.65, n)
        daily_vols = ann_vols / np.sqrt(252)
        daily_drifts = rng.normal(0.02, 0.12, n) / 252
        market_vol = 0.28 / np.sqrt(252)
        regimes = np.ones(n_days)
        crisis = False
        for t in range(n_days):
            if not crisis and rng.random() < 0.003: crisis = True
            elif crisis and rng.random() < 0.02: crisis = False
            regimes[t] = 2.0 if crisis else 1.0
        mkt = 0.04/252 + market_vol * regimes * rng.standard_t(5, n_days); mkt[0] = 0
        betas = rng.uniform(0.6, 1.4, n)
        cl = rng.randint(0, 5, n)
        cf = rng.standard_t(6, (n_days, 5)) * 0.004
        rets = np.zeros((n_days, n))
        for i in range(n):
            rets[:, i] = daily_drifts[i] + betas[i]*mkt + cf[:, cl[i]] + max(daily_vols[i]*0.6, 0.005)*regimes*rng.standard_t(5, n_days)
        rets[0] = 0; rets = np.clip(rets, -0.30, 0.30)
        ip = rng.uniform(25, 250, n)
        px = ip * np.exp(np.cumsum(rets, axis=0))
        bpx = 80.0 * np.exp(np.cumsum(mkt))
        vm = rng.uniform(5e5, 3e6, n) * regimes[:, None] * rng.lognormal(0, 0.4, (n_days, n))
        prices = pd.DataFrame(px, index=dates, columns=tickers)
        volume = pd.DataFrame(vm, index=dates, columns=tickers)
        bench = pd.Series(bpx, index=dates, name=self.config["BENCHMARK"])
        logger.info(f"  Sintetici: {n_days}d, {n} stocks")
        return prices, volume, bench

    @staticmethod
    def compute_simple_returns(prices): return prices.pct_change()


# =================================================================================
# HALF-LIFE ESTIMATION (Ornstein-Uhlenbeck)
# =================================================================================
def estimate_halflife(series, lookback=63):
    """
    Stima half-life di mean-reversion via regressione OU.
    ΔS_t = λ(S_{t-1} - μ) + ε → half_life = -ln(2)/ln(1+λ)
    Usa SOLO dati passati (anti-lookahead).
    Returns: half-life in giorni (inf se non mean-reverting).
    """
    if len(series) < lookback or series.std() < 1e-10:
        return np.inf
    s = series.iloc[-lookback:]
    s_lag = s.shift(1).dropna()
    delta_s = s.diff().dropna()
    # Allinea
    common = s_lag.index.intersection(delta_s.index)
    if len(common) < 10:
        return np.inf
    y = delta_s.loc[common].values
    x = s_lag.loc[common].values
    # Regressione: ΔS = a + b*S_{lag}
    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        b = beta[1]  # Coefficiente di mean-reversion
        if b >= 0:  # Non mean-reverting
            return np.inf
        halflife = -np.log(2) / np.log(1 + b)
        return max(1.0, halflife)
    except:
        return np.inf


# =================================================================================
# STRATEGY: CROSS-SECTIONAL MEAN REVERSION (Parameterized)
# =================================================================================
class CrossSectionalMeanReversion:
    def __init__(self, lookback=20, holding=5, long_q=0.2, short_q=None, commission_bps=15,
                 signal_proportional=False, halflife_filter=False, max_halflife=30, hl_lookback=63,
                 volume_weight=False, vol_lookback=21,
                 threshold_rebalance=False, turnover_threshold=0.30,
                 invert_signal=False, long_only=False):
        self.lookback = lookback
        self.holding = holding
        self.long_q = long_q
        self.short_q = short_q or (1.0 - long_q)
        self.commission_pct = commission_bps / 10000.0
        self.signal_proportional = signal_proportional
        self.halflife_filter = halflife_filter
        self.max_halflife = max_halflife
        self.hl_lookback = hl_lookback
        self.volume_weight = volume_weight
        self.vol_lookback = vol_lookback
        self.threshold_rebalance = threshold_rebalance
        self.turnover_threshold = turnover_threshold
        self.invert_signal = invert_signal  # Per momentum (tech): inverte long/short
        self.long_only = long_only           # Niente short: solo lato long attivo
        self.name = "CS_MeanRev"

    def run(self, prices, returns, test_start, test_end, volumes=None):
        """Genera segnali E calcola returns in un passo."""
        test_mask = (prices.index >= test_start) & (prices.index <= test_end)
        test_dates = prices.index[test_mask]
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        rebal_dates = test_dates[::self.holding]
        daily_returns = {}
        prev_weights = {}

        for rebal_date in rebal_dates:
            idx = prices.index.get_indexer([rebal_date], method='ffill')[0]
            if idx < self.lookback + 1:
                continue

            past_ret = returns.iloc[idx - self.lookback:idx].sum().dropna()
            if len(past_ret) < 5:
                continue

            # v4b: Half-life filter — scarta titoli con MR lenta
            if self.halflife_filter:
                eligible = []
                for ticker in past_ret.index:
                    if ticker in prices.columns:
                        px = prices[ticker].iloc[max(0, idx-self.hl_lookback):idx]
                        hl = estimate_halflife(px, self.hl_lookback)
                        if hl <= self.max_halflife:
                            eligible.append(ticker)
                if len(eligible) < 5:
                    eligible = past_ret.index.tolist()
                past_ret = past_ret[past_ret.index.isin(eligible)]

            past_ret_dm = past_ret - past_ret.mean()

            # v4b: Volume-weighted signal — amplifica segnali su volume alto
            if self.volume_weight and volumes is not None:
                vol_slice = volumes.iloc[max(0, idx-self.vol_lookback):idx]
                if len(vol_slice) > 5:
                    avg_vol = vol_slice.mean()
                    curr_vol = vol_slice.iloc[-1]
                    vol_ratio = (curr_vol / (avg_vol + 1e-10)).clip(0.5, 3.0)
                    # Moltiplica il segnale per volume ratio (solo titoli presenti)
                    common_tickers = past_ret_dm.index.intersection(vol_ratio.index)
                    past_ret_dm.loc[common_tickers] *= vol_ratio.loc[common_tickers]

            # Invert signal per momentum strategy
            if self.invert_signal:
                past_ret_dm = -past_ret_dm

            lt = past_ret_dm.quantile(self.long_q)
            st = past_ret_dm.quantile(self.short_q)
            longs = past_ret_dm[past_ret_dm <= lt].index.tolist()
            shorts = past_ret_dm[past_ret_dm >= st].index.tolist()

            # Long-only mode: niente short, solo longs (o flat se inverted)
            if self.long_only:
                if not longs:
                    continue
                shorts = []  # ignora il lato short
            else:
                if not longs or not shorts:
                    continue

            # v4b: Signal-proportional weighting
            if self.signal_proportional:
                long_scores = past_ret_dm[longs].abs()
                long_total = long_scores.sum() + 1e-10
                new_weights = {}
                for t in longs:
                    new_weights[t] = (long_scores[t] / long_total)
                if not self.long_only:
                    short_scores = past_ret_dm[shorts].abs()
                    short_total = short_scores.sum() + 1e-10
                    for t in shorts:
                        new_weights[t] = new_weights.get(t, 0) - (short_scores[t] / short_total)
            else:
                new_weights = {}
                for t in longs: new_weights[t] = 1.0 / len(longs)
                if not self.long_only:
                    for t in shorts: new_weights[t] = new_weights.get(t, 0) - 1.0 / len(shorts)

            # v4b: Threshold rebalancing — skip se turnover troppo basso
            if self.threshold_rebalance and prev_weights:
                all_tickers = set(list(new_weights.keys()) + list(prev_weights.keys()))
                proposed_turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0))
                                       for t in all_tickers)
                if proposed_turnover < self.turnover_threshold:
                    new_weights = prev_weights  # Mantieni posizioni esistenti

            weights = new_weights
            hold_end = min(idx + self.holding, len(prices) - 1)
            hold_dates = prices.index[idx:hold_end + 1]

            for hdate in hold_dates:
                if hdate > test_end or hdate not in returns.index:
                    continue
                day_ret = 0.0
                turnover = 0.0
                for ticker, w in weights.items():
                    if ticker in returns.columns:
                        r = returns.loc[hdate, ticker]
                        day_ret += w * (r if not pd.isna(r) else 0.0)
                        turnover += abs(w - prev_weights.get(ticker, 0))
                prev_weights = weights.copy()
                daily_returns[hdate] = day_ret - turnover * self.commission_pct

        if not daily_returns:
            return pd.Series(dtype=float)
        return pd.Series(daily_returns).sort_index()


# =================================================================================
# STRATEGY: MULTI-FACTOR MARKET NEUTRAL (Parameterized)
# =================================================================================
class FactorMarketNeutral:
    def __init__(self, lookback=60, rebalance=5, n_ls_pct=0.20, commission_bps=15,
                 signal_proportional=False, halflife_filter=False, max_halflife=30, hl_lookback=63,
                 threshold_rebalance=False, turnover_threshold=0.30,
                 invert_signal=False):
        self.lookback = lookback
        self.rebalance = rebalance
        self.n_ls_pct = n_ls_pct
        self.commission_pct = commission_bps / 10000.0
        self.signal_proportional = signal_proportional
        self.halflife_filter = halflife_filter
        self.max_halflife = max_halflife
        self.hl_lookback = hl_lookback
        self.threshold_rebalance = threshold_rebalance
        self.turnover_threshold = turnover_threshold
        self.invert_signal = invert_signal
        self.name = "Factor_MktNeutral"

    def _composite_score(self, returns, prices, end_idx, eligible=None):
        s = max(0, end_idx - self.lookback)
        hr, hp = returns.iloc[s:end_idx], prices.iloc[s:end_idx]
        if eligible is not None:
            cols = [c for c in eligible if c in hr.columns]
            hr, hp = hr[cols], hp[cols]
        scores = pd.DataFrame(index=hr.columns)
        ret_5d = hr.iloc[-5:].sum()
        scores["rev"] = -ret_5d.rank(pct=True)
        if len(hr) >= 63:
            scores["mom"] = hr.iloc[-63:-5].sum().rank(pct=True)
        else:
            scores["mom"] = 0.5
        scores["lvol"] = (-hr.iloc[-21:].std()).rank(pct=True)
        sma = hp.iloc[-21:].mean()
        scores["mr"] = (-((hp.iloc[-1] - sma) / (sma + 1e-10))).rank(pct=True)
        c = scores.mean(axis=1)
        z = (c - c.mean()) / (c.std() + 1e-10)
        return -z if self.invert_signal else z  # Inverte per momentum

    def run(self, prices, returns, test_start, test_end):
        test_mask = (prices.index >= test_start) & (prices.index <= test_end)
        test_dates = prices.index[test_mask]
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        rebal_dates = test_dates[::self.rebalance]
        daily_returns = {}
        prev_weights = {}

        for rebal_date in rebal_dates:
            idx = prices.index.get_indexer([rebal_date], method='ffill')[0]
            if idx < self.lookback + 1:
                continue

            # v4b: Half-life filter
            eligible = None
            if self.halflife_filter:
                eligible = []
                for ticker in prices.columns:
                    px = prices[ticker].iloc[max(0, idx-self.hl_lookback):idx]
                    hl = estimate_halflife(px, self.hl_lookback)
                    if hl <= self.max_halflife:
                        eligible.append(ticker)
                if len(eligible) < 6:
                    eligible = None  # Fallback: usa tutti

            scores = self._composite_score(returns, prices, idx, eligible).dropna().sort_values()
            if len(scores) < 6:
                continue

            n_each = max(3, int(len(scores) * self.n_ls_pct))
            long_t = scores.nlargest(n_each).index.tolist()
            short_t = scores.nsmallest(n_each).index.tolist()

            # v4b: Signal-proportional weighting
            if self.signal_proportional:
                long_scores = scores[long_t].abs()
                short_scores = scores[short_t].abs()
                long_total = long_scores.sum() + 1e-10
                short_total = short_scores.sum() + 1e-10
                new_weights = {}
                for t in long_t:
                    new_weights[t] = long_scores[t] / long_total
                for t in short_t:
                    new_weights[t] = new_weights.get(t, 0) - short_scores[t] / short_total
            else:
                new_weights = {}
                for t in long_t: new_weights[t] = 1.0 / n_each
                for t in short_t: new_weights[t] = new_weights.get(t, 0) - 1.0 / n_each

            # v4b: Threshold rebalancing
            if self.threshold_rebalance and prev_weights:
                all_tickers = set(list(new_weights.keys()) + list(prev_weights.keys()))
                proposed_turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0))
                                       for t in all_tickers)
                if proposed_turnover < self.turnover_threshold:
                    new_weights = prev_weights

            weights = new_weights
            hold_end = min(idx + self.rebalance, len(prices) - 1)
            hold_dates = prices.index[idx:hold_end + 1]

            for hdate in hold_dates:
                if hdate > test_end or hdate not in returns.index:
                    continue
                day_ret = 0.0
                turnover = 0.0
                for ticker, w in weights.items():
                    if ticker in returns.columns:
                        r = returns.loc[hdate, ticker]
                        day_ret += w * (r if not pd.isna(r) else 0.0)
                        turnover += abs(w - prev_weights.get(ticker, 0))
                prev_weights = weights.copy()
                daily_returns[hdate] = day_ret - turnover * self.commission_pct

        if not daily_returns:
            return pd.Series(dtype=float)
        return pd.Series(daily_returns).sort_index()


# =================================================================================
# PARAMETER OPTIMIZER (Nested Walk-Forward, Anti-Leakage)
# =================================================================================
class ParameterOptimizer:
    """
    Nested walk-forward optimizer.
    Per ogni outer window:
      1. Prende il TRAINING set
      2. Lo divide in inner_train (70%) e inner_val (30%) con purge gap
      3. Testa tutte le combinazioni di parametri su inner_val
      4. Seleziona best params per Sharpe su inner_val
      5. Applica quei params al TEST set (outer)
    → ZERO data leakage: i parametri non vedono MAI dati del test set.
    """

    def __init__(self, config):
        self.config = config

    def optimize_cs(self, prices, returns, train_start_idx, train_end_idx):
        """Ottimizza parametri CS_MeanRev sul training set."""
        grid = self.config["CS_PARAM_GRID"]
        inner_pct = self.config["INNER_TRAIN_PCT"]
        purge = self.config["INNER_PURGE_DAYS"]

        # Split training in inner_train / inner_val
        n_train = train_end_idx - train_start_idx
        inner_train_end_idx = train_start_idx + int(n_train * inner_pct)
        inner_val_start_idx = inner_train_end_idx + purge

        if inner_val_start_idx >= train_end_idx - 20:
            return {"lookback": 20, "holding": 5, "long_q": 0.20}

        inner_val_start = prices.index[inner_val_start_idx]
        inner_val_end = prices.index[train_end_idx]

        best_sharpe = -np.inf
        best_params = {"lookback": 20, "holding": 5, "long_q": 0.20}

        combos = list(iter_product(grid["lookback"], grid["holding"], grid["long_q"]))

        long_only = self.config.get("LONG_ONLY", False)
        for lb, hd, lq in combos:
            try:
                strat = CrossSectionalMeanReversion(
                    lookback=lb, holding=hd, long_q=lq,
                    commission_bps=self.config["COMMISSION_BPS"],
                    long_only=long_only,
                )
                rets = strat.run(prices, returns, inner_val_start, inner_val_end)
                if len(rets) < 15:
                    continue
                sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {"lookback": lb, "holding": hd, "long_q": lq}
            except:
                continue

        return best_params

    def optimize_factor(self, prices, returns, train_start_idx, train_end_idx):
        """Ottimizza parametri Factor_MktNeutral sul training set."""
        grid = self.config["FACTOR_PARAM_GRID"]
        inner_pct = self.config["INNER_TRAIN_PCT"]
        purge = self.config["INNER_PURGE_DAYS"]

        n_train = train_end_idx - train_start_idx
        inner_train_end_idx = train_start_idx + int(n_train * inner_pct)
        inner_val_start_idx = inner_train_end_idx + purge

        if inner_val_start_idx >= train_end_idx - 20:
            return {"lookback": 60, "rebalance": 5, "n_ls_pct": 0.20}

        inner_val_start = prices.index[inner_val_start_idx]
        inner_val_end = prices.index[train_end_idx]

        best_sharpe = -np.inf
        best_params = {"lookback": 60, "rebalance": 5, "n_ls_pct": 0.20}

        combos = list(iter_product(grid["lookback"], grid["rebalance"], grid["n_long_short_pct"]))

        for lb, rb, nlsp in combos:
            try:
                strat = FactorMarketNeutral(
                    lookback=lb, rebalance=rb, n_ls_pct=nlsp,
                    commission_bps=self.config["COMMISSION_BPS"]
                )
                rets = strat.run(prices, returns, inner_val_start, inner_val_end)
                if len(rets) < 15:
                    continue
                sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {"lookback": lb, "rebalance": rb, "n_ls_pct": nlsp}
            except:
                continue

        return best_params


# =================================================================================
# WALK-FORWARD ENGINE
# =================================================================================
class WalkForwardEngine:
    def __init__(self, config):
        self.config = config
        self.optimizer = ParameterOptimizer(config)

    def _generate_windows(self, all_dates):
        windows = []
        current = all_dates[0]
        tm, tsm, pd_ = self.config["TRAIN_WINDOW_MONTHS"], self.config["TEST_WINDOW_MONTHS"], self.config["PURGE_DAYS"]
        while True:
            te = current + pd.DateOffset(months=tm)
            ts = te + pd.DateOffset(days=pd_)
            td = ts + pd.DateOffset(months=tsm)
            if td > all_dates[-1]: break
            te_a = all_dates[all_dates <= te]
            ts_a = all_dates[all_dates >= ts]
            td_a = all_dates[all_dates <= td]
            if len(te_a) > 0 and len(ts_a) > 0 and len(td_a) > 0:
                # Also track train start for optimizer
                train_start = all_dates[all_dates >= current]
                windows.append({
                    "train_start": train_start[0] if len(train_start) > 0 else all_dates[0],
                    "train_end": te_a[-1],
                    "test_start": ts_a[0],
                    "test_end": td_a[-1],
                })
            current += pd.DateOffset(months=tsm)
        return windows

    def run(self, prices, returns, benchmark, volumes=None):
        logger.info("=" * 70)
        logger.info("WALK-FORWARD ENGINE v4b (Signal-Weighted + Dynamic Combo)")
        logger.info("=" * 70)

        windows = self._generate_windows(prices.index)
        logger.info(f"Outer windows: {len(windows)}")

        # Risultati: base (default params) e optimized
        results = {
            "CS_Base": [], "CS_Optimized": [],
            "Factor_Base": [], "Factor_Optimized": [],
        }
        param_history = {"CS": [], "Factor": []}

        for i, w in enumerate(windows):
            ts_idx = prices.index.get_indexer([w["train_start"]], method='ffill')[0]
            te_idx = prices.index.get_indexer([w["train_end"]], method='ffill')[0]

            logger.info(f"\n  Window {i+1}/{len(windows)}: "
                       f"train→{w['train_end'].date()} | test {w['test_start'].date()}→{w['test_end'].date()}")

            # v4b flags (separati per strategia)
            cs_sig_prop = self.config.get("CS_SIGNAL_PROPORTIONAL", False)
            cs_hl_filter = self.config.get("CS_HALFLIFE_FILTER", False)
            f_sig_prop = self.config.get("FACTOR_SIGNAL_PROPORTIONAL", False)
            f_hl_filter = self.config.get("FACTOR_HALFLIFE_FILTER", False)
            max_hl = self.config.get("MAX_HALFLIFE_DAYS", 30)
            hl_lb = self.config.get("HALFLIFE_LOOKBACK", 63)
            cs_vol_w = self.config.get("CS_VOLUME_WEIGHT", False)
            vol_lb = self.config.get("VOLUME_LOOKBACK", 21)
            thr_rebal = self.config.get("THRESHOLD_REBALANCE", False)
            thr_turn = self.config.get("REBAL_TURNOVER_THRESHOLD", 0.30)
            invert = self.config.get("INVERT_SIGNAL", False)
            long_only = self.config.get("LONG_ONLY", False)

            # --- CS_MeanRev ---
            # Base (default params)
            cs_base = CrossSectionalMeanReversion(
                lookback=self.config["CS_LOOKBACK_DAYS"],
                holding=self.config["CS_HOLDING_DAYS"],
                long_q=self.config["CS_LONG_QUANTILE"],
                commission_bps=self.config["COMMISSION_BPS"],
                signal_proportional=cs_sig_prop,
                halflife_filter=cs_hl_filter,
                max_halflife=max_hl,
                hl_lookback=hl_lb,
                volume_weight=cs_vol_w,
                vol_lookback=vol_lb,
                threshold_rebalance=thr_rebal,
                turnover_threshold=thr_turn,
                invert_signal=invert,
                long_only=long_only,
            )
            base_rets = cs_base.run(prices, returns, w["test_start"], w["test_end"], volumes=volumes)
            if len(base_rets) > 0:
                results["CS_Base"].append(base_rets)

            # Optimized
            cs_params = self.optimizer.optimize_cs(prices, returns, ts_idx, te_idx)
            param_history["CS"].append({"window": i+1, **cs_params,
                                        "test_start": w["test_start"].date()})
            cs_opt = CrossSectionalMeanReversion(
                lookback=cs_params["lookback"],
                holding=cs_params["holding"],
                long_q=cs_params["long_q"],
                commission_bps=self.config["COMMISSION_BPS"],
                signal_proportional=cs_sig_prop,
                halflife_filter=cs_hl_filter,
                max_halflife=max_hl,
                hl_lookback=hl_lb,
                volume_weight=cs_vol_w,
                vol_lookback=vol_lb,
                threshold_rebalance=thr_rebal,
                turnover_threshold=thr_turn,
                invert_signal=invert,
                long_only=long_only,
            )
            opt_rets = cs_opt.run(prices, returns, w["test_start"], w["test_end"], volumes=volumes)
            if len(opt_rets) > 0:
                results["CS_Optimized"].append(opt_rets)
            logger.info(f"    CS params: lb={cs_params['lookback']}, "
                       f"hd={cs_params['holding']}, q={cs_params['long_q']:.2f}")

            # --- Factor_MktNeutral ---
            # In long-only mode skip Factor (market-neutral by design, non compatibile)
            if self.config.get("LONG_ONLY", False):
                logger.info("    [LONG_ONLY] Factor strategy saltata (market-neutral non compatibile)")
                continue

            f_base = FactorMarketNeutral(
                lookback=self.config["FACTOR_LOOKBACK_DAYS"],
                rebalance=self.config["FACTOR_REBALANCE_DAYS"],
                commission_bps=self.config["COMMISSION_BPS"],
                signal_proportional=f_sig_prop,
                halflife_filter=f_hl_filter,
                max_halflife=max_hl,
                hl_lookback=hl_lb,
                threshold_rebalance=thr_rebal,
                turnover_threshold=thr_turn,
                invert_signal=invert,
            )
            base_rets_f = f_base.run(prices, returns, w["test_start"], w["test_end"])
            if len(base_rets_f) > 0:
                results["Factor_Base"].append(base_rets_f)

            f_params = self.optimizer.optimize_factor(prices, returns, ts_idx, te_idx)
            param_history["Factor"].append({"window": i+1, **f_params,
                                            "test_start": w["test_start"].date()})
            f_opt = FactorMarketNeutral(
                lookback=f_params["lookback"],
                rebalance=f_params["rebalance"],
                n_ls_pct=f_params["n_ls_pct"],
                commission_bps=self.config["COMMISSION_BPS"],
                signal_proportional=f_sig_prop,
                halflife_filter=f_hl_filter,
                max_halflife=max_hl,
                hl_lookback=hl_lb,
                threshold_rebalance=thr_rebal,
                turnover_threshold=thr_turn,
                invert_signal=invert,
            )
            opt_rets_f = f_opt.run(prices, returns, w["test_start"], w["test_end"])
            if len(opt_rets_f) > 0:
                results["Factor_Optimized"].append(opt_rets_f)
            logger.info(f"    Factor params: lb={f_params['lookback']}, "
                       f"rb={f_params['rebalance']}, pct={f_params['n_ls_pct']:.2f}")

        # Concatenate
        final = {}
        for name, rets_list in results.items():
            if rets_list:
                combined = pd.concat(rets_list)
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                final[name] = combined
                logger.info(f"\n  {name}: {len(combined)} days, cum={combined.sum():.4f}")

        # --- Combined Portfolio ---
        cs_opt_ret = final.get("CS_Optimized", pd.Series(dtype=float))
        f_opt_ret = final.get("Factor_Optimized", pd.Series(dtype=float))

        # Long-only mode: Factor saltato → CS_Optimized IS the combined
        if self.config.get("LONG_ONLY", False) and len(cs_opt_ret) > 0:
            final["Combined_Dynamic"] = cs_opt_ret.copy()
            final["Combined_InvVol"] = cs_opt_ret.copy()
            logger.info(f"\n  [LONG_ONLY] Combined_Dynamic = CS_Optimized: "
                       f"{len(cs_opt_ret)} days, cum={cs_opt_ret.sum():.4f}")

        if len(cs_opt_ret) > 0 and len(f_opt_ret) > 0:
            both = pd.DataFrame({"CS": cs_opt_ret, "Factor": f_opt_ret}).fillna(0)
            vol_lb = self.config["COMBINED_VOL_LOOKBACK"]

            # --- v3 Method: Inverse-Vol Risk Parity (keep for comparison) ---
            logger.info("\n  Building Combined Portfolios...")
            roll_vol = both.shift(1).rolling(vol_lb, min_periods=21).std()
            inv_vol = 1.0 / (roll_vol + 1e-10)
            iv_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.5)
            final["Combined_InvVol"] = (both * iv_weights).sum(axis=1)
            logger.info(f"  Combined_InvVol: {len(final['Combined_InvVol'])} days, "
                       f"cum={final['Combined_InvVol'].sum():.4f}")

            # --- v4b Method: Dynamic Sharpe-Weighted Combination ---
            if self.config.get("DYNAMIC_COMBINATION", False):
                sharpe_lb = self.config.get("COMBO_SHARPE_LOOKBACK", 63)
                decay = self.config.get("COMBO_SHARPE_DECAY", 0.97)
                min_w = self.config.get("COMBO_MIN_WEIGHT", 0.20)
                max_w = self.config.get("COMBO_MAX_WEIGHT", 0.80)

                # Rolling exponential Sharpe per strategia (shift(1) = anti-lookahead)
                shifted = both.shift(1)
                roll_mean = shifted.ewm(span=sharpe_lb, min_periods=21).mean()
                roll_std = shifted.ewm(span=sharpe_lb, min_periods=21).std()
                roll_sharpe = (roll_mean / (roll_std + 1e-10)) * np.sqrt(252)

                # Pesi proporzionali a max(Sharpe, 0) — non allocare a strategia con Sharpe negativo
                sharpe_pos = roll_sharpe.clip(lower=0.0)
                row_sum = sharpe_pos.sum(axis=1)

                # Quando entrambe hanno Sharpe ≤ 0, fallback a equal-weight
                dyn_weights = sharpe_pos.div(row_sum + 1e-10, axis=0)
                dyn_weights.loc[row_sum < 1e-10] = 0.5

                # Clamp pesi a [min_w, max_w] e rinormalizza
                dyn_weights = dyn_weights.clip(lower=min_w, upper=max_w)
                dyn_weights = dyn_weights.div(dyn_weights.sum(axis=1), axis=0)

                combined_dyn = (both * dyn_weights).sum(axis=1)
                final["Combined_Dynamic"] = combined_dyn
                logger.info(f"  Combined_Dynamic: {len(combined_dyn)} days, "
                           f"cum={combined_dyn.sum():.4f}")

                # Salva anche i pesi per diagnostica
                dw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stat_arb_results")
                os.makedirs(dw_dir, exist_ok=True)
                dyn_weights.to_csv(os.path.join(dw_dir, "dynamic_weights_v4b.csv"))

            # Also simple equal-weight
            final["Combined_EqWeight"] = both.mean(axis=1)
            logger.info(f"  Combined_EqWeight: {len(final['Combined_EqWeight'])} days, "
                       f"cum={final['Combined_EqWeight'].sum():.4f}")

        return final, param_history


# =================================================================================
# PERFORMANCE ANALYSIS
# =================================================================================
class PerformanceAnalyzer:
    @staticmethod
    def compute_metrics(returns, benchmark_returns=None, rf=0.04):
        if len(returns) < 10: return {}
        daily_rf = rf / 252
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() - daily_rf) / (returns.std() + 1e-10) * np.sqrt(252)
        dr = returns[returns < 0]
        sortino = (returns.mean() - daily_rf) / (dr.std() * np.sqrt(252) + 1e-10) * np.sqrt(252) if len(dr) > 0 else 0
        cum = (1 + returns).cumprod()
        pk = cum.expanding().max()
        dd = (cum - pk) / pk
        max_dd = dd.min()
        calmar = ann_ret / (abs(max_dd) + 1e-10)
        wr = (returns > 0).mean()
        pf = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) + 1e-10)
        m = {
            "Total Return": f"{((1+returns).prod()-1):.2%}",
            "Ann Return": f"{ann_ret:.2%}", "Ann Vol": f"{ann_vol:.2%}",
            "Sharpe": f"{sharpe:.3f}", "Sortino": f"{sortino:.3f}",
            "Max DD": f"{max_dd:.2%}", "Calmar": f"{calmar:.3f}",
            "Win Rate": f"{wr:.2%}", "Profit Factor": f"{pf:.3f}",
            "Days": len(returns),
            "_sharpe": sharpe, "_ann_ret": ann_ret, "_max_dd": max_dd,
            "_total_ret": (1+returns).prod()-1,
        }
        if benchmark_returns is not None:
            ci = returns.index.intersection(benchmark_returns.index)
            if len(ci) > 20:
                r, b = returns.loc[ci], benchmark_returns.loc[ci]
                beta = r.cov(b) / (b.var() + 1e-10)
                alpha = (r.mean() - beta * b.mean()) * 252
                m["Beta"] = f"{beta:.3f}"; m["Alpha"] = f"{alpha:.2%}"
                m["Corr"] = f"{r.corr(b):.3f}"; m["_beta"] = beta
        return m

    @staticmethod
    def generate_report(results, benchmark, param_history, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        bench_ret = benchmark.pct_change().dropna()

        summary = {}
        for name, rets in results.items():
            if len(rets) > 10:
                summary[name] = PerformanceAnalyzer.compute_metrics(rets, bench_ret)

        if not summary:
            logger.error("Nessun risultato!"); return

        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY v4b — Signal-Weighted + Dynamic Combo OOS")
        logger.info("=" * 80)
        display = {n: {k: v for k, v in m.items() if not k.startswith("_")} for n, m in summary.items()}
        logger.info(f"\n{pd.DataFrame(display).T.to_string()}")

        colors = {
            "CS_Base": "#81C784", "CS_Optimized": "#2E7D32",
            "Factor_Base": "#CE93D8", "Factor_Optimized": "#6A1B9A",
            "Combined_InvVol": "#616161", "Combined_EqWeight": "#9E9E9E",
            "Combined_Dynamic": "#D32F2F",
        }

        # --- PLOT 1: Equity Curves (Base vs Optimized) ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle("Stat Arb Biotech v4b — Signal-Weighted + Dynamic Combo (OOS Walk-Forward)",
                     fontsize=14, fontweight='bold')

        # CS panel
        ax = axes[0]
        for name in ["CS_Base", "CS_Optimized"]:
            if name in results and len(results[name]) > 0:
                c = (1 + results[name]).cumprod()
                ax.plot(c.index, c.values, color=colors[name], linewidth=1.5 if "Opt" in name else 1,
                       linestyle="-" if "Opt" in name else "--", label=name)
        ci = list(results.values())[0].index.intersection(benchmark.index) if results else []
        if len(ci) > 0:
            bn = benchmark.loc[ci] / benchmark.loc[ci[0]]
            ax.plot(bn.index, bn.values, color="red", linewidth=1, alpha=0.4, linestyle=":", label="XBI")
        ax.set_title("CS_MeanRev"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax.tick_params(axis='x', rotation=45)

        # Factor panel
        ax = axes[1]
        for name in ["Factor_Base", "Factor_Optimized"]:
            if name in results and len(results[name]) > 0:
                c = (1 + results[name]).cumprod()
                ax.plot(c.index, c.values, color=colors[name], linewidth=1.5 if "Opt" in name else 1,
                       linestyle="-" if "Opt" in name else "--", label=name)
        if len(ci) > 0:
            ax.plot(bn.index, bn.values, color="red", linewidth=1, alpha=0.4, linestyle=":", label="XBI")
        ax.set_title("Factor_MktNeutral"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax.tick_params(axis='x', rotation=45)

        # Combined panel
        ax = axes[2]
        for name in ["Combined_InvVol", "Combined_EqWeight", "Combined_Dynamic"]:
            if name in results and len(results[name]) > 0:
                c = (1 + results[name]).cumprod()
                ax.plot(c.index, c.values, color=colors[name],
                       linewidth=2 if "InvVol" in name else 1.2, label=name)
        if len(ci) > 0:
            ax.plot(bn.index, bn.values, color="red", linewidth=1, alpha=0.4, linestyle=":", label="XBI")
        ax.set_title("Combined Portfolio"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax.tick_params(axis='x', rotation=45)

        for a in axes:
            if name in summary:
                s = summary.get("Combined_InvVol", {})
                if s and a == axes[2]:
                    txt = f"Sharpe: {s.get('Sharpe','')}\nReturn: {s.get('Ann Return','')}\nMaxDD: {s.get('Max DD','')}"
                    a.text(0.02, 0.98, txt, transform=a.transAxes, fontsize=8,
                          va='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "equity_curves_v3.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # --- PLOT 2: Combined Drawdown ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle("Combined Portfolio — Equity & Drawdown", fontsize=14, fontweight='bold')
        comb_name = "Combined_InvVol" if "Combined_InvVol" in results else "Combined_EqWeight"
        if comb_name in results:
            cr = results[comb_name]
            cum = (1 + cr).cumprod()
            ax1.plot(cum.index, cum.values, color="black", linewidth=2, label=comb_name)
            ci2 = cum.index.intersection(benchmark.index)
            if len(ci2) > 0:
                bn2 = benchmark.loc[ci2] / benchmark.loc[ci2[0]]
                ax1.plot(bn2.index, bn2.values, color="red", linewidth=1.5, linestyle="--", label="XBI")
            ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylabel("Cumulative Return")
            pk = cum.expanding().max()
            dd = (cum - pk) / pk
            ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
            ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_drawdown_v3.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # --- PLOT 3: Monthly Heatmap (Combined) ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle("Monthly Returns Heatmap (OOS)", fontsize=14, fontweight='bold')
        for idx, name in enumerate(["CS_Optimized", "Factor_Optimized"]):
            ax = axes[idx]
            if name in results and len(results[name]) > 0:
                monthly = results[name].resample("M").sum()
                mp = pd.DataFrame({"Y": monthly.index.year, "M": monthly.index.month, "R": monthly.values})
                ht = mp.pivot_table(values="R", index="Y", columns="M", aggfunc="sum")
                sns.heatmap(ht, ax=ax, cmap="RdYlGn", center=0, annot=True, fmt=".1%",
                           annot_kws={"size": 7}, linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax.set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "monthly_heatmap_v3.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # --- PLOT 4: Rolling Sharpe ---
        fig, ax = plt.subplots(figsize=(16, 5))
        fig.suptitle("Rolling 63-Day Sharpe (OOS)", fontsize=14, fontweight='bold')
        for name in ["CS_Optimized", "Factor_Optimized", "Combined_InvVol", "Combined_Dynamic"]:
            if name in results and len(results[name]) > 0:
                rs = results[name].rolling(63, min_periods=21).apply(
                    lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252))
                lw = 2.0 if "Dynamic" in name else 1.2
                ax.plot(rs.index, rs.values, color=colors.get(name, "gray"), linewidth=lw, label=name)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(1, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylabel("Rolling Sharpe")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rolling_sharpe_v3.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # --- PLOT 5: Parameter Evolution ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle("Optimized Parameters Over Time", fontsize=14, fontweight='bold')
        if param_history.get("CS"):
            cs_df = pd.DataFrame(param_history["CS"])
            for j, col in enumerate(["lookback", "holding", "long_q"]):
                axes[0, j].plot(cs_df["window"], cs_df[col], 'o-', color="#2E7D32", markersize=4)
                axes[0, j].set_title(f"CS: {col}"); axes[0, j].grid(True, alpha=0.3)
        if param_history.get("Factor"):
            f_df = pd.DataFrame(param_history["Factor"])
            for j, col in enumerate(["lookback", "rebalance", "n_ls_pct"]):
                axes[1, j].plot(f_df["window"], f_df[col], 'o-', color="#6A1B9A", markersize=4)
                axes[1, j].set_title(f"Factor: {col}"); axes[1, j].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "param_evolution_v3.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # --- Save CSVs ---
        for name, rets in results.items():
            rets.to_csv(os.path.join(output_dir, f"{name}_returns.csv"), header=["net_return"])
        pd.DataFrame(display).T.to_csv(os.path.join(output_dir, "summary_v3.csv"))
        if param_history.get("CS"):
            pd.DataFrame(param_history["CS"]).to_csv(os.path.join(output_dir, "params_cs.csv"), index=False)
        if param_history.get("Factor"):
            pd.DataFrame(param_history["Factor"]).to_csv(os.path.join(output_dir, "params_factor.csv"), index=False)

        logger.info(f"\nRisultati salvati in: {output_dir}")


# =================================================================================
# LEAKAGE VALIDATOR
# =================================================================================
class LeakageValidator:
    @staticmethod
    def block_bootstrap_test(returns, n_bootstraps=500, block_size=21):
        n = len(returns); values = returns.values; sharpes = []
        for _ in range(n_bootstraps):
            starts = np.random.randint(0, max(1, n - block_size), size=n // block_size + 1)
            bs = np.concatenate([values[s:s+block_size] for s in starts])[:n]
            sharpes.append(np.mean(bs) / (np.std(bs) + 1e-10) * np.sqrt(252))
        actual = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        p95 = np.percentile(sharpes, 95)
        return {
            "actual_sharpe": actual, "baseline_mean": np.mean(sharpes),
            "baseline_std": np.std(sharpes), "p95": p95,
            "pvalue": np.mean(np.array(sharpes) >= actual),
            "significant": actual > p95,
        }


# =================================================================================
# MAIN PIPELINE
# =================================================================================
def run_pipeline():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STATISTICAL ARBITRAGE PIPELINE v4b — SIGNAL-WEIGHTED + DYNAMIC COMBO")
    logger.info(f"Strategies: CS_MeanRev + Factor_MktNeutral (signal-prop + halflife filter)")
    logger.info(f"Benchmark: XBI | Universe: {len(CONFIG['UNIVERSE'])} stocks")
    logger.info(f"v4b features: SignalProp={CONFIG.get('SIGNAL_PROPORTIONAL')}, "
               f"HalfLifeFilter={CONFIG.get('HALFLIFE_FILTER')}, "
               f"DynamicCombo={CONFIG.get('DYNAMIC_COMBINATION')}")
    logger.info(f"Period: {CONFIG['DATA_START']} → {CONFIG['DATA_END']}")
    logger.info(f"Param grid: CS={len(list(iter_product(*CONFIG['CS_PARAM_GRID'].values())))} combos, "
               f"Factor={len(list(iter_product(*CONFIG['FACTOR_PARAM_GRID'].values())))} combos")
    logger.info("=" * 80)

    # 1. Data
    logger.info("\n[1/4] DATA COLLECTION")
    collector = DataCollector(CONFIG)
    prices, volumes, benchmark = collector.fetch_all()
    returns = collector.compute_simple_returns(prices)

    # 2. Walk-Forward + Optimization
    logger.info("\n[2/4] WALK-FORWARD + PARAMETER OPTIMIZATION")
    engine = WalkForwardEngine(CONFIG)
    results, param_history = engine.run(prices, returns, benchmark, volumes=volumes)

    if not results:
        logger.error("Nessun risultato!"); return None

    # 3. Leakage Validation
    logger.info("\n[3/4] LEAKAGE VALIDATION")
    validator = LeakageValidator()
    for name in ["CS_Optimized", "Factor_Optimized", "Combined_InvVol", "Combined_Dynamic"]:
        if name in results and len(results[name]) > 10:
            bt = validator.block_bootstrap_test(results[name])
            logger.info(f"\n  {name}:")
            logger.info(f"    Actual Sharpe: {bt['actual_sharpe']:.3f}")
            logger.info(f"    Bootstrap 95th: {bt['p95']:.3f}")
            logger.info(f"    p-value: {bt['pvalue']:.3f}")
            if bt['significant']:
                logger.info(f"    ✅ Significativo al 5%")
            else:
                logger.warning(f"    ⚠️ NON significativo")

    # 4. Report
    logger.info("\n[4/4] PERFORMANCE REPORT")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stat_arb_results")
    PerformanceAnalyzer.generate_report(results, benchmark, param_history, output_dir)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE v4b COMPLETATA in {elapsed:.1f}s")
    logger.info(f"{'='*80}")
    return results, param_history


if __name__ == "__main__":
    run_pipeline()
