#!/usr/bin/env python3
# =================================================================================
# MEGA-PORTFOLIO — Biotech Mean-Reversion + Tech Momentum
#
# Combina due strategie decorrelate:
#   1. Biotech MR (Combined_Dynamic): Sharpe 0.599, +316%, Beta ~0
#   2. Tech Momentum (Combined_Dynamic): Sharpe 0.234, +70.5%, Beta ~0.08
#
# La correlazione attesa è ~0 (universi diversi, logiche opposte).
# Sharpe combinato atteso: sqrt(0.599² + 0.234²) ≈ 0.643 (se corr=0)
#
# Metodi di combinazione testati:
#   - Equal-Weight (50/50)
#   - Inverse-Volatility
#   - Dynamic Sharpe-Weighted
#   - Risk-Parity (equal risk contribution)
#
# ANTI-LEAKAGE: tutti i pesi calcolati con .shift(1)
# =================================================================================

import os
import sys
import warnings
import time
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =================================================================================
# LOAD RETURNS
# =================================================================================
def load_returns():
    """Carica i returns delle pipeline: Biotech MR + Tech Momentum + Macro + Commodity."""
    bio_dir       = os.path.join(SCRIPT_DIR, "stat_arb_results")
    bio_cheap_dir = os.path.join(SCRIPT_DIR, "stat_arb_results_bio_cheap")
    tech_dir      = os.path.join(SCRIPT_DIR, "stat_arb_results_tech")
    macro_dir     = os.path.join(SCRIPT_DIR, "stat_arb_results_macro")
    commodity_dir = os.path.join(SCRIPT_DIR, "stat_arb_results_commodity")

    # Cerca il miglior combined per ogni pipeline
    bio_candidates      = ["Combined_Dynamic_returns.csv", "Combined_InvVol_returns.csv"]
    tech_candidates     = ["Combined_Dynamic_returns.csv", "Combined_InvVol_returns.csv"]
    macro_candidates    = ["Macro_Blend_returns.csv", "TSMOM_returns.csv"]
    # Factor_Optimized prima: su large-cap il fattore composite funziona, CS no
    bio_cheap_candidates = ["Factor_Optimized_returns.csv", "Combined_Dynamic_returns.csv"]

    bio_ret = None
    for fname in bio_candidates:
        path = os.path.join(bio_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            bio_ret = df.iloc[:, 0].dropna()
            logger.info(f"  Biotech MR loaded: {fname} ({len(bio_ret)} days)")
            break

    tech_ret = None
    for fname in tech_candidates:
        path = os.path.join(tech_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            tech_ret = df.iloc[:, 0].dropna()
            logger.info(f"  Tech Momentum loaded: {fname} ({len(tech_ret)} days)")
            break

    # v6: Terza gamba — Macro Momentum
    macro_ret = None
    for fname in macro_candidates:
        path = os.path.join(macro_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            macro_ret = df.iloc[:, 0].dropna()
            logger.info(f"  Macro Momentum loaded: {fname} ({len(macro_ret)} days)")
            break

    # v7: Quarta opzione — Cheap Biotech (long-only, liquid universe)
    bio_cheap_ret = None
    for fname in bio_cheap_candidates:
        path = os.path.join(bio_cheap_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            bio_cheap_ret = df.iloc[:, 0].dropna()
            logger.info(f"  Bio Cheap (Factor L/S) loaded: {fname} ({len(bio_cheap_ret)} days)")
            break
    if bio_cheap_ret is None:
        logger.info(f"  Bio Cheap non trovato in {bio_cheap_dir} — run stat_arb_biotech_cheap.py prima")
    logger.info(f"    (carica Factor_Optimized: Factor funziona su large-cap, CS no)")

    if bio_ret is None:
        logger.error(f"Biotech returns non trovati in {bio_dir}")
        sys.exit(1)
    if tech_ret is None:
        logger.error(f"Tech returns non trovati in {tech_dir}")
        sys.exit(1)
    if macro_ret is None:
        logger.warning(f"Macro returns non trovati in {macro_dir} — running 2-leg mode")

    # Carica anche le sotto-strategie per analisi più granulare
    components = {}
    for name, directory in [("Bio_CS", bio_dir), ("Bio_Factor", bio_dir),
                            ("Tech_CS", tech_dir), ("Tech_Factor", tech_dir)]:
        prefix = "CS_Optimized" if "CS" in name else "Factor_Optimized"
        path = os.path.join(directory, f"{prefix}_returns.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            components[name] = df.iloc[:, 0].dropna()

    # Aggiungi sotto-componenti macro
    for fname in ["TSMOM_returns.csv", "CSMOM_returns.csv"]:
        path = os.path.join(macro_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            name = fname.replace("_returns.csv", "")
            components[f"Macro_{name}"] = df.iloc[:, 0].dropna()

    # v8: Commodity TSMOM (quarta gamba genuinamente decorrelata)
    commodity_ret = None
    for fname in ["Commodity_TSMOM_returns.csv"]:
        path = os.path.join(commodity_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            commodity_ret = df.iloc[:, 0].dropna()
            logger.info(f"  Commodity TSMOM loaded: {fname} ({len(commodity_ret)} days)")
            break
    if commodity_ret is None:
        logger.info(f"  Commodity non trovato — run stat_arb_commodity.py prima")

    return bio_ret, tech_ret, macro_ret, components, bio_cheap_ret, commodity_ret


# =================================================================================
# PERFORMANCE METRICS
# =================================================================================
def compute_metrics(returns, name="", rf=0.04):
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns = returns.squeeze()
    if len(returns) < 10:
        return {}
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
    return {
        "Strategy": name,
        "Total Return": f"{((1+returns).prod()-1):.2%}",
        "Ann Return": f"{ann_ret:.2%}",
        "Ann Vol": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.3f}",
        "Sortino": f"{sortino:.3f}",
        "Max DD": f"{max_dd:.2%}",
        "Calmar": f"{calmar:.3f}",
        "Win Rate": f"{wr:.2%}",
        "Profit Factor": f"{pf:.3f}",
        "Days": len(returns),
        "_sharpe": sharpe, "_ann_ret": ann_ret, "_max_dd": max_dd,
        "_total_ret": (1+returns).prod()-1, "_ann_vol": ann_vol,
    }


# =================================================================================
# COMBINATION METHODS
# =================================================================================
def combine_equal_weight(bio, tech):
    """50/50 equal weight."""
    both = pd.DataFrame({"Bio": bio, "Tech": tech}).fillna(0)
    return both.mean(axis=1)


def combine_inverse_vol(bio, tech, lookback=63):
    """Inverse-volatility risk parity."""
    both = pd.DataFrame({"Bio": bio, "Tech": tech}).fillna(0)
    roll_vol = both.shift(1).rolling(lookback, min_periods=21).std()
    inv_vol = 1.0 / (roll_vol + 1e-10)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.5)
    return (both * weights).sum(axis=1), weights


def combine_dynamic_sharpe(bio, tech, lookback=63, min_w=0.15, max_w=0.85):
    """Dynamic Sharpe-weighted combination."""
    both = pd.DataFrame({"Bio": bio, "Tech": tech}).fillna(0)
    shifted = both.shift(1)
    roll_mean = shifted.ewm(span=lookback, min_periods=21).mean()
    roll_std = shifted.ewm(span=lookback, min_periods=21).std()
    roll_sharpe = (roll_mean / (roll_std + 1e-10)) * np.sqrt(252)

    sharpe_pos = roll_sharpe.clip(lower=0.0)
    row_sum = sharpe_pos.sum(axis=1)
    dyn_weights = sharpe_pos.div(row_sum + 1e-10, axis=0)
    dyn_weights.loc[row_sum < 1e-10] = 0.5
    dyn_weights = dyn_weights.clip(lower=min_w, upper=max_w)
    dyn_weights = dyn_weights.div(dyn_weights.sum(axis=1), axis=0)

    return (both * dyn_weights).sum(axis=1), dyn_weights


def combine_risk_parity(bio, tech, lookback=63):
    """
    Equal Risk Contribution — ogni strategia contribuisce lo stesso rischio.
    w_i ∝ 1/σ_i, ma aggiustato per correlazione.
    """
    both = pd.DataFrame({"Bio": bio, "Tech": tech}).fillna(0)
    shifted = both.shift(1)
    roll_vol = shifted.rolling(lookback, min_periods=21).std()
    roll_corr = shifted["Bio"].rolling(lookback, min_periods=21).corr(shifted["Tech"])

    # Risk parity weights: w_i = 1/σ_i normalizzato
    # (semplificato: per 2 asset, questo è equivalente a inverse-vol
    #  ma con aggiustamento per correlazione)
    inv_vol = 1.0 / (roll_vol + 1e-10)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.5)

    # Aggiustamento correlazione: quando corr è alta, riduci entrambi
    # (meno diversificazione → meno rischio totale desiderato)
    corr_adj = (1 - roll_corr.fillna(0).clip(-0.5, 0.9)) / 2 + 0.5  # [0.55, 0.75]
    total_exposure = corr_adj.clip(0.5, 1.0)
    weights = weights.multiply(total_exposure, axis=0)

    return (both * weights).sum(axis=1), weights, roll_corr


# =================================================================================
# v6 N-ASSET COMBINATION (supports 2 or 3 legs)
# =================================================================================

def combine_n_equal_weight(legs):
    """Equal-weight N assets."""
    df = pd.DataFrame(legs).fillna(0)
    return df.mean(axis=1)


def combine_n_inverse_vol(legs, lookback=63):
    """Inverse-volatility N assets."""
    df = pd.DataFrame(legs).fillna(0)
    shifted = df.shift(1)
    roll_vol = shifted.rolling(lookback, min_periods=21).std()
    inv_vol = 1.0 / (roll_vol + 1e-10)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1.0 / len(legs))
    return (df * weights).sum(axis=1), weights


def combine_n_min_variance(legs, lookback=63):
    """
    Minimum Variance per N assets — usa rolling covariance matrix.
    Per N>2, closed-form: w = Σ^{-1} * 1 / (1' * Σ^{-1} * 1)
    """
    df = pd.DataFrame(legs).fillna(0)
    shifted = df.shift(1)
    n = len(legs)

    # Rolling covariance matrix
    all_weights = []
    for i in range(len(df)):
        if i < max(lookback, 21):
            all_weights.append([1.0 / n] * n)
            continue

        window = shifted.iloc[max(0, i - lookback):i].dropna()
        if len(window) < 21:
            all_weights.append([1.0 / n] * n)
            continue

        cov = window.cov().values
        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-10)
            ones = np.ones(n)
            w = inv_cov @ ones / (ones @ inv_cov @ ones + 1e-10)
            # Clamp
            w = np.clip(w, 0.05, 0.80)
            w = w / w.sum()
        except:
            w = np.ones(n) / n

        all_weights.append(w.tolist())

    weights = pd.DataFrame(all_weights, index=df.index, columns=df.columns)
    return (df * weights).sum(axis=1), weights


def combine_n_max_sharpe(legs, lookback=63, rf=0.04):
    """
    Maximum Sharpe Ratio portfolio per N assets.
    w = Σ^{-1} * (μ - rf) / (1' * Σ^{-1} * (μ - rf))
    Più aggressivo del MinVar: massimizza il rendimento per unità di rischio.
    """
    df = pd.DataFrame(legs).fillna(0)
    shifted = df.shift(1)
    n = len(legs)
    daily_rf = rf / 252

    all_weights = []
    for i in range(len(df)):
        if i < max(lookback, 42):
            all_weights.append([1.0 / n] * n)
            continue

        window = shifted.iloc[max(0, i - lookback):i].dropna()
        if len(window) < 42:
            all_weights.append([1.0 / n] * n)
            continue

        cov = window.cov().values
        mu = window.mean().values - daily_rf  # excess return

        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-10)
            w = inv_cov @ mu
            # Se tutti i rendimenti sono negativi, fallback a MinVar
            if w.sum() < 1e-10:
                ones = np.ones(n)
                w = inv_cov @ ones / (ones @ inv_cov @ ones + 1e-10)
            else:
                w = w / (w.sum() + 1e-10)
            # Clamp + allow short (ma limitato)
            w = np.clip(w, -0.10, 0.80)
            w = w / (np.abs(w).sum() + 1e-10)  # normalize by gross exposure
        except:
            w = np.ones(n) / n

        all_weights.append(w.tolist())

    weights = pd.DataFrame(all_weights, index=df.index, columns=df.columns)
    return (df * weights).sum(axis=1), weights


# =================================================================================
# v5 NEW TECHNIQUES — Sharpe improvement
# =================================================================================

def combine_min_variance(bio, tech, lookback=63):
    """
    Minimum Variance Portfolio — usa la matrice di covarianza completa.
    Per 2 asset: w_bio = (σ²_tech - ρ*σ_bio*σ_tech) / (σ²_bio + σ²_tech - 2*ρ*σ_bio*σ_tech)
    Questo è il portafoglio con la volatilità più bassa possibile.
    """
    both = pd.DataFrame({"Bio": bio, "Tech": tech}).fillna(0)
    shifted = both.shift(1)

    roll_var_bio = shifted["Bio"].rolling(lookback, min_periods=21).var()
    roll_var_tech = shifted["Tech"].rolling(lookback, min_periods=21).var()
    roll_cov = shifted["Bio"].rolling(lookback, min_periods=21).cov(shifted["Tech"])

    # Minimum variance closed-form per 2 asset
    denom = roll_var_bio + roll_var_tech - 2 * roll_cov
    w_bio = (roll_var_tech - roll_cov) / (denom + 1e-10)

    # Clamp weights tra 0.15 e 0.85
    w_bio = w_bio.clip(0.15, 0.85).fillna(0.5)
    w_tech = 1.0 - w_bio

    weights = pd.DataFrame({"Bio": w_bio, "Tech": w_tech}, index=both.index)
    return (both * weights).sum(axis=1), weights


def apply_vol_target(returns, target_vol=0.15, lookback=42, max_leverage=1.5, min_leverage=0.3):
    """
    Volatility Targeting — post-processing overlay.
    Scala l'esposizione per mantenere la volatilità annualizzata vicina al target.
    Letteratura: aggiunge ~0.1-0.2 Sharpe in media (Moreira & Muir 2017).

    NOTA: questo è un OVERLAY sul portafoglio combinato, NON sulle singole strategie.
    Non interferisce con la generazione dei segnali MR.
    """
    shifted_ret = returns.shift(1)
    realized_vol = shifted_ret.ewm(span=lookback, min_periods=10).std() * np.sqrt(252)

    # Leverage = target_vol / realized_vol
    leverage = target_vol / (realized_vol + 1e-10)
    leverage = leverage.clip(min_leverage, max_leverage).fillna(1.0)

    scaled = returns * leverage
    return scaled, leverage


def apply_drawdown_budget(returns, max_dd_budget=-0.20, recovery_speed=0.05):
    """
    Drawdown Budget — riduce esposizione man mano che ci avviciniamo al budget di DD.
    Se il DD corrente è -15% e il budget è -20%, l'esposizione scende al 25%.
    Questo protegge il capitale nei periodi peggiori.
    """
    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak

    # Esposizione = quanto DD budget rimane (shifted per anti-leakage)
    dd_shifted = dd.shift(1).fillna(0)
    dd_ratio = dd_shifted / max_dd_budget  # 0 = nessun DD, 1 = al budget
    exposure = (1 - dd_ratio).clip(0.1, 1.0)  # Min 10% esposizione

    # Gradual recovery: quando il DD si riduce, rientra lentamente
    exposure = exposure.ewm(span=int(1/recovery_speed), min_periods=1).mean()

    scaled = returns * exposure
    return scaled, exposure


def combine_strategy_momentum(bio, tech, lookback=63, fast_lb=21):
    """
    Strategy Momentum Filter — TAA-style overlay.
    Alloca di più alla strategia che sta performando meglio (momentum delle equity curves).
    Usa due timeframe: veloce (21d) per timing, lento (63d) per trend.
    """
    both = pd.DataFrame({"Bio": bio, "Tech": tech}).fillna(0)
    shifted = both.shift(1)

    # Rolling cumulative return su due orizzonti
    cum_slow = shifted.rolling(lookback, min_periods=21).sum()  # ~3 mesi
    cum_fast = shifted.rolling(fast_lb, min_periods=10).sum()   # ~1 mese

    # Score = blend dei due segnali (0.6 slow + 0.4 fast)
    momentum_score = 0.6 * cum_slow + 0.4 * cum_fast

    # Converti in pesi: softmax con temperatura
    temperature = 0.02  # Controlla quanto aggressivo è il tilting
    exp_scores = np.exp(momentum_score / temperature)
    weights = exp_scores.div(exp_scores.sum(axis=1), axis=0).fillna(0.5)

    # Clamp per evitare concentrazione estrema
    weights = weights.clip(0.20, 0.80)
    weights = weights.div(weights.sum(axis=1), axis=0)

    return (both * weights).sum(axis=1), weights


def combine_adaptive(bio, tech, lookback=63, target_vol=0.15, dd_budget=-0.22):
    """
    METODO ADATTIVO COMPLETO — Combina il meglio di tutto:
    1. Minimum Variance per i pesi base (usa correlazione)
    2. Strategy Momentum tilt (favorisce chi sta andando meglio)
    3. Vol Targeting overlay (mantiene vol costante)
    4. Drawdown Budget (protezione tail risk)

    Questo è il metodo più sofisticato del mega-portfolio.
    """
    both = pd.DataFrame({"Bio": bio, "Tech": tech}).fillna(0)
    shifted = both.shift(1)

    # --- Step 1: Minimum Variance base weights ---
    roll_var_bio = shifted["Bio"].rolling(lookback, min_periods=21).var()
    roll_var_tech = shifted["Tech"].rolling(lookback, min_periods=21).var()
    roll_cov = shifted["Bio"].rolling(lookback, min_periods=21).cov(shifted["Tech"])

    denom = roll_var_bio + roll_var_tech - 2 * roll_cov
    w_bio_mv = (roll_var_tech - roll_cov) / (denom + 1e-10)
    w_bio_mv = w_bio_mv.clip(0.20, 0.80).fillna(0.5)

    # --- Step 2: Strategy Momentum tilt ---
    cum_ret = shifted.rolling(lookback, min_periods=21).sum()
    # Tilt: se bio sta andando meglio, sposta peso verso bio
    bio_advantage = cum_ret["Bio"] - cum_ret["Tech"]
    # Sigmoid per convertire in aggiustamento [-0.15, +0.15]
    momentum_tilt = 0.15 * np.tanh(bio_advantage / 0.05)
    momentum_tilt = momentum_tilt.fillna(0)

    # Pesi finali = MinVar + Momentum Tilt
    w_bio = (w_bio_mv + momentum_tilt).clip(0.20, 0.80)
    w_tech = 1.0 - w_bio

    weights = pd.DataFrame({"Bio": w_bio, "Tech": w_tech}, index=both.index)
    base_returns = (both * weights).sum(axis=1)

    # --- Step 3: Volatility Targeting ---
    vol_scaled, leverage = apply_vol_target(base_returns, target_vol=target_vol)

    # --- Step 4: Drawdown Budget ---
    final_returns, dd_exposure = apply_drawdown_budget(vol_scaled, max_dd_budget=dd_budget)

    return final_returns, weights, leverage, dd_exposure


# =================================================================================
# 4-STRATEGY GRANULAR COMBINATION
# =================================================================================
def combine_four_strategies(components, lookback=63, min_w=0.10, max_w=0.50):
    """
    Combina le 4 sotto-strategie: Bio_CS, Bio_Factor, Tech_CS, Tech_Factor.
    Usa dynamic Sharpe-weighting su 4 assets.
    """
    df = pd.DataFrame(components).fillna(0)
    if df.shape[1] < 4:
        return None, None

    shifted = df.shift(1)
    roll_mean = shifted.ewm(span=lookback, min_periods=21).mean()
    roll_std = shifted.ewm(span=lookback, min_periods=21).std()
    roll_sharpe = (roll_mean / (roll_std + 1e-10)) * np.sqrt(252)

    sharpe_pos = roll_sharpe.clip(lower=0.0)
    row_sum = sharpe_pos.sum(axis=1)
    dyn_weights = sharpe_pos.div(row_sum + 1e-10, axis=0)
    dyn_weights.loc[row_sum < 1e-10] = 0.25
    dyn_weights = dyn_weights.clip(lower=min_w, upper=max_w)
    dyn_weights = dyn_weights.div(dyn_weights.sum(axis=1), axis=0)

    combined = (df * dyn_weights).sum(axis=1)
    return combined, dyn_weights


# =================================================================================
# REPORT
# =================================================================================
def generate_report(results, weights_dict, correlation, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # --- Summary Table ---
    summary = {}
    for name, rets in results.items():
        if len(rets) > 10:
            summary[name] = compute_metrics(rets, name)

    logger.info("\n" + "=" * 80)
    logger.info("MEGA-PORTFOLIO — PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    display = {n: {k: v for k, v in m.items() if not k.startswith("_")} for n, m in summary.items()}
    logger.info(f"\n{pd.DataFrame(display).T.to_string()}")

    # --- PLOT 1: All Equity Curves ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("MEGA-PORTFOLIO v6: Biotech MR + Tech Momentum + Macro", fontsize=16, fontweight='bold')

    colors = {
        "Biotech_MR": "#2E7D32", "Tech_Momentum": "#1565C0",
        "Mega_EqWeight": "#9E9E9E", "Mega_InvVol": "#616161",
        "Mega_DynSharpe": "#D32F2F", "Mega_RiskParity": "#FF6F00",
        "Mega_4Strategy": "#6A1B9A",
        "Macro_Momentum": "#FF6F00",
        "Mega_MinVar": "#00838F", "Mega_StratMom": "#AD1457",
        "Mega_InvVol_VT": "#E65100", "Mega_InvVol_DD": "#4A148C",
        "Mega_Adaptive": "#FF1744", "Mega_BestVT": "#00C853",
        "Mega_N_EqWeight": "#78909C", "Mega_N_InvVol": "#455A64",
        "Mega_N_MinVar": "#006064", "Mega_N_MaxSharpe": "#B71C1C",
        "Mega_N_MinVar_VT": "#E65100",
    }

    # Panel 1: Individual legs + best combined
    ax = axes[0, 0]
    for name in ["Biotech_MR", "Tech_Momentum", "Macro_Momentum",
                 "Mega_N_MinVar", "Mega_BestVT"]:
        if name in results:
            cum = (1 + results[name]).cumprod()
            lw = 2.5 if "Mega" in name else 1.0
            ls = "-" if "Mega" in name else "--"
            ax.plot(cum.index, cum.values, color=colors.get(name, "gray"),
                   linewidth=lw, linestyle=ls, label=name, alpha=0.9 if "Mega" in name else 0.5)
    ax.set_title("Individual Legs vs Best Combined")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

    # Panel 2: N-leg methods comparison
    ax = axes[0, 1]
    v6_methods = ["Mega_N_EqWeight", "Mega_N_InvVol", "Mega_N_MinVar",
                  "Mega_N_MaxSharpe", "Mega_N_MinVar_VT", "Mega_BestVT"]
    for name in v6_methods:
        if name in results:
            cum = (1 + results[name]).cumprod()
            lw = 2.5 if name in ("Mega_N_MinVar", "Mega_BestVT") else 1.0
            ax.plot(cum.index, cum.values, color=colors.get(name, "gray"),
                   linewidth=lw, label=name)
    ax.set_title("v6 N-Leg Methods Comparison")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

    # Panel 3: Drawdown comparison
    ax = axes[1, 0]
    for name in ["Mega_MinVar", "Mega_N_MinVar", "Mega_N_MaxSharpe", "Mega_BestVT"]:
        if name in results:
            cum = (1 + results[name]).cumprod()
            pk = cum.expanding().max()
            dd = (cum - pk) / pk
            ax.fill_between(dd.index, dd.values, 0, alpha=0.2,
                           color=colors.get(name, "gray"), label=name)
            ax.plot(dd.index, dd.values, color=colors.get(name, "gray"),
                   linewidth=0.8, alpha=0.6)
    ax.set_title("Drawdown: 2-Leg vs 3-Leg")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylabel("Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

    # Panel 4: Rolling Correlation + Weights
    ax = axes[1, 1]
    if correlation is not None:
        ax.plot(correlation.index, correlation.values, color="purple",
               linewidth=1.2, alpha=0.7, label="Rolling Corr (Bio/Tech)")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(correlation.mean(), color="purple", linewidth=0.5,
                   linestyle=":", alpha=0.5, label=f"Mean: {correlation.mean():.3f}")
    ax.set_title("Rolling Correlation: Biotech vs Tech")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylabel("Correlation")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mega_portfolio.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\nCharts saved to {output_dir}/mega_portfolio.png")

    # --- PLOT 2: 4-Strategy weights over time ---
    if "4strat_weights" in weights_dict and weights_dict["4strat_weights"] is not None:
        fig, ax = plt.subplots(figsize=(16, 5))
        w4 = weights_dict["4strat_weights"]
        ax.stackplot(w4.index, [w4[c].values for c in w4.columns],
                    labels=w4.columns,
                    colors=["#2E7D32", "#81C784", "#1565C0", "#64B5F6"],
                    alpha=0.8)
        ax.set_title("4-Strategy Dynamic Weights Over Time", fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylabel("Weight"); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mega_4strat_weights.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"4-Strategy weights chart saved")

    # Save CSVs
    for name, rets in results.items():
        rets.to_csv(os.path.join(output_dir, f"{name}_returns.csv"), header=["net_return"])
    pd.DataFrame(display).T.to_csv(os.path.join(output_dir, "mega_summary.csv"))


# =================================================================================
# MAIN
# =================================================================================
def main():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("MEGA-PORTFOLIO COMBINER v6")
    logger.info("Biotech MR + Tech Momentum + Macro Momentum")
    logger.info("=" * 80)

    # 1. Load
    logger.info("\n[1/4] CARICAMENTO RETURNS")
    bio_ret, tech_ret, macro_ret, components, bio_cheap_ret, commodity_ret = load_returns()
    three_leg = macro_ret is not None

    # Allinea alle stesse date
    if three_leg:
        common = bio_ret.index.intersection(tech_ret.index).intersection(macro_ret.index)
        macro_ret = macro_ret.loc[common]
    else:
        common = bio_ret.index.intersection(tech_ret.index)
    bio_ret = bio_ret.loc[common]
    tech_ret = tech_ret.loc[common]
    logger.info(f"  Date comuni: {len(common)} days ({common[0].date()} → {common[-1].date()})")
    logger.info(f"  Mode: {'3-LEG (Bio + Tech + Macro)' if three_leg else '2-LEG (Bio + Tech)'}")

    # 2. Correlation Analysis
    logger.info("\n[2/4] ANALISI CORRELAZIONE")
    corr_bt = bio_ret.corr(tech_ret)
    logger.info(f"  Bio/Tech:   {corr_bt:+.4f}")
    if three_leg:
        corr_bm = bio_ret.corr(macro_ret)
        corr_tm = tech_ret.corr(macro_ret)
        logger.info(f"  Bio/Macro:  {corr_bm:+.4f}")
        logger.info(f"  Tech/Macro: {corr_tm:+.4f}")
        avg_corr = (corr_bt + corr_bm + corr_tm) / 3
        logger.info(f"  Media:      {avg_corr:+.4f}")

    roll_corr = bio_ret.rolling(63, min_periods=21).corr(tech_ret)

    # Individual metrics
    m_bio = compute_metrics(bio_ret, "Biotech_MR")
    m_tech = compute_metrics(tech_ret, "Tech_Momentum")
    logger.info(f"\n  Biotech MR:     Sharpe={m_bio['_sharpe']:.3f}, "
               f"AnnRet={m_bio['_ann_ret']:.2%}, MaxDD={m_bio['_max_dd']:.2%}")
    logger.info(f"  Tech Momentum:  Sharpe={m_tech['_sharpe']:.3f}, "
               f"AnnRet={m_tech['_ann_ret']:.2%}, MaxDD={m_tech['_max_dd']:.2%}")
    if three_leg:
        m_macro = compute_metrics(macro_ret, "Macro_Momentum")
        logger.info(f"  Macro Momentum: Sharpe={m_macro['_sharpe']:.3f}, "
                   f"AnnRet={m_macro['_ann_ret']:.2%}, MaxDD={m_macro['_max_dd']:.2%}")

    # 3. Combine
    logger.info("\n[3/4] COMBINAZIONE STRATEGIE")
    results = {
        "Biotech_MR": bio_ret,
        "Tech_Momentum": tech_ret,
    }
    if three_leg:
        results["Macro_Momentum"] = macro_ret
    weights_dict = {}

    # --- Build legs dict for N-asset methods ---
    legs = {"Bio": bio_ret, "Tech": tech_ret}
    if three_leg:
        legs["Macro"] = macro_ret

    # === 2-LEG METHODS (backward compatible) ===
    logger.info("\n  --- 2-Leg Methods ---")
    results["Mega_EqWeight"] = combine_equal_weight(bio_ret, tech_ret)
    m = compute_metrics(results["Mega_EqWeight"])
    logger.info(f"  EqWeight(2):     Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

    results["Mega_InvVol"], w_iv = combine_inverse_vol(bio_ret, tech_ret)
    m = compute_metrics(results["Mega_InvVol"])
    logger.info(f"  InvVol(2):       Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

    results["Mega_MinVar"], w_mv = combine_min_variance(bio_ret, tech_ret)
    m = compute_metrics(results["Mega_MinVar"])
    logger.info(f"  MinVar(2):       Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

    # === N-LEG METHODS (work for 2 or 3 legs) ===
    logger.info(f"\n  --- {len(legs)}-Leg Methods ---")

    results["Mega_N_EqWeight"] = combine_n_equal_weight(legs)
    m = compute_metrics(results["Mega_N_EqWeight"])
    logger.info(f"  N_EqWeight:      Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

    results["Mega_N_InvVol"], w_niv = combine_n_inverse_vol(legs)
    m = compute_metrics(results["Mega_N_InvVol"])
    logger.info(f"  N_InvVol:        Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

    results["Mega_N_MinVar"], w_nmv = combine_n_min_variance(legs)
    m = compute_metrics(results["Mega_N_MinVar"])
    logger.info(f"  N_MinVar:        Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")
    if w_nmv is not None:
        avg_w = w_nmv.mean()
        logger.info(f"    Pesi medi: {', '.join(f'{c}={avg_w[c]:.1%}' for c in w_nmv.columns)}")

    results["Mega_N_MaxSharpe"], w_nms = combine_n_max_sharpe(legs)
    m = compute_metrics(results["Mega_N_MaxSharpe"])
    logger.info(f"  N_MaxSharpe:     Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")
    if w_nms is not None:
        avg_w = w_nms.mean()
        logger.info(f"    Pesi medi: {', '.join(f'{c}={avg_w[c]:.1%}' for c in w_nms.columns)}")

    # === OVERLAYS on best N-leg method ===
    logger.info("\n  --- Overlays su N_MinVar ---")

    # Vol Target su N_MinVar
    results["Mega_N_MinVar_VT"], lev_vt = apply_vol_target(results["Mega_N_MinVar"], target_vol=0.15)
    m = compute_metrics(results["Mega_N_MinVar_VT"])
    logger.info(f"  N_MinVar+VT:     Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

    # Grid search vol target su N_MinVar
    logger.info("\n  --- Grid Search Vol Target (su N_MinVar) ---")
    best_vt_sharpe = -99
    best_vt_params = {}
    for tv in [0.10, 0.12, 0.14, 0.15, 0.17, 0.20, 0.25]:
        for ml in [1.2, 1.5, 2.0]:
            r_vt, _ = apply_vol_target(results["Mega_N_MinVar"], target_vol=tv, max_leverage=ml)
            m_vt = compute_metrics(r_vt)
            if m_vt['_sharpe'] > best_vt_sharpe:
                best_vt_sharpe = m_vt['_sharpe']
                best_vt_params = {'target_vol': tv, 'max_leverage': ml,
                                  'ann_ret': m_vt['_ann_ret'], 'max_dd': m_vt['_max_dd']}
    logger.info(f"  Best: tv={best_vt_params['target_vol']}, max_lev={best_vt_params['max_leverage']}, "
                f"Sharpe={best_vt_sharpe:.3f}, AnnRet={best_vt_params['ann_ret']:.2%}, "
                f"MaxDD={best_vt_params['max_dd']:.2%}")

    results["Mega_BestVT"], _ = apply_vol_target(
        results["Mega_N_MinVar"],
        target_vol=best_vt_params['target_vol'],
        max_leverage=best_vt_params['max_leverage'])
    m = compute_metrics(results["Mega_BestVT"])
    logger.info(f"  BEST VOL TGT:    Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

    # Method 5b: 4-Strategy Granular (if all components available)
    if len(components) == 4:
        # Allinea components
        comp_aligned = {}
        for name, ret in components.items():
            comp_aligned[name] = ret.reindex(common).fillna(0)
        results["Mega_4Strategy"], w_4s = combine_four_strategies(comp_aligned)
        if results["Mega_4Strategy"] is not None:
            weights_dict["4strat_weights"] = w_4s
            m = compute_metrics(results["Mega_4Strategy"])
            logger.info(f"  4-Strategy:      Sharpe={m['_sharpe']:.3f}, AnnRet={m['_ann_ret']:.2%}, MaxDD={m['_max_dd']:.2%}")

            # Pesi medi
            if w_4s is not None:
                avg_w = w_4s.mean()
                logger.info(f"\n  Pesi medi 4-Strategy:")
                for col in avg_w.index:
                    logger.info(f"    {col}: {avg_w[col]:.1%}")
    else:
        logger.info(f"  4-Strategy: skip (solo {len(components)} componenti disponibili)")

    # === CHEAP BIO COMPARISON (v7) ===
    if bio_cheap_ret is not None:
        logger.info("\n" + "="*80)
        logger.info("  CONFRONTO: BIOTECH ORIGINALE (CS+Factor) vs CHEAP FACTOR-ONLY (large-cap)")
        logger.info("="*80)

        # Allinea cheap bio alle stesse date comuni
        cheap_common = bio_cheap_ret.index.intersection(tech_ret.index)
        if three_leg:
            cheap_common = cheap_common.intersection(macro_ret.index)
        bio_cheap_al = bio_cheap_ret.loc[cheap_common]
        tech_cheap   = tech_ret.loc[cheap_common]
        bio_orig_al  = bio_ret.reindex(cheap_common).fillna(0)

        # Metriche individuali
        m_orig  = compute_metrics(bio_orig_al,  "Bio_Original (L/S)")
        m_cheap = compute_metrics(bio_cheap_al, "Bio_Cheap (Factor)")
        logger.info(f"  {'Strategia':<25} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} {'Vol':>8}")
        logger.info(f"  {'-'*60}")
        for m in [m_orig, m_cheap]:
            if m:
                logger.info(f"  {m['Strategy']:<25} {m['_sharpe']:>8.3f} "
                           f"{m['_ann_ret']:>8.2%} {m['_max_dd']:>8.2%} "
                           f"{m['_ann_vol']:>8.2%}")

        # Correlazione cheap_bio vs tech, macro
        corr_cb_t = bio_cheap_al.corr(tech_cheap)
        logger.info(f"\n  Bio_Cheap / Tech corr: {corr_cb_t:+.4f}")
        if three_leg:
            macro_cheap = macro_ret.loc[cheap_common]
            corr_cb_m = bio_cheap_al.corr(macro_cheap)
            logger.info(f"  Bio_Cheap / Macro corr: {corr_cb_m:+.4f}")

        # N_MinVar con cheap bio (sostituisce originale)
        legs_cheap = {"BioCheap": bio_cheap_al, "Tech": tech_cheap}
        if three_leg:
            legs_cheap["Macro"] = macro_cheap

        r_cheap_nmv, w_cheap_nmv = combine_n_min_variance(legs_cheap)
        m_cnmv = compute_metrics(r_cheap_nmv, "Cheap_N_MinVar")
        logger.info(f"\n  N_MinVar con Bio_Cheap: Sharpe={m_cnmv['_sharpe']:.3f}, "
                   f"AnnRet={m_cnmv['_ann_ret']:.2%}, MaxDD={m_cnmv['_max_dd']:.2%}")
        if w_cheap_nmv is not None:
            avg_w = w_cheap_nmv.mean()
            logger.info(f"    Pesi medi: {', '.join(f'{c}={avg_w[c]:.1%}' for c in w_cheap_nmv.columns)}")

        r_cheap_vt, _ = apply_vol_target(r_cheap_nmv, target_vol=0.15)
        m_cvt = compute_metrics(r_cheap_vt, "Cheap_N_MinVar+VT")
        logger.info(f"  N_MinVar+VT (Bio_Cheap): Sharpe={m_cvt['_sharpe']:.3f}, "
                   f"AnnRet={m_cvt['_ann_ret']:.2%}, MaxDD={m_cvt['_max_dd']:.2%}")

        # N_MinVar con TUTTI e 4 (bio_orig + bio_cheap + tech + macro)
        if three_leg:
            legs_4 = {"Bio": bio_orig_al, "BioCheap": bio_cheap_al,
                      "Tech": tech_cheap,  "Macro": macro_cheap}
            r_4leg_nmv, w_4leg = combine_n_min_variance(legs_4)
            m_4leg = compute_metrics(r_4leg_nmv, "4-Leg_N_MinVar")
            logger.info(f"\n  4-Leg (Bio+Cheap+Tech+Macro) N_MinVar: Sharpe={m_4leg['_sharpe']:.3f}, "
                       f"AnnRet={m_4leg['_ann_ret']:.2%}, MaxDD={m_4leg['_max_dd']:.2%}")
            if w_4leg is not None:
                avg_w = w_4leg.mean()
                logger.info(f"    Pesi medi: {', '.join(f'{c}={avg_w[c]:.1%}' for c in w_4leg.columns)}")

            r_4leg_vt, _ = apply_vol_target(r_4leg_nmv, target_vol=0.15)
            m_4leg_vt = compute_metrics(r_4leg_vt, "4-Leg_N_MinVar+VT")
            logger.info(f"  4-Leg+VT N_MinVar:         Sharpe={m_4leg_vt['_sharpe']:.3f}, "
                       f"AnnRet={m_4leg_vt['_ann_ret']:.2%}, MaxDD={m_4leg_vt['_max_dd']:.2%}")

            # Aggiungi al report
            results["Cheap_N_MinVar"] = r_cheap_nmv
            results["Cheap_N_MinVar_VT"] = r_cheap_vt
            results["4Leg_N_MinVar"] = r_4leg_nmv
            results["4Leg_N_MinVar_VT"] = r_4leg_vt

        logger.info("="*80)

    # === COMMODITY TSMOM (v8) — 4a gamba genuinamente decorrelata ===
    if commodity_ret is not None and three_leg:
        logger.info("\n" + "="*80)
        logger.info("  COMMODITY TSMOM — 4a GAMBA (CL,GC,SI,HG,NG,ZC,ZS,ZW,KC,CC)")
        logger.info("="*80)

        # Allinea commodity alle date comuni
        comm_common = bio_ret.index.intersection(tech_ret.index)\
                                   .intersection(macro_ret.index)\
                                   .intersection(commodity_ret.index)
        if len(comm_common) < 100:
            logger.warning(f"  Overlap insufficiente ({len(comm_common)} giorni) — skip")
        else:
            bio_c    = bio_ret.loc[comm_common]
            tech_c   = tech_ret.loc[comm_common]
            macro_c  = macro_ret.loc[comm_common]
            comm_c   = commodity_ret.loc[comm_common]

            m_comm = compute_metrics(comm_c, "Commodity_TSMOM")
            logger.info(f"  Commodity standalone: Sharpe={m_comm['_sharpe']:.3f}, "
                       f"AnnRet={m_comm['_ann_ret']:.2%}, MaxDD={m_comm['_max_dd']:.2%}, "
                       f"Vol={m_comm['_ann_vol']:.2%}")

            # Correlazioni
            logger.info(f"\n  Correlazioni Commodity vs altre gambe:")
            logger.info(f"    Comm / Bio:   {comm_c.corr(bio_c):+.4f}")
            logger.info(f"    Comm / Tech:  {comm_c.corr(tech_c):+.4f}")
            logger.info(f"    Comm / Macro: {comm_c.corr(macro_c):+.4f}")

            # 4-Leg: Bio + Tech + Macro + Commodity
            legs_4c = {"Bio": bio_c, "Tech": tech_c, "Macro": macro_c, "Comm": comm_c}
            r_4c_nmv, w_4c = combine_n_min_variance(legs_4c)
            m_4c = compute_metrics(r_4c_nmv, "4Leg+Comm_N_MinVar")
            logger.info(f"\n  4-Leg (Bio+Tech+Macro+Comm) N_MinVar:")
            logger.info(f"    Sharpe={m_4c['_sharpe']:.3f}, AnnRet={m_4c['_ann_ret']:.2%}, "
                       f"MaxDD={m_4c['_max_dd']:.2%}")
            if w_4c is not None:
                avg_w = w_4c.mean()
                logger.info(f"    Pesi medi: {', '.join(f'{c}={avg_w[c]:.1%}' for c in w_4c.columns)}")

            r_4c_vt, _ = apply_vol_target(r_4c_nmv, target_vol=0.15)
            m_4c_vt = compute_metrics(r_4c_vt, "4Leg+Comm_N_MinVar+VT")
            logger.info(f"  4-Leg+VT: Sharpe={m_4c_vt['_sharpe']:.3f}, "
                       f"AnnRet={m_4c_vt['_ann_ret']:.2%}, MaxDD={m_4c_vt['_max_dd']:.2%}")

            # Confronto diretto: 3-leg vs 4-leg+commodity
            r_3leg_ref, _ = combine_n_min_variance({"Bio": bio_c, "Tech": tech_c, "Macro": macro_c})
            r_3leg_vt, _  = apply_vol_target(r_3leg_ref, target_vol=0.15)
            m_3ref = compute_metrics(r_3leg_vt, "3Leg_N_MinVar+VT (ref)")
            logger.info(f"\n  CONFRONTO DIRETTO (stesse date):")
            logger.info(f"    3-Leg N_MinVar+VT:  Sharpe={m_3ref['_sharpe']:.3f}")
            logger.info(f"    4-Leg+Comm +VT:     Sharpe={m_4c_vt['_sharpe']:.3f}")
            delta = m_4c_vt['_sharpe'] - m_3ref['_sharpe']
            sign = "✅ migliora" if delta > 0.02 else ("➡ neutro" if abs(delta) <= 0.02 else "❌ peggiora")
            logger.info(f"    Delta Sharpe: {delta:+.3f} → {sign}")

            results["Commodity_TSMOM"]   = comm_c
            results["4Leg_Comm_N_MinVar"]    = r_4c_nmv
            results["4Leg_Comm_N_MinVar_VT"] = r_4c_vt
        logger.info("="*80)

    # 4. Report
    logger.info("\n[4/4] GENERAZIONE REPORT")
    output_dir = os.path.join(SCRIPT_DIR, "mega_portfolio_results")
    generate_report(results, weights_dict, roll_corr, output_dir)

    # Final summary
    best_name = max(
        [(n, m) for n, m in
         {n: compute_metrics(r) for n, r in results.items() if "Mega" in n}.items()],
        key=lambda x: x[1].get("_sharpe", -99)
    )
    logger.info(f"\n  🏆 MIGLIOR MEGA-PORTFOLIO: {best_name[0]}")
    logger.info(f"     Sharpe: {best_name[1]['_sharpe']:.3f}")
    logger.info(f"     Total Return: {best_name[1]['_total_ret']:.2%}")
    logger.info(f"     Max DD: {best_name[1]['_max_dd']:.2%}")
    logger.info(f"     Ann Return: {best_name[1]['_ann_ret']:.2%}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"MEGA-PORTFOLIO COMPLETATO in {elapsed:.1f}s")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
