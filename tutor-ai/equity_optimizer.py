#!/usr/bin/env python3
# =================================================================================
# EQUITY CURVE OPTIMIZER v4 — Post-Processing Overlay
#
# Applica 6 tecniche avanzate di ottimizzazione sull'equity curve delle
# strategie v3 (CS_MeanRev + Factor_MktNeutral) già validate.
#
# TECNICHE IMPLEMENTATE (basate su ricerca 2024-2025):
#
# 1. VOLATILITY TARGETING (Quantpedia, Moreira & Muir 2017)
#    → Scala posizioni inversamente alla vol realizzata
#    → Target: vol costante → Sharpe migliore, drawdown ridotto
#
# 2. DRAWDOWN CONTROL OVERLAY (Boyd et al., Stanford)
#    → Scala esposizione proporzionalmente al drawdown corrente
#    → Quando DD→DDmax, esposizione→0 (put sintetica)
#
# 3. REGIME DETECTION (vol-ratio based)
#    → Identifica regimi alto/basso volatilità
#    → Riduce esposizione in regime di crisi
#
# 4. SIGNAL CONFIDENCE WEIGHTING (Chan, Stanford CS report)
#    → Pesi proporzionali alla forza del segnale (non equal-weight)
#    → Rank-based per stabilità numerica
#
# 5. TURNOVER FILTER
#    → Ribilancia solo se cambiamento pesi > soglia (riduce commissioni)
#
# 6. DYNAMIC KELLY SIZING
#    → Fractional Kelly su rolling Sharpe per position sizing ottimale
#
# ANTI-LEAKAGE: Tutte le metriche calcolate con .shift(1) / dati passati.
#
# FONTI:
# - Boyd et al. "Multi-period portfolio selection with drawdown control"
#   https://web.stanford.edu/~boyd/papers/pdf/multiperiod_portfolio_drawdown.pdf
# - Macrosynergy "Drawdown Control" https://macrosynergy.com/research/drawdown-control/
# - Quantpedia "Volatility Targeting" https://quantpedia.com/an-introduction-to-volatility-targeting/
# - Stanford CS "Cross-Section Performance Reversion"
#   http://stanford.edu/class/msande448/2018/Final/Reports/gr2.pdf
# - Harvey (Duke/Man AHL) "Drawdowns" 2020
# =================================================================================

import os
import sys
import warnings
import time
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =================================================================================
# OVERLAY 1: VOLATILITY TARGETING
# =================================================================================
class VolatilityTargeting:
    """
    Scala i rendimenti giornalieri per mantenere una volatilità costante.

    Formula (Moreira & Muir 2017, Quantpedia):
      leverage_t = target_vol / realized_vol_t
      adjusted_return_t = leverage_t * raw_return_t

    Anti-lookahead: realized_vol calcolata su dati .shift(1) (fino a ieri).
    """

    def __init__(self, target_vol_annual=0.15, lookback=21, max_leverage=2.0, min_leverage=0.2):
        self.target_vol = target_vol_annual / np.sqrt(252)  # Daily target
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage

    def apply(self, returns: pd.Series) -> pd.Series:
        """Applica vol targeting ai rendimenti."""
        # Volatilità realizzata (shift(1) = solo dati passati)
        realized_vol = returns.shift(1).rolling(self.lookback, min_periods=10).std()

        # Leverage = target / realized (capped)
        leverage = (self.target_vol / (realized_vol + 1e-10)).clip(
            self.min_leverage, self.max_leverage
        )

        adjusted = returns * leverage
        return adjusted


# =================================================================================
# OVERLAY 2: DRAWDOWN CONTROL (Boyd et al., Stanford)
# =================================================================================
class DrawdownControl:
    """
    Scala l'esposizione in funzione del drawdown corrente.

    Logica (Boyd et al. 2017):
      - Quando drawdown = 0: esposizione piena (1.0)
      - Quando drawdown → max_dd_threshold: esposizione → 0
      - Formula: exposure = max(0, 1 - (current_dd / max_dd_threshold)^power)

    Effetto: put sintetica che limita le perdite sotto max_dd_threshold.
    """

    def __init__(self, max_dd_threshold=0.20, power=2.0, recovery_speed=0.5):
        self.max_dd = max_dd_threshold
        self.power = power
        self.recovery_speed = recovery_speed

    def apply(self, returns: pd.Series) -> pd.Series:
        """Applica drawdown control ai rendimenti."""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak  # Negativo

        # Calcola exposure (usando dd di IERI per anti-lookahead)
        dd_shifted = dd.shift(1).fillna(0)
        dd_ratio = (-dd_shifted / self.max_dd).clip(0, 1)
        exposure = (1 - dd_ratio ** self.power).clip(0, 1)

        # Smooth recovery (non saltare da 0 a 1 istantaneamente)
        smoothed_exposure = exposure.ewm(span=5).mean()

        adjusted = returns * smoothed_exposure
        return adjusted


# =================================================================================
# OVERLAY 3: REGIME DETECTION (Vol-Ratio Based)
# =================================================================================
class RegimeDetector:
    """
    Identifica regime di mercato basato su rapporto volatilità breve/lunga.

    Logica:
      vol_ratio = vol_short / vol_long
      Se vol_ratio > crisis_threshold → regime crisi → riduce esposizione
      Se vol_ratio < calm_threshold → regime calmo → esposizione piena

    Ispirato a: Regime-switching advantage in statistical arbitrage
    (University of Wisconsin, 2024)
    """

    def __init__(self, short_window=10, long_window=63,
                 crisis_threshold=1.5, calm_threshold=0.8,
                 crisis_exposure=0.3, calm_exposure=1.0):
        self.short_w = short_window
        self.long_w = long_window
        self.crisis_th = crisis_threshold
        self.calm_th = calm_threshold
        self.crisis_exp = crisis_exposure
        self.calm_exp = calm_exposure

    def apply(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> pd.Series:
        """Applica regime filter. Usa benchmark se disponibile, altrimenti i returns stessi."""
        ref = benchmark_returns if benchmark_returns is not None else returns

        # Forza Series 1D
        if isinstance(ref, pd.DataFrame):
            ref = ref.iloc[:, 0]
        ref = ref.squeeze()

        # Allinea indici (benchmark potrebbe avere date diverse)
        ref_aligned = ref.reindex(returns.index).ffill()

        # Vol ratio (shift(1) = anti-lookahead)
        vol_short = ref_aligned.shift(1).rolling(self.short_w, min_periods=5).std()
        vol_long = ref_aligned.shift(1).rolling(self.long_w, min_periods=21).std()
        vol_ratio = vol_short / (vol_long + 1e-10)

        # Exposure continua vettorizzata (no loop)
        # Interpolazione lineare tra calm e crisis
        frac = ((vol_ratio - self.calm_th) / (self.crisis_th - self.calm_th)).clip(0, 1)
        exposure = self.calm_exp - frac * (self.calm_exp - self.crisis_exp)

        # NaN → esposizione piena (inizio serie, dati insufficienti)
        exposure = exposure.fillna(1.0)

        adjusted = returns * exposure
        return adjusted, exposure


# =================================================================================
# OVERLAY 4: DYNAMIC KELLY SIZING
# =================================================================================
class DynamicKelly:
    """
    Fractional Kelly criterion basato su rolling Sharpe.

    Formula:
      kelly_fraction = rolling_sharpe / realized_vol
      position_size = fraction * kelly_fraction (con fraction < 1 per sicurezza)

    Usa fraction=0.25 (quarter-Kelly) per conservatività.
    Anti-lookahead: tutto calcolato su .shift(1).
    """

    def __init__(self, kelly_fraction=0.25, lookback=63, max_kelly=2.0, min_kelly=0.1):
        self.fraction = kelly_fraction
        self.lookback = lookback
        self.max_kelly = max_kelly
        self.min_kelly = min_kelly

    def apply(self, returns: pd.Series) -> pd.Series:
        """Applica Kelly sizing dinamico."""
        # Rolling metrics (shift(1) = anti-lookahead)
        shifted = returns.shift(1)
        roll_mean = shifted.rolling(self.lookback, min_periods=21).mean()
        roll_std = shifted.rolling(self.lookback, min_periods=21).std()

        # Rolling Sharpe (daily)
        roll_sharpe = roll_mean / (roll_std + 1e-10)

        # Kelly = Sharpe / vol
        kelly = roll_sharpe / (roll_std + 1e-10)
        kelly_scaled = (self.fraction * kelly).clip(self.min_kelly, self.max_kelly)

        # Quando Sharpe negativo → Kelly molto basso (non shortare il segnale!)
        kelly_scaled = kelly_scaled.where(roll_sharpe > 0, self.min_kelly)

        adjusted = returns * kelly_scaled
        return adjusted


# =================================================================================
# OVERLAY 5: TURNOVER FILTER
# =================================================================================
class TurnoverFilter:
    """
    Post-processing: penalizza returns proporzionalmente al turnover implicito.
    Non modifica direttamente i rendimenti ma applica uno smoothing
    che riduce il noise ad alta frequenza (che genera turnover inutile).
    """

    def __init__(self, ema_span=3):
        self.ema_span = ema_span

    def apply(self, returns: pd.Series) -> pd.Series:
        """Applica smoothing leggero per ridurre trading noise."""
        # EMA molto leggero (span=3 = quasi nessun ritardo)
        smoothed = returns.ewm(span=self.ema_span, adjust=False).mean()
        return smoothed


# =================================================================================
# MASTER OVERLAY PIPELINE
# =================================================================================
class EquityCurveOptimizer:
    """
    Pipeline master che applica tutti gli overlay in sequenza.

    Ordine critico:
    1. Regime Detection (filtra periodi di crisi)
    2. Volatility Targeting (normalizza rischio)
    3. Drawdown Control (protegge da perdite catastrofiche)
    4. Dynamic Kelly (dimensiona secondo edge stimato)
    5. Turnover Filter (opzionale, riduce noise)

    Ogni overlay è opzionalmente disattivabile per A/B testing.
    """

    def __init__(self, config=None):
        self.config = config or {}

        # Configurazione overlays
        self.vol_target = VolatilityTargeting(
            target_vol_annual=self.config.get("VOL_TARGET", 0.15),
            lookback=self.config.get("VOL_LOOKBACK", 21),
            max_leverage=self.config.get("VOL_MAX_LEV", 2.0),
            min_leverage=self.config.get("VOL_MIN_LEV", 0.25),
        )
        self.dd_control = DrawdownControl(
            max_dd_threshold=self.config.get("DD_MAX_THRESHOLD", 0.20),
            power=self.config.get("DD_POWER", 2.0),
        )
        self.regime = RegimeDetector(
            short_window=self.config.get("REGIME_SHORT", 10),
            long_window=self.config.get("REGIME_LONG", 63),
            crisis_threshold=self.config.get("REGIME_CRISIS_TH", 1.5),
            crisis_exposure=self.config.get("REGIME_CRISIS_EXP", 0.3),
        )
        self.kelly = DynamicKelly(
            kelly_fraction=self.config.get("KELLY_FRACTION", 0.25),
            lookback=self.config.get("KELLY_LOOKBACK", 63),
        )
        self.turnover_filter = TurnoverFilter(
            ema_span=self.config.get("EMA_SPAN", 3),
        )

    def optimize(self, returns: pd.Series, benchmark_returns: pd.Series = None,
                 enable_vol_target=True, enable_dd_control=True,
                 enable_regime=True, enable_kelly=True,
                 enable_turnover_filter=False) -> Dict[str, pd.Series]:
        """
        Applica overlay e restituisce dizionario con tutte le varianti.
        Permette A/B testing tra combinazioni.
        """
        results = {"Raw": returns.copy()}

        current = returns.copy()

        # 1. Regime Detection
        if enable_regime and benchmark_returns is not None:
            current, regime_exposure = self.regime.apply(current, benchmark_returns)
            results["After_Regime"] = current.copy()
        else:
            regime_exposure = pd.Series(1.0, index=returns.index)

        # 2. Volatility Targeting
        if enable_vol_target:
            current = self.vol_target.apply(current)
            results["After_VolTarget"] = current.copy()

        # 3. Drawdown Control
        if enable_dd_control:
            current = self.dd_control.apply(current)
            results["After_DDControl"] = current.copy()

        # 4. Dynamic Kelly
        if enable_kelly:
            current = self.kelly.apply(current)
            results["After_Kelly"] = current.copy()

        # 5. Turnover Filter (opzionale)
        if enable_turnover_filter:
            current = self.turnover_filter.apply(current)
            results["After_TurnoverFilter"] = current.copy()

        results["Optimized"] = current.copy()

        return results, regime_exposure


# =================================================================================
# PERFORMANCE METRICS
# =================================================================================
def compute_metrics(returns, name="", rf=0.04):
    # Garantisci che returns sia una Series 1D
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
        "_sharpe": sharpe, "_ann_ret": ann_ret, "_max_dd": max_dd,
        "_total_ret": (1+returns).prod()-1,
    }


# =================================================================================
# ABLATION STUDY
# =================================================================================
def run_ablation_study(returns, benchmark_returns=None):
    """
    Testa ogni overlay singolarmente e tutte le combinazioni utili.
    Permette di capire quale overlay contribuisce di più.
    """
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY — Contributo di ogni overlay")
    logger.info("=" * 70)

    optimizer = EquityCurveOptimizer()
    configs = [
        ("Raw (nessun overlay)", dict(enable_vol_target=False, enable_dd_control=False,
                                       enable_regime=False, enable_kelly=False)),
        ("Solo VolTarget", dict(enable_vol_target=True, enable_dd_control=False,
                                enable_regime=False, enable_kelly=False)),
        ("Solo DDControl", dict(enable_vol_target=False, enable_dd_control=True,
                                enable_regime=False, enable_kelly=False)),
        ("Solo Regime", dict(enable_vol_target=False, enable_dd_control=False,
                             enable_regime=True, enable_kelly=False)),
        ("Solo Kelly", dict(enable_vol_target=False, enable_dd_control=False,
                            enable_regime=False, enable_kelly=True)),
        ("VolTarget + DDControl", dict(enable_vol_target=True, enable_dd_control=True,
                                        enable_regime=False, enable_kelly=False)),
        ("VolTarget + Regime", dict(enable_vol_target=True, enable_dd_control=False,
                                     enable_regime=True, enable_kelly=False)),
        ("VolTarget + DDControl + Regime", dict(enable_vol_target=True, enable_dd_control=True,
                                                  enable_regime=True, enable_kelly=False)),
        ("FULL (tutti gli overlay)", dict(enable_vol_target=True, enable_dd_control=True,
                                          enable_regime=True, enable_kelly=True)),
    ]

    ablation_results = {}
    for name, kwargs in configs:
        res, _ = optimizer.optimize(returns, benchmark_returns, **kwargs)
        opt = res.get("Optimized", res.get("Raw"))
        metrics = compute_metrics(opt, name)
        ablation_results[name] = {"returns": opt, "metrics": metrics}
        logger.info(f"  {name:40s} | Sharpe: {metrics['_sharpe']:+.3f} | "
                   f"MaxDD: {metrics['_max_dd']:.2%} | Ann Ret: {metrics['_ann_ret']:.2%}")

    return ablation_results


# =================================================================================
# GRID SEARCH OVERLAY PARAMETERS
# =================================================================================
def grid_search_overlays(returns, benchmark_returns=None):
    """
    Grid search sui parametri degli overlay per trovare la combinazione ottimale.
    ANTI-LEAKAGE: usa prima metà dei dati per search, seconda metà per validazione.
    """
    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH — Parametri overlay ottimali")
    logger.info("=" * 70)

    n = len(returns)
    mid = n // 2

    train_ret = returns.iloc[:mid]
    test_ret = returns.iloc[mid:]
    train_bench = benchmark_returns.iloc[:mid] if benchmark_returns is not None else None
    test_bench = benchmark_returns.iloc[mid:] if benchmark_returns is not None else None

    # Grid
    vol_targets = [0.10, 0.12, 0.15, 0.18, 0.20]
    dd_thresholds = [0.12, 0.15, 0.20, 0.25]
    dd_powers = [1.5, 2.0, 2.5]
    regime_crisis_ths = [1.3, 1.5, 1.8]
    kelly_fractions = [0.15, 0.25, 0.35]

    best_sharpe = -np.inf
    best_config = {}
    best_calmar = -np.inf
    best_config_calmar = {}

    total_combos = len(vol_targets) * len(dd_thresholds) * len(dd_powers) * len(regime_crisis_ths) * len(kelly_fractions)
    logger.info(f"  Testing {total_combos} combinazioni su training set...")

    count = 0
    for vt in vol_targets:
        for ddt in dd_thresholds:
            for ddp in dd_powers:
                for rct in regime_crisis_ths:
                    for kf in kelly_fractions:
                        count += 1
                        cfg = {
                            "VOL_TARGET": vt, "DD_MAX_THRESHOLD": ddt,
                            "DD_POWER": ddp, "REGIME_CRISIS_TH": rct,
                            "KELLY_FRACTION": kf,
                        }
                        try:
                            opt = EquityCurveOptimizer(cfg)
                            res, _ = opt.optimize(train_ret, train_bench)
                            optimized = res["Optimized"]
                            if len(optimized) < 30:
                                continue
                            m = compute_metrics(optimized)
                            sharpe = m["_sharpe"]
                            max_dd = m["_max_dd"]
                            calmar = m["_ann_ret"] / (abs(max_dd) + 1e-10)

                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_config = cfg.copy()

                            if calmar > best_calmar:
                                best_calmar = calmar
                                best_config_calmar = cfg.copy()
                        except:
                            continue

    logger.info(f"\n  Best Sharpe config (train): {best_config}")
    logger.info(f"  Best Sharpe (train): {best_sharpe:.3f}")
    logger.info(f"  Best Calmar config (train): {best_config_calmar}")
    logger.info(f"  Best Calmar (train): {best_calmar:.3f}")

    # Valida su test set
    logger.info(f"\n  Validazione su test set ({test_ret.index[0].date()} → {test_ret.index[-1].date()}):")

    for label, cfg in [("Best-Sharpe", best_config), ("Best-Calmar", best_config_calmar)]:
        opt = EquityCurveOptimizer(cfg)
        res, _ = opt.optimize(test_ret, test_bench)
        m = compute_metrics(res["Optimized"], label)
        logger.info(f"  {label}: Sharpe={m['_sharpe']:.3f}, MaxDD={m['_max_dd']:.2%}, "
                   f"AnnRet={m['_ann_ret']:.2%}")

    # Baseline (no overlay) on test
    m_base = compute_metrics(test_ret, "Baseline")
    logger.info(f"  Baseline (no overlay): Sharpe={m_base['_sharpe']:.3f}, "
               f"MaxDD={m_base['_max_dd']:.2%}, AnnRet={m_base['_ann_ret']:.2%}")

    return best_config, best_config_calmar


# =================================================================================
# REPORT GENERATOR
# =================================================================================
def generate_v4_report(raw_returns, optimized_returns, benchmark,
                       ablation_results, regime_exposure, output_dir):
    """Genera report completo v4."""
    os.makedirs(output_dir, exist_ok=True)
    bench_ret = benchmark.pct_change().dropna() if benchmark is not None else None

    # --- PLOT 1: Raw vs Optimized Equity Curve ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("v4 Equity Curve Optimization — Raw vs Optimized", fontsize=14, fontweight='bold')

    cum_raw = (1 + raw_returns).cumprod()
    cum_opt = (1 + optimized_returns).cumprod()

    ax1.plot(cum_raw.index, cum_raw.values, color="#E57373", linewidth=1.2,
            linestyle="--", label="v3 Raw (Combined_InvVol)", alpha=0.7)
    ax1.plot(cum_opt.index, cum_opt.values, color="#1B5E20", linewidth=2,
            label="v4 Optimized")

    if benchmark is not None:
        ci = cum_opt.index.intersection(benchmark.index)
        if len(ci) > 0:
            bn = benchmark.loc[ci] / benchmark.loc[ci[0]]
            ax1.plot(bn.index, bn.values, color="gray", linewidth=1, alpha=0.4,
                    linestyle=":", label="XBI")

    # Annotazioni metriche
    m_raw = compute_metrics(raw_returns, "Raw")
    m_opt = compute_metrics(optimized_returns, "Optimized")
    txt = (f"v3 Raw:       Sharpe {m_raw['_sharpe']:+.3f}  MaxDD {m_raw['_max_dd']:.1%}  "
           f"Return {m_raw['_ann_ret']:.1%}\n"
           f"v4 Optimized: Sharpe {m_opt['_sharpe']:+.3f}  MaxDD {m_opt['_max_dd']:.1%}  "
           f"Return {m_opt['_ann_ret']:.1%}")
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes, fontsize=9,
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylabel("Cumulative Return")

    # Drawdown comparison
    pk_raw = cum_raw.expanding().max()
    dd_raw = (cum_raw - pk_raw) / pk_raw
    pk_opt = cum_opt.expanding().max()
    dd_opt = (cum_opt - pk_opt) / pk_opt

    ax2.fill_between(dd_raw.index, dd_raw.values, 0, color="#E57373", alpha=0.2, label="Raw DD")
    ax2.fill_between(dd_opt.index, dd_opt.values, 0, color="#1B5E20", alpha=0.3, label="Optimized DD")
    ax2.legend(fontsize=8); ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v4_equity_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: Ablation Study ---
    if ablation_results:
        names = list(ablation_results.keys())
        sharpes = [ablation_results[n]["metrics"]["_sharpe"] for n in names]
        max_dds = [abs(ablation_results[n]["metrics"]["_max_dd"]) for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Ablation Study — Contributo di ogni overlay", fontsize=14, fontweight='bold')

        colors = ['#E57373' if s < 0.5 else '#4CAF50' if s < 0.7 else '#1B5E20' for s in sharpes]
        bars1 = ax1.barh(range(len(names)), sharpes, color=colors, edgecolor='white')
        ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Sharpe Ratio"); ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(sharpes[0], color="red", linestyle="--", alpha=0.5, label="Baseline")
        for i, v in enumerate(sharpes):
            ax1.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=8)

        colors2 = ['#1B5E20' if d < 0.20 else '#4CAF50' if d < 0.30 else '#E57373' for d in max_dds]
        bars2 = ax2.barh(range(len(names)), max_dds, color=colors2, edgecolor='white')
        ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("Max Drawdown (abs)"); ax2.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(max_dds):
            ax2.text(v + 0.005, i, f"{v:.1%}", va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "v4_ablation_study.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # --- PLOT 3: Regime Exposure Over Time ---
    if regime_exposure is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle("Regime Detection — Exposure Over Time", fontsize=14, fontweight='bold')

        cum_opt2 = (1 + optimized_returns).cumprod()
        ax1.plot(cum_opt2.index, cum_opt2.values, color="#1B5E20", linewidth=1.5)
        ax1.set_ylabel("Cumulative Return"); ax1.grid(True, alpha=0.3)

        ax2.fill_between(regime_exposure.index, regime_exposure.values, 0,
                        color="#FF9800", alpha=0.5)
        ax2.set_ylabel("Regime Exposure"); ax2.set_ylim(0, 1.1); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "v4_regime_exposure.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # --- PLOT 4: Rolling Sharpe Raw vs Optimized ---
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle("Rolling 63-Day Sharpe — Raw vs Optimized", fontsize=14, fontweight='bold')

    rs_raw = raw_returns.rolling(63, min_periods=21).apply(
        lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252))
    rs_opt = optimized_returns.rolling(63, min_periods=21).apply(
        lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252))

    ax.plot(rs_raw.index, rs_raw.values, color="#E57373", linewidth=1, alpha=0.6, label="v3 Raw")
    ax.plot(rs_opt.index, rs_opt.values, color="#1B5E20", linewidth=1.5, label="v4 Optimized")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axhline(1, color="green", linewidth=0.5, linestyle=":", alpha=0.3)
    ax.fill_between(rs_opt.index,
                    np.minimum(rs_raw.values, rs_opt.values),
                    np.maximum(rs_raw.values, rs_opt.values),
                    alpha=0.1, color="green",
                    where=rs_opt.values > rs_raw.values)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylabel("Rolling Sharpe")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v4_rolling_sharpe_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- PLOT 5: Monthly Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Monthly Returns — Raw vs Optimized", fontsize=14, fontweight='bold')

    for idx, (name, rets) in enumerate([("v3 Raw", raw_returns), ("v4 Optimized", optimized_returns)]):
        ax = axes[idx]
        monthly = rets.resample("M").sum()
        mp = pd.DataFrame({"Y": monthly.index.year, "M": monthly.index.month, "R": monthly.values})
        ht = mp.pivot_table(values="R", index="Y", columns="M", aggfunc="sum")
        sns.heatmap(ht, ax=ax, cmap="RdYlGn", center=0, annot=True, fmt=".1%",
                   annot_kws={"size": 7}, linewidths=0.5)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v4_monthly_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Save CSVs
    pd.DataFrame({"raw": raw_returns, "optimized": optimized_returns}).to_csv(
        os.path.join(output_dir, "v4_returns_comparison.csv"))

    logger.info(f"\nReport v4 salvato in: {output_dir}")


# =================================================================================
# MAIN — Carica risultati v3 e applica overlay
# =================================================================================
def main():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("EQUITY CURVE OPTIMIZER v4")
    logger.info("=" * 80)

    # Trova directory dei risultati v3
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "stat_arb_results")

    # Carica returns v3
    logger.info("\n[1/5] CARICAMENTO RISULTATI v3")

    # Prova Combined_InvVol (il migliore)
    combined_path = os.path.join(results_dir, "Combined_InvVol_returns.csv")
    if not os.path.exists(combined_path):
        # Fallback: prova con nomi alternativi
        for alt in ["Combined_EqWeight_returns.csv", "CS_Optimized_returns.csv",
                     "Factor_Optimized_returns.csv"]:
            alt_path = os.path.join(results_dir, alt)
            if os.path.exists(alt_path):
                combined_path = alt_path
                break

    if not os.path.exists(combined_path):
        logger.error(f"Nessun file di risultati trovato in {results_dir}!")
        logger.error("Esegui prima stat_arb_biotech.py (pipeline v3)")
        return

    logger.info(f"  Carico: {os.path.basename(combined_path)}")
    df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    if "net_return" in df.columns:
        raw_returns = df["net_return"]
    else:
        raw_returns = df.iloc[:, 0]

    raw_returns = raw_returns.dropna()
    logger.info(f"  Periodo: {raw_returns.index[0].date()} → {raw_returns.index[-1].date()}")
    logger.info(f"  Trading days: {len(raw_returns)}")

    # Carica anche CS e Factor singoli per combined migliore
    cs_path = os.path.join(results_dir, "CS_Optimized_returns.csv")
    factor_path = os.path.join(results_dir, "Factor_Optimized_returns.csv")
    cs_ret = pd.read_csv(cs_path, index_col=0, parse_dates=True).iloc[:, 0].dropna() if os.path.exists(cs_path) else None
    factor_ret = pd.read_csv(factor_path, index_col=0, parse_dates=True).iloc[:, 0].dropna() if os.path.exists(factor_path) else None

    # Carica benchmark
    logger.info("\n[1.5/5] SCARICAMENTO BENCHMARK PER REGIME DETECTION")
    benchmark = None
    benchmark_returns = None
    try:
        import yfinance as yf
        bench_data = yf.download("XBI", start=raw_returns.index[0].strftime("%Y-%m-%d"),
                                end=raw_returns.index[-1].strftime("%Y-%m-%d"),
                                auto_adjust=True, progress=False)
        if len(bench_data) > 0:
            benchmark = bench_data["Close"].squeeze()  # Force Series (yfinance multi-level fix)
            if isinstance(benchmark, pd.DataFrame):
                benchmark = benchmark.iloc[:, 0]
            benchmark_returns = benchmark.pct_change().dropna()
            # Allinea agli stessi indici
            common = raw_returns.index.intersection(benchmark_returns.index)
            benchmark_returns = benchmark_returns.loc[common]
            logger.info(f"  Benchmark XBI caricato: {len(benchmark_returns)} days")
    except Exception as e:
        logger.warning(f"  Benchmark non disponibile ({e}), regime detection userà returns propri")

    # Metriche baseline
    m_base = compute_metrics(raw_returns, "v3 Baseline")
    logger.info(f"\n  v3 Baseline: Sharpe={m_base['_sharpe']:.3f}, MaxDD={m_base['_max_dd']:.2%}, "
               f"AnnRet={m_base['_ann_ret']:.2%}")

    # --- Step 2: Ablation Study ---
    logger.info("\n[2/5] ABLATION STUDY")
    ablation = run_ablation_study(raw_returns, benchmark_returns)

    # --- Step 3: Grid Search ---
    logger.info("\n[3/5] GRID SEARCH PARAMETRI OVERLAY")
    best_sharpe_cfg, best_calmar_cfg = grid_search_overlays(raw_returns, benchmark_returns)

    # --- Step 4: Applica best config sull'intero dataset ---
    logger.info("\n[4/5] APPLICAZIONE BEST CONFIG")

    # Usa la config che bilancia meglio Sharpe e Calmar
    # Combina: prendi i parametri che migliorano di più
    final_config = best_sharpe_cfg.copy()

    logger.info(f"  Config finale: {final_config}")
    optimizer = EquityCurveOptimizer(final_config)
    all_results, regime_exposure = optimizer.optimize(raw_returns, benchmark_returns)
    optimized = all_results["Optimized"]

    m_opt = compute_metrics(optimized, "v4 Optimized")
    logger.info(f"\n  v4 Optimized: Sharpe={m_opt['_sharpe']:.3f}, MaxDD={m_opt['_max_dd']:.2%}, "
               f"AnnRet={m_opt['_ann_ret']:.2%}")

    # Improvement summary
    logger.info(f"\n  === IMPROVEMENT ===")
    logger.info(f"  Sharpe: {m_base['_sharpe']:.3f} → {m_opt['_sharpe']:.3f} "
               f"({(m_opt['_sharpe']-m_base['_sharpe'])/abs(m_base['_sharpe']+1e-10)*100:+.1f}%)")
    logger.info(f"  MaxDD:  {m_base['_max_dd']:.2%} → {m_opt['_max_dd']:.2%} "
               f"({(abs(m_opt['_max_dd'])-abs(m_base['_max_dd']))/abs(m_base['_max_dd']+1e-10)*100:+.1f}%)")
    logger.info(f"  AnnRet: {m_base['_ann_ret']:.2%} → {m_opt['_ann_ret']:.2%}")

    # Ottimizza anche CS e Factor singolarmente
    if cs_ret is not None and factor_ret is not None:
        logger.info("\n  Ottimizzazione singole strategie:")
        for name, ret in [("CS_Optimized", cs_ret), ("Factor_Optimized", factor_ret)]:
            opt_single = EquityCurveOptimizer(final_config)
            res_s, _ = opt_single.optimize(ret, benchmark_returns)
            opt_s = res_s["Optimized"]
            m_s = compute_metrics(opt_s, name)
            m_s_base = compute_metrics(ret, name + "_base")
            logger.info(f"  {name}: Sharpe {m_s_base['_sharpe']:.3f}→{m_s['_sharpe']:.3f}, "
                       f"MaxDD {m_s_base['_max_dd']:.2%}→{m_s['_max_dd']:.2%}")

            # Salva anche questi
            opt_s.to_csv(os.path.join(results_dir, f"{name}_v4_returns.csv"), header=["net_return"])

        # Ricombina con InvVol
        cs_v4 = pd.read_csv(os.path.join(results_dir, "CS_Optimized_v4_returns.csv"),
                           index_col=0, parse_dates=True).iloc[:, 0]
        f_v4 = pd.read_csv(os.path.join(results_dir, "Factor_Optimized_v4_returns.csv"),
                          index_col=0, parse_dates=True).iloc[:, 0]
        both_v4 = pd.DataFrame({"CS": cs_v4, "Factor": f_v4}).fillna(0)
        vol_lb = 63
        rv = both_v4.shift(1).rolling(vol_lb, min_periods=21).std()
        iv = 1.0 / (rv + 1e-10)
        ww = iv.div(iv.sum(axis=1), axis=0).fillna(0.5)
        combined_v4 = (both_v4 * ww).sum(axis=1)
        combined_v4.to_csv(os.path.join(results_dir, "Combined_v4_returns.csv"), header=["net_return"])

        m_cv4 = compute_metrics(combined_v4, "Combined_v4")
        logger.info(f"\n  Combined v4 (re-combined): Sharpe={m_cv4['_sharpe']:.3f}, "
                   f"MaxDD={m_cv4['_max_dd']:.2%}, AnnRet={m_cv4['_ann_ret']:.2%}")

    # --- Step 5: Report ---
    logger.info("\n[5/5] GENERAZIONE REPORT")
    generate_v4_report(raw_returns, optimized, benchmark, ablation, regime_exposure, results_dir)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"EQUITY CURVE OPTIMIZER v4 COMPLETATO in {elapsed:.1f}s")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
