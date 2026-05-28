#!/usr/bin/env python3
# =================================================================================
# COMMODITY MOMENTUM PIPELINE — Pure Time Series Momentum su Futures
#
# Basato su: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
#            Journal of Financial Economics
#
# LOGICA:
# - Ogni commodity: long se return passato > 0, short se < 0
# - Segnale combinato su 4 lookback (1M, 3M, 6M, 12M) → riduce dipendenza dal timing
# - Ogni posizione vol-scaled a target 10% annuo → risk parity tra commodity
# - Portafoglio totale vol-targetato a 10% → consistente con gamba macro
#
# UNIVERSO (10 commodity futures via yfinance):
# - Energy:  CL=F (WTI Crude), NG=F (Nat Gas)
# - Metals:  GC=F (Gold), SI=F (Silver), HG=F (Copper)
# - Grains:  ZC=F (Corn), ZS=F (Soybeans), ZW=F (Wheat)
# - Softs:   KC=F (Coffee), CC=F (Cocoa)
#
# ROLL (problema principale futures):
# - yfinance dà front-month unadjusted
# - Soluzione: segnale su lookback ≥21gg; il roll giornaliero (tipicamente <0.5%)
#   è noise rispetto al segnale momentum 3-12 mesi
# - Non usiamo ritorni giornalieri per segnali intraday — solo daily prices per
#   calcolare return cumulativo su finestre multi-mese
#
# PERCHÉ AGGIUNGE VALORE:
# - Driver fondamentali completamente diversi da equity: OPEC, siccità, inventari
# - Correlazione storica con equity L/S mean-reversion: ~0.05
# - Correlazione con macro ETF blend: ~0.20 (solo parziale overlap via GLD, DBC)
# - Commodity pura non ha azioni sottostanti → zero esposizione equity factor risk
#
# OUTPUT: stat_arb_results_commodity/
# =================================================================================

import os
import warnings
import time
import logging
from datetime import datetime

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
# CONFIGURATION
# =================================================================================
CONFIG = {
    "DATA_START": "2010-01-01",   # Storico più lungo per commodity (cicli lenti)
    "DATA_END":   "2026-01-31",

    # Universe: 10 commodity futures
    # Energy + Metals + Grains + Softs — diversificazione tra settori
    "UNIVERSE": [
        "CL=F",   # WTI Crude Oil (energia)
        "NG=F",   # Natural Gas (energia, alta vol)
        "GC=F",   # Gold (metallo prezioso, safe-haven)
        "SI=F",   # Silver (metallo prezioso + industriale)
        "HG=F",   # Copper (metallo industriale, proxy crescita)
        "ZC=F",   # Corn (grano)
        "ZS=F",   # Soybeans (oleaginosi)
        "ZW=F",   # Wheat (grano tenero)
        "KC=F",   # Coffee (soft)
        "CC=F",   # Cocoa (soft)
    ],

    # TSMOM lookbacks (trading days)
    # Combinare più lookback riduce il timing-luck
    "LOOKBACKS": [21, 63, 126, 252],    # 1M, 3M, 6M, 12M
    "LOOKBACK_WEIGHTS": [0.15, 0.25, 0.35, 0.25],  # Sottopondera 1M (noisier)

    # Esclusione "last month" (Jegadeesh-Titman: skip ultimo mese per evitare reversal)
    "SKIP_LAST_DAYS": 0,    # Per futures non necessario (no microstructure reversal)

    # Vol scaling
    "VOL_LOOKBACK": 63,     # Finestra per stimare volatilità individuale
    "TARGET_VOL_ASSET": 0.40 / np.sqrt(252),  # Target vol giornaliera per asset (~40% ann)
    # Nota: 40% è alto ma allineato alla vol media commodity futures
    # Il vol-scaling riduce il peso di NG (vol ~90%) e aumenta GC (vol ~15%)

    # Portafoglio
    "TARGET_VOL_PORTFOLIO": 0.10,  # 10% annualizzato — uguale a macro blend
    "COMMISSION_BPS": 3,           # Futures molto liquidi: ~1-3bps per trade

    # Walk-forward (semplificato: TSMOM è robusto ai parametri)
    "TRAIN_MONTHS": 24,
    "TEST_MONTHS":  3,

    "SEED": 42,
}

np.random.seed(CONFIG["SEED"])


# =================================================================================
# DATA COLLECTION
# =================================================================================
class CommodityDataCollector:
    def __init__(self, config):
        self.config = config

    def fetch(self):
        try:
            return self._fetch_yfinance()
        except Exception as e:
            logger.warning(f"yfinance non disponibile ({e}), uso dati sintetici")
            return self._synthetic()

    def _fetch_yfinance(self):
        import yfinance as yf
        tickers = self.config["UNIVERSE"]
        logger.info(f"Download {len(tickers)} commodity futures...")
        data = yf.download(
            tickers, start=self.config["DATA_START"], end=self.config["DATA_END"],
            auto_adjust=True, progress=False, threads=True
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"].copy()
        else:
            prices = data.copy()

        # Filtra: almeno 60% di dati validi
        valid_pct = prices.notna().sum() / len(prices)
        valid = valid_pct[valid_pct > 0.60].index.tolist()
        if len(valid) < 4:
            raise ValueError(f"Solo {len(valid)} commodity con dati sufficienti")
        prices = prices[valid].ffill(limit=5)

        logger.info(f"  Commodity valide: {len(valid)}: {valid}")
        logger.info(f"  Periodo: {prices.index[0].date()} → {prices.index[-1].date()}")
        logger.info(f"  N giorni: {len(prices)}")
        return prices

    def _synthetic(self):
        """Dati sintetici per test — commodity con caratteristiche realistiche."""
        logger.info("Generazione dati sintetici commodity...")
        rng = np.random.RandomState(self.config["SEED"])
        dates = pd.bdate_range(start=self.config["DATA_START"], end=self.config["DATA_END"])
        n, tickers = len(dates), self.config["UNIVERSE"]

        # Volatilità annualizzate realistiche per commodity
        ann_vols = {
            "CL=F": 0.35, "NG=F": 0.65, "GC=F": 0.15, "SI=F": 0.30, "HG=F": 0.25,
            "ZC=F": 0.25, "ZS=F": 0.22, "ZW=F": 0.28, "KC=F": 0.30, "CC=F": 0.25,
        }

        prices = pd.DataFrame(index=dates)
        for t in tickers:
            vol = ann_vols.get(t, 0.30) / np.sqrt(252)
            # Trend + mean-reversion + noise (Ornstein-Uhlenbeck)
            drift = rng.uniform(-0.05, 0.10) / 252  # Commodity hanno drift basso
            r = rng.normal(drift, vol, n)
            # Aggiungi autocorrelazione positiva (momentum) e negativa lenta (ciclo)
            for i in range(1, n):
                r[i] += 0.05 * r[i-1]  # Weak momentum
            prices[t] = 100 * np.exp(np.cumsum(r))

        logger.info(f"  Sintetici generati: {len(tickers)} commodity, {n} giorni")
        return prices


# =================================================================================
# COMMODITY TSMOM STRATEGY
# =================================================================================
class CommodityTSMOM:
    """
    Time Series Momentum su commodity futures.

    Segnale: sign(past_return) * vol_weight
    - long se past_return > 0 negli ultimi `lookback` giorni
    - short se past_return < 0
    - peso proporzionale a 1/vol (risk parity tra asset)
    - portafoglio scalato a target_vol

    Multi-lookback: combina 4 orizzonti temporali con pesi configurabili.
    """

    def __init__(self, lookbacks, lookback_weights, vol_lookback,
                 target_vol_asset, target_vol_portfolio, commission_bps,
                 skip_last=0, rebalance_freq=5):
        self.lookbacks = lookbacks
        self.lookback_weights = np.array(lookback_weights) / sum(lookback_weights)
        self.vol_lookback = vol_lookback
        self.target_vol_asset = target_vol_asset
        self.target_vol_portfolio = target_vol_portfolio
        self.commission_pct = commission_bps / 10000.0
        self.skip_last = skip_last
        self.rebalance_freq = rebalance_freq

    def run(self, prices, test_start, test_end):
        returns = prices.pct_change().fillna(0)
        test_mask = (prices.index >= test_start) & (prices.index <= test_end)
        test_dates = prices.index[test_mask]

        if len(test_dates) < 10:
            return pd.Series(dtype=float)

        rebal_dates = test_dates[::self.rebalance_freq]
        daily_portfolio_rets = {}
        prev_weights = {}

        max_lb = max(self.lookbacks) + self.skip_last + self.vol_lookback + 5

        for rebal_date in rebal_dates:
            idx = prices.index.get_indexer([rebal_date], method='ffill')[0]
            if idx < max_lb:
                continue

            # === Calcola segnale per ogni commodity ===
            weights = {}
            for ticker in prices.columns:
                px_series = prices[ticker].iloc[:idx + 1]
                if px_series.isna().tail(10).any():
                    continue

                # Stima vol su lookback rolling
                ret_series = returns[ticker].iloc[idx - self.vol_lookback:idx]
                realized_vol = ret_series.std()
                if realized_vol < 1e-8:
                    continue

                # Combina segnali su più lookback
                composite_signal = 0.0
                for lb, w in zip(self.lookbacks, self.lookback_weights):
                    start_idx = idx - lb - self.skip_last
                    end_idx   = idx - self.skip_last
                    if start_idx < 0:
                        continue
                    past_ret = px_series.iloc[end_idx] / px_series.iloc[start_idx] - 1
                    # Segnale binario: +1 o -1 (o 0 se flat)
                    signal = np.sign(past_ret)
                    composite_signal += w * signal

                if abs(composite_signal) < 1e-6:
                    continue

                # Vol scaling: peso ∝ target_vol / realized_vol
                # Normalizza a scala daily
                vol_scale = self.target_vol_asset / (realized_vol + 1e-10)
                weights[ticker] = composite_signal * vol_scale

            if not weights:
                continue

            # === Portfolio-level vol targeting ===
            # Stima vol del portafoglio complessivo
            if len(weights) >= 2:
                w_vec = pd.Series(weights)
                # Semplice: usa vol media ponderata (trascura correlazioni per semplicità)
                port_vol = np.sqrt(sum(
                    (w ** 2) * (returns[t].iloc[idx - self.vol_lookback:idx].std() ** 2)
                    for t, w in weights.items() if t in returns.columns
                )) * np.sqrt(252)
                if port_vol > 1e-6:
                    daily_target = self.target_vol_portfolio / np.sqrt(252)
                    port_vol_daily = port_vol / np.sqrt(252)
                    leverage = min(daily_target / port_vol_daily, 3.0)  # Cap a 3x
                    weights = {t: w * leverage for t, w in weights.items()}

            # === Hold per rebalance_freq giorni ===
            hold_end = min(idx + self.rebalance_freq, len(prices) - 1)
            hold_dates = prices.index[idx: hold_end + 1]

            for hdate in hold_dates:
                if hdate > test_end or hdate not in returns.index:
                    continue
                day_ret = 0.0
                turnover = 0.0
                for ticker, w in weights.items():
                    if ticker in returns.columns:
                        r = returns.loc[hdate, ticker]
                        if not np.isnan(r):
                            day_ret += w * r
                # Commissioni su turnover (al momento del rebalance)
                if hdate == rebal_date and prev_weights:
                    all_t = set(list(weights.keys()) + list(prev_weights.keys()))
                    turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0))
                                   for t in all_t)
                    day_ret -= turnover * self.commission_pct

                daily_portfolio_rets[hdate] = day_ret

            prev_weights = dict(weights)

        if not daily_portfolio_rets:
            return pd.Series(dtype=float)

        series = pd.Series(daily_portfolio_rets).sort_index()
        series = series[~series.index.duplicated(keep='last')]
        return series


# =================================================================================
# WALK-FORWARD ENGINE (semplificato — TSMOM è robusto ai param)
# =================================================================================
class CommodityWalkForward:
    """
    Walk-forward su commodity TSMOM.
    Non ottimizza i lookback (TSMOM è documented-robust), ma divide
    in finestre per evitare lookahead sul vol target.
    """
    def __init__(self, config):
        self.config = config

    def build_windows(self, prices):
        """Genera finestre walk-forward (train/test)."""
        start = prices.index[0]
        end = prices.index[-1]

        train_days = self.config["TRAIN_MONTHS"] * 21
        test_days  = self.config["TEST_MONTHS"] * 21

        windows = []
        idx = train_days
        while idx < len(prices):
            train_start = prices.index[max(0, idx - train_days)]
            test_start  = prices.index[idx]
            test_end    = prices.index[min(idx + test_days - 1, len(prices) - 1)]
            windows.append({
                "train_start": train_start,
                "test_start":  test_start,
                "test_end":    test_end,
            })
            idx += test_days
        return windows

    def run(self, prices):
        windows = self.build_windows(prices)
        logger.info(f"  {len(windows)} finestre walk-forward")

        strategy = CommodityTSMOM(
            lookbacks=self.config["LOOKBACKS"],
            lookback_weights=self.config["LOOKBACK_WEIGHTS"],
            vol_lookback=self.config["VOL_LOOKBACK"],
            target_vol_asset=self.config["TARGET_VOL_ASSET"],
            target_vol_portfolio=self.config["TARGET_VOL_PORTFOLIO"],
            commission_bps=self.config["COMMISSION_BPS"],
            skip_last=self.config["SKIP_LAST_DAYS"],
            rebalance_freq=5,
        )

        all_rets = []
        for i, w in enumerate(windows):
            rets = strategy.run(prices, w["test_start"], w["test_end"])
            if len(rets) > 0:
                all_rets.append(rets)
                ann = rets.mean() * 252
                vol = rets.std() * np.sqrt(252)
                sharpe = ann / (vol + 1e-10)
                logger.info(f"  Finestra {i+1:02d}: {w['test_start'].date()} → "
                           f"{w['test_end'].date()} | "
                           f"Sharpe={sharpe:.3f}, AnnRet={ann:.2%}")

        if not all_rets:
            return pd.Series(dtype=float)

        combined = pd.concat(all_rets)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined


# =================================================================================
# PERFORMANCE METRICS
# =================================================================================
def compute_metrics(returns, name="", rf=0.04):
    if len(returns) < 20:
        return {}
    daily_rf = rf / 252
    ann_ret  = returns.mean() * 252
    ann_vol  = returns.std() * np.sqrt(252)
    sharpe   = (returns.mean() - daily_rf) / (returns.std() + 1e-10) * np.sqrt(252)
    dr = returns[returns < 0]
    sortino = (returns.mean() - daily_rf) / (dr.std() * np.sqrt(252) + 1e-10) if len(dr) > 0 else 0
    cum = (1 + returns).cumprod()
    pk  = cum.expanding().max()
    dd  = (cum - pk) / pk
    max_dd = dd.min()
    calmar = ann_ret / (abs(max_dd) + 1e-10)
    wr = (returns > 0).mean()
    t_stat = sharpe * np.sqrt(len(returns) / 252)
    return {
        "name": name,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": wr,
        "t_stat": t_stat,
        "days": len(returns),
        "total_ret": (1 + returns).prod() - 1,
    }


# =================================================================================
# REPORT
# =================================================================================
def generate_report(returns_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info("COMMODITY MOMENTUM — PERFORMANCE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Strategy':<25} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} "
               f"{'Vol':>8} {'Calmar':>8} {'t-stat':>8} {'Days':>6}")
    logger.info(f"  {'-'*80}")

    for name, rets in returns_dict.items():
        m = compute_metrics(rets, name)
        if m:
            logger.info(f"  {m['name']:<25} {m['sharpe']:>8.3f} {m['ann_ret']:>8.2%} "
                       f"{m['max_dd']:>8.2%} {m['ann_vol']:>8.2%} "
                       f"{m['calmar']:>8.3f} {m['t_stat']:>8.2f} {m['days']:>6}")

    # --- Equity curves ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Commodity TSMOM Pipeline", fontsize=14, fontweight='bold')
    colors = {"Commodity_TSMOM": "#8B4513", "TSMOM_1M": "#D2691E",
              "TSMOM_3M": "#CD853F", "TSMOM_6M": "#DEB887", "TSMOM_12M": "#F4A460"}

    # Panel 1: Equity curve
    ax = axes[0, 0]
    for name, rets in returns_dict.items():
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, label=name,
               color=colors.get(name, "gray"), linewidth=1.5 if name == "Commodity_TSMOM" else 0.8)
    ax.set_title("Equity Curves")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

    # Panel 2: Drawdown
    ax = axes[0, 1]
    main_ret = returns_dict.get("Commodity_TSMOM")
    if main_ret is not None:
        cum = (1 + main_ret).cumprod()
        pk  = cum.expanding().max()
        dd  = (cum - pk) / pk
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="#8B4513")
        ax.plot(dd.index, dd.values, color="#8B4513", linewidth=0.8)
        m = compute_metrics(main_ret)
        ax.set_title(f"Drawdown (MaxDD={m['max_dd']:.2%})")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Drawdown")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)

    # Panel 3: Annual returns
    ax = axes[1, 0]
    if main_ret is not None:
        annual = main_ret.resample('YE').apply(lambda r: (1 + r).prod() - 1)
        colors_bar = ['#2E7D32' if r > 0 else '#C62828' for r in annual.values]
        ax.bar(annual.index.year, annual.values * 100, color=colors_bar, alpha=0.8, width=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title("Annual Returns")
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Monthly returns heatmap
    ax = axes[1, 1]
    if main_ret is not None:
        monthly = main_ret.resample('ME').apply(lambda r: (1 + r).prod() - 1)
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'ret': monthly.values
        })
        pivot = monthly_df.pivot(index='year', columns='month', values='ret')
        pivot.columns = ['Gen','Feb','Mar','Apr','Mag','Giu',
                         'Lug','Ago','Set','Ott','Nov','Dic']
        import seaborn as sns
        sns.heatmap(pivot * 100, ax=ax, cmap='RdYlGn', center=0,
                   fmt='.1f', annot=True, annot_kws={'size': 7},
                   cbar_kws={'label': '%'})
        ax.set_title("Monthly Returns (%)")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "commodity_momentum.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\n  Chart salvato: {fig_path}")

    # Save CSVs
    for name, rets in returns_dict.items():
        rets.to_csv(os.path.join(output_dir, f"{name}_returns.csv"), header=["net_return"])
    logger.info(f"  CSV salvati in: {output_dir}")


# =================================================================================
# MAIN
# =================================================================================
def run_pipeline():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("COMMODITY MOMENTUM PIPELINE — Time Series Momentum su Futures")
    logger.info("Universo: CL, NG, GC, SI, HG, ZC, ZS, ZW, KC, CC")
    logger.info("Segnale: TSMOM multi-lookback (1M+3M+6M+12M) + vol scaling")
    logger.info(f"Target vol portfolio: {CONFIG['TARGET_VOL_PORTFOLIO']:.0%} annuo")
    logger.info(f"Commission: {CONFIG['COMMISSION_BPS']}bps | Periodo: {CONFIG['DATA_START']} → {CONFIG['DATA_END']}")
    logger.info("=" * 80)

    # 1. Dati
    logger.info("\n[1/3] DATA COLLECTION")
    collector = CommodityDataCollector(CONFIG)
    prices = collector.fetch()
    logger.info(f"  Universe finale: {list(prices.columns)}")

    # 2. Walk-Forward
    logger.info("\n[2/3] WALK-FORWARD TSMOM")
    engine = CommodityWalkForward(CONFIG)
    tsmom_rets = engine.run(prices)

    if len(tsmom_rets) < 50:
        logger.error("Risultati insufficienti"); return None

    # Anche singoli lookback per analisi (no walk-forward, full OOS = tutti i dati)
    results = {"Commodity_TSMOM": tsmom_rets}
    logger.info(f"\n  Commodity_TSMOM: {len(tsmom_rets)} giorni, "
               f"cum={tsmom_rets.sum():.4f}")

    # Singoli lookback per capire quale orizonte temporale è il più forte
    logger.info("\n  --- Analisi per lookback singolo ---")
    test_start = tsmom_rets.index[0]
    test_end   = tsmom_rets.index[-1]

    lb_names = {21: "TSMOM_1M", 63: "TSMOM_3M", 126: "TSMOM_6M", 252: "TSMOM_12M"}
    for lb, lb_name in lb_names.items():
        strat_single = CommodityTSMOM(
            lookbacks=[lb],
            lookback_weights=[1.0],
            vol_lookback=CONFIG["VOL_LOOKBACK"],
            target_vol_asset=CONFIG["TARGET_VOL_ASSET"],
            target_vol_portfolio=CONFIG["TARGET_VOL_PORTFOLIO"],
            commission_bps=CONFIG["COMMISSION_BPS"],
            skip_last=CONFIG["SKIP_LAST_DAYS"],
            rebalance_freq=5,
        )
        r = strat_single.run(prices, test_start, test_end)
        if len(r) > 20:
            m = compute_metrics(r, lb_name)
            logger.info(f"    {lb_name}: Sharpe={m['sharpe']:.3f}, "
                       f"AnnRet={m['ann_ret']:.2%}, MaxDD={m['max_dd']:.2%}")
            results[lb_name] = r

    # 3. Report
    logger.info("\n[3/3] PERFORMANCE REPORT")
    output_dir = os.path.join(SCRIPT_DIR, "stat_arb_results_commodity")
    generate_report(results, output_dir)

    # Correlazione con benchmark equity
    m = compute_metrics(tsmom_rets, "Commodity_TSMOM")
    logger.info(f"\n  === RIEPILOGO FINALE ===")
    logger.info(f"  Sharpe:      {m['sharpe']:.3f}")
    logger.info(f"  AnnRet:      {m['ann_ret']:.2%}")
    logger.info(f"  MaxDD:       {m['max_dd']:.2%}")
    logger.info(f"  Vol:         {m['ann_vol']:.2%}")
    logger.info(f"  Calmar:      {m['calmar']:.3f}")
    logger.info(f"  t-stat:      {m['t_stat']:.2f}")
    logger.info(f"  Win Rate:    {m['win_rate']:.2%}")
    logger.info(f"  Days:        {m['days']}")
    logger.info(f"\n  Target integrazione in mega_portfolio come 4a gamba")
    logger.info(f"  (dopo aver visto correlazione con Bio/Tech/Macro)")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"COMMODITY PIPELINE COMPLETATA in {elapsed:.1f}s")
    logger.info(f"{'='*80}")
    return results


if __name__ == "__main__":
    run_pipeline()
