#!/usr/bin/env python3
# =================================================================================
# CROSS-ASSET MACRO MOMENTUM — Third Leg del Mega-Portfolio
#
# Strategia: Time-Series + Cross-Sectional Momentum su ETF macro
# Basata su: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
#            Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere"
#
# Universo: 8 ETF che rappresentano asset class decorrelate
#   GLD  — Oro (risk-off, inflation hedge)
#   TLT  — Treasury 20y (flight-to-quality, duration)
#   VNQ  — REITs (real estate, yield)
#   EEM  — Emerging Markets (global growth)
#   XLE  — Energy (commodity cycle)
#   UUP  — US Dollar (FX, carry)
#   EFA  — Developed Intl (geographic diversification)
#   DBC  — Commodity Index (real inflation)
#
# Segnale:
#   1. Time-Series Momentum (TSMOM): long se return > 0 su finestra, short altrimenti
#   2. Cross-Sectional Momentum (CSMOM): long top N, short bottom N per return
#   3. Blend: 60% TSMOM + 40% CSMOM (TSMOM più robusto su macro)
#   4. Vol-scaling: ogni posizione scalata per 1/σ (risk parity across positions)
#
# Walk-forward: stesse finestre del biotech/tech per allineamento
#
# ANTI-LEAKAGE: tutti i segnali calcolati su dati PRIMA del periodo di test
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
# CONFIG
# =================================================================================
CONFIG = {
    "UNIVERSE": ["GLD", "TLT", "VNQ", "EEM", "XLE", "UUP", "EFA", "DBC"],
    "BENCHMARK": "SPY",
    "DATA_START": "2016-01-01",
    "DATA_END": "2026-01-31",

    # Walk-forward
    "WF_TRAIN_YEARS": 2,
    "WF_TEST_MONTHS": 3,
    "WF_MIN_TRAIN_DAYS": 252,

    # Signal parameters (da ottimizzare in WF)
    "TSMOM_LOOKBACKS": [21, 42, 63, 126, 252],   # 1m, 2m, 3m, 6m, 12m
    "CSMOM_LOOKBACKS": [21, 42, 63, 126],
    "LONG_TOP_N": [2, 3, 4],
    "SHORT_BOTTOM_N": [2, 3, 4],

    # Blend
    "TSMOM_WEIGHT": 0.60,
    "CSMOM_WEIGHT": 0.40,

    # Risk management
    "VOL_SCALE": True,           # Risk parity per posizione
    "VOL_LOOKBACK": 42,
    "TARGET_VOL": 0.10,          # Target vol annualizzata per l'intera strategia
    "COMMISSION_BPS": 5,         # ETF = costi bassissimi

    # Anti-leakage
    "SIGNAL_LAG": 1,             # Segnale calcolato T-1, trade a T
}


# =================================================================================
# DATA
# =================================================================================
def fetch_data(config):
    """Scarica prezzi ETF da yfinance."""
    import yfinance as yf

    all_tickers = config["UNIVERSE"] + [config["BENCHMARK"]]
    logger.info(f"Download {len(all_tickers)} ETF...")

    data = yf.download(all_tickers, start=config["DATA_START"],
                       end=config["DATA_END"], auto_adjust=True,
                       progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = all_tickers[:1]

    # Separa benchmark
    benchmark = None
    if config["BENCHMARK"] in prices.columns:
        benchmark = prices[config["BENCHMARK"]].dropna()
        prices = prices.drop(columns=[config["BENCHMARK"]])

    # Rimuovi colonne con troppi NaN
    valid = prices.columns[prices.isna().mean() < 0.3]
    prices = prices[valid].dropna(how='all')

    # Forward-fill per ETF (basse assenze)
    prices = prices.ffill().dropna(how='any')

    returns = prices.pct_change().dropna()

    logger.info(f"  Dati: {prices.shape[0]} giorni, {prices.shape[1]} ETF")
    logger.info(f"  Range: {prices.index[0].date()} → {prices.index[-1].date()}")
    logger.info(f"  ETF validi: {list(prices.columns)}")

    if benchmark is not None:
        benchmark = benchmark.reindex(prices.index).ffill()

    return prices, returns, benchmark


# =================================================================================
# TSMOM — Time-Series Momentum
# =================================================================================
class TimeSeriesMomentum:
    """
    Long se rendimento passato > 0, short altrimenti.
    Vol-scaled: ogni posizione ha uguale contributo di rischio.
    """
    def __init__(self, lookback=126, vol_lookback=42, target_vol=0.10,
                 commission_bps=5, vol_scale=True):
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol
        self.commission_pct = commission_bps / 10000.0
        self.vol_scale = vol_scale
        self.name = "TSMOM"

    def generate_signals(self, prices, returns, signal_date):
        """Genera segnali TSMOM per una data specifica."""
        idx = prices.index.get_indexer([signal_date], method='ffill')[0]
        if idx < self.lookback + 1:
            return None

        # Rendimento passato per ogni ETF
        past_ret = returns.iloc[idx - self.lookback:idx].sum()

        # Segnale: +1 se positivo, -1 se negativo
        signal = np.sign(past_ret)

        # Vol-scaling: peso ∝ 1/σ
        if self.vol_scale:
            vol = returns.iloc[max(0, idx - self.vol_lookback):idx].std() * np.sqrt(252)
            inv_vol = 1.0 / (vol + 1e-10)
            weights = signal * inv_vol
            # Normalizza per target vol
            total_vol = np.sqrt((weights**2 * vol**2).sum()) / np.sqrt(252) * np.sqrt(252)
            if total_vol > 1e-10:
                weights = weights * (self.target_vol / total_vol)
        else:
            weights = signal / len(signal)

        return weights

    def run(self, prices, returns, test_start, test_end, holding=5):
        """Run TSMOM su un periodo di test."""
        test_mask = (prices.index >= test_start) & (prices.index <= test_end)
        test_dates = prices.index[test_mask]
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        rebal_dates = test_dates[::holding]
        daily_returns = {}
        prev_weights = None

        for rebal_date in rebal_dates:
            weights = self.generate_signals(prices, returns, rebal_date)
            if weights is None:
                continue

            idx = prices.index.get_indexer([rebal_date], method='ffill')[0]
            hold_end = min(idx + holding, len(prices) - 1)
            hold_dates = prices.index[idx:hold_end + 1]

            # Costi di transazione
            if prev_weights is not None:
                turnover = (weights - prev_weights.reindex(weights.index, fill_value=0)).abs().sum()
            else:
                turnover = weights.abs().sum()

            for hdate in hold_dates:
                if hdate > test_end or hdate not in returns.index:
                    continue
                day_ret = (weights * returns.loc[hdate]).sum()
                # Costo proporzionale al primo giorno
                if hdate == hold_dates[0]:
                    day_ret -= turnover * self.commission_pct
                daily_returns[hdate] = day_ret

            prev_weights = weights

        return pd.Series(daily_returns).sort_index()


# =================================================================================
# CSMOM — Cross-Sectional Momentum
# =================================================================================
class CrossSectionalMomentum:
    """
    Long top N performer, short bottom N performer.
    Classico cross-sectional momentum (Jegadeesh & Titman 1993) su ETF.
    """
    def __init__(self, lookback=63, long_n=3, short_n=3,
                 vol_lookback=42, commission_bps=5, vol_scale=True):
        self.lookback = lookback
        self.long_n = long_n
        self.short_n = short_n
        self.vol_lookback = vol_lookback
        self.commission_pct = commission_bps / 10000.0
        self.vol_scale = vol_scale
        self.name = "CSMOM"

    def generate_signals(self, prices, returns, signal_date):
        idx = prices.index.get_indexer([signal_date], method='ffill')[0]
        if idx < self.lookback + 1:
            return None

        past_ret = returns.iloc[idx - self.lookback:idx].sum().dropna()
        if len(past_ret) < self.long_n + self.short_n:
            return None

        # Ranking
        ranked = past_ret.sort_values()
        shorts = ranked.index[:self.short_n].tolist()
        longs = ranked.index[-self.long_n:].tolist()

        weights = pd.Series(0.0, index=past_ret.index)

        if self.vol_scale:
            vol = returns.iloc[max(0, idx - self.vol_lookback):idx].std() * np.sqrt(252)
            inv_vol = 1.0 / (vol + 1e-10)
            for t in longs:
                weights[t] = inv_vol.get(t, 1.0)
            for t in shorts:
                weights[t] = -inv_vol.get(t, 1.0)
            # Normalizza
            total = weights.abs().sum()
            if total > 1e-10:
                weights = weights / total
        else:
            for t in longs:
                weights[t] = 1.0 / self.long_n
            for t in shorts:
                weights[t] = -1.0 / self.short_n

        return weights

    def run(self, prices, returns, test_start, test_end, holding=5):
        test_mask = (prices.index >= test_start) & (prices.index <= test_end)
        test_dates = prices.index[test_mask]
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        rebal_dates = test_dates[::holding]
        daily_returns = {}
        prev_weights = None

        for rebal_date in rebal_dates:
            weights = self.generate_signals(prices, returns, rebal_date)
            if weights is None:
                continue

            idx = prices.index.get_indexer([rebal_date], method='ffill')[0]
            hold_end = min(idx + holding, len(prices) - 1)
            hold_dates = prices.index[idx:hold_end + 1]

            if prev_weights is not None:
                turnover = (weights - prev_weights.reindex(weights.index, fill_value=0)).abs().sum()
            else:
                turnover = weights.abs().sum()

            for hdate in hold_dates:
                if hdate > test_end or hdate not in returns.index:
                    continue
                day_ret = (weights * returns.loc[hdate]).sum()
                if hdate == hold_dates[0]:
                    day_ret -= turnover * self.commission_pct
                daily_returns[hdate] = day_ret

            prev_weights = weights

        return pd.Series(daily_returns).sort_index()


# =================================================================================
# WALK-FORWARD ENGINE
# =================================================================================
class WalkForwardMacro:
    def __init__(self, config):
        self.config = config

    def _generate_windows(self, prices):
        """Genera finestre train/test walk-forward."""
        min_date = prices.index[0]
        max_date = prices.index[-1]
        train_days = self.config["WF_TRAIN_YEARS"] * 252

        windows = []
        test_start = min_date + pd.DateOffset(years=self.config["WF_TRAIN_YEARS"])

        while test_start < max_date:
            test_end = test_start + pd.DateOffset(months=self.config["WF_TEST_MONTHS"])
            if test_end > max_date:
                test_end = max_date

            train_end = test_start - pd.DateOffset(days=1)
            train_start = min_date

            windows.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

            test_start = test_end + pd.DateOffset(days=1)

        return windows

    def _optimize_tsmom(self, prices, returns, train_end):
        """Ottimizza lookback per TSMOM su dati di training."""
        best_sharpe = -99
        best_lb = 126

        for lb in self.config["TSMOM_LOOKBACKS"]:
            strat = TimeSeriesMomentum(
                lookback=lb,
                vol_lookback=self.config["VOL_LOOKBACK"],
                target_vol=self.config["TARGET_VOL"],
                commission_bps=self.config["COMMISSION_BPS"],
                vol_scale=self.config["VOL_SCALE"],
            )
            # Train: ultimo anno dei dati di training
            train_test_start = train_end - pd.DateOffset(years=1)
            rets = strat.run(prices, returns, train_test_start, train_end, holding=5)
            if len(rets) > 20:
                sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_lb = lb

        return best_lb

    def _optimize_csmom(self, prices, returns, train_end):
        """Ottimizza lookback e N per CSMOM su dati di training."""
        best_sharpe = -99
        best_params = (63, 3, 3)

        for lb in self.config["CSMOM_LOOKBACKS"]:
            for ln in self.config["LONG_TOP_N"]:
                for sn in self.config["SHORT_BOTTOM_N"]:
                    if ln + sn > len(self.config["UNIVERSE"]):
                        continue
                    strat = CrossSectionalMomentum(
                        lookback=lb, long_n=ln, short_n=sn,
                        vol_lookback=self.config["VOL_LOOKBACK"],
                        commission_bps=self.config["COMMISSION_BPS"],
                        vol_scale=self.config["VOL_SCALE"],
                    )
                    train_test_start = train_end - pd.DateOffset(years=1)
                    rets = strat.run(prices, returns, train_test_start, train_end, holding=5)
                    if len(rets) > 20:
                        sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = (lb, ln, sn)

        return best_params

    def run(self, prices, returns):
        """Esegue walk-forward completo."""
        windows = self._generate_windows(prices)
        logger.info(f"  Walk-Forward: {len(windows)} finestre")

        all_tsmom = {}
        all_csmom = {}
        all_blend = {}

        for i, w in enumerate(windows):
            logger.info(f"\n  Window {i+1}/{len(windows)}: "
                        f"train→{w['train_end'].date()} | "
                        f"test {w['test_start'].date()}→{w['test_end'].date()}")

            # Ottimizza TSMOM
            best_tsmom_lb = self._optimize_tsmom(prices, returns, w['train_end'])
            tsmom = TimeSeriesMomentum(
                lookback=best_tsmom_lb,
                vol_lookback=self.config["VOL_LOOKBACK"],
                target_vol=self.config["TARGET_VOL"],
                commission_bps=self.config["COMMISSION_BPS"],
                vol_scale=self.config["VOL_SCALE"],
            )
            tsmom_rets = tsmom.run(prices, returns, w['test_start'], w['test_end'])
            logger.info(f"    TSMOM: lb={best_tsmom_lb}, {len(tsmom_rets)} days")

            # Ottimizza CSMOM
            cs_lb, cs_ln, cs_sn = self._optimize_csmom(prices, returns, w['train_end'])
            csmom = CrossSectionalMomentum(
                lookback=cs_lb, long_n=cs_ln, short_n=cs_sn,
                vol_lookback=self.config["VOL_LOOKBACK"],
                commission_bps=self.config["COMMISSION_BPS"],
                vol_scale=self.config["VOL_SCALE"],
            )
            csmom_rets = csmom.run(prices, returns, w['test_start'], w['test_end'])
            logger.info(f"    CSMOM: lb={cs_lb}, long={cs_ln}, short={cs_sn}, {len(csmom_rets)} days")

            for d, r in tsmom_rets.items():
                all_tsmom[d] = r
            for d, r in csmom_rets.items():
                all_csmom[d] = r

        # Converti in Series
        tsmom_series = pd.Series(all_tsmom).sort_index()
        csmom_series = pd.Series(all_csmom).sort_index()

        # Blend
        tw = self.config["TSMOM_WEIGHT"]
        cw = self.config["CSMOM_WEIGHT"]
        common = tsmom_series.index.intersection(csmom_series.index)
        blend_series = tw * tsmom_series.loc[common] + cw * csmom_series.loc[common]

        return tsmom_series, csmom_series, blend_series


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
# REPORT
# =================================================================================
def generate_report(results, benchmark, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Summary table
    summary = {}
    for name, rets in results.items():
        if len(rets) > 10:
            summary[name] = compute_metrics(rets, name)

    logger.info("\n" + "=" * 80)
    logger.info("MACRO MOMENTUM — PERFORMANCE SUMMARY (Walk-Forward OOS)")
    logger.info("=" * 80)
    display = {n: {k: v for k, v in m.items() if not k.startswith("_")} for n, m in summary.items()}
    logger.info(f"\n{pd.DataFrame(display).T.to_string()}")

    # Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MACRO MOMENTUM — Cross-Asset ETF Strategy", fontsize=14, fontweight='bold')

    colors = {"TSMOM": "#2E7D32", "CSMOM": "#1565C0", "Macro_Blend": "#D32F2F", "SPY": "#9E9E9E"}

    # Panel 1: Equity curves
    ax = axes[0, 0]
    for name, rets in results.items():
        cum = (1 + rets).cumprod()
        lw = 2.5 if "Blend" in name else 1.2
        ax.plot(cum.index, cum.values, color=colors.get(name, "gray"),
               linewidth=lw, label=f"{name} ({summary.get(name, {}).get('Sharpe', '?')})")
    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        common_idx = bench_ret.index.intersection(list(results.values())[0].index)
        bench_cum = (1 + bench_ret.loc[common_idx]).cumprod()
        ax.plot(bench_cum.index, bench_cum.values, color=colors["SPY"],
               linewidth=1.0, linestyle="--", label="SPY (benchmark)", alpha=0.5)
    ax.set_title("Equity Curves (Walk-Forward OOS)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: Drawdown
    ax = axes[0, 1]
    for name, rets in results.items():
        cum = (1 + rets).cumprod()
        pk = cum.expanding().max()
        dd = (cum - pk) / pk
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3,
                       color=colors.get(name, "gray"), label=name)
    ax.set_title("Drawdown"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 3: Rolling Sharpe
    ax = axes[1, 0]
    for name, rets in results.items():
        roll_sharpe = rets.rolling(126).mean() / rets.rolling(126).std() * np.sqrt(252)
        ax.plot(roll_sharpe.index, roll_sharpe.values, color=colors.get(name, "gray"),
               linewidth=1.0, alpha=0.7, label=name)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Rolling 6M Sharpe"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 4: Monthly returns heatmap per il Blend
    ax = axes[1, 1]
    if "Macro_Blend" in results:
        blend = results["Macro_Blend"]
        monthly = (1 + blend).resample('ME').prod() - 1
        years = sorted(monthly.index.year.unique())
        months = range(1, 13)
        heatmap_data = np.full((len(years), 12), np.nan)
        for i, y in enumerate(years):
            for m in months:
                mask = (monthly.index.year == y) & (monthly.index.month == m)
                vals = monthly[mask]
                if len(vals) > 0:
                    heatmap_data[i, m - 1] = vals.iloc[0]

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
                       vmin=-0.10, vmax=0.10)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, fontsize=7)
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=7)
        ax.set_title("Macro_Blend Monthly Returns")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "macro_momentum.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\nCharts saved to {output_dir}/macro_momentum.png")

    # Save CSVs
    for name, rets in results.items():
        rets.to_csv(os.path.join(output_dir, f"{name}_returns.csv"), header=["net_return"])

    pd.DataFrame(display).T.to_csv(os.path.join(output_dir, "macro_summary.csv"))


# =================================================================================
# MAIN
# =================================================================================
def main():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("CROSS-ASSET MACRO MOMENTUM")
    logger.info("Time-Series + Cross-Sectional Momentum su ETF")
    logger.info("=" * 80)

    # 1. Fetch data
    logger.info("\n[1/3] DOWNLOAD DATI")
    try:
        prices, returns, benchmark = fetch_data(CONFIG)
    except Exception as e:
        logger.error(f"Errore download dati: {e}")
        logger.error("yfinance non disponibile. Esegui su macchina con internet.")
        sys.exit(1)

    if prices.shape[1] < 4:
        logger.error(f"Solo {prices.shape[1]} ETF validi, servono almeno 4")
        sys.exit(1)

    # 2. Walk-forward
    logger.info("\n[2/3] WALK-FORWARD OPTIMIZATION")
    engine = WalkForwardMacro(CONFIG)
    tsmom_rets, csmom_rets, blend_rets = engine.run(prices, returns)

    logger.info(f"\n  TSMOM: {len(tsmom_rets)} days OOS")
    logger.info(f"  CSMOM: {len(csmom_rets)} days OOS")
    logger.info(f"  Blend: {len(blend_rets)} days OOS")

    results = {
        "TSMOM": tsmom_rets,
        "CSMOM": csmom_rets,
        "Macro_Blend": blend_rets,
    }

    # 3. Report
    logger.info("\n[3/3] PERFORMANCE REPORT")
    output_dir = os.path.join(SCRIPT_DIR, "stat_arb_results_macro")
    generate_report(results, benchmark, output_dir)

    # Best strategy
    best_name = max(
        [(n, compute_metrics(r)) for n, r in results.items()],
        key=lambda x: x[1].get("_sharpe", -99)
    )
    logger.info(f"\n  🏆 MIGLIOR MACRO STRATEGY: {best_name[0]}")
    logger.info(f"     Sharpe: {best_name[1]['_sharpe']:.3f}")
    logger.info(f"     Total Return: {best_name[1]['_total_ret']:.2%}")
    logger.info(f"     Max DD: {best_name[1]['_max_dd']:.2%}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"MACRO MOMENTUM COMPLETATO in {elapsed:.1f}s")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
