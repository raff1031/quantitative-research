#!/usr/bin/env python3
# =================================================================================
# Statistical Arbitrage Pipeline — CHEAP BIOTECH (Long-Only, Liquid Universe)
#
# Differenze vs stat_arb_biotech.py (full XBI universe):
#
# 1. LONG-ONLY: niente short selling → no margin, no stock borrow fees (~50-150bps/anno)
#    → adatto a conti da <10k EUR senza margin enablement
#
# 2. LIQUID UNIVERSE: solo le 10 più liquide (AMGN, GILD, REGN, VRTX, MRNA, BIIB,
#    ILMN, ALNY, NBIX, INCY) → bid-ask spread 2-5bps vs 20-50bps small-cap
#
# 3. COMMISSION RIDOTTE: 5bps (da 15bps) grazie alla liquidità
#
# 4. HOLDING PERIOD MAGGIORE: 10 giorni (da 5) → meno turnover → meno friction
#    Coerente con long-only: mean-reversion su large-cap è più lenta
#
# 5. LOOKBACK PIÙ LUNGO: per catturare reversal su titoli meno volatili
#
# LOGICA ECONOMICA:
# - Long-only MR: quando un titolo sottoperforma i suoi peer large-cap del settore,
#   compri aspettando un rimbalzo verso la media del gruppo
# - Non serve shorted: il PnL viene solo dalla gamba long
# - Risk: non market-neutral! Esposizione netta lunga al settore biotech
#   → MA: su larga scala questo è accettabile con position sizing piccolo
#
# OUTPUT: stat_arb_results_bio_cheap/
# =================================================================================

import os
import sys
import copy

# Aggiungi la directory corrente al path per importare stat_arb_biotech
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Importa tutto da stat_arb_biotech (inclusi tutti i moduli)
from stat_arb_biotech import (
    CONFIG as CONFIG_ORIGINAL,
    DataCollector,
    CrossSectionalMeanReversion,
    FactorMarketNeutral,
    ParameterOptimizer,
    WalkForwardEngine,
    LeakageValidator,
    PerformanceAnalyzer,
    estimate_halflife,
    np, pd, plt, sns, stats, logger, logging,
    time, warnings, iter_product
)

# =================================================================================
# CONFIGURATION: CHEAP BIOTECH (override selettivo)
# =================================================================================
CONFIG = copy.deepcopy(CONFIG_ORIGINAL)

CONFIG.update({
    # --- Universe: solo large-cap liquidi ---
    "UNIVERSE": [
        "AMGN",   # ~130B mcap, ~2-3bps spread
        "GILD",   # ~90B mcap, ~2-3bps spread
        "REGN",   # ~80B mcap, ~3-4bps spread
        "VRTX",   # ~120B mcap, ~2-3bps spread
        "MRNA",   # ~15B mcap, ~4-5bps spread (volatiler)
        "BIIB",   # ~30B mcap, ~4-5bps spread
        "ILMN",   # ~25B mcap, ~4-5bps spread
        "ALNY",   # ~20B mcap, ~5-6bps spread
        "NBIX",   # ~12B mcap, ~5-7bps spread
        "INCY",   # ~12B mcap, ~5-7bps spread
    ],

    # --- Benchmark invariato (XBI) ---
    "BENCHMARK": "XBI",

    # --- Long-Short (mercato neutrale come l'originale) ---
    # LONG_ONLY rimosso → L/S standard
    # Borrow fee su AMGN/GILD/REGN: ~0.3-0.5%/anno (easy-to-borrow, quasi zero)
    # vs small-cap originali: 2-5%/anno — vantaggio reale

    # --- Commissioni ridotte: large-cap liquid (2-5bps spread vs 20-50bps) ---
    "COMMISSION_BPS": 5,

    # --- Holding period maggiore: MR più lenta su large-cap efficienti ---
    "CS_LOOKBACK_DAYS": 20,       # Lookback invariato
    "CS_HOLDING_DAYS": 10,        # 10 giorni vs 5 originale → meno turnover
    "CS_LONG_QUANTILE": 0.30,     # 30% → 3 longs, 3 shorts su 10 titoli

    # --- Grid: adattata per universe piccolo ---
    "CS_PARAM_GRID": {
        "lookback": [10, 15, 20, 30],
        "holding":  [5, 7, 10, 15],
        "long_q":   [0.20, 0.25, 0.30, 0.40],  # 2-4 titoli per lato su 10
    },

    # --- Signal processing ---
    "CS_SIGNAL_PROPORTIONAL": True,
    "CS_HALFLIFE_FILTER": True,
    "CS_VOLUME_WEIGHT": False,
    "MAX_HALFLIFE_DAYS": 40,      # Più permissivo: large-cap MR più lenta
    "HALFLIFE_LOOKBACK": 63,

    # --- Factor strategy (ora attiva anche sul lato short) ---
    "FACTOR_PARAM_GRID": {
        "lookback": [30, 45, 60, 90],
        "rebalance": [5, 7, 10],
        "n_long_short_pct": [0.20, 0.25, 0.30],
    },

    # --- Walk-forward invariato ---
    "TRAIN_WINDOW_MONTHS": 24,
    "TEST_WINDOW_MONTHS": 3,
    "PURGE_DAYS": 10,
    "INNER_TRAIN_PCT": 0.70,
    "INNER_PURGE_DAYS": 5,

    # --- Dynamic combination ---
    "DYNAMIC_COMBINATION": True,
    "COMBO_SHARPE_LOOKBACK": 63,
    "COMBO_SHARPE_DECAY": 0.97,
    "COMBO_MIN_WEIGHT": 0.20,
    "COMBO_MAX_WEIGHT": 0.80,

    # --- Threshold rebalancing: no (universe piccolo, abbastanza turnover) ---
    "THRESHOLD_REBALANCE": False,
    "REBAL_TURNOVER_THRESHOLD": 0.30,

    # --- Seed ---
    "SEED": 42,
})

np.random.seed(CONFIG["SEED"])


# =================================================================================
# MAIN PIPELINE (wrapper attorno a run_pipeline di stat_arb_biotech)
# =================================================================================
def run_pipeline():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("CHEAP BIOTECH PIPELINE — LONG/SHORT, LIQUID LARGE-CAP UNIVERSE")
    logger.info(f"Universe: {CONFIG['UNIVERSE']}")
    logger.info(f"Mode: Long/Short (market-neutral) | Commission: {CONFIG['COMMISSION_BPS']}bps")
    logger.info(f"Borrow fee vantaggio: ~0.3-0.5%/anno (easy-to-borrow) vs 2-5% small-cap")
    logger.info(f"Holding: {CONFIG['CS_HOLDING_DAYS']} giorni | Lookback: {CONFIG['CS_LOOKBACK_DAYS']} giorni")
    logger.info(f"Period: {CONFIG['DATA_START']} → {CONFIG['DATA_END']}")
    logger.info(f"Param grid: CS={len(list(iter_product(*CONFIG['CS_PARAM_GRID'].values())))} combos")
    logger.info("=" * 80)

    # 1. Dati
    logger.info("\n[1/4] DATA COLLECTION")
    collector = DataCollector(CONFIG)
    prices, volumes, benchmark = collector.fetch_all()
    returns = collector.compute_simple_returns(prices)

    logger.info(f"  Universe effettivo: {list(prices.columns)}")
    logger.info(f"  N ticker: {len(prices.columns)}")

    # 2. Walk-Forward
    logger.info("\n[2/4] WALK-FORWARD + PARAMETER OPTIMIZATION")
    engine = WalkForwardEngine(CONFIG)
    results, param_history = engine.run(prices, returns, benchmark, volumes=volumes)

    if not results:
        logger.error("Nessun risultato!"); return None

    # 3. Leakage Validation
    logger.info("\n[3/4] LEAKAGE VALIDATION")
    validator = LeakageValidator()
    for name in ["CS_Optimized", "Combined_Dynamic"]:
        if name in results and len(results[name]) > 10:
            bt = validator.block_bootstrap_test(results[name])
            logger.info(f"\n  {name}:")
            logger.info(f"    Actual Sharpe: {bt['actual_sharpe']:.3f}")
            logger.info(f"    Bootstrap 95th: {bt['p95']:.3f}")
            logger.info(f"    p-value: {bt['pvalue']:.3f}")
            sig = "✅ Significativo al 5%" if bt['significant'] else "⚠️ NON significativo"
            logger.info(f"    {sig}")

    # 4. Report
    logger.info("\n[4/4] PERFORMANCE REPORT")
    output_dir = os.path.join(_this_dir, "stat_arb_results_bio_cheap")
    PerformanceAnalyzer.generate_report(results, benchmark, param_history, output_dir)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"CHEAP BIOTECH PIPELINE COMPLETATA in {elapsed:.1f}s")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*80}")
    return results, param_history


if __name__ == "__main__":
    run_pipeline()
