#!/usr/bin/env python3
# =================================================================================
# Cross-Sectional MOMENTUM Pipeline — Mega-Cap Tech (QQQ Universe)
#
# LOGICA INVERTITA rispetto al stat_arb_biotech.py:
# - Biotech: compra i perdenti, shorta i vincitori (mean-reversion)
# - Tech:    compra i VINCITORI, shorta i PERDENTI (momentum/trend-following)
#
# Motivo: le mega-cap tech sono momentum-driven (NVDA, META, TSLA ecc.)
# La mean-reversion perdeva -85% su tech → il contrario dovrebbe funzionare.
#
# Usa lo stesso engine del biotech con INVERT_SIGNAL=True
# =================================================================================

import sys
import os

# Importa tutto dal file biotech
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stat_arb_biotech import (
    DataCollector, CrossSectionalMeanReversion, FactorMarketNeutral,
    ParameterOptimizer, WalkForwardEngine, PerformanceAnalyzer,
    LeakageValidator, estimate_halflife, CONFIG as BASE_CONFIG,
    logger, np, pd, time, os as _os
)
from itertools import product as iter_product

# Override CONFIG per Tech Momentum
CONFIG = BASE_CONFIG.copy()
CONFIG.update({
    "BENCHMARK": "QQQ",
    "UNIVERSE": [
        "AAPL", "MSFT", "GOOGL", "META", "AMZN",
        "NVDA", "AMD", "INTC", "TSLA", "AVGO",
        "QCOM", "TXN", "ADBE", "CRM", "CSCO",
        "NFLX", "AMAT", "MU", "LRCX", "INTU",
        "CMCSA", "PEP", "COST", "TMUS", "PYPL",
    ],

    # INVERSIONE SEGNALE → Momentum
    "INVERT_SIGNAL": True,

    # Commissioni più basse per mega-cap
    "COMMISSION_BPS": 10,

    # Disabilita half-life filter (non ha senso per momentum)
    "CS_HALFLIFE_FILTER": False,
    "FACTOR_HALFLIFE_FILTER": False,

    # Signal-proportional OK per momentum (concentra sui winner più forti)
    "CS_SIGNAL_PROPORTIONAL": True,
    "FACTOR_SIGNAL_PROPORTIONAL": False,

    # Volume weight e threshold rebalancing
    "CS_VOLUME_WEIGHT": True,
    "THRESHOLD_REBALANCE": True,

    # Dynamic combination
    "DYNAMIC_COMBINATION": True,
})


def run_pipeline():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("CROSS-SECTIONAL MOMENTUM PIPELINE — MEGA-CAP TECH (INVERTED)")
    logger.info(f"Benchmark: QQQ | Universe: {len(CONFIG['UNIVERSE'])} stocks")
    logger.info(f"Period: {CONFIG['DATA_START']} → {CONFIG['DATA_END']}")
    logger.info(f"INVERT_SIGNAL: {CONFIG['INVERT_SIGNAL']} (MOMENTUM MODE)")
    logger.info("=" * 80)

    # 1. Data
    logger.info("\n[1/4] DATA COLLECTION")
    collector = DataCollector(CONFIG)
    prices, volumes, benchmark = collector.fetch_all()
    returns = collector.compute_simple_returns(prices)

    # 2. Walk-Forward
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stat_arb_results_tech")
    PerformanceAnalyzer.generate_report(results, benchmark, param_history, output_dir)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"TECH MOMENTUM PIPELINE COMPLETATA in {elapsed:.1f}s")
    logger.info(f"{'='*80}")
    return results, param_history


if __name__ == "__main__":
    run_pipeline()
