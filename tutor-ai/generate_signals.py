#!/usr/bin/env python3
# =================================================================================
# GENERATE SIGNALS — Genera current_signals.csv per ogni gamba del mega-portfolio
#
# Da runnare PRIMA di tws_executor.py per aggiornare i segnali correnti.
# Usa yfinance per scaricare i prezzi recenti e replica la logica delle strategie.
#
# USO:
#   python generate_signals.py              # Genera segnali per tutte le gambe
#   python generate_signals.py --leg bio    # Solo biotech
#   python generate_signals.py --leg tech   # Solo tech
#   python generate_signals.py --leg comm   # Solo commodity
#
# OUTPUT (per ogni gamba, nella rispettiva cartella pipeline):
#   current_signals.csv  con colonne: ticker, weight (± normalizzati a somma |w|=1)
#
# WORKFLOW CONSIGLIATO (giornaliero):
#   1. python generate_signals.py     ← calcola segnali da prezzi recenti
#   2. python tws_executor.py --mode paper --equity 10000 --dry-run  ← verifica
#   3. python tws_executor.py --mode paper --equity 10000            ← esegui
# =================================================================================

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================================
# CONFIGURAZIONE UNIVERSE — deve corrispondere ai pipeline di backtest
# =================================================================================

# Biotech "cheap" (large-cap, L/S via Factor composite)
BIO_UNIVERSE = [
    "AMGN", "GILD", "REGN", "VRTX", "MRNA",
    "BIIB", "ILMN", "ALNY", "NBIX", "INCY"
]
BIO_DIR = os.path.join(SCRIPT_DIR, "stat_arb_results_bio_cheap")

# Biotech "standard" (25 titoli, L/S via CS+Factor blend) — OPZIONALE
# Nota: SGEN e BLUE rimossi (delisted)
BIO_STANDARD_UNIVERSE = [
    "AMGN", "GILD", "REGN", "VRTX", "MRNA", "BIIB", "ILMN", "ALNY", "NBIX", "INCY",
    "BMRN", "EXEL", "ACAD", "SRPT", "RARE", "FOLD", "IONS", "HALO",
    "PCVX", "RCKT", "ARWR", "LEGN", "CRSP", "BEAM", "EDIT"
]
BIO_STANDARD_DIR = os.path.join(SCRIPT_DIR, "stat_arb_results")

# Tech Momentum
TECH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "AMAT", "LRCX",
    "KLAC", "ASML", "INTC", "MRVL", "NXPI", "ANET", "CDNS", "SNPS", "ORCL", "CRM",
    "ADBE", "NOW", "WDAY", "PANW", "CRWD", "ZS", "OKTA", "DDOG", "NET", "FTNT"
]
TECH_DIR = os.path.join(SCRIPT_DIR, "stat_arb_results_tech")

# Commodity TSMOM
COMMODITY_UNIVERSE = {
    "CL=F": "Crude Oil",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "NG=F": "Natural Gas",
    "ZC=F": "Corn",
    "ZS=F": "Soybeans",
    "ZW=F": "Wheat",
    "KC=F": "Coffee",
    "CC=F": "Cocoa",
}
COMMODITY_DIR = os.path.join(SCRIPT_DIR, "stat_arb_results_commodity")


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def download_prices(tickers: list, period: str = "6mo") -> pd.DataFrame:
    """Scarica prezzi di chiusura adjusted da yfinance."""
    logger.info(f"  Downloading {len(tickers)} tickers ({period})...")
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True,
                         progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw
        prices = prices.dropna(how="all")
        n_ok = prices.notna().any().sum()
        logger.info(f"  → {n_ok}/{len(tickers)} ticker con dati validi, "
                   f"{len(prices)} giorni")
        return prices
    except Exception as e:
        logger.error(f"Download fallito: {e}")
        return pd.DataFrame()


def save_signals(signals: pd.DataFrame, output_dir: str, leg_name: str):
    """Salva current_signals.csv nella cartella pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "current_signals.csv")
    signals.to_csv(out_path)
    logger.info(f"\n  ✅ Salvato: {out_path}")
    logger.info(f"     {len(signals)} posizioni per {leg_name}")
    logger.info(f"     Long: {(signals['weight'] > 0).sum()}  "
               f"Short: {(signals['weight'] < 0).sum()}")
    # Mostra top posizioni
    top = signals.reindex(signals['weight'].abs().sort_values(ascending=False).index).head(10)
    for ticker, row in top.iterrows():
        w = row['weight']
        bar = "+" * int(abs(w) * 30) if w > 0 else "-" * int(abs(w) * 30)
        sign = "▲" if w > 0 else "▼"
        logger.info(f"     {sign} {ticker:8}: {w:+.3f}  {bar}")


# =================================================================================
# BIOTECH — CrossSectional Mean Reversion (versione semplificata per live)
# =================================================================================

def generate_bio_signals(universe: list, output_dir: str, leg_name: str,
                          holding_days: int = 10,
                          long_quantile: float = 0.30,
                          lookback_days: int = 5,
                          price_cap: float = None):
    """
    Replica la logica CrossSectionalMeanReversion dell'ultimo rebalancing.

    Segnale: z-score del return normalizzato nell'ultimo `lookback_days`.
    Long top `long_quantile`, Short bottom `long_quantile`.
    Peso proporzionale al rank (normalizzato a |sum| = 1).

    Nota: la strategia biotech usa CS MR (mean reversion), non momentum.
    I "vincitori" recenti vengono shorted, i "perdenti" comprati.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERAZIONE SEGNALI: {leg_name}")
    logger.info(f"  Universe: {len(universe)} ticker")
    logger.info(f"  Lookback: {lookback_days}d  Holding: {holding_days}d  "
               f"Quantile: {long_quantile:.0%}")

    prices = download_prices(universe, period="3mo")
    if prices.empty:
        logger.error("  Nessun dato disponibile")
        return

    # Filtra per prezzo massimo accessibile (utile per account con capitale limitato)
    if price_cap is not None and price_cap > 0:
        current_prices = prices.iloc[-1].dropna()
        affordable = [t for t in prices.columns
                      if t in current_prices.index and current_prices[t] <= price_cap]
        too_expensive = [t for t in prices.columns if t not in affordable]
        if too_expensive:
            logger.info(f"  💡 Esclusi per prezzo > ${price_cap:.0f}: "
                       f"{', '.join(too_expensive)}")
        if len(affordable) < 4:
            logger.warning(f"  ⚠️  Solo {len(affordable)} titoli accessibili con "
                          f"price_cap=${price_cap:.0f} — "
                          f"usa --equity maggiore oppure rimuovi il filtro")
            if len(affordable) == 0:
                return
        prices = prices[[t for t in affordable if t in prices.columns]]
        logger.info(f"     Universo filtrato: {affordable}")

    # Rimuovi colonne con troppi NaN (ticker delisted o non scaricati)
    prices_clean = prices.dropna(axis=1, thresh=max(5, len(prices) // 2))
    if prices_clean.empty:
        logger.error("  Nessun ticker con dati sufficienti")
        return

    # Calcola returns sull'holding period
    rets = prices_clean.pct_change(lookback_days, fill_method=None).dropna(how="all")
    if rets.empty or len(rets) < 3:
        logger.error("  Dati insufficienti per calcolare returns")
        return

    # Usa gli ultimi N segnali e media per stabilità
    # (evita di basarsi su un solo giorno di dati)
    recent_window = min(holding_days, len(rets))
    avg_ret = rets.iloc[-recent_window:].mean()

    # Z-score cross-sectional
    mu = avg_ret.mean()
    sigma = avg_ret.std()
    if sigma < 1e-8:
        logger.warning("  Volatilità cross-sectionale troppo bassa — segnale flat")
        return

    z_scores = (avg_ret - mu) / sigma

    # Classifica
    n = len(z_scores.dropna())
    n_pos = max(1, int(n * long_quantile))

    sorted_z = z_scores.dropna().sort_values()
    short_tickers = sorted_z.index[:n_pos].tolist()   # Bottom quantile → SHORT
    long_tickers  = sorted_z.index[-n_pos:].tolist()  # Top quantile → LONG

    # Nota: CS MR = CONTRARIAN. Ret alti → short, ret bassi → long
    # (opposto al momentum)

    weights = {}
    for t in long_tickers:
        # Long i titoli "perdenti" (mean reversion: torneranno su)
        weights[t] = abs(z_scores[t])    # Peso proporzionale a |z|

    for t in short_tickers:
        # Short i titoli "vincitori" (mean reversion: scenderanno)
        weights[t] = -abs(z_scores[t])

    # Normalizza a |sum| = 1
    total = sum(abs(v) for v in weights.values()) or 1.0
    weights = {t: w / total for t, w in weights.items()}

    # Crea DataFrame e salva
    df = pd.DataFrame([{"ticker": t, "weight": w} for t, w in weights.items()])
    df = df.set_index("ticker").sort_values("weight", ascending=False)
    save_signals(df, output_dir, leg_name)


# =================================================================================
# TECH — Momentum (CrossSectional, ma usando momentum invece di MR)
# =================================================================================

def generate_tech_signals(universe: list, output_dir: str, leg_name: str,
                           momentum_lookback: int = 63,
                           long_quantile: float = 0.30):
    """
    Tech Momentum: long i titoli con momentum più forte (top quantile),
    short i più deboli (bottom quantile).

    Usa lookback 3M (63 trading days) come segnale principale.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERAZIONE SEGNALI: {leg_name}")
    logger.info(f"  Universe: {len(universe)} ticker")
    logger.info(f"  Momentum lookback: {momentum_lookback}d  Quantile: {long_quantile:.0%}")

    prices = download_prices(universe, period="6mo")
    if prices.empty:
        logger.error("  Nessun dato disponibile")
        return

    # Calcola return su momentum_lookback
    rets_clean = prices.dropna(axis=1, thresh=max(10, len(prices) // 2))
    if len(rets_clean.columns) == 0:
        logger.error("  Troppi ticker con dati mancanti")
        return

    mom = rets_clean.pct_change(momentum_lookback, fill_method=None).iloc[-1].dropna()
    if len(mom) < 6:
        logger.error("  Dati insufficienti")
        return

    # Z-score e rank
    mu = mom.mean()
    sigma = mom.std()
    if sigma < 1e-8:
        logger.warning("  Volatilità cross-sectionale troppo bassa")
        return

    z_scores = (mom - mu) / sigma
    n = len(z_scores)
    n_pos = max(1, int(n * long_quantile))

    sorted_z = z_scores.sort_values()
    short_tickers = sorted_z.index[:n_pos].tolist()   # Bottom → SHORT
    long_tickers  = sorted_z.index[-n_pos:].tolist()  # Top → LONG

    weights = {}
    for t in long_tickers:
        weights[t] = abs(z_scores[t])    # Momentum: long i vincitori
    for t in short_tickers:
        weights[t] = -abs(z_scores[t])   # Short i perdenti

    total = sum(abs(v) for v in weights.values()) or 1.0
    weights = {t: w / total for t, w in weights.items()}

    df = pd.DataFrame([{"ticker": t, "weight": w} for t, w in weights.items()])
    df = df.set_index("ticker").sort_values("weight", ascending=False)
    save_signals(df, output_dir, leg_name)


# =================================================================================
# COMMODITY — TSMOM multi-lookback (uguale a stat_arb_commodity.py)
# =================================================================================

def generate_commodity_signals(universe: dict, output_dir: str):
    """
    TSMOM blended 1M+3M+6M+12M con vol-scaling per ogni commodity.
    Coerente con stat_arb_commodity.py (CONFIG["LOOKBACKS"] e LOOKBACK_WEIGHTS]).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERAZIONE SEGNALI: Commodity TSMOM")
    logger.info(f"  Universe: {len(universe)} futures")

    tickers = list(universe.keys())
    lookbacks = [21, 63, 126, 252]
    lb_weights = np.array([0.15, 0.25, 0.35, 0.25])
    target_vol = 0.40     # vol annualizzata target per asset
    max_leverage = 3.0
    vol_lookback = 63

    prices = download_prices(tickers, period="2y")
    if prices.empty:
        logger.error("  Nessun dato disponibile")
        return

    results = []
    for sym in tickers:
        name = universe[sym]
        if sym not in prices.columns or prices[sym].isna().all():
            logger.warning(f"  {sym} ({name}): nessun dato — skip")
            continue

        price_series = prices[sym].dropna()
        rets = price_series.pct_change().dropna()

        if len(rets) < max(lookbacks) // 2:
            logger.warning(f"  {sym} ({name}): dati insufficienti ({len(rets)} days) — skip")
            continue

        # Vol annualizzata
        vol = rets.iloc[-vol_lookback:].std() * np.sqrt(252) if len(rets) >= vol_lookback else 0.25
        vol = max(vol, 0.05)

        # Segnale TSMOM blended
        blend = 0.0
        details = []
        for lb, w in zip(lookbacks, lb_weights):
            if len(rets) < lb:
                details.append(f"{lb//21}M=N/A")
                continue
            ret_lb = (1 + rets.iloc[-lb:]).prod() - 1
            sign_lb = np.sign(ret_lb)
            blend += w * sign_lb
            details.append(f"{lb//21}M={ret_lb:+.0%}")

        # Vol-scaling
        vol_scale = min(target_vol / vol, max_leverage)
        raw_signal = blend * vol_scale

        direction = "LONG  ✅" if raw_signal > 0 else ("SHORT 🔴" if raw_signal < 0 else "FLAT  ⬜")
        logger.info(f"  {sym:6} {name:15}: "
                   f"blend={blend:+.2f}  vol={vol:.0%}  "
                   f"scale={vol_scale:.1f}x  sig={raw_signal:+.2f}  → {direction}")
        logger.info(f"           [{', '.join(details)}]")

        results.append({"ticker": sym, "signal": raw_signal, "weight": 0.0})

    if not results:
        logger.warning("  Nessun segnale commodity calcolato")
        return

    df = pd.DataFrame(results).set_index("ticker")

    # Normalizza pesi proporzionali al |signal|
    total_abs = df["signal"].abs().sum() or 1.0
    df["weight"] = df["signal"] / total_abs

    df = df.sort_values("weight", ascending=False)
    save_signals(df[["weight", "signal"]], output_dir, "Commodity TSMOM")


# =================================================================================
# MAIN
# =================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera current_signals.csv per ogni gamba del mega-portfolio"
    )
    parser.add_argument("--leg", choices=["bio", "bio-standard", "tech", "comm", "all"],
                        default="all", help="Gamba da generare (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra segnali senza salvare i CSV")
    parser.add_argument("--equity", type=float, default=10_000,
                        help="Equity totale in USD (usato per filtrare titoli non "
                             "accessibili col budget disponibile, default: 10000)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"GENERATE SIGNALS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    run_all = (args.leg == "all")

    # Calcola cap prezzo per Bio: stima budget bio / posizioni minime
    # Con peso Bio ~10% e 3 posizioni per lato, max prezzo = equity × 2.5%
    # Esempio: $10k → cap $250 (esclude AMGN $375, REGN $900, VRTX $350, ALNY $336)
    bio_price_cap = args.equity * 0.025
    logger.info(f"\n  💰 Bio price cap: ${bio_price_cap:.0f} "
               f"({args.equity:,.0f} × 2.5% = max prezzo per 1 azione)")

    if run_all or args.leg == "bio":
        generate_bio_signals(
            universe=BIO_UNIVERSE,
            output_dir=BIO_DIR,
            leg_name="Biotech Cheap L/S",
            holding_days=10,
            long_quantile=0.40,   # 40% su ~6 stock accessibili → 2 pos/lato
            lookback_days=5,
            price_cap=bio_price_cap
        )

    if run_all or args.leg == "bio-standard":
        generate_bio_signals(
            universe=BIO_STANDARD_UNIVERSE,
            output_dir=BIO_STANDARD_DIR,
            leg_name="Biotech Standard L/S",
            holding_days=5,
            long_quantile=0.25,
            lookback_days=3
        )

    if run_all or args.leg == "tech":
        generate_tech_signals(
            universe=TECH_UNIVERSE,
            output_dir=TECH_DIR,
            leg_name="Tech Momentum L/S",
            momentum_lookback=63,
            long_quantile=0.30
        )

    if run_all or args.leg == "comm":
        generate_commodity_signals(
            universe=COMMODITY_UNIVERSE,
            output_dir=COMMODITY_DIR
        )

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETATO")
    logger.info("Prossimo step:")
    logger.info("  python tws_executor.py --mode paper --equity 10000 --dry-run")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
