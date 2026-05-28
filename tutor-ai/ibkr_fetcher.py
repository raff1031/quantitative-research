#!/usr/bin/env python3
# =================================================================================
# IBKR FETCHER — Scarica dati storici da Interactive Brokers via reqHistoricalData
#
# Drop-in replacement per yfinance nei backtest: stessa fonte dati usata in live,
# nessun vendor mismatch, prezzi adjusted per split e dividendi.
#
# PREREQUISITI:
#   pip install ib_insync
#   IB Gateway in esecuzione (porta 4002 paper / 4001 live)
#
# USO STANDALONE (download dati):
#   python ibkr_fetcher.py --tickers MRNA NBIX ILMN GLD TLT --period 1y
#   python ibkr_fetcher.py --tickers MRNA --period 5y --out prezzi_mrna.csv
#
# USO COME LIBRERIA in generate_signals.py:
#   from ibkr_fetcher import IBKRFetcher
#   fetcher = IBKRFetcher()
#   fetcher.connect()
#   prices = fetcher.download(["MRNA", "NBIX", "ILMN"], period="1y")  # stessa API di yf.download()
#   fetcher.disconnect()
#
# LIMITI IB (pacing rules):
#   - Max 60 richieste storiche / 10 min su paper account
#   - ADJUSTED_LAST disponibile solo per azioni US (per ETF usa TRADES)
#   - Dati disponibili fino a 1 anno su account paper (5 anni su live)
#
# NOTA: se IB Gateway non è disponibile, il fetcher cade automaticamente
#       su yfinance come fallback (comportamento identico a prima).
# =================================================================================

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Configurazione connessione ────────────────────────────────────────────────
_DEFAULT_HOST      = "127.0.0.1"
_DEFAULT_PORT      = 4002      # IB Gateway paper
_DEFAULT_CLIENT_ID = 10        # client_id diverso dall'executor (usa 1)
_PACING_DELAY      = 0.5       # secondi tra richieste (evita pacing violation)
_PACING_DELAY_LONG = 2.0       # delay per richieste lunghe (1+ anno)

# Durate supportate da IB (reqHistoricalData durationStr)
_PERIOD_TO_DURATION = {
    "1d": "2 D",    "5d": "5 D",
    "1mo": "1 M",   "3mo": "3 M",  "6mo": "6 M",
    "1y":  "1 Y",   "2y": "2 Y",   "5y": "5 Y",
    "10y": "10 Y",  "max": "10 Y",
    # alias comuni
    "3d": "3 D",    "2d": "2 D",
}


# =================================================================================
# IBKR FETCHER CLASS
# =================================================================================
class IBKRFetcher:
    """
    Scarica dati storici OHLCV da Interactive Brokers.
    Interfaccia compatibile con yfinance per drop-in replacement.
    """

    def __init__(self, host: str = _DEFAULT_HOST,
                 port: int = _DEFAULT_PORT,
                 client_id: int = _DEFAULT_CLIENT_ID,
                 fallback_to_yfinance: bool = True):
        """
        host/port: connessione a IB Gateway
        client_id: usa un ID diverso dall'executor (default 10)
        fallback_to_yfinance: se True, usa yfinance se IB non è disponibile
        """
        self.host      = host
        self.port      = port
        self.client_id = client_id
        self.fallback  = fallback_to_yfinance
        self.ib        = None
        self._cache: Dict[str, pd.DataFrame] = {}  # ticker -> DataFrame OHLCV

        # Controlla disponibilità ib_insync
        try:
            from ib_insync import IB, Stock, util as ib_util
            self._IB       = IB
            self._Stock    = Stock
            self._ib_util  = ib_util
            self._ib_avail = True
        except ImportError:
            self._ib_avail = False
            logger.warning("ib_insync non disponibile — fallback su yfinance")

    # -----------------------------------------------------------------
    # CONNECTION
    # -----------------------------------------------------------------
    def connect(self) -> bool:
        """Connette a IB Gateway. Ritorna True se connesso con successo."""
        if not self._ib_avail:
            return False
        try:
            self.ib = self._IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"✅ IBKRFetcher connesso (porta {self.port}, "
                        f"clientId {self.client_id})")
            return True
        except Exception as e:
            logger.warning(f"IBKRFetcher: connessione fallita ({e}) "
                           f"— fallback su yfinance")
            self.ib = None
            return False

    def disconnect(self):
        """Disconnette da IB Gateway."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKRFetcher disconnesso.")
        self.ib = None

    def is_connected(self) -> bool:
        return self.ib is not None and self.ib.isConnected()

    # -----------------------------------------------------------------
    # CORE: reqHistoricalData per un singolo ticker
    # -----------------------------------------------------------------
    def _fetch_one(self, ticker: str, duration: str,
                   bar_size: str = "1 day",
                   what_to_show: str = "ADJUSTED_LAST") -> pd.DataFrame:
        """
        Scarica OHLCV per un singolo ticker da IB.
        Ritorna DataFrame con colonne [Open, High, Low, Close, Volume].
        """
        if not self.is_connected():
            raise RuntimeError("IBKRFetcher: non connesso a IB")

        contract = self._Stock(ticker, "SMART", "USD")

        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",          # dati fino ad oggi
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,             # solo regular trading hours
                formatDate=1,            # date come stringa YYYYMMDD
                keepUpToDate=False,
            )
        except Exception as e:
            # Alcuni ETF non supportano ADJUSTED_LAST → retry con TRADES
            if what_to_show == "ADJUSTED_LAST":
                logger.debug(f"  {ticker}: ADJUSTED_LAST fallito, retry con TRADES")
                try:
                    bars = self.ib.reqHistoricalData(
                        contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=1,
                        keepUpToDate=False,
                    )
                except Exception as e2:
                    logger.warning(f"  {ticker}: reqHistoricalData fallito — {e2}")
                    return pd.DataFrame()
            else:
                logger.warning(f"  {ticker}: reqHistoricalData fallito — {e}")
                return pd.DataFrame()

        if not bars:
            logger.warning(f"  {ticker}: nessun dato ricevuto da IB")
            return pd.DataFrame()

        df = self._ib_util.df(bars)
        df.index = pd.to_datetime(df["date"])
        df.index.name = "Date"
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df.dropna(subset=["Close"])

        # Pacing delay per rispettare i limiti IB
        delay = _PACING_DELAY_LONG if "Y" in duration else _PACING_DELAY
        time.sleep(delay)

        return df

    # -----------------------------------------------------------------
    # PUBLIC: download() — interfaccia compatibile con yf.download()
    # -----------------------------------------------------------------
    def download(self, tickers: List[str], period: str = "1y",
                 bar_size: str = "1 day") -> pd.DataFrame:
        """
        Scarica prezzi storici per una lista di ticker.

        Interfaccia identica a yfinance.download():
          returns DataFrame con MultiIndex (metrica, ticker) se >1 ticker,
          oppure DataFrame flat se 1 ticker.

        Esempio:
          prices = fetcher.download(["MRNA", "NBIX"], period="1y")
          closes = prices["Close"]   # DataFrame con colonne [MRNA, NBIX]
        """
        duration = _PERIOD_TO_DURATION.get(period, "1 Y")

        if isinstance(tickers, str):
            tickers = [tickers]

        results = {}
        for ticker in tickers:
            # Usa cache se disponibile
            cache_key = f"{ticker}_{period}"
            if cache_key in self._cache:
                results[ticker] = self._cache[cache_key]
                continue

            if self.is_connected():
                logger.info(f"  IB: scarico {ticker} ({duration})...")
                df = self._fetch_one(ticker, duration, bar_size)
                if not df.empty:
                    self._cache[cache_key] = df
                    results[ticker] = df
                    continue

            # Fallback yfinance
            if self.fallback:
                results[ticker] = self._yf_fallback_ticker(ticker, period)

        if not results:
            return pd.DataFrame()

        if len(tickers) == 1:
            return results.get(tickers[0], pd.DataFrame())

        # Multi-ticker: ricostruisce MultiIndex come yf.download()
        all_dates = sorted(set(
            d for df in results.values() for d in df.index
        ))
        idx = pd.DatetimeIndex(all_dates)
        metrics = ["Open", "High", "Low", "Close", "Volume"]
        arrays  = [[m for m in metrics for _ in tickers],
                   [t for _ in metrics for t in tickers]]
        multi_idx = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
        out = pd.DataFrame(index=idx, columns=multi_idx, dtype=float)

        for ticker, df in results.items():
            for metric in metrics:
                if metric in df.columns:
                    out[(metric, ticker)] = df[metric].reindex(idx)

        return out

    # -----------------------------------------------------------------
    # PUBLIC: history() — interfaccia compatibile con yf.Ticker().history()
    # -----------------------------------------------------------------
    def history(self, ticker: str, period: str = "3d") -> pd.DataFrame:
        """
        Scarica history per un singolo ticker.
        Compatibile con yf.Ticker(ticker).history(period=period).
        """
        duration = _PERIOD_TO_DURATION.get(period, "3 D")
        cache_key = f"{ticker}_{period}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.is_connected():
            df = self._fetch_one(ticker, duration)
            if not df.empty:
                self._cache[cache_key] = df
                return df

        if self.fallback:
            return self._yf_fallback_ticker(ticker, period)

        return pd.DataFrame()

    def last_price(self, ticker: str) -> float:
        """Ritorna l'ultimo prezzo di chiusura disponibile."""
        df = self.history(ticker, period="3d")
        if df.empty or "Close" not in df.columns:
            return 0.0
        return float(df["Close"].iloc[-1])

    # -----------------------------------------------------------------
    # FALLBACK: yfinance
    # -----------------------------------------------------------------
    @staticmethod
    def _yf_fallback_ticker(ticker: str, period: str) -> pd.DataFrame:
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            logger.warning(f"  yfinance fallback fallito per {ticker}: {e}")
            return pd.DataFrame()

    # -----------------------------------------------------------------
    # BULK CLOSE PRICES — utile per generate_signals.py
    # -----------------------------------------------------------------
    def get_close_prices(self, tickers: List[str],
                         period: str = "1y") -> pd.DataFrame:
        """
        Ritorna DataFrame con sole colonne Close per ogni ticker.
        Equivale a yf.download(tickers, period)["Close"].

        Esempio:
          closes = fetcher.get_close_prices(["MRNA","NBIX","ILMN"], "1y")
          # DataFrame date×ticker con prezzi adjusted
        """
        raw = self.download(tickers, period)
        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            try:
                return raw["Close"]
            except KeyError:
                return pd.DataFrame()
        else:
            # Single ticker
            if "Close" in raw.columns:
                return raw[["Close"]]
            return pd.DataFrame()

    def clear_cache(self):
        """Svuota la cache in-memory."""
        self._cache.clear()


# =================================================================================
# CONVENIENCE FUNCTION — istanzia, scarica e disconnette in un colpo solo
# =================================================================================
def fetch_prices_ibkr(tickers: List[str], period: str = "1y",
                       host: str = _DEFAULT_HOST,
                       port: int = _DEFAULT_PORT,
                       client_id: int = _DEFAULT_CLIENT_ID,
                       fallback: bool = True) -> pd.DataFrame:
    """
    Funzione convenience: scarica close prices da IB (o yfinance come fallback).
    Ritorna DataFrame date×ticker con prezzi Close.

    Uso in generate_signals.py:
        from ibkr_fetcher import fetch_prices_ibkr
        prices = fetch_prices_ibkr(tickers, period="1y")
    """
    fetcher = IBKRFetcher(host=host, port=port, client_id=client_id,
                           fallback_to_yfinance=fallback)
    fetcher.connect()
    try:
        closes = fetcher.get_close_prices(tickers, period)
    finally:
        fetcher.disconnect()
    return closes


# =================================================================================
# CLI — download e salvataggio CSV
# =================================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Scarica dati storici da IB Gateway e li salva in CSV"
    )
    parser.add_argument("--tickers", nargs="+", required=True,
                        help="Lista di ticker, es: MRNA NBIX ILMN GLD TLT")
    parser.add_argument("--period", default="1y",
                        choices=list(_PERIOD_TO_DURATION.keys()),
                        help="Periodo storico (default: 1y)")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT,
                        help="Porta IB Gateway (default: 4002)")
    parser.add_argument("--client-id", type=int, default=_DEFAULT_CLIENT_ID,
                        help="Client ID IB (default: 10)")
    parser.add_argument("--out", default=None,
                        help="File CSV output (default: ibkr_prices_TICKER_PERIODO.csv)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Non usare yfinance se IB non disponibile")
    args = parser.parse_args()

    fetcher = IBKRFetcher(
        port=args.port,
        client_id=args.client_id,
        fallback_to_yfinance=not args.no_fallback,
    )
    connected = fetcher.connect()
    if not connected:
        logger.warning("IB non disponibile — uso yfinance come fallback")

    logger.info(f"Scaricando {args.tickers} (periodo={args.period})...")
    closes = fetcher.get_close_prices(args.tickers, args.period)
    fetcher.disconnect()

    if closes.empty:
        logger.error("Nessun dato scaricato.")
        sys.exit(1)

    logger.info(f"\n{closes.tail()}")
    logger.info(f"\nShape: {closes.shape} | Date: {closes.index[0].date()} → "
                f"{closes.index[-1].date()}")

    out_file = args.out
    if out_file is None:
        t_str = "_".join(args.tickers[:3])
        out_file = f"ibkr_prices_{t_str}_{args.period}.csv"

    closes.to_csv(out_file)
    logger.info(f"✅ Salvato: {out_file}")
