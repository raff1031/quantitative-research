#!/usr/bin/env python3
# =================================================================================
# TWS EXECUTOR — Esecuzione live/paper via Interactive Brokers API
#
# Collega il mega-portfolio (4 gambe) a Interactive Brokers TWS/Gateway.
# Legge i segnali dai CSV del pipeline, calcola i target weights via N_MinVar,
# confronta con le posizioni attuali e piazza gli ordini necessari.
#
# PREREQUISITI:
#   pip install ib_insync pandas numpy
#   IB Gateway in esecuzione su porta 4002 (paper) o 4001 (live)
#
# USO:
#   python tws_executor.py --mode paper --equity 10000 --dry-run
#   python tws_executor.py --mode paper --equity 10000
#
# GAMBE:
#   1. Biotech MR     — equity L/S su 27 titoli (margin account)
#   2. Tech Momentum  — equity L/S su ~15 titoli (margin account)
#   3. Macro ETF      — long-only su 8 ETF (no margin)
#   4. Commodity TSMOM— futures (CL, GC, ZC, ZS, ZW, SI, HG, NG, KC, CC)
#
# NOTA FUTURES:
#   Per account <$50k si usano automaticamente i micro-futures dove disponibili
#   (MCL, MGC). Per grains (ZC, ZS, ZW) non esistono micro → si skippa se il
#   notional supera il budget allocato.
# =================================================================================

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# yfinance — prezzi gratuiti (evita problemi subscription IB per paper account)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("⚠️  yfinance non installato. Installa con: pip install yfinance")
    print("   Prezzi live non disponibili per macro/commodity.")

# ib_insync — wrapper semplificato per IB API
try:
    from ib_insync import IB, Stock, Future, Contract, Order, LimitOrder, MarketOrder
    from ib_insync import util as ib_util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("⚠️  ib_insync non installato. Installa con: pip install ib_insync")
    print("   In dry-run mode puoi comunque vedere i segnali calcolati.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================================
# CONFIGURATION
# =================================================================================
CONFIG = {
    # IB Gateway / TWS connection
    "HOST":      "127.0.0.1",
    "PORT_PAPER": 4002,    # IB Gateway paper
    "PORT_LIVE":  4001,    # IB Gateway live
    "PORT_TWS_PAPER": 7497,  # TWS paper (alternativa a Gateway)
    "PORT_TWS_LIVE":  7496,  # TWS live
    "CLIENT_ID":  1,

    # Pipeline output directories (legge i CSV del backtest)
    # BIO_DIR: usa stat_arb_results_bio_cheap/ se esiste (large-cap L/S via Factor),
    # altrimenti stat_arb_results/ (27-ticker standard pipeline)
    "BIO_DIR":        os.path.join(SCRIPT_DIR, "stat_arb_results_bio_cheap"),
    "BIO_DIR_STD":    os.path.join(SCRIPT_DIR, "stat_arb_results"),
    "TECH_DIR":       os.path.join(SCRIPT_DIR, "stat_arb_results_tech"),
    "MACRO_DIR":      os.path.join(SCRIPT_DIR, "stat_arb_results_macro"),
    "COMMODITY_DIR":  os.path.join(SCRIPT_DIR, "stat_arb_results_commodity"),

    # Mega-portfolio weights (N_MinVar stimati dal backtest)
    # Questi vengono aggiornati automaticamente se disponibili i CSV
    # oppure si usano questi come fallback fissi
    "FALLBACK_WEIGHTS": {
        "Bio":       0.11,
        "Tech":      0.13,
        "Macro":     0.65,
        "Commodity": 0.11,
    },

    # Risk management
    "MAX_POSITION_PCT":  0.15,   # Max % del portfolio su singolo titolo
    "MAX_SECTOR_PCT":    0.35,   # Max % per gamba
    "MIN_ORDER_USD":     100,    # Ordini sotto questa soglia vengono ignorati
    "ORDER_TYPE":       "LMT",   # LMT (limit) o MKT (market)
    "LMT_OFFSET_PCT":   0.002,   # 0.2% offset dal mid per limit orders
    "SLIPPAGE_EST_BPS":  5,      # Stima slippage per logging (non modifica ordini)

    # Rebalancing
    "REBAL_THRESHOLD":  0.05,    # Rebalancia solo se peso attuale diverge >5%
    "MIN_DAYS_BETWEEN_REBAL": 3, # Non rebalancia più spesso di così

    # Commodity futures — contratti front-month
    # Per account piccoli (<$50k) si usano micro dove disponibili
    "FUTURES_CONTRACTS": {
        "CL=F": {"symbol": "CL",  "exchange": "NYMEX", "currency": "USD",
                  "micro": "MCL",  "micro_exchange": "NYMEX",
                  "multiplier": 1000, "micro_multiplier": 100},
        "GC=F": {"symbol": "GC",  "exchange": "COMEX", "currency": "USD",
                  "micro": "MGC",  "micro_exchange": "COMEX",
                  "multiplier": 100,  "micro_multiplier": 10},
        "SI=F": {"symbol": "SI",  "exchange": "COMEX", "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 5000, "micro_multiplier": None},
        "HG=F": {"symbol": "HG",  "exchange": "COMEX", "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 25000, "micro_multiplier": None},
        "NG=F": {"symbol": "NG",  "exchange": "NYMEX", "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 10000, "micro_multiplier": None},
        "ZC=F": {"symbol": "ZC",  "exchange": "CBOT",  "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 5000, "micro_multiplier": None},
        "ZS=F": {"symbol": "ZS",  "exchange": "CBOT",  "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 5000, "micro_multiplier": None},
        "ZW=F": {"symbol": "ZW",  "exchange": "CBOT",  "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 5000, "micro_multiplier": None},
        "KC=F": {"symbol": "KC",  "exchange": "NYBOT", "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 37500, "micro_multiplier": None},
        "CC=F": {"symbol": "CC",  "exchange": "NYBOT", "currency": "USD",
                  "micro": None,   "micro_exchange": None,
                  "multiplier": 10, "micro_multiplier": None},
    },

    # Macro ETF universe
    "MACRO_ETFS": ["GLD", "TLT", "VNQ", "EEM", "XLE", "UUP", "EFA", "DBC"],
}


# =================================================================================
# SIGNAL LOADER — legge i CSV del pipeline
# =================================================================================
class SignalLoader:
    """
    Legge i returns storici dai CSV del backtest per stimare il segnale corrente.
    Il "segnale" è implicito nell'ultimo rebalancing del walk-forward:
    la direzione e il peso delle posizioni attuali è già codificata nel CSV.

    Nota: per un sistema live vero, dovresti runnare la pipeline giornalmente
    e leggere i weights direttamente. Qui usiamo un'approssimazione:
    calcoliamo il segnale attuale runnando la strategia sui prezzi recenti.
    """

    def __init__(self, config: dict, equity: float):
        self.config = config
        self.equity = equity

    def load_leg_returns(self, directory: str, candidates: list) -> Optional[pd.Series]:
        """Carica il CSV di returns migliore disponibile."""
        for fname in candidates:
            path = os.path.join(directory, fname)
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                rets = df.iloc[:, 0].dropna()
                logger.info(f"  Loaded {fname}: {len(rets)} days, "
                           f"last={rets.index[-1].date()}")
                return rets
        return None

    def compute_n_minvar_weights(self, legs: Dict[str, pd.Series],
                                  lookback: int = 63) -> Dict[str, float]:
        """
        Calcola i pesi N_MinVar rolling sull'ultimo lookback di dati.
        Replica la logica di mega_portfolio.combine_n_min_variance().
        """
        # Allinea date
        common_idx = legs[list(legs.keys())[0]].index
        for s in legs.values():
            common_idx = common_idx.intersection(s.index)

        if len(common_idx) < lookback:
            logger.warning(f"Dati insufficienti ({len(common_idx)} giorni), uso fallback weights")
            return self.config["FALLBACK_WEIGHTS"]

        # Usa gli ultimi `lookback` giorni
        df = pd.DataFrame({k: v.loc[common_idx] for k, v in legs.items()})
        recent = df.iloc[-lookback:]

        cov = recent.cov().values
        n = len(legs)
        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
            ones = np.ones(n)
            raw_w = inv_cov @ ones / (ones @ inv_cov @ ones + 1e-10)
            raw_w = np.clip(raw_w, 0.05, 0.80)
            raw_w = raw_w / raw_w.sum()
        except Exception:
            raw_w = np.ones(n) / n

        leg_names = list(legs.keys())
        weights = {leg_names[i]: raw_w[i] for i in range(n)}
        return weights

    def load_all(self) -> Tuple[Dict[str, float], Dict[str, pd.Series]]:
        """
        Carica tutti i legs e calcola i pesi N_MinVar correnti.
        Ritorna: (weights, leg_returns)
        """
        legs = {}

        bio = self.load_leg_returns(
            self.config["BIO_DIR"],
            ["Combined_Dynamic_returns.csv", "Combined_InvVol_returns.csv"]
        )
        if bio is not None:
            legs["Bio"] = bio

        tech = self.load_leg_returns(
            self.config["TECH_DIR"],
            ["Combined_Dynamic_returns.csv", "Combined_InvVol_returns.csv"]
        )
        if tech is not None:
            legs["Tech"] = tech

        macro = self.load_leg_returns(
            self.config["MACRO_DIR"],
            ["Macro_Blend_returns.csv", "TSMOM_returns.csv"]
        )
        if macro is not None:
            legs["Macro"] = macro

        commodity = self.load_leg_returns(
            self.config["COMMODITY_DIR"],
            ["Commodity_TSMOM_returns.csv"]
        )
        if commodity is not None:
            legs["Commodity"] = commodity

        if not legs:
            logger.error("Nessun leg disponibile — runnare prima i pipeline!")
            sys.exit(1)

        weights = self.compute_n_minvar_weights(legs)
        logger.info("\n  === TARGET WEIGHTS (N_MinVar) ===")
        for leg, w in weights.items():
            budget = w * self.equity
            logger.info(f"    {leg:<12}: {w:.1%}  (${budget:,.0f})")

        return weights, legs


# =================================================================================
# CONTRACT FACTORY — crea i contratti IB corretti per ogni strumento
# =================================================================================
class ContractFactory:
    def __init__(self, equity: float, config: dict):
        self.equity = equity
        self.config = config
        self.use_micro = equity < 100_000  # Micro futures per account <100k

    def make_stock(self, symbol: str) -> "Stock":
        """Azione o ETF quotata su SMART exchange."""
        if not IB_AVAILABLE:
            # Restituisce un mock object per dry-run senza IB installato
            from types import SimpleNamespace
            return SimpleNamespace(symbol=symbol, exchange="SMART", currency="USD")
        return Stock(symbol, "SMART", "USD")

    def make_future(self, yf_symbol: str) -> Optional[Tuple["Future", int]]:
        """
        Crea un contratto Future per il front-month.
        Ritorna (contract, multiplier) o None se non fattibile.
        """
        if not IB_AVAILABLE:
            return None  # Futures non eseguibili senza ib_insync

        if yf_symbol not in self.config["FUTURES_CONTRACTS"]:
            logger.warning(f"Futures spec non trovata per {yf_symbol}")
            return None

        spec = self.config["FUTURES_CONTRACTS"][yf_symbol]

        # Front-month: mese successivo
        now = datetime.now()
        front_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
        expiry = front_month.strftime("%Y%m")

        # Usa micro se account piccolo e micro disponibile
        if self.use_micro and spec["micro"] is not None:
            contract = Future(
                symbol=spec["micro"],
                lastTradeDateOrContractMonth=expiry,
                exchange=spec["micro_exchange"],
                currency=spec["currency"]
            )
            multiplier = spec["micro_multiplier"]
            logger.info(f"  {yf_symbol}: usando MICRO ({spec['micro']}, "
                       f"exp={expiry}, mult={multiplier})")
        else:
            contract = Future(
                symbol=spec["symbol"],
                lastTradeDateOrContractMonth=expiry,
                exchange=spec["exchange"],
                currency=spec["currency"]
            )
            multiplier = spec["multiplier"]

        return contract, multiplier


# =================================================================================
# POSITION SIZER — converte weights in quantità ordini
# =================================================================================
class PositionSizer:
    def __init__(self, equity: float, config: dict):
        self.equity = equity
        self.config = config

    def compute_stock_shares(self, budget_usd: float, price: float,
                              long_short_signal: float) -> int:
        """
        Calcola il numero di azioni da comprare/vendere.
        long_short_signal: +1 (long), -1 (short), 0 (flat)
        """
        if price <= 0 or abs(long_short_signal) < 0.01:
            return 0
        shares = int(budget_usd / price)
        max_shares = int(self.equity * self.config["MAX_POSITION_PCT"] / price)
        shares = min(shares, max_shares)
        return int(shares * np.sign(long_short_signal))

    def compute_futures_contracts(self, budget_usd: float, price: float,
                                   multiplier: int, signal: float) -> int:
        """
        Calcola il numero di contratti futures.
        Notional = price × multiplier × n_contracts
        """
        if price <= 0 or multiplier <= 0 or abs(signal) < 0.01:
            return 0
        notional_per_contract = price * multiplier
        n_contracts = int(budget_usd / notional_per_contract)
        if notional_per_contract > budget_usd * 0.5:
            logger.warning(f"  Notional/contract (${notional_per_contract:,.0f}) > "
                          f"50% budget (${budget_usd:,.0f}) — skip")
            return 0
        return int(n_contracts * np.sign(signal))

    def min_order_check(self, budget_usd: float) -> bool:
        return budget_usd >= self.config["MIN_ORDER_USD"]


# =================================================================================
# TWS EXECUTOR — connette, legge posizioni, piazza ordini
# =================================================================================
class TWSExecutor:
    def __init__(self, mode: str, equity: float, dry_run: bool,
                 use_tws: bool = False):
        """
        mode: 'paper' o 'live'
        equity: capitale totale da gestire (USD)
        dry_run: se True, calcola ordini ma non li invia
        use_tws: se True, connette a TWS invece di IB Gateway
        """
        self.mode = mode
        self.equity = equity
        self.dry_run = dry_run
        self.ib = None

        # Seleziona porta
        if use_tws:
            self.port = CONFIG["PORT_TWS_PAPER"] if mode == "paper" else CONFIG["PORT_TWS_LIVE"]
        else:
            self.port = CONFIG["PORT_PAPER"] if mode == "paper" else CONFIG["PORT_LIVE"]

        self.factory        = ContractFactory(equity, CONFIG)
        self.sizer          = PositionSizer(equity, CONFIG)
        self.loader         = SignalLoader(CONFIG, equity)
        self.pending_symbols: set = set()   # simboli con ordini aperti (anti-dup)

    # -----------------------------------------------------------------
    # CONNECTION
    # -----------------------------------------------------------------
    def connect(self) -> bool:
        if not IB_AVAILABLE:
            logger.warning("ib_insync non disponibile — modalità dry-run forzata")
            self.dry_run = True
            return False
        self.ib = IB()
        try:
            self.ib.connect(CONFIG["HOST"], self.port, clientId=CONFIG["CLIENT_ID"])
            logger.info(f"✅ Connesso a IB {self.mode.upper()} (porta {self.port})")
            return True
        except Exception as e:
            logger.error(f"❌ Connessione fallita: {e}")
            logger.error(f"   Assicurati che IB Gateway sia in esecuzione su porta {self.port}")
            if not self.dry_run:
                sys.exit(1)
            return False

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnesso da IB")

    # -----------------------------------------------------------------
    # ACCOUNT INFO
    # -----------------------------------------------------------------
    def get_account_equity(self) -> float:
        if not self.ib or not self.ib.isConnected():
            return self.equity
        account_values = self.ib.accountValues()
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                real_equity = float(av.value)
                logger.info(f"  Account equity reale: ${real_equity:,.2f}")
                return real_equity
        return self.equity

    def get_current_positions(self) -> Dict[str, float]:
        """Ritorna {symbol: net_quantity} delle posizioni attuali."""
        if not self.ib or not self.ib.isConnected():
            return {}
        positions = {}
        for pos in self.ib.positions():
            symbol = pos.contract.symbol
            positions[symbol] = pos.position
        return positions

    def load_open_orders(self):
        """
        Carica gli ordini aperti da IB e popola self.pending_symbols.
        Chiamato all'inizio di run() per evitare ordini duplicati quando
        il pipeline viene eseguito più volte senza aver chiuso gli ordini
        precedenti.

        Se un simbolo ha già un ordine aperto (Submitted / PreSubmitted /
        PendingSubmit), viene aggiunto a pending_symbols e saltato in
        place_order().
        """
        self.pending_symbols = set()
        if not self.ib or not self.ib.isConnected():
            return
        try:
            # reqAllOpenOrders recupera TUTTI gli ordini aperti da tutte le
            # sessioni client (non solo quella corrente). Poi sleep(1) per
            # dare tempo ai callback openOrder di popolare ib.trades().
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)
            open_trades = self.ib.openTrades()
            if not open_trades:
                logger.info("  Nessun ordine aperto.")
                return
            logger.info(f"\n  ⚠️  Ordini aperti trovati: {len(open_trades)}")
            for trade in open_trades:
                sym    = trade.contract.symbol
                status = trade.orderStatus.status
                action = trade.order.action
                qty    = trade.order.totalQuantity
                logger.info(f"    ⛔  {sym:8} {action:4} {qty:.0f}  [{status}]")
                self.pending_symbols.add(sym)
            logger.warning(
                f"\n  ⛔  {len(self.pending_symbols)} simboli già in portafoglio/ordini "
                f"→ verranno SALTATI per evitare duplicati:\n"
                f"     {sorted(self.pending_symbols)}"
            )
            logger.info(
                "  💡 Per forzare nuovi ordini: cancella gli ordini aperti in TWS, "
                "poi ri-esegui."
            )
        except Exception as e:
            logger.warning(f"  Impossibile caricare ordini aperti: {e}")

    def get_price(self, contract) -> float:
        """Ottieni mid-price per un contratto (via IB API)."""
        if not self.ib or not self.ib.isConnected():
            return 0.0
        try:
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)
            mid = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else ticker.last
            return float(mid) if mid and mid > 0 else 0.0
        except Exception as e:
            logger.warning(f"  Prezzo non disponibile per {contract.symbol}: {e}")
            return 0.0

    def get_price_yf(self, yf_ticker: str) -> float:
        """
        Ottieni l'ultimo prezzo di chiusura da yfinance.
        Usato come alternativa gratuita ai dati IB (nessun abbonamento richiesto).
        Funziona per: azioni (AAPL), ETF (GLD, TLT), futures (CL=F, GC=F).
        """
        if not YF_AVAILABLE:
            return 0.0
        try:
            data = yf.Ticker(yf_ticker).history(period="3d")
            if data.empty:
                logger.warning(f"  yfinance: nessun dato per {yf_ticker}")
                return 0.0
            price = float(data["Close"].iloc[-1])
            return price
        except Exception as e:
            logger.warning(f"  yfinance error per {yf_ticker}: {e}")
            return 0.0

    def get_yf_returns(self, yf_ticker: str, days: int = 252) -> Optional[pd.Series]:
        """
        Scarica returns storici da yfinance per calcolo segnali TSMOM.
        days: numero di giorni di storia richiesta.
        """
        if not YF_AVAILABLE:
            return None
        try:
            # Scarica un po' di più per avere buffer su weekends/holidays
            period = f"{days + 30}d"
            data = yf.Ticker(yf_ticker).history(period=period)
            if data.empty or len(data) < 10:
                return None
            returns = data["Close"].pct_change().dropna()
            return returns
        except Exception as e:
            logger.warning(f"  yfinance error per {yf_ticker}: {e}")
            return None

    # -----------------------------------------------------------------
    # ORDER PLACEMENT
    # -----------------------------------------------------------------
    def create_order(self, quantity: int, price: float) -> "Order":
        """
        Crea un ordine limit o market.
        Limit: offset di LMT_OFFSET_PCT dal mid per ridurre market impact.
        """
        if CONFIG["ORDER_TYPE"] == "MKT":
            return MarketOrder("BUY" if quantity > 0 else "SELL", abs(quantity))
        else:
            # Limit order: buy leggermente sopra mid, sell leggermente sotto
            if quantity > 0:  # BUY
                lmt_price = round(price * (1 + CONFIG["LMT_OFFSET_PCT"]), 2)
                return LimitOrder("BUY", abs(quantity), lmt_price)
            else:  # SELL / SHORT
                lmt_price = round(price * (1 - CONFIG["LMT_OFFSET_PCT"]), 2)
                return LimitOrder("SELL", abs(quantity), lmt_price)

    def place_order(self, contract, quantity: int, price: float, description: str):
        """Piazza un singolo ordine, con logging dettagliato."""
        if quantity == 0:
            return

        # ── Anti-duplicato: salta se c'è già un ordine aperto sullo stesso simbolo ──
        sym = getattr(contract, "symbol", description)
        if sym in self.pending_symbols:
            logger.warning(
                f"  ⛔ SKIP {description}: ordine aperto già presente "
                f"(cancellalo in TWS per piazzarne uno nuovo)"
            )
            return

        action = "BUY" if quantity > 0 else "SELL/SHORT"
        notional = abs(quantity) * price

        if notional < CONFIG["MIN_ORDER_USD"]:
            logger.info(f"  ⬛ SKIP {description}: notional ${notional:.0f} < "
                       f"min ${CONFIG['MIN_ORDER_USD']}")
            return

        logger.info(f"  {'🟢' if quantity > 0 else '🔴'} {action:10} "
                   f"{abs(quantity):5} × {description:<20} "
                   f"@ ${price:.2f}  notional=${notional:,.0f}")

        if self.dry_run:
            logger.info(f"    [DRY RUN] Ordine NON inviato")
            return

        if not self.ib or not self.ib.isConnected():
            logger.warning(f"    [OFFLINE] Connessione non disponibile")
            return

        order = self.create_order(quantity, price)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"    ✅ Ordine inviato: orderId={trade.order.orderId}")

    # -----------------------------------------------------------------
    # STRATEGY: BIOTECH / TECH EQUITY L/S
    # -----------------------------------------------------------------
    def execute_equity_leg(self, leg_name: str, leg_dir: str, budget_usd: float):
        """
        Legge il CSV dei pesi della strategia e piazza ordini equity.
        Nota: i pesi L/S sono impliciti nei returns — qui usiamo un'approssimazione
        basata sull'ultimo rebalancing disponibile dai CSV dei pesi.
        """
        logger.info(f"\n  --- {leg_name} (budget: ${budget_usd:,.0f}) ---")

        # Cerca il CSV dei pesi se disponibile
        weight_file = os.path.join(leg_dir, "last_weights.csv")
        signal_file = os.path.join(leg_dir, "current_signals.csv")

        if os.path.exists(signal_file):
            # Leggi segnali generati dalla pipeline
            signals = pd.read_csv(signal_file, index_col=0)
            logger.info(f"    Segnali caricati: {len(signals)} posizioni")

            # Calcola prezzi via yfinance direttamente (più rapido — IB paper
            # non ha abbonamento per la maggior parte dei ticker)
            ticker_prices = {}
            tickers_list = [str(t) for t in signals.index]
            if YF_AVAILABLE and tickers_list:
                try:
                    raw = yf.download(tickers_list, period="2d",
                                      auto_adjust=True, progress=False,
                                      threads=True)
                    if isinstance(raw.columns, pd.MultiIndex):
                        closes = raw["Close"].iloc[-1]
                    else:
                        closes = raw.iloc[-1]
                    for t in tickers_list:
                        if t in closes.index and not np.isnan(closes[t]):
                            ticker_prices[t] = float(closes[t])
                except Exception as e:
                    logger.warning(f"    yfinance batch download: {e}")

            # --- Pass 1: identifica posizioni eseguibili ---
            executable   = {}   # ticker -> (weight, price)
            skipped_long  = []
            skipped_short = []

            for ticker, row in signals.iterrows():
                ticker = str(ticker)
                weight = float(row.get("weight", 0))
                if abs(weight) < 0.01:
                    continue
                position_usd = budget_usd * abs(weight)

                # Prezzo: batch yfinance già scaricato, oppure IB come fallback
                price = ticker_prices.get(ticker, 0.0)
                if price <= 0 and IB_AVAILABLE:
                    price = self.get_price(self.factory.make_stock(ticker))
                if price <= 0:
                    logger.warning(f"    Prezzo non disponibile: {ticker}")
                    continue

                shares = self.sizer.compute_stock_shares(position_usd, price, weight)
                if shares != 0:
                    executable[ticker] = (weight, price)
                else:
                    entry = f"{ticker}(w={weight:+.2f} ${position_usd:.0f}@${price:.0f})"
                    if weight > 0:
                        skipped_long.append(entry)
                    else:
                        skipped_short.append(entry)

            # --- Log posizioni skippate ---
            all_skipped = skipped_long + skipped_short
            if all_skipped:
                logger.warning(
                    f"    ⚠️  {len(all_skipped)} pos. skippate (capitale insufficiente):"
                )
                for s in all_skipped[:6]:
                    logger.warning(f"       {s}")
                if len(all_skipped) > 6:
                    logger.warning(f"       ... e altri {len(all_skipped)-6}")

            # --- Controllo bilanciamento L/S ---
            n_exec_long  = sum(1 for w, _ in executable.values() if w > 0)
            n_exec_short = sum(1 for w, _ in executable.values() if w < 0)

            if skipped_long and n_exec_long == 0 and n_exec_short > 0:
                logger.warning(
                    f"\n    ⚠️  PORTAFOGLIO SBILANCIATO: 0 LONG eseguibili, "
                    f"{n_exec_short} SHORT → gamba net-short (non market-neutral!)"
                )
                long_sigs = signals[signals["weight"] > 0]
                if not long_sigs.empty:
                    max_long_w = long_sigs["weight"].max()
                    logger.warning(
                        f"       Budget max per singolo long: "
                        f"${int(budget_usd * max_long_w)} "
                        f"(peso={max_long_w:.2f} × ${budget_usd:.0f})"
                    )
                logger.info(
                    f"    💡 Rigenera segnali con filtro prezzo: "
                    f"python generate_signals.py --equity {int(self.equity)}"
                )
            elif skipped_short and n_exec_short == 0 and n_exec_long > 0:
                logger.warning(
                    f"\n    ⚠️  PORTAFOGLIO SBILANCIATO: {n_exec_long} LONG, "
                    f"0 SHORT eseguibili → gamba net-long"
                )

            if not executable:
                # Nessuna posizione eseguibile — calcola budget minimo
                top_w = signals["weight"].abs().max()
                top_ticker = signals["weight"].abs().idxmax()
                top_price = ticker_prices.get(str(top_ticker), 0)
                if top_price > 0:
                    min_budget = top_price / top_w
                    logger.info(
                        f"\n    💡 Per eseguire almeno 1 pos. servono ≥ ${min_budget:,.0f} totali."
                    )
                    logger.info(
                        f"       Oppure: python generate_signals.py --equity {int(self.equity)}"
                    )
                return

            # --- Pass 2: ri-normalizza pesi tra posizioni eseguibili e piazza ordini ---
            # Ri-normalizzare usa il budget intero sulle sole posizioni accessibili
            total_exec_abs = sum(abs(w) for w, _ in executable.values()) or 1.0
            orders_placed = 0

            for ticker, (weight, price) in executable.items():
                new_weight   = weight / total_exec_abs          # peso ri-normalizzato
                position_usd = budget_usd * abs(new_weight)
                shares = self.sizer.compute_stock_shares(position_usd, price, new_weight)
                if shares == 0:
                    continue
                contract = self.factory.make_stock(ticker)
                self.place_order(contract, shares, price, ticker)
                orders_placed += 1
        else:
            # Nessun file di segnali — avviso e skip
            logger.warning(f"    ⚠️  Segnali non trovati in {leg_dir}")
            logger.warning(f"    Crea 'current_signals.csv' con colonne [ticker, weight]")
            logger.warning(f"    Esempio: AMGN,+0.15 | GILD,-0.12 | ...")
            self._show_signal_template(leg_name)

    def _show_signal_template(self, leg_name: str):
        """Mostra un esempio di formato segnali atteso."""
        logger.info(f"\n    TEMPLATE current_signals.csv per {leg_name}:")
        logger.info(f"    ticker,weight")
        logger.info(f"    AMGN,+0.15")
        logger.info(f"    GILD,-0.12")
        logger.info(f"    (weight positivo = long, negativo = short)")

    # -----------------------------------------------------------------
    # STRATEGY: MACRO ETF (long-only, TSMOM segnale binario)
    # -----------------------------------------------------------------
    def execute_macro_leg(self, budget_usd: float):
        """
        Macro TSMOM: long gli ETF con return positivo negli ultimi 3 mesi,
        flat quelli con return negativo. Equal-weight sugli ETF long.

        Usa yfinance per prezzi e calcolo segnali — nessun abbonamento IB
        richiesto. Questo è il modo corretto per paper account che non hanno
        sottoscrizioni ai dati live.
        """
        logger.info(f"\n  --- Macro ETF (budget: ${budget_usd:,.0f}) ---")

        longs = []
        prices = {}

        for etf in CONFIG["MACRO_ETFS"]:
            # Scarica returns ultimi 6 mesi via yfinance
            ret_series = self.get_yf_returns(etf, days=130)
            price = self.get_price_yf(etf)

            if ret_series is None or len(ret_series) < 63 or price <= 0:
                logger.warning(f"    {etf}: dati insufficienti — skip")
                continue

            # Segnale TSMOM 3M (63 giorni di trading)
            ret_3m = (1 + ret_series.iloc[-63:]).prod() - 1
            direction = "LONG  ✅" if ret_3m > 0 else "flat  ⬜"
            logger.info(f"    {etf:6}: ret_3M={ret_3m:+.1%}  ${price:.2f}  → {direction}")

            if ret_3m > 0:
                longs.append(etf)
                prices[etf] = price

        if not longs:
            logger.info("    Nessun ETF con segnale positivo — posizione flat")
            return

        per_etf_usd = budget_usd / len(longs)
        logger.info(f"\n    → ETF selezionati ({len(longs)}): {longs}")
        logger.info(f"    → Budget/ETF: ${per_etf_usd:,.0f}")

        for etf in longs:
            contract = self.factory.make_stock(etf)
            price = prices[etf]
            shares = self.sizer.compute_stock_shares(per_etf_usd, price, +1)
            self.place_order(contract, shares, price, etf)

    # -----------------------------------------------------------------
    # STRATEGY: COMMODITY FUTURES (TSMOM)
    # -----------------------------------------------------------------
    def execute_commodity_leg(self, budget_usd: float):
        """
        Commodity TSMOM multi-lookback: segnale calcolato da yfinance
        (CL=F, GC=F, etc. sono ticker yfinance nativi per futures front-month).

        Segnale: blend 1M+3M+6M+12M (pesi [0.15,0.25,0.35,0.25]) con
        vol-scaling individuale e portafoglio equalizzato per notionale.

        Gli ordini IB usano i contratti futures corretti (micro se equity<100k).
        """
        logger.info(f"\n  --- Commodity TSMOM (budget: ${budget_usd:,.0f}) ---")

        # Lookback e pesi (coerenti con stat_arb_commodity.py)
        lookbacks = [21, 63, 126, 252]
        lb_weights = np.array([0.15, 0.25, 0.35, 0.25])
        target_vol_asset = 0.40   # vol annualizzata target per singolo asset
        max_leverage = 3.0

        # Carica segnali da current_signals.csv se disponibile (da generate_signals.py)
        signal_override_file = os.path.join(CONFIG["COMMODITY_DIR"], "current_signals.csv")
        signal_override = {}
        if os.path.exists(signal_override_file):
            try:
                sig_df = pd.read_csv(signal_override_file, index_col=0)
                if "signal" in sig_df.columns:
                    signal_override = sig_df["signal"].to_dict()
                    logger.info(f"    Caricati segnali da {signal_override_file}")
            except Exception:
                pass

        # Calcola segnali e prezzi per ogni commodity
        signals = {}    # yf_sym -> segnale vol-scaled
        prices  = {}    # yf_sym -> prezzo

        for yf_sym in CONFIG["FUTURES_CONTRACTS"].keys():
            # Override da file se disponibile
            if yf_sym in signal_override:
                signals[yf_sym] = float(signal_override[yf_sym])
                price = self.get_price_yf(yf_sym)
                if price > 0:
                    prices[yf_sym] = price
                continue

            # Calcola da yfinance
            ret_series = self.get_yf_returns(yf_sym, days=300)
            price = self.get_price_yf(yf_sym)

            if ret_series is None or len(ret_series) < max(lookbacks) // 2 or price <= 0:
                logger.warning(f"    {yf_sym}: dati insufficienti — skip")
                continue

            # Volatilità annualizzata (63 giorni)
            vol = ret_series.iloc[-63:].std() * np.sqrt(252) if len(ret_series) >= 63 else 0.25
            vol = max(vol, 0.05)  # floor

            # Segnale TSMOM blended
            blend = 0.0
            for lb, w in zip(lookbacks, lb_weights):
                if len(ret_series) < lb:
                    blend += w * 0  # Mancano dati per questo lookback
                    continue
                ret_lb = (1 + ret_series.iloc[-lb:]).prod() - 1
                blend += w * np.sign(ret_lb)

            # Vol-scaling
            vol_scale = min(target_vol_asset / vol, max_leverage)
            raw_signal = blend * vol_scale

            direction = "LONG  ✅" if raw_signal > 0 else ("SHORT 🔴" if raw_signal < 0 else "FLAT  ⬜")
            logger.info(f"    {yf_sym:6}: blend={blend:+.2f}  vol={vol:.0%}  "
                       f"scale={vol_scale:.1f}x  sig={raw_signal:+.2f}  ${price:.2f}  → {direction}")

            if abs(raw_signal) > 0.01:
                signals[yf_sym] = raw_signal
                prices[yf_sym] = price

        if not signals:
            logger.info("    Nessun segnale commodity — posizione flat")
            return

        # Normalizza per somma abs = 1 (allocazione equal-notional)
        total_abs = sum(abs(s) for s in signals.values()) or 1.0

        # Stima capitale minimo richiesto per il contratto più economico
        min_notional = float("inf")
        for yf_sym in signals:
            spec = CONFIG["FUTURES_CONTRACTS"].get(yf_sym, {})
            price = prices.get(yf_sym, 0)
            if price <= 0:
                continue
            if self.factory.use_micro and spec.get("micro"):
                mult = spec.get("micro_multiplier", spec.get("multiplier", 1))
            else:
                mult = spec.get("multiplier", 1)
            notional = price * mult
            min_notional = min(min_notional, notional)

        if min_notional != float("inf") and min_notional > budget_usd * 0.5:
            logger.warning(
                f"\n    ⚠️  CAPITALE INSUFFICIENTE per futures:"
            )
            logger.warning(
                f"       Contratto più economico: ${min_notional:,.0f}"
            )
            logger.warning(
                f"       Budget allocato: ${budget_usd:,.0f}"
            )
            logger.info(
                f"\n    💡 Opzioni:"
            )
            logger.info(
                f"       1. Aumenta equity a ≥ ${int(min_notional * 2 / 0.122):,} "
                f"(per avere budget commodity ≥ ${int(min_notional * 2):,})"
            )
            logger.info(
                f"       2. Usa ETF commodity come proxy: PDBC, DJP, BCI, COMB"
            )
            logger.info(
                f"          (GC→GLD, CL→USO, già parzialmente coperto da DBC in Macro)"
            )
            return

        # Piazza ordini IB futures con segnali calcolati da yfinance
        orders_placed = 0
        for yf_sym, signal in signals.items():
            result = self.factory.make_future(yf_sym)
            if result is None:
                continue
            contract, multiplier = result

            weight = abs(signal) / total_abs
            position_usd = budget_usd * weight
            price = prices.get(yf_sym, 0)
            if price <= 0:
                logger.warning(f"    Prezzo non disponibile: {yf_sym} — skip")
                continue

            n_contracts = self.sizer.compute_futures_contracts(
                position_usd, price, multiplier, signal
            )

            if n_contracts == 0:
                continue

            spec = CONFIG["FUTURES_CONTRACTS"][yf_sym]
            sym_display = (spec["micro"] if self.factory.use_micro and spec["micro"]
                          else spec["symbol"])
            self.place_order(contract, n_contracts, price, f"{sym_display}(fut)")
            orders_placed += 1

        if orders_placed == 0:
            logger.warning("    Nessun ordine commodity eseguito (posizioni troppo piccole)")

    # -----------------------------------------------------------------
    # MAIN EXECUTION
    # -----------------------------------------------------------------
    def run(self):
        logger.info("=" * 80)
        logger.info(f"TWS EXECUTOR — {self.mode.upper()} {'[DRY RUN]' if self.dry_run else '[LIVE]'}")
        logger.info(f"Equity: ${self.equity:,.0f} | Porta: {self.port}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # 1. Connessione
        connected = self.connect()
        if connected:
            self.equity = self.get_account_equity()

        # 1b. Carica ordini aperti (anti-duplicato)
        logger.info("\n[0/3] CONTROLLO ORDINI APERTI (ANTI-DUPLICATO)")
        self.load_open_orders()

        # 2. Carica weights N_MinVar
        logger.info("\n[1/3] CARICAMENTO SEGNALI E PESI")
        weights, legs = self.loader.load_all()

        # 3. Current positions
        logger.info("\n[2/3] POSIZIONI ATTUALI")
        current_positions = self.get_current_positions()
        if current_positions:
            for sym, qty in current_positions.items():
                logger.info(f"  {sym}: {qty:+.0f}")
        else:
            logger.info("  Nessuna posizione aperta (o offline)")

        # 4. Esecuzione per gamba
        logger.info("\n[3/3] ESECUZIONE ORDINI")

        if "Bio" in weights:
            bio_budget = weights["Bio"] * self.equity
            self.execute_equity_leg("Biotech MR", CONFIG["BIO_DIR"], bio_budget)

        if "Tech" in weights:
            tech_budget = weights["Tech"] * self.equity
            self.execute_equity_leg("Tech Momentum", CONFIG["TECH_DIR"], tech_budget)

        if "Macro" in weights:
            macro_budget = weights["Macro"] * self.equity
            self.execute_macro_leg(macro_budget)

        if "Commodity" in weights:
            comm_budget = weights["Commodity"] * self.equity
            self.execute_commodity_leg(comm_budget)

        # 5. Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"ESECUZIONE COMPLETATA — {datetime.now().strftime('%H:%M:%S')}")
        if self.dry_run:
            logger.info("⚠️  DRY RUN: nessun ordine reale è stato piazzato")
        logger.info(f"{'='*80}")

        self.disconnect()


# =================================================================================
# SIGNAL EXPORTER — genera i CSV di segnali correnti dalla pipeline
# =================================================================================
def export_current_signals():
    """
    Esporta i segnali correnti dal pipeline in formato leggibile dall'executor.
    Runnare dopo ogni walk-forward per aggiornare i segnali.

    TODO: integrare direttamente con stat_arb_biotech.py e stat_arb_macro.py
    per estrarre i pesi dell'ultimo rebalancing invece di usare un file separato.
    """
    logger.info("Esportazione segnali correnti...")
    # Per ora questo è un placeholder — da completare con l'integrazione
    # diretta ai pipeline
    logger.info("⚠️  export_current_signals() non ancora implementato.")
    logger.info("   Crea manualmente 'current_signals.csv' in ciascuna cartella pipeline.")
    logger.info("   Formato: ticker,weight (es: AMGN,+0.15)")


# =================================================================================
# CLI
# =================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TWS Executor — esegue il mega-portfolio via IB API"
    )
    parser.add_argument("--mode",     choices=["paper", "live"], default="paper",
                        help="Paper o live trading (default: paper)")
    parser.add_argument("--equity",   type=float, default=10_000,
                        help="Capitale totale in USD (default: 10000)")
    parser.add_argument("--dry-run",  action="store_true", default=False,
                        help="Calcola ordini senza inviarli")
    parser.add_argument("--use-tws",  action="store_true", default=False,
                        help="Connette a TWS invece di IB Gateway")
    parser.add_argument("--export-signals", action="store_true", default=False,
                        help="Esporta segnali correnti e termina")
    args = parser.parse_args()

    if args.export_signals:
        export_current_signals()
        sys.exit(0)

    # Safety check: non permettere live senza conferma esplicita
    if args.mode == "live" and not args.dry_run:
        confirm = input("\n⚠️  ATTENZIONE: stai per tradare LIVE con denaro reale.\n"
                       f"   Equity: ${args.equity:,.0f}\n"
                       "   Digita 'CONFERMO' per procedere: ")
        if confirm != "CONFERMO":
            print("Operazione annullata.")
            sys.exit(0)

    executor = TWSExecutor(
        mode=args.mode,
        equity=args.equity,
        dry_run=args.dry_run,
        use_tws=args.use_tws,
    )
    executor.run()
