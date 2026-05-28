"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AVELLANEDA-STOIKOV MARKET MAKER — Anti-Adverse Selection Edition           ║
║                                                                            ║
║  Framework : Hummingbot Script V2 (ScriptStrategyBase)                     ║
║  Autore    : Sviluppatore Quantitativo Senior                              ║
║  Versione  : 1.0.0                                                         ║
║                                                                            ║
║  Logiche difensive implementate:                                           ║
║    1. AS Vanilla  – Reservation Price + Optimal Spread (γ, σ, κ)           ║
║    2. Trend Filter – EMA momentum con skewing direzionale                  ║
║    3. OBI          – Order Book Imbalance (protezione dump/pump)           ║
║    4. Kill-Switch  – Volatilità estrema → pausa totale                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

ISTRUZIONI DI DEPLOY:
  1. Copia questo file in: hummingbot/scripts/
  2. Avvia Hummingbot → digita: start --script as_anti_adverse_selection.py
  3. I parametri sono definiti come costanti di classe in fondo alla sezione
     "PARAMETRI CONFIGURABILI" e possono essere modificati senza toccare la logica.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import logging                              # Logging standard di Python
import time                                 # Per i timestamp del kill-switch
import math                                 # Funzioni matematiche (log, sqrt)
from decimal import Decimal                 # Aritmetica a precisione arbitraria
from typing import Dict, List, Optional     # Type hints
from collections import deque               # Buffer circolare per i prezzi storici

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
# ↑ Classe base per tutti gli script V2 di Hummingbot.
#   Fornisce: on_tick(), buy(), sell(), cancel(), connectors, active_orders, ecc.

from hummingbot.core.data_type.common import OrderType, TradeType
# ↑ OrderType.LIMIT / MARKET, TradeType.BUY / SELL – enumerazioni per ordini

# Logger dedicato a questa strategia
logger = logging.getLogger(__name__)


class AvellanedaStoikovAntiAS(ScriptStrategyBase):
    """
    Market Maker basato sul modello di Avellaneda-Stoikov con quattro livelli
    di difesa contro la Adverse Selection (flusso tossico).

    Architettura del ciclo on_tick():
    ┌─────────────┐
    │  on_tick()   │ ← chiamato ogni `tick_size` secondi (default 1s)
    └──────┬──────┘
           │
           ▼
    ┌──────────────────┐
    │ 1) Aggiorna dati │  mid_price, volatilità, EMA, OBI
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ 2) Kill-Switch?  │──── SÌ → cancel_all + pausa 5 min
    └──────┬───────────┘
           │ NO
           ▼
    ┌──────────────────┐
    │ 3) Calcola AS    │  reservation_price, optimal_spread
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ 4) Trend Filter  │  Applica skewing direzionale (EMA)
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ 5) OBI Filter    │  Allarga spread su lato esposto
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ 6) Piazza ordini │  bid_price, ask_price → buy()/sell()
    └──────────────────┘
    """

    # ═════════════════════════════════════════════════════════════════════════
    #  PARAMETRI CONFIGURABILI
    #  Modifica questi valori per adattare la strategia al tuo mercato.
    # ═════════════════════════════════════════════════════════════════════════

    # ── Connessione all'exchange ──
    exchange: str = "binance_perpetual"      # Nome del connector Hummingbot
    trading_pair: str = "ETH-USDT"           # Coppia di trading
    markets = {"binance_perpetual": {"ETH-USDT"}}
    # ↑ Dizionario obbligatorio: dice a Hummingbot quali mercati sottoscrivere.
    #   Formato: {connector_name: {set_of_trading_pairs}}

    # ── Dimensione ordini ──
    order_amount: Decimal = Decimal("0.01")  # Quantità base per ordine (in ETH)
    max_order_amount: Decimal = Decimal("0.05")  # Quantità massima per ordine

    # ── Parametri Avellaneda-Stoikov ──
    gamma: float = 0.1
    # ↑ γ (gamma) — Coefficiente di avversione al rischio di inventario.
    #   Valori più alti → spread più ampio + reservation price più aggressivo
    #   nel riportare l'inventario a zero.
    #   Range tipico crypto: 0.01 (aggressivo) → 1.0 (molto conservativo)

    kappa: float = 1.5
    # ↑ κ (kappa) — Parametro di liquidità dell'order book.
    #   Rappresenta la "densità" del book. Valori più alti = book più denso
    #   = spread ottimale più stretto.
    #   Può essere stimato empiricamente dal book reale o impostato manualmente.
    #   Range tipico: 0.5 → 5.0

    session_duration: float = 28800.0
    # ↑ T — Durata della sessione di trading in secondi (default: 8 ore = 28800s).
    #   Nel modello AS, (T - t) rappresenta il tempo rimanente nella sessione.
    #   Quando T - t → 0, lo spread si comprime e il reservation price converge
    #   al mid-price, forzando il "flattening" dell'inventario.

    inventory_target: float = 0.0
    # ↑ q_target — Inventario obiettivo (in unità di base, es. ETH).
    #   0.0 = market neutral. Il modello AS aggiusta il reservation price
    #   per riportare l'inventario verso questo target.

    # ── Parametri Volatilità ──
    vol_window: int = 60
    # ↑ Finestra (in tick/secondi) per il calcolo della volatilità realizzata.
    #   60 = usa gli ultimi 60 mid-price per calcolare σ.

    vol_min_samples: int = 20
    # ↑ Numero minimo di campioni prima di iniziare a quotare.
    #   Evita di piazzare ordini con una stima σ instabile.

    # ── Parametri EMA (Trend Filter) ──
    ema_period: int = 60
    # ↑ Periodo della EMA in tick (≈ secondi se tick_size = 1s).
    #   EMA a 60s → filtro di momentum a brevissimo termine.
    #   Se il prezzo è sotto la EMA → trend ribassista → proteggi il bid.

    ema_crash_threshold: float = -0.002
    # ↑ Soglia di "crash" come variazione percentuale (mid_price / EMA - 1).
    #   Se il deviation < -0.2% → consideriamo un crollo improvviso.
    #   Il bot rimuove TUTTI i bid e piazza solo ask per scaricare inventario.
    #   Nota: questo valore è NEGATIVO. Es. -0.002 = -0.2%

    ema_skew_factor: float = 2.0
    # ↑ Moltiplicatore per lo skewing dello spread basato sulla deviazione
    #   del prezzo rispetto alla EMA.
    #   Più alto = spread bid/ask più asimmetrico in presenza di trend.

    # ── Parametri Order Book Imbalance (OBI) ──
    obi_levels: int = 10
    # ↑ Numero di livelli dell'order book da analizzare per l'OBI.

    obi_threshold: float = 0.80
    # ↑ Soglia critica di squilibrio. Se ask_volume / total_volume > 80%
    #   → pressione venditrice dominante → allarga lo spread bid.
    #   Simmetricamente: se bid_volume / total_volume > 80% → allarga ask.

    obi_spread_multiplier: float = 2.5
    # ↑ Moltiplicatore dello spread quando l'OBI supera la soglia.
    #   Es. 2.5 = lo spread sul lato esposto diventa 2.5x il normale.

    # ── Parametri Kill-Switch ──
    kill_vol_multiplier: float = 3.0
    # ↑ Se la volatilità istantanea (ultimi 10 tick) supera
    #   kill_vol_multiplier × volatilità_media → KILL SWITCH ATTIVATO.
    #   3.0 = la vol breve deve essere 3x la vol media per triggerare.

    kill_vol_short_window: int = 10
    # ↑ Finestra brevissima (tick) per la volatilità istantanea del kill-switch.

    kill_pause_duration: float = 300.0
    # ↑ Durata della pausa in secondi dopo l'attivazione del kill-switch.
    #   300 = 5 minuti. Durante la pausa, nessun ordine viene piazzato.

    # ── Spread Minimo ──
    min_spread_bps: float = 5.0
    # ↑ Spread minimo in basis points (1 bps = 0.01%).
    #   Garantisce che anche con volatilità bassa, non quoteremo mai
    #   con uno spread inferiore a questo valore (protezione costi di trading).

    # ═════════════════════════════════════════════════════════════════════════
    #  STATO INTERNO (non modificare direttamente)
    # ═════════════════════════════════════════════════════════════════════════

    def __init__(self, connectors: Dict):
        """
        Costruttore: inizializza lo stato interno della strategia.
        Viene chiamato una sola volta all'avvio dello script.

        Args:
            connectors: Dizionario {nome_connector: oggetto_connector}
                        fornito automaticamente da Hummingbot.
        """
        super().__init__(connectors)

        # Buffer circolare per la serie storica dei mid-price.
        # Dimensione = max(vol_window, ema_period) + margine di sicurezza.
        self._price_buffer: deque = deque(maxlen=max(self.vol_window, self.ema_period) + 100)

        # Valore corrente della EMA (Exponential Moving Average).
        self._ema: Optional[float] = None

        # Timestamp di inizio sessione (per calcolare T - t nel modello AS).
        self._session_start: float = time.time()

        # Stato del kill-switch.
        self._kill_switch_active: bool = False
        # Timestamp di attivazione del kill-switch (per calcolare la durata della pausa).
        self._kill_switch_timestamp: float = 0.0

        # Contatore dei tick (usato per il logging periodico).
        self._tick_count: int = 0

        # Inventario corrente in unità di base (es. ETH).
        # Viene aggiornato ad ogni tick leggendo il saldo dal connector.
        self._current_inventory: float = 0.0

    # ═════════════════════════════════════════════════════════════════════════
    #  METODO PRINCIPALE: on_tick()
    # ═════════════════════════════════════════════════════════════════════════

    def on_tick(self):
        """
        Metodo principale chiamato ad ogni tick dal framework Hummingbot.
        Frequenza: ogni ~1 secondo (configurabile via tick_size nella config).

        Questo metodo orchestra l'intera pipeline della strategia:
        aggiornamento dati → controllo kill-switch → calcolo AS →
        applicazione filtri difensivi → piazzamento ordini.
        """
        self._tick_count += 1

        # ─── STEP 0: Riferimento al connector ───
        # Recuperiamo l'oggetto connector per interagire con l'exchange.
        connector = self.connectors[self.exchange]

        # ─── STEP 1: Lettura del mid-price corrente ───
        # get_mid_price() restituisce (best_bid + best_ask) / 2
        mid_price = connector.get_mid_price(self.trading_pair)
        if mid_price is None or mid_price <= Decimal("0"):
            # Se il prezzo non è disponibile (es. book vuoto), skippa il tick.
            logger.warning("[AS] Mid-price non disponibile, skip tick.")
            return

        mid_price_float = float(mid_price)

        # ─── STEP 2: Aggiorna il buffer dei prezzi ───
        self._price_buffer.append(mid_price_float)

        # ─── STEP 3: Aggiorna la EMA (Exponential Moving Average) ───
        self._update_ema(mid_price_float)

        # ─── STEP 4: Aggiorna l'inventario corrente ───
        self._update_inventory(connector)

        # ─── STEP 5: Controlla se abbiamo abbastanza dati ───
        if len(self._price_buffer) < self.vol_min_samples:
            # Non abbiamo abbastanza campioni per stimare la volatilità.
            # Loggiamo e aspettiamo che il buffer si riempia.
            logger.info(
                f"[AS] Raccolta dati in corso: {len(self._price_buffer)}/{self.vol_min_samples} campioni. "
                f"Mid-price: {mid_price_float:.4f}"
            )
            return

        # ─── STEP 6: Calcola la volatilità realizzata ───
        sigma = self._calc_realized_volatility(window=self.vol_window)
        if sigma is None or sigma <= 0:
            logger.warning("[AS] Volatilità calcolata <= 0, skip tick.")
            return

        # ═════════════════════════════════════════════════════════════════════
        #  KILL-SWITCH: Protezione da volatilità estrema
        # ═════════════════════════════════════════════════════════════════════

        # ─── STEP 7: Controlla il Kill-Switch ───
        if self._check_kill_switch(sigma):
            # Il kill-switch è attivo: tutti gli ordini sono stati cancellati.
            # Non facciamo nulla fino alla fine della pausa.
            return

        # ═════════════════════════════════════════════════════════════════════
        #  MODELLO AVELLANEDA-STOIKOV
        # ═════════════════════════════════════════════════════════════════════

        # ─── STEP 8: Calcola il tempo residuo nella sessione ───
        elapsed = time.time() - self._session_start
        # T - t: tempo rimanente. Clampato a minimo 1 secondo per evitare
        # divisione per zero o comportamento erratico a fine sessione.
        time_remaining = max(self.session_duration - elapsed, 1.0)

        # ─── STEP 9: Calcola l'inventario relativo ───
        # q = inventory_corrente - inventory_target
        # Se q > 0: siamo "lunghi" → il reservation price deve scendere
        # Se q < 0: siamo "corti" → il reservation price deve salire
        q = self._current_inventory - self.inventory_target

        # ─── STEP 10: Calcola il Reservation Price ───
        # Formula AS: r = s - q * γ * σ² * (T - t)
        #
        # Dove:
        #   s     = mid-price corrente
        #   q     = inventario netto (rispetto al target)
        #   γ     = avversione al rischio (self.gamma)
        #   σ²    = varianza del prezzo (sigma al quadrato)
        #   (T-t) = tempo residuo nella sessione
        #
        # Intuizione: se siamo long (q > 0), il reservation price si abbassa,
        # rendendo il nostro ask più aggressivo (prezzo più basso = più facile vendere)
        # e il bid meno aggressivo (prezzo più basso = compriamo meno).
        reservation_price = mid_price_float - q * self.gamma * (sigma ** 2) * time_remaining

        # ─── STEP 11: Calcola lo Spread Ottimale ───
        # Formula AS: δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/κ)
        #
        # Primo termine (γ * σ² * (T-t)):
        #   Componente di "rischio puro". Cresce con la volatilità e il tempo residuo.
        #   Rappresenta il premio per il rischio di detenere inventario.
        #
        # Secondo termine ((2/γ) * ln(1 + γ/κ)):
        #   Componente di "liquidità del book". Dipende dalla densità del book (κ).
        #   Book più denso → κ alto → termine piccolo → spread più stretto.
        #   Book più rado → κ basso → termine grande → spread più ampio.
        optimal_spread = (
            self.gamma * (sigma ** 2) * time_remaining
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        )

        # ─── STEP 12: Applica lo spread minimo ───
        # Converte il min_spread_bps in valore assoluto e garantisce che
        # lo spread non sia mai inferiore.
        min_spread_abs = mid_price_float * (self.min_spread_bps / 10000.0)
        optimal_spread = max(optimal_spread, min_spread_abs)

        # ─── STEP 13: Calcola bid e ask "grezzi" ───
        # Centrati sul reservation price, non sul mid-price.
        # Questo è il cuore del modello AS: lo skewing dell'inventario
        # è già incorporato nel reservation_price.
        raw_bid = reservation_price - optimal_spread / 2.0
        raw_ask = reservation_price + optimal_spread / 2.0

        # ═════════════════════════════════════════════════════════════════════
        #  FILTRO 1: TREND FILTER (EMA Skewing)
        # ═════════════════════════════════════════════════════════════════════

        # ─── STEP 14: Applica lo skewing basato sulla EMA ───
        # Calcoliamo la deviazione percentuale del prezzo rispetto alla EMA:
        #   deviation = (mid_price / EMA) - 1
        #
        # Se deviation < 0 → prezzo sotto la EMA → trend ribassista
        # Se deviation > 0 → prezzo sopra la EMA → trend rialzista
        bid_price, ask_price, skip_bid, skip_ask = self._apply_trend_filter(
            raw_bid, raw_ask, mid_price_float, optimal_spread
        )

        # ═════════════════════════════════════════════════════════════════════
        #  FILTRO 2: ORDER BOOK IMBALANCE (OBI)
        # ═════════════════════════════════════════════════════════════════════

        # ─── STEP 15: Leggi l'OBI e aggiusta gli spread ───
        bid_price, ask_price = self._apply_obi_filter(
            connector, bid_price, ask_price, mid_price_float, optimal_spread
        )

        # ═════════════════════════════════════════════════════════════════════
        #  PIAZZAMENTO ORDINI
        # ═════════════════════════════════════════════════════════════════════

        # ─── STEP 16: Cancella tutti gli ordini esistenti ───
        # Prima di piazzare nuovi ordini, rimuoviamo quelli vecchi.
        # Questo approccio "cancel-and-replace" è standard nel market making
        # per garantire che le quote riflettano sempre le condizioni correnti.
        self._cancel_all_active_orders()

        # ─── STEP 17: Sanity check sui prezzi ───
        # Verifica che bid < mid < ask (ordini non "incrociati").
        if bid_price >= mid_price_float and not skip_bid:
            logger.warning(
                f"[AS] Bid ({bid_price:.4f}) >= mid ({mid_price_float:.4f}), skip bid."
            )
            skip_bid = True

        if ask_price <= mid_price_float and not skip_ask:
            logger.warning(
                f"[AS] Ask ({ask_price:.4f}) <= mid ({mid_price_float:.4f}), skip ask."
            )
            skip_ask = True

        # ─── STEP 18: Piazza l'ordine BID (acquisto) ───
        if not skip_bid and bid_price > 0:
            # Converte il prezzo a Decimal per la compatibilità con Hummingbot.
            bid_decimal = Decimal(str(round(bid_price, 8)))
            self.buy(
                connector_name=self.exchange,        # Nome del connector
                trading_pair=self.trading_pair,      # Coppia di trading
                amount=self.order_amount,            # Quantità
                order_type=OrderType.LIMIT_MAKER,    # LIMIT_MAKER = post-only
                price=bid_decimal                    # Prezzo bid
            )
            # ↑ LIMIT_MAKER garantisce che l'ordine venga piazzato come maker
            #   (e mai come taker), evitando fee più alte e fill indesiderati.

        # ─── STEP 19: Piazza l'ordine ASK (vendita) ───
        if not skip_ask and ask_price > 0:
            ask_decimal = Decimal(str(round(ask_price, 8)))
            self.sell(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=self.order_amount,
                order_type=OrderType.LIMIT_MAKER,
                price=ask_decimal
            )

        # ─── STEP 20: Logging periodico (ogni 30 tick ≈ 30 secondi) ───
        if self._tick_count % 30 == 0:
            self._log_status(
                mid_price_float, reservation_price, sigma, optimal_spread,
                bid_price, ask_price, skip_bid, skip_ask, q, time_remaining
            )

    # ═════════════════════════════════════════════════════════════════════════
    #  METODI PRIVATI: CALCOLI E FILTRI
    # ═════════════════════════════════════════════════════════════════════════

    def _update_ema(self, price: float) -> None:
        """
        Aggiorna la EMA (Exponential Moving Average) con il nuovo prezzo.

        Formula EMA:
            EMA_t = α * price + (1 - α) * EMA_{t-1}

        dove α (alfa) = 2 / (period + 1)

        La EMA è un filtro passa-basso esponenziale: dà più peso ai prezzi
        recenti rispetto alla SMA (media semplice), permettendo di reagire
        più velocemente ai cambiamenti di trend.

        Args:
            price: Mid-price corrente (float).
        """
        # Calcola il coefficiente di smoothing α
        alpha = 2.0 / (self.ema_period + 1)

        if self._ema is None:
            # Prima osservazione: inizializza la EMA con il prezzo corrente.
            self._ema = price
        else:
            # Aggiornamento incrementale standard della EMA.
            self._ema = alpha * price + (1.0 - alpha) * self._ema

    def _update_inventory(self, connector) -> None:
        """
        Legge il saldo corrente dell'asset base dal connector e aggiorna
        l'inventario interno.

        Per "ETH-USDT", l'asset base è "ETH".
        Usa get_balance() del connector per ottenere il saldo totale (inclusi
        gli ordini aperti) dell'asset base.

        Args:
            connector: Oggetto connector di Hummingbot.
        """
        # Estrae il nome dell'asset base dalla coppia (es. "ETH" da "ETH-USDT")
        base_asset = self.trading_pair.split("-")[0]

        # get_balance() restituisce il saldo TOTALE dell'asset (disponibile + in ordini).
        # Per futures/perpetual, restituisce la posizione netta.
        balance = connector.get_balance(base_asset)
        self._current_inventory = float(balance) if balance else 0.0

    def _calc_realized_volatility(self, window: int) -> Optional[float]:
        """
        Calcola la volatilità realizzata (σ) come deviazione standard dei
        log-returns sulla finestra specificata.

        Formula:
            r_i = ln(P_i / P_{i-1})           (log-return)
            σ   = std(r_1, r_2, ..., r_n)     (deviazione standard campionaria)

        I log-returns sono preferiti ai returns semplici perché:
        1. Sono additivi nel tempo (proprietà fondamentale per la teoria AS)
        2. Sono approssimativamente normali per variazioni piccole
        3. Sono simmetrici rispetto a rialzi e ribassi

        Args:
            window: Numero di prezzi da considerare per il calcolo.

        Returns:
            Volatilità realizzata (float), o None se dati insufficienti.
        """
        # Prendiamo gli ultimi `window` prezzi dal buffer.
        prices = list(self._price_buffer)[-window:]

        if len(prices) < 2:
            return None

        # Calcola i log-returns: r_i = ln(P_i / P_{i-1})
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:  # Protezione da prezzi zero/negativi
                log_returns.append(math.log(prices[i] / prices[i - 1]))

        if len(log_returns) < 2:
            return None

        # Calcola media e varianza dei log-returns
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        # ↑ Varianza campionaria (divisione per n-1, correzione di Bessel)

        # σ = sqrt(varianza)
        sigma = math.sqrt(variance)

        return sigma

    def _check_kill_switch(self, sigma_long: float) -> bool:
        """
        Kill-Switch: controlla se la volatilità a brevissimo termine è
        anomala rispetto alla volatilità media.

        Logica:
            σ_short = volatilità sugli ultimi `kill_vol_short_window` tick
            σ_long  = volatilità sugli ultimi `vol_window` tick

            Se σ_short > kill_vol_multiplier * σ_long:
                → ATTIVA IL KILL-SWITCH
                → Cancella TUTTI gli ordini
                → Metti in pausa per `kill_pause_duration` secondi

        Questa protezione è essenziale per i mercati crypto, dove eventi
        come liquidazioni a cascata, flash crash o news improvvise possono
        causare movimenti del 5-10% in pochi secondi, rendendo qualsiasi
        spread quotato inadeguato.

        Args:
            sigma_long: Volatilità calcolata sulla finestra lunga (vol_window).

        Returns:
            True se il kill-switch è attivo (e gli ordini sono stati cancellati),
            False altrimenti.
        """
        now = time.time()

        # ── CASO 1: Kill-switch già attivo → controlla se la pausa è finita ──
        if self._kill_switch_active:
            time_in_pause = now - self._kill_switch_timestamp
            if time_in_pause < self.kill_pause_duration:
                # Ancora in pausa. Loggiamo il tempo rimanente.
                remaining = self.kill_pause_duration - time_in_pause
                if self._tick_count % 30 == 0:
                    logger.warning(
                        f"[KILL-SWITCH] ⏸ Pausa attiva. "
                        f"Riavvio tra {remaining:.0f}s."
                    )
                return True
            else:
                # La pausa è finita. Riattiva il bot.
                self._kill_switch_active = False
                logger.info("[KILL-SWITCH] ✅ Pausa terminata. Ripresa operazioni.")
                # Resetta il timer della sessione AS per evitare artefatti
                self._session_start = time.time()
                return False

        # ── CASO 2: Controlla se bisogna attivare il kill-switch ──
        # Calcola la volatilità a brevissimo termine.
        sigma_short = self._calc_realized_volatility(window=self.kill_vol_short_window)

        if sigma_short is None:
            return False  # Non abbiamo abbastanza dati short-term

        # Confronta la volatilità breve con quella lunga.
        if sigma_long > 0 and sigma_short > self.kill_vol_multiplier * sigma_long:
            # ── ATTIVA IL KILL-SWITCH ──
            self._kill_switch_active = True
            self._kill_switch_timestamp = now

            # Cancella TUTTI gli ordini attivi immediatamente.
            self._cancel_all_active_orders()

            logger.critical(
                f"[KILL-SWITCH] 🚨 ATTIVATO! "
                f"σ_short={sigma_short:.6f} > "
                f"{self.kill_vol_multiplier:.1f} × σ_long={sigma_long:.6f} "
                f"(soglia={self.kill_vol_multiplier * sigma_long:.6f}). "
                f"Tutti gli ordini cancellati. Pausa di {self.kill_pause_duration:.0f}s."
            )
            return True

        return False

    def _apply_trend_filter(
        self,
        raw_bid: float,
        raw_ask: float,
        mid_price: float,
        optimal_spread: float
    ) -> tuple:
        """
        Filtro di Trend basato sulla EMA: applica uno skewing direzionale
        ai prezzi bid/ask in base alla deviazione del prezzo dalla EMA.

        Logica dettagliata:
        1. Calcola la deviazione: dev = (mid_price / EMA) - 1
           - dev < 0 → prezzo SOTTO la EMA → downtrend
           - dev > 0 → prezzo SOPRA la EMA → uptrend

        2. Se dev < ema_crash_threshold (es. -0.2%):
           → CRASH DETECT: rimuovi TUTTI i bid, piazza solo ask.
           Razionale: durante un crollo, ogni acquisto è adverse selection.
           Il market maker deve SOLO scaricare inventario.

        3. Altrimenti: applica un "skewing" continuo.
           - In downtrend: allarga il bid, stringi l'ask
           - In uptrend: allarga l'ask, stringi il bid
           Formula:
             skew = ema_skew_factor * dev * optimal_spread
             bid_price = raw_bid - |skew|  (se dev < 0, allarga il bid)
             ask_price = raw_ask - |skew|  (se dev < 0, ask più aggressivo)

        Args:
            raw_bid: Prezzo bid calcolato dal modello AS.
            raw_ask: Prezzo ask calcolato dal modello AS.
            mid_price: Mid-price corrente.
            optimal_spread: Spread ottimale AS.

        Returns:
            Tupla (bid_price, ask_price, skip_bid: bool, skip_ask: bool).
        """
        skip_bid = False
        skip_ask = False
        bid_price = raw_bid
        ask_price = raw_ask

        if self._ema is None or self._ema <= 0:
            # EMA non ancora inizializzata → nessun filtro.
            return bid_price, ask_price, skip_bid, skip_ask

        # Calcola la deviazione percentuale del prezzo rispetto alla EMA.
        # deviation = (P / EMA) - 1
        deviation = (mid_price / self._ema) - 1.0

        # ── CASO CRASH: deviazione sotto la soglia critica ──
        if deviation < self.ema_crash_threshold:
            # Il prezzo è in caduta libera rispetto alla EMA.
            # Strategia di sopravvivenza:
            #   - NESSUN BID: non vogliamo comprare durante un crash
            #   - SOLO ASK: vendiamo tutto l'inventario possibile
            skip_bid = True

            # Rendi l'ask più aggressivo (più vicino al mid-price) per
            # aumentare la probabilità di vendere rapidamente.
            ask_price = mid_price + optimal_spread * 0.25
            # ↑ Ask a solo il 25% dello spread → molto aggressivo in vendita

            logger.warning(
                f"[TREND] 🔴 CRASH DETECTATO! "
                f"Deviation={deviation:.4%} < soglia={self.ema_crash_threshold:.4%}. "
                f"Bid DISABILITATO, Ask aggressivo a {ask_price:.4f}."
            )
            return bid_price, ask_price, skip_bid, skip_ask

        # ── CASO NORMALE: skewing continuo ──
        # Lo skew è proporzionale alla deviazione dalla EMA.
        # Positivo → prezzo sopra EMA → trend up → allarga ask, stringi bid
        # Negativo → prezzo sotto EMA → trend down → allarga bid, stringi ask
        skew = self.ema_skew_factor * deviation * optimal_spread

        # Applica lo skew al bid e all'ask.
        # L'effetto è OPPOSTO per bid e ask:
        #   - skew > 0 (uptrend): bid_price SALE (più aggressivo in acquisto),
        #                          ask_price SALE (meno aggressivo in vendita)
        #     → Acchiappa il momentum rialzista comprando + facilmente
        #   - skew < 0 (downtrend): bid_price SCENDE (meno aggressivo in acquisto),
        #                            ask_price SCENDE (più aggressivo in vendita)
        #     → Protegge dal downside allontanando i bid
        bid_price = raw_bid + skew
        ask_price = raw_ask + skew

        return bid_price, ask_price, skip_bid, skip_ask

    def _apply_obi_filter(
        self,
        connector,
        bid_price: float,
        ask_price: float,
        mid_price: float,
        optimal_spread: float
    ) -> tuple:
        """
        Filtro Order Book Imbalance (OBI): analizza i primi N livelli del
        book degli ordini per rilevare squilibri di liquidità.

        Logica:
            1. Leggi i primi `obi_levels` livelli di bid e ask.
            2. Calcola:
               bid_volume = Σ volume(bid_i) per i = 1..N
               ask_volume = Σ volume(ask_i) per i = 1..N
               total_volume = bid_volume + ask_volume
            3. Calcola l'imbalance ratio:
               ask_ratio = ask_volume / total_volume
               bid_ratio = bid_volume / total_volume

            4. Se ask_ratio > obi_threshold (es. 80%):
               → Troppa pressione in vendita → dump imminente?
               → ALLARGA lo spread bid (compra a prezzo più basso)
               → Protegge il market maker da acquisti che saranno immediatamente
                 in perdita a causa della pressione venditrice.

            5. Simmetricamente: se bid_ratio > obi_threshold:
               → Troppa pressione in acquisto → pump imminente?
               → ALLARGA lo spread ask (vendi a prezzo più alto)
               → Protegge da vendite allo scoperto durante un pump.

        Args:
            connector: Oggetto connector Hummingbot.
            bid_price: Prezzo bid corrente (post-trend filter).
            ask_price: Prezzo ask corrente (post-trend filter).
            mid_price: Mid-price corrente.
            optimal_spread: Spread ottimale calcolato dal modello AS.

        Returns:
            Tupla (bid_price_adjusted, ask_price_adjusted).
        """
        try:
            # Recupera l'order book dal connector.
            order_book = connector.get_order_book(self.trading_pair)

            if order_book is None:
                return bid_price, ask_price

            # ── Estrai i primi N livelli di bid e ask ──
            # snapshot() o iterate() dipendono dalla versione di Hummingbot.
            # Usiamo un approccio robusto con try/except.

            # I bid sono ordinati dal più alto al più basso (best bid first).
            # Gli ask sono ordinati dal più basso al più alto (best ask first).
            bid_entries = list(order_book.bid_entries())[:self.obi_levels]
            ask_entries = list(order_book.ask_entries())[:self.obi_levels]

            if not bid_entries or not ask_entries:
                return bid_price, ask_price

            # ── Calcola i volumi aggregati ──
            # Ogni entry ha .price e .amount (o .quantity a seconda della versione).
            bid_volume = sum(float(entry.amount) for entry in bid_entries)
            ask_volume = sum(float(entry.amount) for entry in ask_entries)

            total_volume = bid_volume + ask_volume
            if total_volume <= 0:
                return bid_price, ask_price

            # ── Calcola i ratio ──
            ask_ratio = ask_volume / total_volume  # % di volume in vendita
            bid_ratio = bid_volume / total_volume  # % di volume in acquisto

            # ── Applica protezione anti-dump ──
            if ask_ratio > self.obi_threshold:
                # Pressione venditrice dominante → allarga il bid
                # Il market maker non vuole comprare davanti a un muro di vendite.
                bid_spread = mid_price - bid_price  # Spread corrente lato bid
                bid_price = mid_price - bid_spread * self.obi_spread_multiplier
                # ↑ Lo spread bid viene moltiplicato → bid più lontano dal mid

                logger.info(
                    f"[OBI] ⚠️ Ask-heavy: ask_ratio={ask_ratio:.1%} > "
                    f"soglia={self.obi_threshold:.1%}. "
                    f"Bid allargato a {bid_price:.4f} "
                    f"(x{self.obi_spread_multiplier:.1f})"
                )

            # ── Applica protezione anti-pump ──
            if bid_ratio > self.obi_threshold:
                # Pressione compratrice dominante → allarga l'ask
                ask_spread = ask_price - mid_price  # Spread corrente lato ask
                ask_price = mid_price + ask_spread * self.obi_spread_multiplier
                # ↑ Lo spread ask viene moltiplicato → ask più lontano dal mid

                logger.info(
                    f"[OBI] ⚠️ Bid-heavy: bid_ratio={bid_ratio:.1%} > "
                    f"soglia={self.obi_threshold:.1%}. "
                    f"Ask allargato a {ask_price:.4f} "
                    f"(x{self.obi_spread_multiplier:.1f})"
                )

        except Exception as e:
            # Se l'order book non è accessibile (es. exchange down), logga l'errore
            # ma non interrompere la strategia.
            logger.error(f"[OBI] Errore lettura order book: {e}")

        return bid_price, ask_price

    def _cancel_all_active_orders(self) -> None:
        """
        Cancella tutti gli ordini attivi sui mercati gestiti da questa strategia.

        Itera su tutti i tracking order del framework e invia una richiesta
        di cancellazione per ciascuno. Hummingbot gestisce internamente
        la conferma asincrona della cancellazione dall'exchange.
        """
        # self.get_active_orders() restituisce tutti gli ordini attivi
        # piazzati da questa strategia su tutti i connector.
        for connector_name, orders in self.active_orders.items():
            for order in orders:
                self.cancel(
                    connector_name=connector_name,
                    trading_pair=order.trading_pair,
                    order_id=order.client_order_id
                )

    # ═════════════════════════════════════════════════════════════════════════
    #  CALLBACK DEGLI ORDINI
    #  Questi metodi vengono chiamati automaticamente da Hummingbot quando
    #  un ordine viene fillato, cancellato o fallisce.
    # ═════════════════════════════════════════════════════════════════════════

    def did_fill_order(self, event):
        """
        Callback invocato quando un ordine viene eseguito (fill).

        Usiamo questo callback per:
        1. Loggare l'esecuzione con tutti i dettagli
        2. Aggiornare immediatamente l'inventario (senza aspettare il prossimo tick)

        Il fill è il momento più critico per un market maker: è quando
        si materializza il rischio di adverse selection. Se veniamo fillati
        costantemente su un solo lato, stiamo subendo flusso tossico.

        Args:
            event: Oggetto BuyOrderCompletedEvent o SellOrderCompletedEvent.
        """
        # Determina il tipo di ordine dal nome dell'evento.
        trade_type = "BUY" if "Buy" in type(event).__name__ else "SELL"

        logger.info(
            f"[FILL] {'🟢' if trade_type == 'BUY' else '🔴'} "
            f"{trade_type} eseguito: "
            f"prezzo={event.price:.6f}, "
            f"quantità={event.base_asset_amount:.6f}, "
            f"pair={self.trading_pair}"
        )

    def did_cancel_order(self, event):
        """
        Callback invocato quando un ordine viene cancellato con successo.
        Utile per il debugging; in produzione si può disabilitare.

        Args:
            event: Oggetto OrderCancelledEvent.
        """
        pass  # Silenzioso in produzione per evitare log flood

    # ═════════════════════════════════════════════════════════════════════════
    #  LOGGING E DIAGNOSTICA
    # ═════════════════════════════════════════════════════════════════════════

    def _log_status(
        self,
        mid_price: float,
        reservation_price: float,
        sigma: float,
        optimal_spread: float,
        bid_price: float,
        ask_price: float,
        skip_bid: bool,
        skip_ask: bool,
        q: float,
        time_remaining: float
    ) -> None:
        """
        Logga lo stato completo della strategia per diagnostica.
        Chiamato ogni 30 tick (~30 secondi) per evitare log flood.

        Il log include tutte le variabili chiave del modello AS e lo stato
        di ogni filtro difensivo, permettendo il debugging in tempo reale.

        Args:
            mid_price: Mid-price corrente.
            reservation_price: Reservation price AS.
            sigma: Volatilità realizzata.
            optimal_spread: Spread ottimale AS.
            bid_price: Prezzo bid finale (post-filtri).
            ask_price: Prezzo ask finale (post-filtri).
            skip_bid: Se True, il bid è stato disabilitato.
            skip_ask: Se True, l'ask è stato disabilitato.
            q: Inventario netto (rispetto al target).
            time_remaining: Tempo residuo nella sessione.
        """
        # Calcola lo spread effettivo in basis points per un confronto rapido.
        effective_spread_bps = ((ask_price - bid_price) / mid_price) * 10000 if mid_price > 0 else 0.0

        # Calcola la deviazione EMA se disponibile.
        ema_dev = ((mid_price / self._ema) - 1.0) if self._ema and self._ema > 0 else 0.0

        logger.info(
            f"\n{'═' * 60}\n"
            f"  [AS Market Maker] Status Report\n"
            f"{'─' * 60}\n"
            f"  Mid-Price      : {mid_price:.4f}\n"
            f"  Reservation Pr.: {reservation_price:.4f} "
            f"(delta: {reservation_price - mid_price:+.4f})\n"
            f"  Volatilità (σ) : {sigma:.6f}\n"
            f"  Spread Ottimale: {optimal_spread:.4f}\n"
            f"  Bid Price      : {bid_price:.4f} {'[SKIP]' if skip_bid else ''}\n"
            f"  Ask Price      : {ask_price:.4f} {'[SKIP]' if skip_ask else ''}\n"
            f"  Spread Effett. : {effective_spread_bps:.1f} bps\n"
            f"  Inventario (q) : {q:+.6f} "
            f"(target: {self.inventory_target:.4f})\n"
            f"  EMA            : {self._ema:.4f if self._ema else 'N/A'} "
            f"(dev: {ema_dev:+.4%})\n"
            f"  Tempo Rimasto  : {time_remaining:.0f}s "
            f"({time_remaining / 3600:.1f}h)\n"
            f"  Kill-Switch    : {'🔴 ATTIVO' if self._kill_switch_active else '🟢 OK'}\n"
            f"{'═' * 60}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  METODO format_status(): visualizzazione nello Hummingbot UI
    # ═════════════════════════════════════════════════════════════════════════

    def format_status(self) -> str:
        """
        Genera una stringa di stato formattata per il comando `status`
        della CLI di Hummingbot.

        Questo metodo viene chiamato quando l'utente digita `status` nella
        console di Hummingbot e permette di visualizzare lo stato della
        strategia in formato leggibile.

        Returns:
            Stringa formattata con lo stato corrente.
        """
        lines = []
        lines.append("\n  ╔══════════════════════════════════════════╗")
        lines.append("  ║  AS Anti-Adverse Selection Market Maker  ║")
        lines.append("  ╚══════════════════════════════════════════╝\n")

        # ── Stato del mercato ──
        connector = self.connectors.get(self.exchange)
        if connector:
            mid_price = connector.get_mid_price(self.trading_pair)
            lines.append(f"  Exchange      : {self.exchange}")
            lines.append(f"  Trading Pair  : {self.trading_pair}")
            lines.append(f"  Mid-Price     : {mid_price}")

        # ── Stato dell'inventario ──
        lines.append(f"  Inventario    : {self._current_inventory:+.6f}")
        lines.append(f"  Target        : {self.inventory_target:.4f}")

        # ── Stato della EMA ──
        if self._ema:
            lines.append(f"  EMA ({self.ema_period}s)   : {self._ema:.4f}")

        # ── Stato della volatilita ──
        sigma = self._calc_realized_volatility(self.vol_window)
        if sigma:
            lines.append(f"  Volatilità    : {sigma:.6f}")

        # ── Stato del kill-switch ──
        if self._kill_switch_active:
            remaining = self.kill_pause_duration - (time.time() - self._kill_switch_timestamp)
            lines.append(f"  Kill-Switch   : 🔴 ATTIVO (riavvio tra {remaining:.0f}s)")
        else:
            lines.append(f"  Kill-Switch   : 🟢 Inattivo")

        # ── Ordini attivi ──
        active_count = sum(len(orders) for orders in self.active_orders.values())
        lines.append(f"  Ordini Attivi : {active_count}")

        # ── Campioni raccolti ──
        lines.append(f"  Campioni prezzo: {len(self._price_buffer)}/{self.vol_min_samples}")

        return "\n".join(lines)
