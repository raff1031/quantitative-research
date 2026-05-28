# Edge Scanner — Multi-Factor Stock Screener

Stock screening engine con **convergence detection** — combina segnali congressuali, insider, fondamentali, tecnici e relative strength per identificare setup ad alta probabilità.

## 100% Free Data Sources

| Source | Data | URL |
|--------|------|-----|
| **Senate Stock Watcher** | Senate trades (JSON) | `github.com/timothycarambat/senate-stock-watcher-data` |
| **House Stock Watcher** | House trades (JSON) | S3 bulk download via `housestockwatcher.com` |
| **OpenInsider** | SEC Form 4 insider trades | `openinsider.com` (scrape) |
| **Yahoo Finance** | Prezzi, fondamentali | `yfinance` (Python) |

Nessuna API key richiesta. Zero costi.

## Quick Start

```bash
pip install -r requirements.txt

# Screen default watchlist (15 stocks)
python run.py

# Screen tickers specifici
python run.py NVDA LMT PLTR TSLA

# Screen i 20 titoli più tradati dal Congresso
python run.py --top-congress
```

## Architettura

```
data_sources.py          <- Fetcher gratuiti (GitHub, S3, OpenInsider)
    |
screener.py              <- 5 analyzer (Fundamental, Technical, RS, Quality, Congress)
    |
run.py                   <- Pipeline runner + convergence output
```

### I 5 Fattori

| Fattore | Peso | Cosa misura |
|---------|------|-------------|
| **Fundamental** | 30% | Valuation (P/E, PEG, EV/EBITDA), Growth, Profitability, Health |
| **Technical** | 25% | Momentum (RSI, MACD), Trend (MA alignment, ADX), Volatility, Volume |
| **Relative Strength** | 20% | Mansfield RS vs sector benchmark + vs SPY |
| **Quality** | 15% | Piotroski F-Score (9 check) |
| **Congressional** | 10% | Net sentiment, trader quality, conviction signals |

### Convergence Framework

Il vero edge non e' un singolo fattore ma la **convergenza**:

- **5/5**: Congress BUY + Insider BUY + Tech uptrend + Fund strong + RS positive = Alta probabilita, size 2-3%
- **3/5**: Congress BUY + Fund strong + Quality high, ma Tech breakdown = Watchlist, aspetta tech confirmation
- **1/5**: Congress BUY, ma tutto il resto negativo = No trade, segnale rumoroso

## Prossimi Step

- Collegare Finnhub free tier (60 call/min) come source addizionale
- Aggiungere 13F institutional holdings (SEC EDGAR, trimestrale)
- Streamlit dashboard interattiva
- Backtest: "compra top-5 convergence, ribilancia mensile" vs SPY
- Alert system: notifica quando convergenza >= 4/5 su un titolo
- Integrazione TFT: congressional + insider come features nel modello
