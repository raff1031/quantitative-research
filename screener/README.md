# Multi-Factor Stock Screener

Stock screening engine che combina **5 fattori** per classificare e rankare azioni:

1. **Fundamental Analysis** (30%) — Valuation (P/E, PEG, EV/EBITDA), Growth (revenue/earnings), Profitability (margins, ROE), Financial Health (D/E, current ratio, FCF)
2. **Technical Analysis** (25%) — Momentum (RSI, MACD, Stochastic), Trend (MA alignment, ADX), Volatility (ATR, Bollinger), Volume (OBV, volume ratio)
3. **Relative Strength** (20%) — Mansfield RS vs sector benchmark (XLK, XLF, etc.) e vs SPY
4. **Quality Score** (15%) — Piotroski F-Score inspired (9 check su profitability, leverage, liquidity, earnings quality)
5. **Congressional Trades** (10%) — Net sentiment, trader quality scoring, conviction signals (opzioni > equity), leadership role weighting

## Setup

```bash
pip install -r requirements.txt
python screener.py
```

## Uso Programmatico

```python
from screener import MultiFactorScreener

# Pesi custom (devono sommare a 1.0)
weights = {
    'fundamental': 0.30,
    'technical': 0.25,
    'relative_strength': 0.20,
    'congressional': 0.10,
    'quality': 0.15,
}

screener = MultiFactorScreener(weights=weights)

# Singola azione
result = screener.analyze_stock('NVDA')
print(f"Score: {result['composite_score']} — {result['signal']}")

# Screening multiplo
tickers = ['NVDA', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
df, full_results = screener.screen(tickers)
print(df)
```

## Congressional Trade Data

Il modulo `CongressionalAnalyzer` accetta trade data in questo formato:

```python
congressional_data = {
    'NVDA': [
        {
            'member': 'Nancy Pelosi',
            'type': 'purchase',          # 'purchase' | 'sale'
            'amount': '$1,000,001 - $5,000,000',
            'date': '2025-01-15',
            'instrument': 'option_call',  # 'stock' | 'option_call' | 'option_put'
        },
    ],
}

df, results = screener.screen(tickers, congressional_data)
```

### Data Sources per Congressional Trades
- **QuiverQuant API**: `https://www.quiverquant.com/congresstrading/` (ha Python SDK)
- **House Clerk XML**: `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/`
- **Senate EFD**: `https://efdsearch.senate.gov/`
- **GitHub**: `neelsomani/senator-filings` (parser pronto)
- **Unusual Whales**: `https://unusualwhales.com/congress` (premium)

## Architettura dei Punteggi

Ogni fattore produce un score 0-100. Il composite è la media pesata:

```
COMPOSITE = Fund(30%) + Tech(25%) + RS(20%) + Quality(15%) + Congress(10%)
```

| Score Range | Signal      |
|-------------|-------------|
| 75-100      | STRONG BUY  |
| 65-74       | BUY         |
| 55-64       | HOLD+       |
| 45-54       | HOLD        |
| 35-44       | HOLD-       |
| 25-34       | SELL        |
| 0-24        | STRONG SELL |

## Estensioni Suggerite

- **Scraper automatico** per House Clerk / Senate EFD (da aggiungere come modulo)
- **Integrazione TFT**: usa i composite scores come features aggiuntive nel Temporal Fusion Transformer
- **HMM Regime Filter**: disabilita il segnale congressuale durante regimi di alta volatilità
- **Backtesting**: testa la strategia "compra top-5 composite, ribilancia mensile" vs SPY
- **Alert system**: notifica quando un nuovo trade congressuale di un leader appare su un titolo con score >65
- **Streamlit dashboard**: visualizzazione interattiva con grafici
