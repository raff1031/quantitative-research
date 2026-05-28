"""
Simulated output example — this is what the screener produces when run locally.
Run `python screener.py` on your machine to get live data.
"""

EXAMPLE_OUTPUT = """
======================================================================
  MULTI-FACTOR STOCK SCREENER
  Congressional Trades × Fundamentals × Technicals × Rel. Strength
======================================================================

Screening 17 stocks...

======================================================================
  RANKING
======================================================================
   Ticker   Name                 Price    COMPOSITE  Signal      Fundamental  Technical  Rel. Strength  Quality  F-Score   RSI
1    NVDA   NVIDIA Corp        $138.25       72.4    BUY              68.5      71.2         78.3        75.6     8       58.3
2    META   Meta Platforms     $612.80       69.8    BUY              72.1      65.8         71.5        80.0     8       52.1
3    AMZN   Amazon.com         $228.15       68.2    BUY              65.3      68.4         72.1        70.0     7       55.7
4    GOOGL  Alphabet Inc       $185.42       66.5    BUY              70.8      62.3         65.4        68.9     7       48.9
5    MSFT   Microsoft Corp     $425.30       65.1    BUY              73.2      58.6         63.8        72.2     8       46.2
6    JPM    JPMorgan Chase     $248.90       63.8    HOLD+            68.4      64.2         62.1        66.7     7       54.8
7    LMT    Lockheed Martin    $478.65       62.4    HOLD+            62.1      60.5         68.3        64.4     7       51.3
8    XOM    Exxon Mobil        $108.20       61.2    HOLD+            65.8      58.1         60.5        62.2     7       49.6
9    PLTR   Palantir Tech      $ 82.40       60.8    HOLD+            48.2      72.5         73.1        44.4     5       62.8
10   GS     Goldman Sachs      $598.30       59.5    HOLD+            61.3      60.8         58.4        64.4     7       50.2
11   AAPL   Apple Inc          $232.10       58.9    HOLD+            62.5      55.3         57.2        68.9     7       44.8
12   RTX    RTX Corp           $128.45       57.6    HOLD+            58.4      56.8         60.1        60.0     6       47.5
13   UNH    UnitedHealth       $512.80       56.2    HOLD+            64.2      48.5         52.3        66.7     7       42.1
14   CVX    Chevron Corp       $ 158.30      55.8    HOLD+            60.1      54.2         55.8        58.9     6       46.3
15   JNJ    Johnson & Johnson  $152.40       54.1    HOLD             58.8      48.2         50.1        64.4     7       43.2
16   TSLA   Tesla Inc          $328.50       52.3    HOLD             42.5      62.8         58.4        38.9     4       58.4
17   SMCI   Super Micro        $ 38.20       45.8    HOLD             35.2      55.6         52.3        33.3     3       55.1

======================================================================
  TOP 5 DETAILED BREAKDOWN
======================================================================

────────────────────────────────────────────────────────
  #1 NVDA — NVIDIA Corporation
  Technology | Semiconductors
  Price: $138.25 | Cap: $3.4T
  ★ COMPOSITE: 72.4/100 → BUY
────────────────────────────────────────────────────────
  Fundamental:   68.5/100  │  Technical:      71.2/100
  Rel Strength:  78.3/100  │  Quality:        75.6/100
  Congressional: 70.0/100  │  F-Score:        8/9

  Key Fundamentals:
    P/E (fwd): 28.5  |  PEG: 0.85
    Revenue Growth: 0.94  |  Profit Margin: 0.55
    ROE: 1.15  |  D/E: 17.2

  Key Technicals:
    RSI(14): 58.3  |  MACD Hist: 1.250
    Above SMA20: True  |  SMA50: True  |  SMA200: True
    ATR%: 2.85%  |  ADX: 32.5
    Returns: 1w=+3.2%  1m=+8.5%  3m=+15.2%  6m=+42.1%

  Relative Strength:
    vs SPY - Mansfield: 8.45
    vs Sector (XLK) - Mansfield: 4.21

  Congressional Activity:
    Trades: 3 (Buy: 3, Sell: 0)
    Net Sentiment: BULLISH
    Notable: Nancy Pelosi, Josh Gottheimer, Dan Crenshaw

======================================================================
  Weights: Fund=30% | Tech=25% | RS=20% | Congress=10% | Quality=15%
======================================================================
"""

print(EXAMPLE_OUTPUT)
