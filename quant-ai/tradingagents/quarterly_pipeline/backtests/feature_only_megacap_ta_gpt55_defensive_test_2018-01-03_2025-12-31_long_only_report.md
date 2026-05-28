# Quarterly TradingAgents Pipeline Backtest

- Tickers: NVDA, MSFT, AAPL, AMZN, GOOGL, META
- Dates: 2018-01-03 to 2025-12-31
- Cadence: every 3 months
- Mode: long_only
- Benchmark: SPY
- Execution: next close after signal date
- Gross exposure cap: 100.0%
- Transaction cost: 5.00 bps per turnover
- TSMOM filter: on
- TSMOM update frequency: monthly
- Risk-off assets: GLD, TLT

## Performance

- Strategy total return: 568.18%
- Benchmark total return: 184.47%
- Total alpha: 383.71%
- Equal-weight universe total return: 746.07%
- Alpha vs equal-weight universe: -177.89%
- Strategy annualized return: 26.92%
- Strategy annualized vol: 24.53%
- Strategy Sharpe, 0 rf: 1.10
- Strategy max drawdown: -33.59%

## Rebalances

| Signal | Entry | Period End | TSMOM | Gross | Turnover | Cost | Weights / Ratings |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2018-01-03 | 2018-01-04 | 2018-04-04 | -1.17% | 100.0% | 300.0% | 0.15% | NVDA: 0.0% (Underweight); MSFT: 0.0% (Buy); AAPL: 0.0% (Hold); AMZN: 0.0% (Overweight); GOOGL: 0.0% (Hold); META: 0.0% (Overweight); GLD: 100.0% (risk-off) |
| 2018-04-03 | 2018-04-04 | 2025-12-31 | 6.24% | 100.0% | 3800.0% | 1.90% | NVDA: 25.0% (Overweight); MSFT: 25.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 50.0% (Buy); GOOGL: 0.0% (Underweight); META: 0.0% (Sell) |

## Look-Ahead Guards

- Signals are generated in chronological order.
- Each rebalance date uses an isolated memory log.
- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.
- Portfolio weights start after the signal date, at the next close.
- Current/future rows in technical indicator data are filtered by trade date.
- TSMOM, when enabled, is computed from closes available on or before the signal date.