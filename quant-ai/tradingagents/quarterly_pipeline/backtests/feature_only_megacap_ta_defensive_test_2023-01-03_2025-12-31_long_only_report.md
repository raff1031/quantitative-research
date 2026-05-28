# Quarterly TradingAgents Pipeline Backtest

- Tickers: NVDA, MSFT, AAPL, AMZN, GOOGL, META
- Dates: 2023-01-03 to 2025-12-31
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

- Strategy total return: 297.83%
- Benchmark total return: 84.85%
- Total alpha: 212.98%
- Equal-weight universe total return: 311.84%
- Alpha vs equal-weight universe: -14.01%
- Strategy annualized return: 59.04%
- Strategy annualized vol: 26.52%
- Strategy Sharpe, 0 rf: 2.23
- Strategy max drawdown: -17.15%

## Rebalances

| Signal | Entry | Period End | TSMOM | Gross | Turnover | Cost | Weights / Ratings |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2023-01-03 | 2023-01-04 | 2023-04-04 | 7.18% | 100.0% | 300.0% | 0.15% | NVDA: 0.0% (Underweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Sell); GOOGL: 0.0% (Hold); META: 50.0% (Overweight); GLD: 50.0% (risk-off) |
| 2023-04-03 | 2023-04-04 | 2025-12-31 | 6.24% | 100.0% | 933.3% | 0.47% | NVDA: 33.3% (Overweight); MSFT: 33.3% (Overweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 33.3% (Overweight) |

## Look-Ahead Guards

- Signals are generated in chronological order.
- Each rebalance date uses an isolated memory log.
- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.
- Portfolio weights start after the signal date, at the next close.
- Current/future rows in technical indicator data are filtered by trade date.
- TSMOM, when enabled, is computed from closes available on or before the signal date.