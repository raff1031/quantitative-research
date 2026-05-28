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

## Performance

- Strategy total return: 187.51%
- Benchmark total return: 84.85%
- Total alpha: 102.66%
- Equal-weight universe total return: 311.84%
- Alpha vs equal-weight universe: -124.32%
- Strategy annualized return: 42.60%
- Strategy annualized vol: 24.46%
- Strategy Sharpe, 0 rf: 1.74
- Strategy max drawdown: -25.33%

## Rebalances

| Signal | Entry | Period End | TSMOM | Gross | Turnover | Cost | Weights / Ratings |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2023-01-03 | 2023-01-04 | 2023-04-04 |  | 50.0% | 50.0% | 0.03% | NVDA: 0.0% (Underweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 50.0% (Overweight) |
| 2023-04-03 | 2023-04-04 | 2023-07-05 | 16.02% | 100.0% | 110.0% | 0.06% | NVDA: 0.0% (Hold); MSFT: 20.0% (Overweight); AAPL: 20.0% (Overweight); AMZN: 20.0% (Overweight); GOOGL: 20.0% (Overweight); META: 20.0% (Overweight) |
| 2023-07-03 | 2023-07-05 | 2023-10-04 | 16.61% | 100.0% | 80.0% | 0.04% | NVDA: 0.0% (Hold); MSFT: 33.3% (Overweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 33.3% (Overweight); META: 33.3% (Overweight) |
| 2023-10-03 | 2023-10-04 | 2024-01-04 | 3.33% | 50.0% | 150.0% | 0.07% | NVDA: 50.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Hold) |
| 2024-01-03 | 2024-01-04 | 2024-04-04 | 6.59% | 100.0% | 83.3% | 0.04% | NVDA: 33.3% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Hold); GOOGL: 33.3% (Overweight); META: 33.3% (Overweight) |
| 2024-04-03 | 2024-04-04 | 2024-07-05 | 22.43% | 100.0% | 50.0% | 0.02% | NVDA: 25.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 25.0% (Overweight); GOOGL: 25.0% (Overweight); META: 25.0% (Overweight) |
| 2024-07-03 | 2024-07-05 | 2024-10-04 | 17.41% | 100.0% | 100.0% | 0.05% | NVDA: 0.0% (Hold); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 50.0% (Overweight); META: 50.0% (Overweight) |
| 2024-10-03 | 2024-10-04 | 2025-01-06 | 11.37% | 100.0% | 100.0% | 0.05% | NVDA: 0.0% (Hold); MSFT: 0.0% (Underweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 100.0% (Buy); META: 0.0% (Hold) |
| 2025-01-03 | 2025-01-06 | 2025-04-04 | 7.41% | 100.0% | 150.0% | 0.07% | NVDA: 25.0% (Overweight); MSFT: 25.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 25.0% (Overweight); META: 25.0% (Overweight) |
| 2025-04-03 | 2025-04-04 | 2025-07-07 | -5.01% | 0.0% | 100.0% | 0.05% | NVDA: 0.0% (Hold); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Hold) |
| 2025-07-03 | 2025-07-07 | 2025-10-06 | 6.95% | 100.0% | 100.0% | 0.05% | NVDA: 25.0% (Overweight); MSFT: 25.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 25.0% (Overweight); GOOGL: 0.0% (Hold); META: 25.0% (Overweight) |
| 2025-10-03 | 2025-10-06 | 2025-12-31 | 25.41% | 50.0% | 100.0% | 0.05% | NVDA: 50.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Hold) |

## Look-Ahead Guards

- Signals are generated in chronological order.
- Each rebalance date uses an isolated memory log.
- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.
- Portfolio weights start after the signal date, at the next close.
- Current/future rows in technical indicator data are filtered by trade date.
- TSMOM, when enabled, is computed from closes available on or before the signal date.