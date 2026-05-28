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

## Performance

- Strategy total return: 145.24%
- Benchmark total return: 84.85%
- Total alpha: 60.38%
- Equal-weight universe total return: 311.84%
- Alpha vs equal-weight universe: -166.60%
- Strategy annualized return: 35.18%
- Strategy annualized vol: 20.79%
- Strategy Sharpe, 0 rf: 1.69
- Strategy max drawdown: -14.41%

## Rebalances

| Signal | Entry | Period End | TSMOM | Gross | Turnover | Cost | Weights / Ratings |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2023-01-03 | 2023-01-04 | 2023-04-04 | 3.30% | 50.0% | 150.0% | 0.07% | NVDA: 0.0% (Underweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 50.0% (Overweight) |
| 2023-04-03 | 2023-04-04 | 2023-07-05 | 6.48% | 100.0% | 110.0% | 0.06% | NVDA: 0.0% (Hold); MSFT: 20.0% (Overweight); AAPL: 20.0% (Overweight); AMZN: 20.0% (Overweight); GOOGL: 20.0% (Overweight); META: 20.0% (Overweight) |
| 2023-07-03 | 2023-07-05 | 2023-10-04 | -4.88% | 0.0% | 180.0% | 0.09% | NVDA: 0.0% (Hold); MSFT: 0.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Overweight); META: 0.0% (Overweight) |
| 2023-10-03 | 2023-10-04 | 2024-01-04 | 4.98% | 50.0% | 50.0% | 0.03% | NVDA: 50.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Hold) |
| 2024-01-03 | 2024-01-04 | 2024-04-04 | 3.64% | 100.0% | 83.3% | 0.04% | NVDA: 33.3% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Hold); GOOGL: 33.3% (Overweight); META: 33.3% (Overweight) |
| 2024-04-03 | 2024-04-04 | 2024-07-05 | 3.78% | 100.0% | 250.0% | 0.12% | NVDA: 25.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 25.0% (Overweight); GOOGL: 25.0% (Overweight); META: 25.0% (Overweight) |
| 2024-07-03 | 2024-07-05 | 2024-10-04 | 3.08% | 100.0% | 100.0% | 0.05% | NVDA: 0.0% (Hold); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 50.0% (Overweight); META: 50.0% (Overweight) |
| 2024-10-03 | 2024-10-04 | 2025-01-06 | -2.41% | 0.0% | 400.0% | 0.20% | NVDA: 0.0% (Hold); MSFT: 0.0% (Underweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Buy); META: 0.0% (Hold) |
| 2025-01-03 | 2025-01-06 | 2025-04-04 | -5.57% | 0.0% | 200.0% | 0.10% | NVDA: 0.0% (Overweight); MSFT: 0.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Overweight); META: 0.0% (Overweight) |
| 2025-04-03 | 2025-04-04 | 2025-07-07 | 5.02% | 0.0% | 0.0% | 0.00% | NVDA: 0.0% (Hold); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Hold) |
| 2025-07-03 | 2025-07-07 | 2025-10-06 | 3.56% | 100.0% | 100.0% | 0.05% | NVDA: 25.0% (Overweight); MSFT: 25.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 25.0% (Overweight); GOOGL: 0.0% (Hold); META: 25.0% (Overweight) |
| 2025-10-03 | 2025-10-06 | 2025-12-31 | -0.58% | 0.0% | 150.0% | 0.07% | NVDA: 0.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Hold) |

## Look-Ahead Guards

- Signals are generated in chronological order.
- Each rebalance date uses an isolated memory log.
- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.
- Portfolio weights start after the signal date, at the next close.
- Current/future rows in technical indicator data are filtered by trade date.
- TSMOM, when enabled, is computed from closes available on or before the signal date.