# Quarterly TradingAgents Pipeline Backtest

- Tickers: NVDA
- Dates: 2024-05-10 to 2024-08-10
- Cadence: every 3 months
- Mode: long_only
- Benchmark: SPY
- Execution: next close after signal date
- Gross exposure cap: 100.0%
- Transaction cost: 5.00 bps per turnover

## Performance

- Strategy total return: 8.93%
- Benchmark total return: 2.65%
- Total alpha: 6.29%
- Strategy annualized return: 42.41%
- Strategy annualized vol: 31.82%
- Strategy Sharpe, 0 rf: 1.33
- Strategy max drawdown: -13.87%

## Rebalances

| Signal | Entry | Period End | Gross | Turnover | Cost | Weights / Ratings |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2024-05-10 | 2024-05-13 | 2024-08-09 | 50.0% | 50.0% | 0.03% | NVDA: 50.0% (Overweight) |

## Look-Ahead Guards

- Signals are generated in chronological order.
- Each rebalance date uses an isolated memory log.
- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.
- Portfolio weights start after the signal date, at the next close.
- Current/future rows in technical indicator data are filtered by trade date.