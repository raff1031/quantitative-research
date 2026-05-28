# Quarterly TradingAgents Pipeline Backtest

- Tickers: NVDA, MSFT, AAPL, AMZN, GOOGL, META
- Dates: 2023-01-03 to 2025-12-31
- Cadence: every 3 months
- Mode: long_only
- Benchmark: SPY
- Execution: next close after signal date
- Gross exposure cap: 100.0%
- Transaction cost: 5.00 bps per turnover

## Performance

- Strategy total return: 210.56%
- Benchmark total return: 84.85%
- Total alpha: 125.71%
- Strategy annualized return: 46.34%
- Strategy annualized vol: 23.16%
- Strategy Sharpe, 0 rf: 2.00
- Strategy max drawdown: -25.26%

## Rebalances

| Signal | Entry | Period End | Gross | Turnover | Cost | Weights / Ratings |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2023-01-03 | 2023-01-04 | 2023-04-04 | 50.0% | 50.0% | 0.03% | NVDA: 0.0% (Underweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 50.0% (Overweight) |
| 2023-04-03 | 2023-04-04 | 2025-12-31 | 100.0% | 110.0% | 0.06% | NVDA: 0.0% (Hold); MSFT: 20.0% (Overweight); AAPL: 20.0% (Overweight); AMZN: 20.0% (Overweight); GOOGL: 20.0% (Overweight); META: 20.0% (Overweight) |

## Look-Ahead Guards

- Signals are generated in chronological order.
- Each rebalance date uses an isolated memory log.
- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.
- Portfolio weights start after the signal date, at the next close.
- Current/future rows in technical indicator data are filtered by trade date.