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

- Strategy total return: 795.30%
- Benchmark total return: 184.47%
- Total alpha: 610.83%
- Equal-weight universe total return: 746.07%
- Alpha vs equal-weight universe: 49.23%
- Strategy annualized return: 31.66%
- Strategy annualized vol: 24.68%
- Strategy Sharpe, 0 rf: 1.28
- Strategy max drawdown: -26.78%

## Rebalances

| Signal | Entry | Period End | TSMOM | Gross | Turnover | Cost | Weights / Ratings |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2018-01-03 | 2018-01-04 | 2018-04-04 | -1.17% | 100.0% | 300.0% | 0.15% | NVDA: 0.0% (Overweight); MSFT: 0.0% (Buy); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Overweight); GLD: 100.0% (risk-off) |
| 2018-04-03 | 2018-04-04 | 2018-07-05 | 5.84% | 100.0% | 400.0% | 0.20% | NVDA: 25.0% (Overweight); MSFT: 25.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 50.0% (Buy); GOOGL: 0.0% (Underweight); META: 0.0% (Sell) |
| 2018-07-03 | 2018-07-05 | 2018-10-04 | 7.65% | 100.0% | 50.0% | 0.02% | NVDA: 0.0% (Hold); MSFT: 33.3% (Overweight); AAPL: 0.0% (Hold); AMZN: 66.7% (Buy); GOOGL: 0.0% (Underweight); META: 0.0% (Underweight) |
| 2018-10-03 | 2018-10-04 | 2019-01-04 | -13.53% | 100.0% | 300.0% | 0.15% | NVDA: 0.0% (Overweight); MSFT: 0.0% (Buy); AAPL: 0.0% (Buy); AMZN: 0.0% (Overweight); GOOGL: 0.0% (Hold); META: 0.0% (Sell); GLD: 100.0% (risk-off) |
| 2019-01-03 | 2019-01-04 | 2019-04-04 | 14.37% | 100.0% | 400.0% | 0.20% | NVDA: 0.0% (Sell); MSFT: 66.7% (Buy); AAPL: 0.0% (Sell); AMZN: 0.0% (Hold); GOOGL: 33.3% (Overweight); META: 0.0% (Underweight) |
| 2019-04-03 | 2019-04-04 | 2019-07-05 | 4.23% | 100.0% | 450.0% | 0.23% | NVDA: 0.0% (Sell); MSFT: 50.0% (Buy); AAPL: 0.0% (Hold); AMZN: 0.0% (Hold); GOOGL: 25.0% (Overweight); META: 25.0% (Overweight) |
| 2019-07-03 | 2019-07-05 | 2019-10-04 | 0.84% | 100.0% | 80.0% | 0.04% | NVDA: 0.0% (Sell); MSFT: 40.0% (Buy); AAPL: 20.0% (Overweight); AMZN: 20.0% (Overweight); GOOGL: 0.0% (Underweight); META: 20.0% (Overweight) |
| 2019-10-03 | 2019-10-04 | 2020-01-06 | 10.30% | 100.0% | 400.0% | 0.20% | NVDA: 0.0% (Hold); MSFT: 33.3% (Overweight); AAPL: 66.7% (Buy); AMZN: 0.0% (Sell); GOOGL: 0.0% (Underweight); META: 0.0% (Underweight) |
| 2020-01-03 | 2020-01-06 | 2020-04-06 | -19.25% | 100.0% | 253.3% | 0.13% | NVDA: 0.0% (Overweight); MSFT: 0.0% (Buy); AAPL: 0.0% (Buy); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 0.0% (Hold); TLT: 100.0% (risk-off) |
| 2020-04-03 | 2020-04-06 | 2020-07-06 | 20.16% | 100.0% | 200.0% | 0.10% | NVDA: 33.3% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 66.7% (Buy); GOOGL: 0.0% (Sell); META: 0.0% (Sell) |
| 2020-07-02 | 2020-07-06 | 2020-10-05 | 8.28% | 100.0% | 100.0% | 0.05% | NVDA: 25.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 50.0% (Buy); AMZN: 25.0% (Overweight); GOOGL: 0.0% (Underweight); META: 0.0% (Sell) |
| 2020-10-02 | 2020-10-05 | 2021-01-04 | 11.40% | 100.0% | 283.3% | 0.14% | NVDA: 66.7% (Buy); MSFT: 0.0% (Hold); AAPL: 33.3% (Overweight); AMZN: 0.0% (Hold); GOOGL: 0.0% (Sell); META: 0.0% (Underweight) |
| 2020-12-31 | 2021-01-04 | 2021-04-05 | 7.04% | 100.0% | 133.3% | 0.07% | NVDA: 0.0% (Hold); MSFT: 33.3% (Overweight); AAPL: 66.7% (Buy); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 0.0% (Underweight) |
| 2021-04-01 | 2021-04-05 | 2021-07-06 | 8.36% | 100.0% | 133.3% | 0.07% | NVDA: 0.0% (Hold); MSFT: 40.0% (Buy); AAPL: 0.0% (Sell); AMZN: 0.0% (Underweight); GOOGL: 40.0% (Buy); META: 20.0% (Overweight) |
| 2021-07-02 | 2021-07-06 | 2021-10-04 | 0.02% | 100.0% | 80.0% | 0.04% | NVDA: 25.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Hold); GOOGL: 50.0% (Buy); META: 25.0% (Overweight) |
| 2021-10-01 | 2021-10-04 | 2022-01-04 | 9.76% | 100.0% | 400.0% | 0.20% | NVDA: 25.0% (Overweight); MSFT: 25.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Sell); GOOGL: 50.0% (Buy); META: 0.0% (Underweight) |
| 2022-01-03 | 2022-01-04 | 2022-04-04 | -4.85% | 100.0% | 300.0% | 0.15% | NVDA: 0.0% (Overweight); MSFT: 0.0% (Overweight); AAPL: 0.0% (Buy); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 0.0% (Hold); GLD: 100.0% (risk-off) |
| 2022-04-01 | 2022-04-04 | 2022-07-05 | -17.40% | 0.0% | 100.0% | 0.05% | NVDA: 0.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Buy); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 0.0% (Sell) |
| 2022-07-01 | 2022-07-05 | 2022-10-04 | -5.93% | 0.0% | 200.0% | 0.10% | NVDA: 0.0% (Sell); MSFT: 0.0% (Buy); AAPL: 0.0% (Overweight); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Hold); META: 0.0% (Sell) |
| 2022-10-03 | 2022-10-04 | 2023-01-04 | 7.56% | 100.0% | 200.0% | 0.10% | NVDA: 0.0% (Sell); MSFT: 0.0% (Hold); AAPL: 50.0% (Overweight); AMZN: 0.0% (Hold); GOOGL: 0.0% (Underweight); META: 0.0% (Sell); GLD: 50.0% (risk-off) |
| 2023-01-03 | 2023-01-04 | 2023-04-04 | 7.18% | 100.0% | 600.0% | 0.30% | NVDA: 0.0% (Hold); MSFT: 66.7% (Buy); AAPL: 0.0% (Sell); AMZN: 0.0% (Sell); GOOGL: 0.0% (Underweight); META: 33.3% (Overweight) |
| 2023-04-03 | 2023-04-04 | 2023-07-05 | 10.21% | 100.0% | 100.0% | 0.05% | NVDA: 50.0% (Buy); MSFT: 25.0% (Overweight); AAPL: 0.0% (Hold); AMZN: 0.0% (Sell); GOOGL: 0.0% (Hold); META: 25.0% (Overweight) |
| 2023-07-03 | 2023-07-05 | 2023-10-04 | -3.22% | 0.0% | 170.0% | 0.09% | NVDA: 0.0% (Buy); MSFT: 0.0% (Hold); AAPL: 0.0% (Overweight); AMZN: 0.0% (Hold); GOOGL: 0.0% (Underweight); META: 0.0% (Buy) |
| 2023-10-03 | 2023-10-04 | 2024-01-04 | 11.64% | 100.0% | 300.0% | 0.15% | NVDA: 25.0% (Overweight); MSFT: 0.0% (Hold); AAPL: 0.0% (Underweight); AMZN: 0.0% (Underweight); GOOGL: 25.0% (Overweight); META: 50.0% (Buy) |
| 2024-01-03 | 2024-01-04 | 2024-04-04 | 10.11% | 100.0% | 100.0% | 0.05% | NVDA: 0.0% (Hold); MSFT: 25.0% (Overweight); AAPL: 0.0% (Underweight); AMZN: 25.0% (Overweight); GOOGL: 0.0% (Hold); META: 50.0% (Buy) |
| 2024-04-03 | 2024-04-04 | 2024-07-05 | 4.38% | 100.0% | 80.0% | 0.04% | NVDA: 40.0% (Buy); MSFT: 20.0% (Overweight); AAPL: 0.0% (Sell); AMZN: 20.0% (Overweight); GOOGL: 0.0% (Hold); META: 20.0% (Overweight) |
| 2024-07-03 | 2024-07-05 | 2024-10-04 | 5.53% | 100.0% | 93.3% | 0.05% | NVDA: 16.7% (Overweight); MSFT: 16.7% (Overweight); AAPL: 33.3% (Buy); AMZN: 33.3% (Buy); GOOGL: 0.0% (Hold); META: 0.0% (Underweight) |
| 2024-10-03 | 2024-10-04 | 2025-01-06 | 3.42% | 100.0% | 100.0% | 0.05% | NVDA: 25.0% (Overweight); MSFT: 0.0% (Underweight); AAPL: 50.0% (Buy); AMZN: 0.0% (Hold); GOOGL: 0.0% (Sell); META: 25.0% (Overweight) |
| 2025-01-03 | 2025-01-06 | 2025-04-04 | -6.70% | 100.0% | 316.7% | 0.16% | NVDA: 0.0% (Buy); MSFT: 0.0% (Underweight); AAPL: 0.0% (Overweight); AMZN: 0.0% (Buy); GOOGL: 0.0% (Overweight); META: 0.0% (Hold); GLD: 100.0% (risk-off) |
| 2025-04-03 | 2025-04-04 | 2025-07-07 | 11.52% | 100.0% | 100.0% | 0.05% | NVDA: 0.0% (Sell); MSFT: 0.0% (Hold); AAPL: 50.0% (Overweight); AMZN: 0.0% (Underweight); GOOGL: 0.0% (Sell); META: 0.0% (Underweight); GLD: 50.0% (risk-off) |
| 2025-07-03 | 2025-07-07 | 2025-10-06 | 8.16% | 100.0% | 200.0% | 0.10% | NVDA: 25.0% (Overweight); MSFT: 50.0% (Buy); AAPL: 0.0% (Sell); AMZN: 0.0% (Hold); GOOGL: 0.0% (Hold); META: 25.0% (Overweight) |
| 2025-10-03 | 2025-10-06 | 2025-12-31 | 6.24% | 100.0% | 150.0% | 0.07% | NVDA: 40.0% (Buy); MSFT: 0.0% (Hold); AAPL: 20.0% (Overweight); AMZN: 0.0% (Underweight); GOOGL: 40.0% (Buy); META: 0.0% (Sell) |

## Look-Ahead Guards

- Signals are generated in chronological order.
- Each rebalance date uses an isolated memory log.
- Strict mode rejects news, social, and fundamentals analysts with Yahoo data.
- Portfolio weights start after the signal date, at the next close.
- Current/future rows in technical indicator data are filtered by trade date.
- TSMOM, when enabled, is computed from closes available on or before the signal date.