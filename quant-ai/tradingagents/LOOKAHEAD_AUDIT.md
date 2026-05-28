# Look-Ahead Bias Audit

Scope: `scripts/quarterly_pipeline.py`, `scripts/strategy_lab.py`, and the
TradingAgents market-only signal path used for the current backtests.

## Current Verdict

The original market-only TradingAgents pipeline is usable for exploratory
research, but it is not institutional-grade point-in-time research. The new
feature-only anonymous pipeline materially reduces LLM-specific look-ahead
risk by removing real tickers, real dates, tool calls, news, and fundamentals
from the model input.

The largest remaining risks are:

1. LLM latent knowledge of future events.
2. Survivorship bias from using today's mega-cap winners as the historical
   universe.
3. Yahoo Finance historical data revisions and adjusted-price behavior.
4. Overlay execution timing assumptions for monthly TSMOM filters.

The pipeline now has several guards, but these risks should be reported with
any results.

## Guards Already In Place

- Signals are generated chronologically by rebalance date.
- Portfolio weights are applied after the signal date, at the next close.
- `--backtest-only` reuses saved JSON signals and does not call the LLM again.
- Strict mode rejects `news`, `social`, and `fundamentals` analysts because
  their default Yahoo paths can use current snapshots rather than true
  point-in-time data.
- Technical indicator data goes through `load_ohlcv(..., curr_date)`, which
  filters rows after the trade date.
- The pipeline isolates each signal date's memory log, preventing later
  rebalances from leaking into earlier prompts.
- `get_stock_data` is now clamped to `current_trade_date`, so accidental
  future OHLC requests from the LLM are capped at the simulated date.
- TSMOM filters are computed from closes available on or before the signal or
  overlay evaluation date.
- A new feature-only path, `scripts/feature_signal_pipeline.py`, sends the LLM
  only anonymous numeric feature tables. The model sees `Asset_001` and
  `Period_001`, not tickers or calendar dates.
- Feature-only signal files include an audit companion file:
  `*.audit.json`. It stores the asset mapping, feature table, prompt hash,
  full prompt, raw response, and parsed response.
- The risk-off overlay can allocate uninvested capital to defensive assets
  such as `GLD` and `TLT`, chosen by point-in-time TSMOM.

## Open Risks

### 1. LLM Latent Future Knowledge

Even if tools only return point-in-time prices, the LLM may already know that
NVDA, META, or the broader AI trade performed well after a historical date.
Prompts can instruct the model not to use future information, but this cannot
be perfectly enforced.

Mitigation:

- Prefer the feature-only anonymous pipeline over the tool-calling
  TradingAgents pipeline for historical backtests.
- Prefer deterministic indicator-to-rating rules for final production tests,
  or compare LLM signals against deterministic baselines.
- Evaluate on older and broader universes where model memorization is less
  likely to map cleanly to the selected assets.

### 2. Survivorship / Universe Selection Bias

The current universe `NVDA MSFT AAPL AMZN GOOGL META` is chosen with today's
knowledge. This is a major source of bias. It omits companies that were large
at the start of the period but later underperformed, and it overrepresents
known winners.

Mitigation:

- Use a point-in-time index constituent list, e.g. S&P 100 / Nasdaq 100 as of
  each rebalance date.
- Or define the universe ex ante and disclose that it is a curated mega-cap
  technology basket, not a market-wide strategy.
- Always compare against equal-weight of the same universe, not only SPY.

### 3. Yahoo Finance Data Revisions And Adjustments

Yahoo data is convenient but not a point-in-time historical database. Data can
be revised. Adjusted series can also encode later corporate action adjustment
factors. This is usually acceptable for quick research, but not ideal for
signal generation.

Mitigation:

- For production-grade tests, use a point-in-time vendor.
- For technical indicators, consider raw OHLC with split-only adjustment
  rather than total-return adjusted prices.
- Preserve downloaded data snapshots by run date.

### 4. Fundamentals / News / Social Disabled In Strict Mode

This is intentional. Yahoo fundamentals and news are not guaranteed
point-in-time for historical simulations. Re-enabling those analysts should be
treated as exploratory unless a point-in-time data source is wired in.

Mitigation:

- Use Alpha Vantage or another vendor only where historical timestamps and
  reporting availability are controlled.
- Record publication dates and enforce release-date cutoffs.

### 5. Overlay Execution Timing

The main LLM signal path enters after the signal date. Monthly TSMOM overlays
are price-only and point-in-time, but their execution timing should be treated
carefully. A conservative version should compute the overlay at a close and
apply it from the next close-to-close interval.

Current implementation:

- For each daily return row, monthly TSMOM uses the previous available close
  as the as-of date, then applies the resulting weight to the next close-to-
  close interval represented by that row. This is a conservative one-bar style
  timing assumption for close-to-close data.

### 6. Multiple Testing / Researcher Degrees Of Freedom

Testing 21D, 63D, 126D filters and defensive assets can overfit the sample.

Mitigation:

- Pre-register a small set of overlays.
- Split sample into design and validation windows.
- Report all tested variants, not only the best one.

## Recommended Next Validation

1. Add equal-weight universe, SPY, QQQ, and 60/40 benchmarks.
2. Use point-in-time universe membership.
3. Run walk-forward tests where overlay parameters are selected on one period
   and evaluated on another.
4. Add deterministic baselines using the same anonymous feature table.
5. Replace Yahoo with a point-in-time data vendor for final conclusions.

## Current Best Practice Command Shape

Generate historical anonymous feature-only signals once:

```bat
.\.venv\Scripts\python.exe scripts\feature_signal_pipeline.py --name "Feature Only MegaCap TA 2018" --tickers NVDA MSFT AAPL AMZN GOOGL META --start-date 2018-01-03 --end-date 2025-12-31 --cadence-months 3 --resume
```

Then run many overlays without new LLM calls:

```bat
.\.venv\Scripts\python.exe scripts\quarterly_pipeline.py --name "Feature Only MegaCap TA 2018 Defensive" --signal-name "Feature Only MegaCap TA 2018" --tickers NVDA MSFT AAPL AMZN GOOGL META --start-date 2018-01-03 --end-date 2025-12-31 --mode long_only --transaction-cost-bps 5 --gross-exposure-cap 1.0 --tsmom-filter --tsmom-asset SPY --tsmom-lookback-days 63 --tsmom-threshold 0 --tsmom-off-exposure 0 --tsmom-update-frequency monthly --risk-off-assets GLD TLT --risk-off-lookback-days 63 --risk-off-threshold 0 --risk-off-update-frequency monthly --signals-dir feature_pipeline\signals --backtest-only
```
