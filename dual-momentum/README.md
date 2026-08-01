# Dual Momentum: what survives statistical testing

An implementation of Antonacci-style dual momentum on US equities, long Treasuries and gold,
built to answer one question honestly: **once you correct for the fact that you tried many
configurations, is there anything left?**

Short answer: the drawdown reduction is real and robust. The return edge is not statistically
significant. This README reports both.

## Headline result

Quarterly rotation across S&P 500 total return, long Treasuries and gold, with a T-bill
absolute-momentum filter and a 12-month lookback. Returns are net of 10bps per switch, and the
headline uses a **three-tranche blend** (three portfolios started in different months, averaged)
rather than a single rebalancing grid, so the number is not an artefact of which month you
happened to start in.

| Jan 1992 – Nov 2018 | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Dual momentum, 3-tranche | **10.3%** | **0.68** | **&minus;22.4%** |
| S&P 500 buy and hold | 9.5% | 0.55 | &minus;50.9% |
| + cross-sectional 12-1 sleeve | 12.8% | 0.72 | |

## Why the headline is not the interesting part

**1. The return edge does not survive multiple-testing correction.**
White's Reality Check and Hansen's SPA across the 12-configuration grid return p ≈ 0.45–0.47.
The Jobson-Korkie-Memmel test on the Sharpe difference gives p = 0.56, and a bootstrap
confidence interval of [0.35, 1.03] straddles the benchmark's 0.55. No configuration reliably
beats buy and hold on return once you account for having searched.

**2. Factor attribution says the stock sleeve is beta, not skill.**
Regressing the cross-sectional 12-1 sleeve on Fama-French 3 factors plus momentum, with HAC
(Newey-West) standard errors: alpha ≈ 0 (t = &minus;0.17), momentum loading 0.41 (t = 13),
R² = 0.87. The sleeve is a momentum-factor vehicle. The rotation itself shows 4.4%/year alpha
(t = 2.2) at 0.38 equity beta, but most of that is bond and gold premia the model cannot price.

**3. Timing luck is measurable, and it decays.**
The ex-post-best single rebalancing grid beats the tranche blend by 0.13 Sharpe in sample.
Out of sample (Dec 2018 onward) that gap collapses to 0.01. This is why the headline uses the
blend: picking the best grid after the fact buys you nothing you can keep.

**4. Out of sample, the rotation underperforms.**
From 2019 to 2026 the S&P 500 returned 15.9% CAGR at Sharpe 0.80 against the rotation's 11-12%
at 0.66-0.67. There was no sustained bear market in that window, which is the only environment
the strategy is built to help in. An equal-weight 1/3 portfolio with no timing at all had the
best out-of-sample Sharpe of the lot (0.90).

## What the strategy is actually for

A risk reducer, not a return generator. It halves the drawdown of an equity-only portfolio and
survives regime changes, and that is defensible. Claiming it generates alpha is not.

## Refinements implemented

- **Barroso–Santa-Clara volatility targeting.** Scale exposure by target volatility over trailing
  realized volatility, capped at 2x, no look-ahead. On the crash-prone momentum book: Sharpe
  0.63 → 0.74, and **excess kurtosis 2.4 → 0.9**, which is the paper's actual claim. On the
  long-short book, skewness &minus;1.36 → &minus;0.04. Honest caveat included in the notebook:
  it helps crashes that follow turbulence and amplifies crashes that arrive out of calm.
- **Residual momentum (Blitz-Huij-Martens).** Signal built from the information ratio of FF3
  regression residuals, via a vectorised rolling 36-month regression. Lower return than plain
  momentum (11.4% vs 14.1% CAGR) but lower drawdown (&minus;45% vs &minus;56%) and lower kurtosis.
- **International GEM rotations.** US vs emerging markets beats US buy and hold on all three axes
  (CAGR 12.0 vs 10.9, Sharpe 0.665 vs 0.606, max drawdown &minus;33% vs &minus;51%) even though
  emerging markets standalone was terrible. Adding EM to the three-asset menu, by contrast,
  **hurts**: Sharpe 0.63 vs 0.69. A marginal asset has to be a better diversifier than whatever
  it displaces.
- **Cost sensitivity.** Break-even is around 41bps per switch against realistic ETF costs of
  2-5bps, so costs are not the binding constraint.

## Data notes

- The equity dataset is survivorship-bias free, with an index-membership mask applied on every
  date. Roughly 48% of the price series is stale outside membership windows and must be masked.
- Prices are **price-only, not total return**. The dividend gap was estimated at about 2.11%/year
  from the total-return versus reconstructed-price difference and added back explicitly.
- Monthly sampling understates drawdowns relative to weekly sampling, because it misses intra-period
  troughs. Any drawdown comparison in the notebook is made at matching frequency.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook Dual_Momentum_Antonacci.ipynb
```

Asset-class series are fetched from public sources and cached in `data/`. **The course equity
dataset is not redistributed here** for licensing reasons; the notebook documents the expected
format so the pipeline can be pointed at an equivalent source.

## References

Antonacci (2014), *Dual Momentum Investing*. Jegadeesh & Titman (1993). Fama & French (1993).
Carhart (1997). White (2000), *A Reality Check for Data Snooping*. Hansen (2005), *A Test for
Superior Predictive Ability*. Barroso & Santa-Clara (2015), *Momentum Has Its Moments*.
Blitz, Huij & Martens (2011), *Residual Momentum*.
