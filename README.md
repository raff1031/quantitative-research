# Quantitative Research & Algorithmic Trading

**Romano Raffaele** · MSc Data Science candidate, Università degli Studi di Milano-Bicocca (graduating Oct 2026)  
📧 romano.raff10@gmail.com · 📍 Milano, Italy

---

## Overview

This repository collects original quantitative research and production-grade trading systems built independently from 2024 to 2026. Projects span systematic equity strategies, crypto market making, statistical arbitrage, and ML-driven alpha generation — covering the full pipeline from raw data ingestion to live execution.

All strategies include out-of-sample validation. Look-ahead bias is explicitly controlled throughout via purged/embargoed cross-validation or strict train/test splits calibrated on training data and evaluated on held-out periods.

---

## Projects

### 1. Avellaneda-Stoikov Market Maker with Anti-Adverse Selection
`Market Making AS/as_anti_adverse_selection.py`

A production-ready market making strategy deployed on Hummingbot (ScriptStrategyBase V2), implementing the Avellaneda-Stoikov (2008) stochastic control framework with four layers of adversarial flow protection:

- **Reservation price + optimal spread** derived from inventory risk parameter γ, mid-price volatility σ, and order arrival intensity κ
- **EMA trend filter** with directional skewing — quote asymmetrically in trending conditions to avoid inventory buildup
- **Order Book Imbalance (OBI)** — real-time bid/ask queue depth ratio triggers quote withdrawal before pump/dump events
- **Kill-switch** — extreme volatility (ATR threshold) suspends quoting entirely to avoid crossing toxic flow

Designed for crypto spot/perp markets with configurable parameters. Includes complete deployment instructions for Hummingbot live execution.

---

### 2. Pairs Trading — Walk-Forward Intra-Sector (v5)
`Market Making AS/pairs_trading_v5.py`

Five-generation research evolution culminating in a walk-forward intra-sector pairs trading system:

| Version | Problem | Fix |
|---------|---------|-----|
| v1 | KO/PEP not cointegrated → losses | — |
| v3 | Only 2 trades → no statistical significance | — |
| v4 | Cross-sector pairs, no economic logic → 609 trades, -12.5% | — |
| **v5** | ✅ Intra-sector only + calibrated OOS | Full rework |

**v5 methodology:**
- Cointegration tested via Engle-Granger on **intra-sector pairs only** (same GICS industry group)
- 60/40 train/test split — all parameters (z-score window = half-life, entry/exit thresholds) calibrated on training, applied frozen on out-of-sample
- **Kalman filter** hedge ratio (δ = 1e-5) for slow, stable adaptation vs. noisy daily recalibration
- Log-price spread (more stationary than price-ratio)
- Entry: z-score > 1.5 | Exit: z-score = 0 | Stop: z-score > 3.5 | TC: 5bps round-trip

---

### 3. Autoencoder Statistical Arbitrage
`Market Making AS/AE smart beta/autoencoder_statarb_v2.py`

Deep learning approach to equity stat-arb: a sparse autoencoder learns a low-dimensional latent representation of cross-sectional return co-movement. Residuals (reconstruction errors) are mean-reverting by construction and tradeable as market-neutral signals.

- **Architecture**: encoder → 8-dimensional bottleneck → decoder, trained on 5 years of daily returns for S&P 500 constituents
- **Signal**: per-stock residual z-score relative to rolling 60-day distribution
- **Attribution**: latent factor decomposition to understand which encoded dimensions drive each signal
- Smart beta variant extends the approach to factor portfolios (value, momentum, low-vol)

Outputs: latent space visualisation, per-stock attribution heatmap, equity curve with drawdown profile.

---

### 4. LSTM Walk-Forward Screener
`screener v2/`

Multi-asset directional screener using a **bidirectional LSTM** trained on weekly OHLCV + technical features. Designed for medium-frequency signals (1–4 week horizon).

**Feature set (10 features, zero look-ahead):**
- Weekly returns (1w, 2w, 4w), RSI-14, MACD signal, Bollinger Band position
- Volume ratio (vs. 20-week MA), ATR-normalised range, trend strength (ADX proxy)
- All features computed strictly from past data; no future information leakage

**Validation:**
- Walk-forward: 80% rolling train, 20% OOS test, 12-week embargo between folds
- Confidence-scaled position sizing (larger position when model conviction is higher)
- Transaction costs: 10bps round-trip

**Tested on:** AAPL, GOOG, AMZN, NFLX, MRNA, GC=F (Gold), BTC-USD  
Outputs: per-asset OOS equity curves, walk-forward prediction charts.

---

### 5. Cross-Asset Correlation Signal Model (v3)
`co-correlazione/model_v3.py`

Multi-layer signal generation pipeline combining ML classification with alternative data overlays. Built for biotech/pharma sector (IBB, MRNA, LLY) but extensible to any sector.

**Signal layers:**
1. **Regime detection** — Gaussian HMM (3 states: bull/bear/high-vol) on GARCH-filtered returns
2. **ML base signal** — Gradient Boosting classifier with PCA-compressed features, isotonic calibration for probability output
3. **Options overlay** — live options chain: put/call ratio, ATM implied volatility, IV skew, term structure slope
4. **Sentiment overlay** — VADER sentiment on live news feed (yfinance.news), no API key required
5. **VIX/VVIX fear regime** — cross-asset volatility regime conditioning

**Composite signal** = ML probability × options confirmation × sentiment confirmation × regime gate  
Interactive Plotly dashboard with all signal layers visualised.

---

### 6. FVG (Fair Value Gap) Strategy
`FVG csv/FVG.ipynb`

Systematic backtest of ICT Fair Value Gap setups across multiple futures contracts: BTC, ES, NQ, FDAX, FESX, GC, NG. Fills identified programmatically from OHLC bars; entry on retest, exit at next imbalance or time stop.

Multi-instrument correlation analysis to identify regime sensitivity and portfolio diversification potential. Includes per-ticker performance breakdown and drawdown analysis.

---

### 7. Open Range Breakout (ORB)
`csv orb/orb.ipynb`

Systematic ORB on NQ (E-mini NASDAQ 100), 5-minute bars. Opening range defined as first N bars after RTH open; breakout entry on confirmed close above/below range, with Bollinger Band confirmation filter.

Parameter optimisation grid with correlation analysis across configurations to avoid overfitting to a single parameter set.

---

### 8. Sweep Retracement Strategy (NinjaTrader 8 / C#)
`NT8 strat/HourSweepRetracement.cs`  
`NT8 strat/HourSweepRetracementInverse.cs`

Live-tradeable NinjaTrader 8 strategy in C#. Detects hourly liquidity sweeps (stops hunted above/below prior hour high/low), then enters on retracement back inside the range. Both long-bias and inverse (short-bias) variants.

Designed for NQ and ES futures during RTH session. Integrated with NinjaTrader's execution engine for live order management.

---

### 9. FCF-Based Equity Screener
`FCF.ipynb` · `screener v2/screener.py`

Fundamental quant screener ranking S&P 500 stocks by Free Cash Flow yield, FCF growth consistency, and sector-neutral z-score. Backtested monthly rebalancing with 120-stock long portfolio vs. SPY benchmark.

Portfolio123-compatible output format included.

---

## Stack

| Domain | Tools |
|--------|-------|
| Language | Python 3.10, C# (.NET, NinjaTrader 8) |
| ML/DL | PyTorch (CUDA), scikit-learn, XGBoost, hmmlearn |
| Data | yfinance, Binance/Bybit WebSocket API, NinjaTrader data feed |
| Execution | Hummingbot (crypto), NinjaTrader 8 (futures) |
| Backtesting | Custom walk-forward engines with purged/embargoed CV |
| Viz | Matplotlib, Plotly, interactive HTML dashboards |
| Quant libs | statsmodels, arch (GARCH), scipy |

---

## Methodology Notes

**On look-ahead bias:** every model in this repository uses strict temporal separation between training and evaluation. For time series, this means either (a) purged + embargoed cross-validation with a gap equal to the signal horizon, or (b) a single fixed train/test split where all hyperparameters are calibrated on the training window only.

**On transaction costs:** all backtests include realistic transaction costs (5–10bps round-trip for equities/futures, 10bps for crypto) and slippage assumptions. Results reported are net of costs.

**On overfitting:** strategy selection across versions is documented (v1→v5 pairs trading, v1→v3 correlation model) to show the research process, including failures, not just the final result.

---

*Independent research. All results are backtested and walk-forward validated; past performance does not guarantee future results.*
