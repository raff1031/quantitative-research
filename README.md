# Quantitative Research & Algorithmic Trading

**Romano Raffaele** · MSc Data Science candidate, Università degli Studi di Milano-Bicocca (graduating Oct 2026)  
📧 romano.raff10@gmail.com · 📍 Milano, Italy · 🔗 [github.com/raff1031/quantitative-research](https://github.com/raff1031/quantitative-research)

---

## Overview

Original quantitative research and production-grade trading systems built independently from 2024–2026. Projects span systematic futures strategies, crypto market making, statistical arbitrage, and ML-driven alpha generation — covering the full pipeline from raw data ingestion to live execution.

All strategies include out-of-sample validation. Look-ahead bias is explicitly controlled via purged/embargoed cross-validation or strict train/test splits calibrated on training data only.

---

## Backtest Results Summary

![Strategy Comparison](charts/strategy_comparison.png)

![FVG Breakdown](charts/fvg_breakdown.png)

![Strategy Radar](charts/strategy_radar.png)

| Strategy | Instruments | Trades | Sharpe | Win Rate | Profit Factor | Max DD |
|----------|-------------|--------|--------|----------|---------------|--------|
| **FVG (Fair Value Gap)** | BTC, ES, NQ, FDAX, FESX, GC, NG | 6,222 | 1.25 | 44.6% | 1.30 | -4.9% |
| **ORB (Open Range Breakout)** | NQ | 923 | 1.55 | 64.4% | 1.30 | -33.2% |
| **Sweep Retracement** | NQ (6–7h) | 399 | **2.45** | **60.4%** | 1.56 | -21.1% |
| **Sweep Retracement** | NQ (9–16h) | 693 | 1.35 | 48.2% | 1.25 | -20.8% |
| **Pattern 5** | ES, NQ | 482 | 1.53 | 52.5% | 1.38 | -22.2% |
| **MASRS (HMM + TFT)** | BTC/ETH Perps | OOS | **0.93** | — | — | -13.4% |

*All results net of transaction costs. FVG/ORB/Sweep/Pattern 5 from NinjaTrader 8 live data exports. MASRS walk-forward OOS on Binance/Bybit.*

---

## FVG Strategy — Per-Instrument Breakdown

| Instrument | Trades | Net Profit | Avg Trade | Sharpe | Win Rate |
|------------|--------|-----------|-----------|--------|----------|
| BTC | 648 | €323,209 | €498.8 | 1.82 | 40.7% |
| FDAX | 1,140 | €227,832 | €199.9 | 1.24 | 50.2% |
| NQ | 824 | €136,075 | €165.1 | 1.95 | 43.1% |
| ES | 1,322 | €120,721 | €91.3 | 1.65 | 32.7% |
| GC (Gold) | 952 | €108,724 | €114.2 | 1.44 | 52.0% |
| FESX | 479 | €20,689 | €43.2 | 2.08 | 55.9% |
| NG (Nat Gas) | 857 | €20,535 | €23.96 | 1.24 | 45.6% |

---

## Projects

### 1. Avellaneda-Stoikov Market Maker with Anti-Adverse Selection
`Market Making AS/as_anti_adverse_selection.py`

Production-ready market making strategy on Hummingbot (ScriptStrategyBase V2), implementing the Avellaneda-Stoikov (2008) stochastic control framework with four layers of adversarial flow protection:

- **Reservation price + optimal spread** from inventory risk γ, volatility σ, and order arrival intensity κ
- **EMA trend filter** with directional skewing — quote asymmetrically in trending conditions
- **Order Book Imbalance (OBI)** — bid/ask queue depth ratio triggers quote withdrawal before pump/dump events
- **Kill-switch** — extreme volatility (ATR threshold) suspends quoting entirely

Deployed on Hummingbot for crypto spot/perp markets.

---

### 2. Pairs Trading — Walk-Forward Intra-Sector (v5)
`Market Making AS/pairs_trading_v5.py`

Five-generation research evolution. v1–v4 failures documented to show methodology:

| Version | Problem | Outcome |
|---------|---------|---------|
| v1 | KO/PEP not cointegrated | Losses |
| v3 | 2 trades only | No significance |
| v4 | Cross-sector pairs | 609 trades, -12.5% |
| **v5** | Intra-sector + OOS calibration | ✅ Profitable OOS |

**v5 methodology:** Engle-Granger cointegration on intra-sector pairs only · 60/40 train/test split · Kalman filter hedge ratio (δ=1e-5) · Log-price spread · Entry z>1.5, exit z=0, stop z>3.5 · 5bps TC

---

### 3. Autoencoder Statistical Arbitrage
`Market Making AS/AE smart beta/autoencoder_statarb_v2.py`

Sparse autoencoder learns low-dimensional latent representation of S&P 500 cross-sectional return co-movement. Residuals (reconstruction errors) are mean-reverting by construction.

- **Architecture**: encoder → 8-dim bottleneck → decoder, trained on 5 years of daily returns
- **Signal**: per-stock residual z-score vs. rolling 60-day distribution
- **Attribution**: latent factor decomposition identifying which dimensions drive each signal
- Smart beta variant extends to factor portfolios (value, momentum, low-vol)

---

### 4. LSTM Walk-Forward Screener
`screener v2/`

Bidirectional LSTM trained on weekly OHLCV + 10 technical features (RSI-14, MACD, BB position, volume ratio, ATR range, ADX, multi-period returns). All features strictly look-ahead free.

**Validation:** 80% rolling train / 20% OOS / 12-week embargo between folds · Confidence-scaled position sizing · 10bps TC  
**Tested on:** AAPL, GOOG, AMZN, NFLX, MRNA, GC=F, BTC-USD  
**Hardware:** NVIDIA RTX 4050 (6.4GB VRAM), PyTorch CUDA

> **Methodology note:** A self-audit (`screener v2/AUDIT_REPORT.md`) identified a lookahead bias in the position sizing normalisation (`np.std(predictions)` computed over the full test set). The issue was documented and a fix proposed (use validation-set std instead). This is included in the repo as an example of the auditing discipline applied to all models.

---

### 5. Cross-Asset Correlation Signal Model (v3)
`co-correlazione/model_v3.py`

Multi-layer composite signal for biotech/pharma sector (IBB, MRNA, LLY):

1. **GARCH + HMM** — volatility-filtered 3-state regime detection (bull/bear/high-vol)
2. **GBM classifier** — PCA-compressed features, isotonic calibration
3. **Options overlay** — live P/C ratio, ATM-IV, IV skew, term structure slope
4. **Sentiment overlay** — VADER on yfinance.news (no API key)
5. **VIX/VVIX regime conditioning**

Interactive Plotly dashboard included.

---

### 6. FVG (Fair Value Gap) Strategy
`FVG csv/FVG.ipynb`

Systematic backtest of ICT Fair Value Gap setups across 7 futures: BTC, ES, NQ, FDAX, FESX, GC, NG. **6,222 trades total, Sharpe 1.25, profit factor 1.30.** NQ achieves the highest Sharpe (1.95) and BTC the highest avg trade (€498.8).

---

### 7. Open Range Breakout (ORB)
`csv orb/orb.ipynb`

Systematic ORB on NQ 5-minute bars with Bollinger Band confirmation. **923 trades, Sharpe 1.55, 64.4% win rate.** Parameter optimisation grid with correlation analysis across configurations.

---

### 8. Sweep Retracement (NinjaTrader 8 / C#)
`NT8 strat/HourSweepRetracement.cs`

Live NinjaTrader 8 C# strategy detecting hourly liquidity sweeps and entering on retracement. **Best configuration: Sharpe 2.45, win rate 60.4%, profit factor 1.56** (NQ 6–7h window). Both long-bias and inverse variants included.

---

### 9. Equity Screener — Multi-Approach Backtest
`backtest_results/`

![Screener Approaches](charts/screener_approaches.png)

Three systematic long-only approaches tested on S&P 500 universe (Oct 2025, 103 tickers):

| Approach | Trades | Win Rate | Sharpe | Total PnL |
|----------|--------|----------|--------|-----------|
| **A (v2)** | 662 | 71.1% | **3.71** | $7,164 |
| **B (v2)** | 501 | 72.7% | **3.73** | $5,649 |
| D (v2) | 1,744 | 58.8% | -0.91 | -$2,026 |

Approach D (high-frequency signals on short holding periods) included deliberately as a negative result — showing that signal quality degrades with over-trading. Approach A and B (medium-frequency, fundamentals-augmented) achieve high win rates and Sharpe ratios above 3.5 net of 10bps commission.

---

### 10. FCF-Based Equity Screener
`FCF.ipynb` · `screener v2/screener.py`

S&P 500 screener ranking by FCF yield and growth consistency. Monthly rebalancing, 120-stock long portfolio. Portfolio123-compatible output.

---

## Stack

| Domain | Tools |
|--------|-------|
| Language | Python 3.10, C# (.NET / NinjaTrader 8) |
| ML/DL | PyTorch (CUDA), scikit-learn, XGBoost, hmmlearn |
| Market Making | Hummingbot ScriptStrategyBase V2 |
| Data | yfinance, Binance/Bybit WebSocket API, NinjaTrader data feed |
| Execution | Hummingbot (crypto), NinjaTrader 8 (futures), custom asyncio engine |
| Validation | Custom walk-forward engines, purged/embargoed CV |
| Viz | Matplotlib, Plotly, interactive HTML dashboards |
| Quant libs | statsmodels, arch (GARCH), scipy, vaderSentiment |

---

## Methodology Notes

**Look-ahead bias:** All models use either (a) purged + embargoed CV with gap equal to signal horizon, or (b) fixed train/test split where all hyperparameters are calibrated on training only and applied frozen to OOS.

**Transaction costs:** All backtests net of realistic costs: 5–10bps round-trip for equities/futures, 10bps for crypto.

**Research transparency:** Strategy evolution is documented (v1→v5 pairs trading, v1→v3 correlation model) including failures, not just the final result.

---

*Independent research. All results are backtested and walk-forward validated; past performance does not guarantee future results.*
