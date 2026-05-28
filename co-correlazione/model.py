"""
=============================================================================
  QUANT SIGNAL MODEL — Direction Prediction via Correlation + Volatility
  IBB | MRNA | LLY  |  Feb 2022 -> Feb 2026
=============================================================================
  Framework:
    - Features: rolling correlations, realized vol, momentum, spread signals
    - Model: Gradient Boosting Classifier (sklearn) + Logistic Regression baseline
    - Validation: Walk-Forward (no lookahead bias)
    - Output: feature importance, backtest equity curve, tomorrow's prediction
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TICKERS    = ["IBB", "MRNA", "LLY"]
END_DATE   = "2026-02-20"
START_DATE = "2022-02-20"

# Rolling windows for feature engineering
SHORT_WIN  = 10    # ~2 weeks
MED_WIN    = 21    # ~1 month
LONG_WIN   = 63    # ~3 months

# Walk-forward params
TRAIN_FRAC = 0.65  # first 65% for in-sample training
# remaining 35% is out-of-sample test

BG    = "#0d1117"
PANEL = "#161b22"
TEXT  = "#e6edf3"
GRID  = "#21262d"
COLORS = {"IBB": "#00d4ff", "MRNA": "#ff6b6b", "LLY": "#ffd93d"}

# ── FETCH DATA ─────────────────────────────────────────────────────────────────
print("[1/5] Fetching data from Yahoo Finance ...")
raw    = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
prices = raw["Close"][TICKERS].dropna()
ret    = np.log(prices / prices.shift(1)).dropna()
print(f"      {len(ret)} trading days | {ret.index[0].date()} -> {ret.index[-1].date()}")

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
print("[2/5] Engineering features ...")

def build_features(ret: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix.
    Features are multi-dimensional signals capturing:
      (A) Momentum / autocorrelation
      (B) Cross-asset correlations (rolling)
      (C) Realized volatility + vol-of-vol
      (D) Divergence / spread (catch-up signal)
      (E) Volatility regime (high/low)
    """
    f = pd.DataFrame(index=ret.index)

    for t in TICKERS:
        r = ret[t]

        # (A) Momentum at multiple horizons
        for w in [1, 2, 3, 5, 10, MED_WIN]:
            f[f"{t}_mom_{w}d"] = r.shift(1).rolling(w).sum()

        # (A) Lagged returns (direct AR signals)
        for lag in [1, 2, 3]:
            f[f"{t}_lag{lag}"] = r.shift(lag)

        # (C) Realized volatility
        for w in [SHORT_WIN, MED_WIN, LONG_WIN]:
            vol = r.shift(1).rolling(w).std() * np.sqrt(252)
            f[f"{t}_vol_{w}d"] = vol

        # (C) Vol-of-vol (volatility regime signal)
        f[f"{t}_volvol_{MED_WIN}d"] = (
            r.shift(1).rolling(SHORT_WIN).std()
             .rolling(MED_WIN).std() * np.sqrt(252)
        )

        # (C) Vol relative to long-window vol (normalised vol level)
        short_vol = r.shift(1).rolling(SHORT_WIN).std()
        long_vol  = r.shift(1).rolling(LONG_WIN).std()
        f[f"{t}_vol_ratio"] = short_vol / (long_vol + 1e-8)

    # (B) Rolling pairwise correlations
    pairs = [("IBB","MRNA"), ("IBB","LLY"), ("MRNA","LLY")]
    for (a, b) in pairs:
        for w in [SHORT_WIN, MED_WIN, LONG_WIN]:
            f[f"corr_{a}_{b}_{w}d"] = (
                ret[a].shift(1).rolling(w).corr(ret[b].shift(1))
            )
        # Correlation change (momentum of correlation)
        corr_short = ret[a].shift(1).rolling(SHORT_WIN).corr(ret[b].shift(1))
        corr_long  = ret[a].shift(1).rolling(LONG_WIN).corr(ret[b].shift(1))
        f[f"corr_delta_{a}_{b}"] = corr_short - corr_long

    # (D) Divergence signals: today asset moved, peer did NOT
    #     = spread = asset_today - peer_vol_adjusted_today
    #     When corr is high but spread is large -> catch-up expected
    for (a, b) in pairs:
        # raw return spread (shifted by 1 to avoid lookahead)
        f[f"spread_{a}_{b}"] = ret[a].shift(1) - ret[b].shift(1)
        # spread / combined vol (normalised divergence)
        combined_vol = (
            ret[a].shift(1).rolling(MED_WIN).std()
          + ret[b].shift(1).rolling(MED_WIN).std()
        ) / 2 + 1e-8
        f[f"spread_norm_{a}_{b}"] = f[f"spread_{a}_{b}"] / combined_vol

    # (E) Correlation regime: is current corr high (>0.5) or low?
    for (a, b) in pairs:
        corr_med = ret[a].shift(1).rolling(MED_WIN).corr(ret[b].shift(1))
        f[f"regime_{a}_{b}"] = (corr_med > 0.5).astype(int)

    # (E) Cross-asset effect: if peer moved strongly, does target follow?
    for t in TICKERS:
        peers = [x for x in TICKERS if x != t]
        for p in peers:
            f[f"peer_lag1_{p}_on_{t}"] = ret[p].shift(1)

    return f.dropna()


features = build_features(ret)
feature_names = features.columns.tolist()
print(f"      {len(feature_names)} features built across {len(features)} days")

# ── TARGETS: Next-day direction (+1 up, 0 down) ────────────────────────────────
targets = {}
for t in TICKERS:
    targets[t] = (ret[t].shift(-1) > 0).astype(int)   # tomorrow's direction

# Align everything
common_idx = features.index.intersection(
    pd.concat([targets[t].rename(t) for t in TICKERS], axis=1).dropna().index
)
X_all = features.loc[common_idx]
Y_all = pd.concat([targets[t].rename(t) for t in TICKERS], axis=1).loc[common_idx]

n_total = len(X_all)
n_train = int(n_total * TRAIN_FRAC)
train_idx = common_idx[:n_train]
test_idx  = common_idx[n_train:]
print(f"      Train: {len(train_idx)} days | Test (OOS): {len(test_idx)} days")

# ── MODEL TRAINING + EVALUATION ────────────────────────────────────────────────
print("[3/5] Training models (Gradient Boosting + Logistic baseline) ...")

results = {}      # per ticker
importances = {}  # per ticker

for ticker in TICKERS:
    y = Y_all[ticker]

    X_train = X_all.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test  = X_all.loc[test_idx]
    y_test  = y.loc[test_idx]

    # -- Logistic Regression baseline (needs scaling)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=500, C=0.1)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_pipe.predict(X_test))

    # -- Gradient Boosting Classifier
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_pred  = gb.predict(X_test)
    gb_prob  = gb.predict_proba(X_test)[:, 1]  # prob of UP
    gb_acc   = accuracy_score(y_test, gb_pred)

    # -- Backtest: simple long/short on signal
    #    +1 if model says UP (prob > 0.5), -1 if DOWN
    signal_oos = pd.Series(np.where(gb_prob > 0.5, 1, -1), index=test_idx)
    oos_returns = ret[ticker].loc[test_idx]
    strategy_ret = signal_oos.shift(1).dropna() * oos_returns.shift(-1).dropna()
    # Align
    common = strategy_ret.dropna().index
    strategy_ret = strategy_ret.loc[common]
    bh_ret       = oos_returns.loc[common]

    # Cumulative
    strat_cum = (1 + strategy_ret).cumprod()
    bh_cum    = (1 + bh_ret).cumprod()

    # Sharpe (annualized)
    sharpe = (strategy_ret.mean() / (strategy_ret.std() + 1e-8)) * np.sqrt(252)

    # Feature importances
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": gb.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)

    results[ticker] = {
        "lr_acc":    lr_acc,
        "gb_acc":    gb_acc,
        "sharpe":    sharpe,
        "strat_cum": strat_cum,
        "bh_cum":    bh_cum,
        "gb_prob":   gb_prob,
        "gb_pred":   gb_pred,
        "y_test":    y_test,
        "signal":    signal_oos,
        "gb_model":  gb,
    }
    importances[ticker] = imp_df

    print(f"      {ticker:>4}  LR acc={lr_acc:.1%}  GB acc={gb_acc:.1%}  Sharpe={sharpe:.2f}")

# ── TOMORROW'S PREDICTION ──────────────────────────────────────────────────────
print("[4/5] Generating tomorrow's prediction ...")

tomorrow_preds = {}
last_X = X_all.iloc[[-1]]  # use the last available row of features
for ticker in TICKERS:
    gb = results[ticker]["gb_model"]
    prob_up = gb.predict_proba(last_X)[0, 1]
    direction = "UP" if prob_up > 0.5 else "DOWN"
    confidence = max(prob_up, 1 - prob_up)
    tomorrow_preds[ticker] = {
        "prob_up":   prob_up,
        "direction": direction,
        "confidence": confidence,
    }
    print(f"      {ticker:>4}: {direction} (prob_UP={prob_up:.1%}, confidence={confidence:.1%})")

# ── BUILD DASHBOARD ────────────────────────────────────────────────────────────
print("[5/5] Building dashboard ...")

# ─── FIG 1: Backtest equity curves ────────────────────────────────────────────
fig1 = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        f"{t} — Strategy vs Buy&Hold  |  OOS Sharpe: {results[t]['sharpe']:.2f}  Acc: {results[t]['gb_acc']:.1%}"
        for t in TICKERS
    ],
    horizontal_spacing=0.07,
)
for j, ticker in enumerate(TICKERS, 1):
    sc = results[ticker]["strat_cum"]
    bh = results[ticker]["bh_cum"]
    fig1.add_trace(go.Scatter(
        x=sc.index, y=sc.values,
        name=f"{ticker} Strategy",
        line=dict(color=COLORS[ticker], width=2),
        hovertemplate=f"<b>{ticker} Strategy</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}x<extra></extra>",
    ), row=1, col=j)
    fig1.add_trace(go.Scatter(
        x=bh.index, y=bh.values,
        name=f"{ticker} B&H",
        line=dict(color="#484f58", width=1.5, dash="dot"),
        hovertemplate=f"<b>{ticker} B&H</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}x<extra></extra>",
    ), row=1, col=j)

fig1.update_layout(
    height=480, width=1400,
    title=dict(
        text="Out-of-Sample Backtest — Corr+Vol Signal Strategy vs Buy & Hold",
        font=dict(color=TEXT, size=18, family="Inter, sans-serif"), x=0.5,
    ),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    legend=dict(bgcolor=PANEL, bordercolor=GRID, borderwidth=1, font=dict(color=TEXT)),
    hovermode="x unified",
)
fig1.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
fig1.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
for ann in fig1.layout.annotations:
    ann.font.color = TEXT

# ─── FIG 2: Feature importance per ticker ─────────────────────────────────────
fig2 = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{t} — Top 15 Feature Importances" for t in TICKERS],
    horizontal_spacing=0.1,
)
for j, ticker in enumerate(TICKERS, 1):
    imp = importances[ticker].iloc[::-1]  # reverse for horizontal bar
    fig2.add_trace(go.Bar(
        y=imp["feature"],
        x=imp["importance"],
        orientation="h",
        marker_color=COLORS[ticker],
        marker_line_color="#0d1117",
        marker_line_width=0.5,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        name=ticker,
    ), row=1, col=j)

fig2.update_layout(
    height=560, width=1400,
    title=dict(
        text="Feature Importance — What drives each prediction?",
        font=dict(color=TEXT, size=18, family="Inter, sans-serif"), x=0.5,
    ),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    showlegend=False,
)
fig2.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
fig2.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=9))
for ann in fig2.layout.annotations:
    ann.font.color = TEXT

# ─── FIG 3: Rolling OOS accuracy over time ────────────────────────────────────
fig3 = go.Figure()

ROLL_ACC_WIN = 40  # rolling window for accuracy

for ticker in TICKERS:
    pred_series = pd.Series(results[ticker]["gb_pred"], index=test_idx)
    true_series = results[ticker]["y_test"]
    correct     = (pred_series == true_series).astype(int)
    roll_acc    = correct.rolling(ROLL_ACC_WIN).mean()

    fig3.add_trace(go.Scatter(
        x=roll_acc.index,
        y=roll_acc.values,
        mode="lines",
        name=ticker,
        line=dict(color=COLORS[ticker], width=2),
        hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>Rolling Acc: %{{y:.1%}}<extra></extra>",
    ))

fig3.add_hline(y=0.5, line_dash="dot", line_color="#888", opacity=0.5)
fig3.update_layout(
    height=400, width=1400,
    title=dict(
        text=f"Rolling {ROLL_ACC_WIN}-Day Prediction Accuracy (OOS) — 50% = random",
        font=dict(color=TEXT, size=17, family="Inter, sans-serif"), x=0.5,
    ),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    legend=dict(bgcolor=PANEL, bordercolor=GRID, borderwidth=1, font=dict(color=TEXT)),
    yaxis=dict(title="Accuracy", tickformat=".0%", range=[0.3, 0.75]),
    xaxis=dict(title="Date"),
    hovermode="x unified",
)
fig3.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
fig3.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)

# ─── FIG 4: Model probability time series (OOS) ───────────────────────────────
fig4 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                     subplot_titles=[f"{t} — P(UP tomorrow) over OOS period" for t in TICKERS],
                     vertical_spacing=0.07)

for j, ticker in enumerate(TICKERS, 1):
    prob = pd.Series(results[ticker]["gb_prob"], index=test_idx)
    actual_up = results[ticker]["y_test"] == 1

    def hex_to_rgba(h, alpha=0.12):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig4.add_trace(go.Bar(
        x=prob.index, y=actual_up.astype(int),
        marker_color="rgba(0,255,100,0.08)",
        showlegend=False, hoverinfo="skip",
    ), row=j, col=1)
    fig4.add_trace(go.Scatter(
        x=prob.index, y=prob.values,
        mode="lines", name=ticker,
        line=dict(color=COLORS[ticker], width=1.5),
        fill="tozeroy",
        fillcolor=hex_to_rgba(COLORS[ticker]),
        hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>P(UP)=%{{y:.2f}}<extra></extra>",
    ), row=j, col=1)
    fig4.add_hline(y=0.5, line_dash="dot", line_color="#888", opacity=0.4, row=j, col=1)

fig4.update_layout(
    height=700, width=1400,
    title=dict(
        text="Model Signal — Predicted P(UP) over Out-of-Sample Period",
        font=dict(color=TEXT, size=17, family="Inter, sans-serif"), x=0.5,
    ),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    showlegend=False,
)
fig4.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
fig4.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, range=[0, 1])
for ann in fig4.layout.annotations:
    ann.font.color = TEXT

# ─── TOMORROW CARD ─────────────────────────────────────────────────────────────
def gauge_color(conf):
    if conf > 0.65: return "#00ffd5"
    if conf > 0.55: return "#ffd93d"
    return "#ff6b6b"

tomorrow_html = ""
for ticker in TICKERS:
    p = tomorrow_preds[ticker]
    arrow = "&#8679;" if p["direction"] == "UP" else "&#8681;"
    dir_color = "#00ffd5" if p["direction"] == "UP" else "#ff6b6b"
    conf_bar_w = int(p["confidence"] * 100)
    g_color = gauge_color(p["confidence"])
    tomorrow_html += f"""
    <div class="pred-card">
      <div class="pred-ticker" style="color:{COLORS[ticker]}">{ticker}</div>
      <div class="pred-direction" style="color:{dir_color}">{arrow} {p['direction']}</div>
      <div class="pred-prob">P(UP) = {p['prob_up']:.1%}</div>
      <div class="conf-label">Confidence: {p['confidence']:.1%}</div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{conf_bar_w}%;background:{g_color}"></div>
      </div>
    </div>"""

# ─── ASSEMBLE HTML ─────────────────────────────────────────────────────────────
html_parts = []
for fig in [fig1, fig3, fig2, fig4]:
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

last_date = X_all.index[-1].strftime("%Y-%m-%d")

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Quant Signal Model — IBB / MRNA / LLY</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background:#0d1117; color:#e6edf3; font-family:'Inter',sans-serif; padding:32px 24px; }}

    h1 {{ text-align:center; font-size:2.2rem; font-weight:800; margin-bottom:6px;
          background:linear-gradient(135deg,#00d4ff,#a29bfe,#ffd93d);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .subtitle {{ text-align:center; color:#8b949e; font-size:0.88rem; margin-bottom:12px; }}

    .method-box {{
      max-width:860px; margin:0 auto 40px; padding:20px 28px;
      background:#161b22; border:1px solid #21262d; border-radius:12px;
      font-size:0.88rem; line-height:1.8; color:#c9d1d9;
    }}
    .method-box h3 {{ color:#00d4ff; margin-bottom:10px; font-size:1rem; }}
    .method-box ul {{ padding-left:20px; }}
    .method-box li {{ margin-bottom:4px; }}
    .method-box .tag {{
      display:inline-block; padding:2px 10px; border-radius:6px;
      font-size:0.72rem; font-weight:700; margin-right:6px;
      background:#7c3aed22; color:#a29bfe; border:1px solid #7c3aed55;
    }}

    /* Tomorrow predictions */
    .pred-section {{ margin-bottom:52px; }}
    .pred-section h2 {{
      font-size:0.95rem; text-transform:uppercase; letter-spacing:1px;
      color:#8b949e; margin-bottom:20px; padding-bottom:8px;
      border-bottom:1px solid #21262d;
    }}
    .pred-row {{ display:flex; gap:20px; justify-content:center; flex-wrap:wrap; }}
    .pred-card {{
      flex:1; min-width:220px; max-width:320px;
      background:#161b22; border:1px solid #21262d; border-radius:14px;
      padding:28px 24px; text-align:center;
      transition:transform 0.2s,box-shadow 0.2s;
    }}
    .pred-card:hover {{ transform:translateY(-4px); box-shadow:0 8px 32px rgba(0,0,0,0.4); }}
    .pred-ticker {{ font-size:1.6rem; font-weight:800; margin-bottom:10px; }}
    .pred-direction {{ font-size:2.8rem; font-weight:800; margin-bottom:8px; }}
    .pred-prob {{ font-size:0.92rem; color:#8b949e; margin-bottom:12px; }}
    .conf-label {{ font-size:0.78rem; color:#484f58; margin-bottom:6px; }}
    .conf-bar-bg {{
      height:6px; background:#21262d; border-radius:999px; overflow:hidden;
    }}
    .conf-bar-fill {{ height:100%; border-radius:999px; transition:width 0.4s; }}

    /* Chart sections */
    section {{ margin-bottom:56px; }}
    section h2 {{
      font-size:1rem; font-weight:600; color:#8b949e;
      text-transform:uppercase; letter-spacing:1px;
      margin-bottom:16px; padding-bottom:8px; border-bottom:1px solid #21262d;
    }}
    .chart-wrap {{
      background:#161b22; border:1px solid #21262d;
      border-radius:12px; overflow:hidden; padding:8px;
    }}
    footer {{ text-align:center; color:#484f58; font-size:0.75rem; margin-top:32px; }}
  </style>
</head>
<body>

  <h1>Quant Signal Model</h1>
  <p class="subtitle">IBB · MRNA · LLY &nbsp;|&nbsp; Features: Correlation + Volatility + Momentum
     &nbsp;|&nbsp; Walk-Forward OOS Validation</p>

  <!-- Method box -->
  <div class="method-box">
    <h3>How it works</h3>
    <ul>
      <li><span class="tag">Features</span> Rolling correlations (10d/21d/63d), realized vol, vol-of-vol,
          momentum (1d…21d), cross-asset divergence spreads, correlation regime flag</li>
      <li><span class="tag">Model</span> Gradient Boosting Classifier — 200 trees, depth 3,
          trained on first 65% of data (in-sample)</li>
      <li><span class="tag">Signal</span> P(UP tomorrow) > 0.5 → Long, else Short</li>
      <li><span class="tag">Validation</span> Walk-forward: predictions generated only on remaining 35%
          (out-of-sample) — no lookahead bias</li>
      <li><span class="tag">Backtest</span> 1-day holding period, no transaction costs,
          Long/Short on signal</li>
    </ul>
  </div>

  <!-- Tomorrow's predictions -->
  <div class="pred-section">
    <h2>Tomorrow's Prediction &nbsp;(signal as of {last_date})</h2>
    <div class="pred-row">
      {tomorrow_html}
    </div>
  </div>

  <section>
    <h2>01 — Out-of-Sample Backtest: Strategy vs Buy &amp; Hold</h2>
    <div class="chart-wrap">{html_parts[0]}</div>
  </section>

  <section>
    <h2>02 — Rolling 40-Day Prediction Accuracy (OOS)</h2>
    <div class="chart-wrap">{html_parts[1]}</div>
  </section>

  <section>
    <h2>03 — Feature Importance: What Drives Each Asset's Signal?</h2>
    <div class="chart-wrap">{html_parts[2]}</div>
  </section>

  <section>
    <h2>04 — Predicted P(UP) Over Out-of-Sample Period</h2>
    <div class="chart-wrap">{html_parts[3]}</div>
  </section>

  <footer>
    Quant Signal Model · Antigravity · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
    Disclaimer: educational only, not financial advice
  </footer>
</body>
</html>"""

with open("model_dashboard.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("=" * 68)
print("  TOMORROW'S SIGNAL SUMMARY")
print("=" * 68)
for ticker in TICKERS:
    p = tomorrow_preds[ticker]
    bar = "#" * int(p["confidence"] * 30) + "." * (30 - int(p["confidence"] * 30))
    print(f"  {ticker:>4}: {p['direction']:>4}  P(UP)={p['prob_up']:.1%}  [{bar}] {p['confidence']:.1%}")
print("=" * 68)
print("\n[OK] Model dashboard saved -> model_dashboard.html")
print("     Open in browser to explore backtest + feature importance.\n")
