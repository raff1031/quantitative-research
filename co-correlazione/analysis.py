"""
Co-Correlation & Cross-Correlation Analysis
IBB | MRNA | LLY - Last 4 Years (Feb 2022 -> Feb 2026)
Split into 8 x 6-month windows + lag-based cross-correlation
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS   = ["IBB", "MRNA", "LLY"]
END_DATE  = "2026-02-20"
START_DATE= "2022-02-20"
LAG_RANGE = range(-30, 31)          # cross-corr lags in trading days
COLORS    = {"IBB": "#00d4ff", "MRNA": "#ff6b6b", "LLY": "#ffd93d"}
PAIRS     = [("IBB","MRNA"), ("IBB","LLY"), ("MRNA","LLY")]

# Pair colours for cross-corr / time-series lines
PAIR_COLORS = {
    ("IBB","MRNA"): "#ff6b6b",
    ("IBB","LLY"):  "#ffd93d",
    ("MRNA","LLY"): "#a29bfe",
}

BG      = "#0d1117"
PANEL   = "#161b22"
TEXT    = "#e6edf3"
GRID    = "#21262d"

# ── FETCH DATA ────────────────────────────────────────────────────────────────
print("[*] Fetching price data ...")
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
prices = raw["Close"][TICKERS].dropna()

# Daily log-returns
returns = np.log(prices / prices.shift(1)).dropna()
print(f"    OK  {len(returns)} trading days loaded ({returns.index[0].date()} -> {returns.index[-1].date()})")

# ── SPLIT INTO 6-MONTH WINDOWS ────────────────────────────────────────────────
def six_month_windows(df):
    """Yield (label, sub_df) for each ~6-month calendar window."""
    windows = []
    start = df.index[0]
    while start < df.index[-1]:
        # advance 6 months
        month = start.month + 6
        year  = start.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        end   = pd.Timestamp(year=year, month=month, day=1)
        end   = min(end, df.index[-1] + pd.Timedelta(days=1))
        chunk = df[(df.index >= start) & (df.index < end)]
        if len(chunk) > 10:
            label = f"{start.strftime('%b %Y')} → {(end - pd.Timedelta(days=1)).strftime('%b %Y')}"
            windows.append((label, chunk))
        start = end
    return windows

windows = six_month_windows(returns)
print(f"    {len(windows)} windows created")

# ── COMPUTE CORRELATION PER WINDOW ───────────────────────────────────────────
window_labels = []
window_corrs  = []   # list of 3×3 DataFrames
pair_ts       = {p: [] for p in PAIRS}  # time-series of pairwise corr

for label, chunk in windows:
    corr = chunk.corr()
    window_labels.append(label)
    window_corrs.append(corr)
    for p in PAIRS:
        pair_ts[p].append(corr.loc[p[0], p[1]])

# ── CROSS-CORRELATION ─────────────────────────────────────────────────────────
def cross_corr(series_x, series_y, lags):
    """Pearson cross-correlation at each lag (positive lag = x leads y)."""
    results = {}
    for lag in lags:
        if lag > 0:
            # x is shifted forward → x at t-lag vs y at t → x leads y
            x_s = series_x.shift(lag)
        elif lag < 0:
            x_s = series_x.shift(lag)
        else:
            x_s = series_x
        valid = pd.concat([x_s, series_y], axis=1).dropna()
        if len(valid) < 5:
            results[lag] = np.nan
        else:
            results[lag] = valid.iloc[:,0].corr(valid.iloc[:,1])
    return results

xcorr = {}
for p in PAIRS:
    xcorr[p] = cross_corr(returns[p[0]], returns[p[1]], LAG_RANGE)

# ── BUILD PLOTLY DASHBOARD ────────────────────────────────────────────────────
print("[*] Building interactive dashboard ...")

def dark_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=20, family="Inter, sans-serif"), x=0.5),
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=60, r=60, t=80, b=60),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

# ─── FIGURE 1: Heatmaps per window ───────────────────────────────────────────
n_win  = len(windows)
ncols  = 4
nrows  = int(np.ceil(n_win / ncols))

fig1 = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=window_labels,
    horizontal_spacing=0.06,
    vertical_spacing=0.14,
)

for i, (label, corr_df) in enumerate(zip(window_labels, window_corrs)):
    row = i // ncols + 1
    col = i % ncols + 1
    z   = corr_df.values
    # Round for annotation text
    text_vals = [[f"{v:.2f}" for v in row_] for row_ in z]
    hm = go.Heatmap(
        z=z,
        x=TICKERS,
        y=TICKERS,
        text=text_vals,
        texttemplate="%{text}",
        colorscale=[
            [0.0,  "#1a0533"],
            [0.25, "#6b21a8"],
            [0.5,  "#1e3a5f"],
            [0.75, "#0ea5e9"],
            [1.0,  "#00ffd5"],
        ],
        zmin=-1, zmax=1,
        showscale=(i == n_win - 1),
        colorbar=dict(
            title=dict(text="rho", font=dict(color=TEXT)),
            tickfont=dict(color=TEXT),
            bgcolor=PANEL,
            bordercolor=GRID,
            x=1.02,
        ) if i == n_win - 1 else None,
    )
    fig1.add_trace(hm, row=row, col=col)

fig1.update_layout(
    height=350 * nrows,
    width=1400,
    title=dict(
        text="Correlation Matrix - 6-Month Windows  |  IBB . MRNA . LLY",
        font=dict(color=TEXT, size=22, family="Inter, sans-serif"),
        x=0.5,
    ),
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
)
for ann in fig1.layout.annotations:
    ann.font.color = TEXT
    ann.font.size  = 11

# ─── FIGURE 2: Pairwise correlation time-series ───────────────────────────────
fig2 = go.Figure()

for p in PAIRS:
    label_pair = f"{p[0]} ↔ {p[1]}"
    fig2.add_trace(go.Scatter(
        x=window_labels,
        y=pair_ts[p],
        mode="lines+markers",
        name=label_pair,
        line=dict(color=PAIR_COLORS[p], width=2.5),
        marker=dict(size=8, symbol="circle"),
        hovertemplate=f"<b>{label_pair}</b><br>Window: %{{x}}<br>ρ = %{{y:.3f}}<extra></extra>",
    ))

fig2.add_hline(y=0, line_dash="dot", line_color=TEXT, opacity=0.3)
fig2 = dark_layout(fig2, "Pairwise Correlation Over Time - 6-Month Windows")
fig2.update_layout(
    height=500, width=1400,
    legend=dict(
        bgcolor=PANEL, bordercolor=GRID, borderwidth=1,
        font=dict(color=TEXT),
    ),
    yaxis=dict(title="Pearson ρ", range=[-1, 1]),
    xaxis=dict(title="Period"),
)

# ─── FIGURE 3: Cross-correlation (lag analysis) ───────────────────────────────
fig3 = make_subplots(
    rows=1, cols=len(PAIRS),
    subplot_titles=[f"{p[0]} ↔ {p[1]}" for p in PAIRS],
    horizontal_spacing=0.08,
)

lag_list = list(LAG_RANGE)
for j, p in enumerate(PAIRS):
    vals = [xcorr[p][l] for l in lag_list]
    # Fill bars: positive = purple, negative = orange
    colors_bar = ["#7c3aed" if v >= 0 else "#f97316" for v in vals]
    fig3.add_trace(
        go.Bar(
            x=lag_list,
            y=vals,
            marker_color=colors_bar,
            name=f"{p[0]}↔{p[1]}",
            hovertemplate=f"<b>{p[0]} ↔ {p[1]}</b><br>Lag: %{{x}} days<br>ρ = %{{y:.3f}}<extra></extra>",
        ),
        row=1, col=j+1,
    )
    # Mark the zero-lag line
    fig3.add_vline(x=0, line_dash="dash", line_color="#00ffd5", opacity=0.6, row=1, col=j+1)

fig3.update_layout(
    height=500, width=1400,
    title=dict(
        text="Cross-Correlation (Lead/Lag Analysis) - Full 4-Year Period | Positive lag = left ticker leads",
        font=dict(color=TEXT, size=18, family="Inter, sans-serif"),
        x=0.5,
    ),
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT,  family="Inter, sans-serif"),
    showlegend=False,
)
for ann in fig3.layout.annotations:
    ann.font.color = TEXT
fig3.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_text="Lag (trading days)")
fig3.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_text="ρ", range=[-1,1])

# ─── FIGURE 4: Price + Returns overview ──────────────────────────────────────
fig4 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=["Normalised Price (base=100)", "Daily Log-Returns (IBB)"],
    vertical_spacing=0.1,
)
norm = (prices / prices.iloc[0]) * 100
for t in TICKERS:
    fig4.add_trace(go.Scatter(
        x=norm.index, y=norm[t],
        mode="lines", name=t,
        line=dict(color=COLORS[t], width=2),
    ), row=1, col=1)

fig4.add_trace(go.Bar(
    x=returns.index, y=returns["IBB"],
    marker_color=np.where(returns["IBB"] >= 0, "#00ffd5", "#ff6b6b"),
    name="IBB Returns",
), row=2, col=1)

fig4.update_layout(
    height=700, width=1400,
    title=dict(
        text="Price & Returns Overview - IBB . MRNA . LLY",
        font=dict(color=TEXT, size=20, family="Inter, sans-serif"), x=0.5,
    ),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    legend=dict(bgcolor=PANEL, bordercolor=GRID, borderwidth=1, font=dict(color=TEXT)),
)
fig4.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
fig4.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
for ann in fig4.layout.annotations:
    ann.font.color = TEXT

# ── ASSEMBLE HTML DASHBOARD ───────────────────────────────────────────────────
html_parts = []
for fig in [fig4, fig1, fig2, fig3]:
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

# ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────────
print("\n" + "="*72)
print("  PAIRWISE CORRELATION SUMMARY BY 6-MONTH WINDOW")
print("="*72)
header = f"{'Period':<32} {'IBB↔MRNA':>10} {'IBB↔LLY':>10} {'MRNA↔LLY':>10}"
print(header)
print("-"*72)
for i, label in enumerate(window_labels):
    row_vals = " ".join([f"{pair_ts[p][i]:>10.3f}" for p in PAIRS])
    print(f"{label:<32} {row_vals}")
print("="*72)

print("\n  CROSS-CORRELATION PEAK LAGS (full 4-year period)")
print("-"*72)
for p in PAIRS:
    vals  = xcorr[p]
    best_lag = max(vals, key=lambda l: abs(vals[l]))
    print(f"  {p[0]:>4} ↔ {p[1]:<4}  peak ρ={vals[best_lag]:.3f} at lag={best_lag:+d} days "
          f"({'leads' if best_lag > 0 else 'lags' if best_lag < 0 else 'simultaneous'})")
print("="*72 + "\n")

# ── WRITE OUTPUT FILE ─────────────────────────────────────────────────────────
HTML_OUT = "dashboard.html"
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Co-Correlation Dashboard — IBB · MRNA · LLY</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117;
      color: #e6edf3;
      font-family: 'Inter', sans-serif;
      padding: 32px 24px;
    }}
    h1 {{
      text-align: center;
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 4px;
      background: linear-gradient(135deg, #00d4ff, #a29bfe, #ffd93d);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{
      text-align: center;
      color: #8b949e;
      font-size: 0.9rem;
      margin-bottom: 40px;
    }}
    .pill-row {{
      display: flex;
      gap: 12px;
      justify-content: center;
      margin-bottom: 40px;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 6px 18px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.5px;
    }}
    .pill.ibb  {{ background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff55; }}
    .pill.mrna {{ background: #ff6b6b22; color: #ff6b6b; border: 1px solid #ff6b6b55; }}
    .pill.lly  {{ background: #ffd93d22; color: #ffd93d; border: 1px solid #ffd93d55; }}
    .pill.meta {{ background: #a29bfe22; color: #a29bfe; border: 1px solid #a29bfe55; }}
    section {{
      margin-bottom: 56px;
    }}
    section h2 {{
      font-size: 1.15rem;
      font-weight: 600;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid #21262d;
    }}
    .chart-wrap {{
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 12px;
      overflow: hidden;
      padding: 8px;
    }}
    footer {{
      text-align: center;
      color: #484f58;
      font-size: 0.78rem;
      margin-top: 32px;
    }}
  </style>
</head>
<body>
  <h1>Co-Correlation Dashboard</h1>
  <p class="subtitle">IBB · MRNA · LLY &nbsp;|&nbsp; Feb 2022 → Feb 2026 &nbsp;|&nbsp; Daily Log-Returns · 6-Month Windows</p>

  <div class="pill-row">
    <span class="pill ibb">IBB — iShares Biotech ETF</span>
    <span class="pill mrna">MRNA — Moderna</span>
    <span class="pill lly">LLY — Eli Lilly</span>
    <span class="pill meta">8 × 6-Month Windows</span>
    <span class="pill meta">±30-Day Cross-Correlation</span>
  </div>

  <section>
    <h2>01 — Price &amp; Returns Overview</h2>
    <div class="chart-wrap">{html_parts[0]}</div>
  </section>

  <section>
    <h2>02 — Correlation Heatmaps by 6-Month Window</h2>
    <div class="chart-wrap">{html_parts[1]}</div>
  </section>

  <section>
    <h2>03 — Pairwise Correlation Evolution</h2>
    <div class="chart-wrap">{html_parts[2]}</div>
  </section>

  <section>
    <h2>04 — Cross-Correlation (Lead / Lag Analysis)</h2>
    <div class="chart-wrap">{html_parts[3]}</div>
  </section>

  <footer>Generated by Antigravity · {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} UTC</footer>
</body>
</html>"""

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"[OK] Dashboard saved -> {HTML_OUT}")
print("     Open it in any browser to explore the interactive charts.\n")
