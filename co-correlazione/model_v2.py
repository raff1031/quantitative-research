"""
=============================================================================
  QUANT SIGNAL MODEL v2 — Professional Grade
  IBB | MRNA | LLY  |  Feb 2022 -> Feb 2026
=============================================================================
  Upgrades over v1:
    [1] GARCH(1,1) per-asset conditional volatility  (arch library)
    [2] DCC-style dynamic conditional correlation     (Engle 2002)
    [3] HMM 3-state regime detection                 (hmmlearn)
    [4] PCA factor decomposition: systematic vs idio  (sklearn)
    [5] Expanding walk-forward (retrain every 21d)   (no lookahead)
    [6] Calibrated probabilities (Platt scaling)
    [7] Full risk metrics: Sharpe, Sortino, Calmar, MaxDD
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# Quant tools
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

# Viz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TICKERS     = ["IBB", "MRNA", "LLY"]
START_DATE  = "2022-02-20"
END_DATE    = "2026-02-20"
N_REGIMES   = 3          # HMM states: bear / sideways / bull
RETRAIN_FREQ = 21        # retrain model every 21 trading days (walk-forward)
MIN_TRAIN   = 200        # minimum days before first prediction
GARCH_P, GARCH_Q = 1, 1

BG    = "#0d1117"
PANEL = "#161b22"
TEXT  = "#e6edf3"
GRID  = "#21262d"
COLORS = {"IBB": "#00d4ff", "MRNA": "#ff6b6b", "LLY": "#ffd93d"}
REGIME_COLORS = ["#ff6b6b", "#ffd93d", "#00ffd5"]  # Bear / Sideways / Bull

# ── FETCH DATA ─────────────────────────────────────────────────────────────────
print("[1/7] Fetching data ...")
raw    = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
prices = raw["Close"][TICKERS].dropna()
ret    = np.log(prices / prices.shift(1)).dropna()
print(f"      {len(ret)} trading days  |  {ret.index[0].date()} -> {ret.index[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — GARCH(1,1) CONDITIONAL VOLATILITY
# Fits a GARCH model to each asset's return series.
# Output: sigma_t for each day (predicted conditional std for next day)
# This is FAR more accurate than rolling std for real vol forecasting.
# ═══════════════════════════════════════════════════════════════════════════════
print("[2/7] Fitting GARCH(1,1) models ...")

garch_vol = {}     # conditional vol per ticker (same index as ret)
garch_resid = {}   # standardised residuals: e_t = r_t / sigma_t

for t in TICKERS:
    r_scaled = ret[t] * 100        # scale to % for GARCH numerical stability
    am = arch_model(r_scaled, vol="Garch", p=GARCH_P, q=GARCH_Q,
                    dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    # conditional_volatility is sigma_t (in-sample fitted)
    sigma = res.conditional_volatility / 100   # back to decimal
    sigma.index = ret.index
    garch_vol[t]   = sigma
    garch_resid[t] = ret[t] / (sigma + 1e-8)
    print(f"      {t:>4}  omega={res.params['omega']:.4f}  "
          f"alpha[1]={res.params['alpha[1]']:.3f}  "
          f"beta[1]={res.params['beta[1]']:.3f}  "
          f"persistence={(res.params['alpha[1]']+res.params['beta[1]']):.3f}")

garch_vol_df  = pd.DataFrame(garch_vol)
garch_resid_df = pd.DataFrame(garch_resid)

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — DCC-STYLE DYNAMIC CONDITIONAL CORRELATION
# Engle (2002): standardise residuals via GARCH, then model their rolling
# covariance with exponential weighting (EWMA) -> Q_t -> R_t (corr matrix)
# We use lambda=0.94 (RiskMetrics standard)
# ═══════════════════════════════════════════════════════════════════════════════
print("[3/7] Computing DCC-style dynamic correlations ...")

LAMBDA = 0.94   # EWMA decay (0.94 = RiskMetrics daily)

def ewma_cov(e: pd.DataFrame, lam: float) -> list:
    """Exponentially weighted covariance matrices."""
    n, k = e.shape
    Q = [e.values[:21].T @ e.values[:21] / 21]  # seed with sample cov
    for i in range(1, n):
        e_t = e.values[i].reshape(-1, 1)
        Q.append(lam * Q[-1] + (1 - lam) * e_t @ e_t.T)
    return Q

Q_list = ewma_cov(garch_resid_df, LAMBDA)

def Q_to_R(Q: np.ndarray) -> np.ndarray:
    """Normalise Q to correlation matrix R."""
    d = np.sqrt(np.diag(Q))
    D_inv = np.diag(1.0 / (d + 1e-10))
    return D_inv @ Q @ D_inv

R_list = [Q_to_R(Q) for Q in Q_list]

# Extract pairwise DCC series
pairs = [("IBB","MRNA"), ("IBB","LLY"), ("MRNA","LLY")]
pair_idx = {
    ("IBB","MRNA"): (0,1),
    ("IBB","LLY"):  (0,2),
    ("MRNA","LLY"): (1,2),
}
dcc_corr = {}
for p, (i, j) in pair_idx.items():
    dcc_corr[p] = pd.Series(
        [R_list[t][i, j] for t in range(len(R_list))],
        index=garch_resid_df.index,
    )

dcc_df = pd.DataFrame(dcc_corr)
print(f"      DCC computed. Last corr matrix:\n{np.round(R_list[-1], 3)}")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — HIDDEN MARKOV MODEL (3-STATE REGIME DETECTION)
# Fits a Gaussian HMM on [IBB_ret, vol_regime] to detect latent states:
#   State 0 = Bear  (negative drift, high vol)
#   State 1 = Sideways (flat, moderate vol)
#   State 2 = Bull  (positive drift, low vol)
# ═══════════════════════════════════════════════════════════════════════════════
print("[4/7] Fitting HMM regime detector ...")

# Features for HMM: IBB return + GARCH vol + overall vol
hmm_features = np.column_stack([
    ret["IBB"].values,
    garch_vol["IBB"].values,
    (garch_vol_df.mean(axis=1)).values,
])
hmm_features = (hmm_features - hmm_features.mean(0)) / (hmm_features.std(0) + 1e-8)

hmm = GaussianHMM(n_components=N_REGIMES, covariance_type="full",
                  n_iter=500, random_state=42)
hmm.fit(hmm_features)
regime_raw = hmm.predict(hmm_features)

# Re-label regimes by IBB mean return within each state (0=worst, 2=best)
regime_series = pd.Series(regime_raw, index=ret.index)
mean_ret_by_regime = {
    s: ret["IBB"][regime_series == s].mean()
    for s in range(N_REGIMES)
}
sorted_states = sorted(mean_ret_by_regime, key=mean_ret_by_regime.get)
remap = {orig: new for new, orig in enumerate(sorted_states)}
regime = regime_series.map(remap)   # 0=bear, 1=sideways, 2=bull

regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
for s in range(N_REGIMES):
    mask = regime == s
    print(f"      Regime {s} [{regime_names[s]:>9}]: "
          f"{mask.sum():3d} days  "
          f"IBB daily mu={ret['IBB'][mask].mean()*100:+.3f}%  "
          f"IBB daily sigma={ret['IBB'][mask].std()*100:.3f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — PCA FACTOR DECOMPOSITION
# Decompose returns into:
#   PC1 = "biotech systematic factor" (the mode all three co-move on)
#   PC2 = "LLY vs MRNA" divergence factor
# Features: PC scores, loadings, idiosyncratic residuals per asset
# ═══════════════════════════════════════════════════════════════════════════════
print("[5/7] PCA factor decomposition ...")

# We'll fit PCA on a rolling 63-day window to capture evolving factor structure
rolling_pc1 = pd.Series(index=ret.index, dtype=float)
rolling_pc1_load = pd.DataFrame(index=ret.index, columns=TICKERS, dtype=float)
idio = pd.DataFrame(index=ret.index, columns=TICKERS, dtype=float)

for i in range(63, len(ret)):
    window = ret.iloc[i-63:i]
    scaler = StandardScaler()
    X_w = scaler.fit_transform(window)
    pca = PCA(n_components=1)
    pc_scores = pca.fit_transform(X_w)
    loadings   = pca.components_[0]
    # PC score for the current day (the last day in window)
    rolling_pc1.iloc[i] = pc_scores[-1, 0]
    for k, t in enumerate(TICKERS):
        rolling_pc1_load.loc[ret.index[i], t] = loadings[k]
        # idiosyncratic: actual return minus systematiccomponent
        systematic_ret = pc_scores[-1, 0] * loadings[k] * window.std()[t]
        idio.loc[ret.index[i], t] = window.iloc[-1][t] - systematic_ret

rolling_pc1 = rolling_pc1.dropna()
idio = idio.dropna()

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — FEATURE MATRIX v2
# Combines all signals: GARCH vol, DCC corr, HMM regime, PCA, momentum
# ═══════════════════════════════════════════════════════════════════════════════
print("[6/7] Building v2 feature matrix + expanding walk-forward ...")

SHORT, MED, LONG = 10, 21, 63

def build_features_v2(ret, garch_vol_df, dcc_df, regime, rolling_pc1, idio):
    f = pd.DataFrame(index=ret.index)

    for t in TICKERS:
        r = ret[t]

        # === Momentum (lagged returns) ===
        for lag in [1, 2, 3, 5]:
            f[f"{t}_lag{lag}"] = r.shift(lag)
        for w in [5, MED, LONG]:
            f[f"{t}_mom_{w}d"] = r.shift(1).rolling(w).mean()

        # === GARCH conditional vol (the real upgrade over rolling std) ===
        gv = garch_vol_df[t]
        f[f"{t}_garch_vol"]     = gv.shift(1)
        f[f"{t}_garch_vol_chg"] = gv.diff(1).shift(1)  # vol acceleration
        # Vol z-score: how extreme is today's vol vs its own history?
        f[f"{t}_vol_zscore"] = (
            (gv - gv.rolling(LONG).mean()) / (gv.rolling(LONG).std() + 1e-8)
        ).shift(1)

        # === Idiosyncratic component (PCA residual) ===
        if t in idio.columns:
            f[f"{t}_idio"]      = idio[t].shift(1)
            f[f"{t}_idio_mom5"] = idio[t].shift(1).rolling(5).mean()

    # === DCC dynamic correlations ===
    for p in pairs:
        key = f"dcc_{p[0]}_{p[1]}"
        dcc = dcc_df[p].shift(1)
        f[key]                   = dcc
        f[f"{key}_chg"]          = dcc.diff(1)       # corr velocity
        f[f"{key}_zscore"]       = (
            (dcc - dcc.rolling(LONG).mean()) / (dcc.rolling(LONG).std() + 1e-8)
        )

    # === HMM regime (lagged by 1, encoded as dummies) ===
    for s in range(N_REGIMES):
        f[f"regime_{s}"] = (regime.shift(1) == s).astype(float)
    f["regime_val"] = regime.shift(1)

    # === PCA systematic factor ===
    f["pc1_score"]     = rolling_pc1.shift(1)
    f["pc1_score_mom"] = rolling_pc1.shift(1).rolling(5).mean()

    # === Cross-asset divergence signals ===
    # Large divergence in high-corr regime -> catch-up
    for (a, b) in pairs:
        spread = (ret[a] - ret[b]).shift(1)
        dcc_ab = dcc_df[(a, b)].shift(1)
        f[f"div_{a}_{b}"]          = spread
        # Weighted by correlation: large signal when corr is high
        f[f"div_corr_{a}_{b}"]     = spread * dcc_ab
        f[f"div_abs_{a}_{b}"]      = spread.abs()

    # === Regime-conditional vol spread ===
    # Vol spread between asset and IBB (benchmark)
    for t in ["MRNA", "LLY"]:
        f[f"vol_vs_ibb_{t}"] = (
            garch_vol_df[t] - garch_vol_df["IBB"]
        ).shift(1)

    return f.dropna()


features = build_features_v2(ret, garch_vol_df, dcc_df, regime, rolling_pc1, idio)
feature_names = features.columns.tolist()
print(f"      {len(feature_names)} features  |  {len(features)} usable days")

# Targets: next-day direction
targets = {}
for t in TICKERS:
    targets[t] = (ret[t].shift(-1) > 0).astype(int)

common_idx = features.index.intersection(
    pd.concat([targets[t].rename(t) for t in TICKERS], axis=1).dropna().index
)
X_all = features.loc[common_idx]
Y_all = pd.concat([targets[t].rename(t) for t in TICKERS], axis=1).loc[common_idx]

# ─── EXPANDING WALK-FORWARD ────────────────────────────────────────────────────
# Start predictions after MIN_TRAIN days; retrain every RETRAIN_FREQ days.
# At each retraining point, use ALL data up to that point (expanding window).
# This mirrors real production deployment.

oos_results = {t: {"idx": [], "prob": [], "pred": [], "true": []} for t in TICKERS}

# GBM hyperparams — same for all tickers (could be tuned per ticker)
GBM_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.75, min_samples_leaf=8,
    max_features=0.6, random_state=42,
)

n_total  = len(common_idx)
retrain_points = list(range(MIN_TRAIN, n_total, RETRAIN_FREQ))

scaler_cache = {t: None for t in TICKERS}
model_cache  = {t: None for t in TICKERS}   # calibrated models
importances  = {t: None for t in TICKERS}

for step_i, train_end in enumerate(retrain_points):
    train_X = X_all.iloc[:train_end]
    train_Y = Y_all.iloc[:train_end]

    # Determine prediction window (until next retrain or end)
    if step_i + 1 < len(retrain_points):
        pred_end = retrain_points[step_i + 1]
    else:
        pred_end = n_total
    test_X  = X_all.iloc[train_end:pred_end]
    test_idx = common_idx[train_end:pred_end]

    if len(test_X) == 0:
        continue

    for t in TICKERS:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(train_X)
        X_te_s = sc.transform(test_X)

        base_clf = GradientBoostingClassifier(**GBM_PARAMS)
        # Calibrate: wraps GBM and returns true probabilities (Platt scaling)
        clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        clf.fit(X_tr_s, train_Y[t])

        prob = clf.predict_proba(X_te_s)[:, 1]
        pred = (prob > 0.5).astype(int)

        oos_results[t]["idx"].extend(test_idx.tolist())
        oos_results[t]["prob"].extend(prob.tolist())
        oos_results[t]["pred"].extend(pred.tolist())
        oos_results[t]["true"].extend(Y_all[t].iloc[train_end:pred_end].tolist())

        # Save last trained model + importances
        model_cache[t]  = (clf, sc)
        scaler_cache[t] = sc
        # Feature importances from the base GBM inside calibrated clf
        importances[t]  = GradientBoostingClassifier(**GBM_PARAMS).fit(
            X_tr_s, train_Y[t]).feature_importances_

    if step_i % 5 == 0:
        pct = train_end / n_total * 100
        print(f"      [{pct:4.0f}%] train_end={train_end:3d}  pred_window={len(test_X):2d}d", end="\r")

print("\n      Expanding walk-forward complete.")

# ─── AGGREGATE METRICS ────────────────────────────────────────────────────────
def risk_metrics(strategy_ret: pd.Series) -> dict:
    annual = np.sqrt(252)
    mu     = strategy_ret.mean()
    sigma  = strategy_ret.std() + 1e-8
    sharpe = (mu / sigma) * annual
    neg    = strategy_ret[strategy_ret < 0]
    sortino = (mu / (neg.std() + 1e-8)) * annual
    cumret = (1 + strategy_ret).cumprod()
    peak   = cumret.cummax()
    dd     = (cumret - peak) / (peak + 1e-8)
    max_dd = dd.min()
    total_ret = cumret.iloc[-1] - 1
    calmar = (total_ret / (abs(max_dd) + 1e-8)) / (len(strategy_ret) / 252)
    return dict(sharpe=sharpe, sortino=sortino, max_dd=max_dd,
                total_ret=total_ret, calmar=calmar)

summary = {}
for t in TICKERS:
    oos = oos_results[t]
    idx_  = pd.DatetimeIndex(oos["idx"])
    prob_ = np.array(oos["prob"])
    pred_ = np.array(oos["pred"])
    true_ = np.array(oos["true"])
    acc   = accuracy_score(true_, pred_)
    ll    = log_loss(true_, prob_)

    # Long/Short strategy: +1 if signal UP, -1 if DOWN
    signal = pd.Series(np.where(pred_ == 1, 1, -1), index=idx_)
    oos_ret_asset = ret[t].reindex(idx_)
    strat_ret = (signal * oos_ret_asset).dropna()
    bh_ret    = oos_ret_asset.dropna()

    strat_cum = (1 + strat_ret).cumprod()
    bh_cum    = (1 + bh_ret).cumprod()
    metrics   = risk_metrics(strat_ret)

    # Rolling 30-day accuracy
    roll_acc = pd.Series((pred_ == true_).astype(float), index=idx_).rolling(30).mean()

    summary[t] = dict(
        acc=acc, ll=ll, metrics=metrics,
        strat_cum=strat_cum, bh_cum=bh_cum,
        roll_acc=roll_acc,
        prob=pd.Series(prob_, index=idx_),
    )

    print(f"      {t:>4}  ACC={acc:.1%}  LogLoss={ll:.3f}  "
          f"Sharpe={metrics['sharpe']:.2f}  Sortino={metrics['sortino']:.2f}  "
          f"MaxDD={metrics['max_dd']:.1%}  Calmar={metrics['calmar']:.2f}")

# ── TOMORROW'S PREDICTION ──────────────────────────────────────────────────────
tomorrow_preds = {}
last_X_raw = X_all.iloc[[-1]]
for t in TICKERS:
    clf, sc = model_cache[t]
    x_s = sc.transform(last_X_raw)
    prob_up = clf.predict_proba(x_s)[0, 1]
    direction = "UP" if prob_up > 0.5 else "DOWN"
    conf = max(prob_up, 1 - prob_up)
    curr_regime = int(regime.iloc[-1])
    tomorrow_preds[t] = dict(
        prob_up=prob_up, direction=direction, conf=conf,
        regime=curr_regime, regime_name=regime_names[curr_regime],
        garch_vol=garch_vol_df[t].iloc[-1],
        dcc_ibb=dcc_df.get(("IBB", t), dcc_df.get((t, "IBB"), pd.Series([np.nan]))).iloc[-1],
    )

print("\n  TOMORROW'S SIGNALS (calibrated):")
print("  " + "="*60)
for t, p in tomorrow_preds.items():
    bar = "#" * int(p["conf"] * 25) + "." * (25 - int(p["conf"] * 25))
    print(f"  {t:>4}: {p['direction']:>4}  P(UP)={p['prob_up']:.1%}  "
          f"conf=[{bar}] {p['conf']:.0%}  "
          f"Regime={p['regime_name']}  GARCH_vol={p['garch_vol']*100:.2f}%/day")
print("  " + "="*60)

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
def hex_to_rgba(h, a=0.15):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def dark_fig(title=None, height=500, width=1400):
    fig = go.Figure()
    fig.update_layout(
        height=height, width=width,
        title=dict(text=title or "", font=dict(color=TEXT, size=17,
                   family="Inter, sans-serif"), x=0.5),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        legend=dict(bgcolor=PANEL, bordercolor=GRID, borderwidth=1,
                    font=dict(color=TEXT)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

# ─── FIG A: GARCH conditional vol vs rolling std ──────────────────────────────
figA = make_subplots(rows=1, cols=3,
                     subplot_titles=[f"{t} — GARCH Vol vs Rolling Std" for t in TICKERS],
                     horizontal_spacing=0.07)
for j, t in enumerate(TICKERS, 1):
    roll_std = ret[t].rolling(21).std() * np.sqrt(252)
    figA.add_trace(go.Scatter(
        x=garch_vol_df.index, y=garch_vol_df[t] * np.sqrt(252),
        name=f"{t} GARCH", line=dict(color=COLORS[t], width=2),
        hovertemplate=f"<b>{t} GARCH</b><br>%{{y:.1%}}<extra></extra>",
    ), row=1, col=j)
    figA.add_trace(go.Scatter(
        x=roll_std.index, y=roll_std.values,
        name=f"{t} Rolling", line=dict(color="#484f58", width=1.3, dash="dot"),
        hovertemplate=f"<b>{t} Rolling</b><br>%{{y:.1%}}<extra></extra>",
    ), row=1, col=j)
figA.update_layout(
    height=420, width=1400,
    title=dict(text="GARCH(1,1) Conditional Volatility vs Rolling Std (annualised)",
               font=dict(color=TEXT, size=17, family="Inter, sans-serif"), x=0.5),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
)
figA.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
figA.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickformat=".0%")
for ann in figA.layout.annotations:
    ann.font.color = TEXT

# ─── FIG B: DCC dynamic correlations vs rolling Pearson ───────────────────────
figB = make_subplots(rows=1, cols=3,
                     subplot_titles=[f"{p[0]} vs {p[1]}" for p in pairs],
                     horizontal_spacing=0.07)
for j, p in enumerate(pairs, 1):
    rolling_p = ret[p[0]].rolling(21).corr(ret[p[1]])
    pair_color = ["#ff6b6b","#ffd93d","#a29bfe"][j-1]
    figB.add_trace(go.Scatter(
        x=dcc_df.index, y=dcc_df[p],
        name=f"DCC {p[0]}/{p[1]}", line=dict(color=pair_color, width=2),
        hovertemplate=f"<b>DCC</b><br>%{{y:.3f}}<extra></extra>",
    ), row=1, col=j)
    figB.add_trace(go.Scatter(
        x=rolling_p.index, y=rolling_p.values,
        name=f"Rolling {p[0]}/{p[1]}", line=dict(color="#484f58", width=1.3, dash="dot"),
        hovertemplate=f"<b>Rolling</b><br>%{{y:.3f}}<extra></extra>",
    ), row=1, col=j)
    figB.add_hline(y=0, line_dash="dot", line_color="#666", opacity=0.4, row=1, col=j)
figB.update_layout(
    height=420, width=1400,
    title=dict(text="DCC Dynamic Conditional Correlation vs Rolling Pearson (21d)",
               font=dict(color=TEXT, size=17, family="Inter, sans-serif"), x=0.5),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
)
figB.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
figB.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, range=[-1, 1])
for ann in figB.layout.annotations:
    ann.font.color = TEXT

# ─── FIG C: HMM Regime timeline ───────────────────────────────────────────────
figC = dark_fig("HMM 3-State Regime Detection — IBB with Regime Shading", height=420)
figC.add_trace(go.Scatter(
    x=ret.index, y=(prices["IBB"] / prices["IBB"].iloc[0]) * 100,
    name="IBB (norm 100)", line=dict(color=COLORS["IBB"], width=1.8),
    hovertemplate="IBB: %{y:.1f}<extra></extra>",
))

for s in range(N_REGIMES):
    mask = regime == s
    # Add vertical bands for each regime period
    changes = mask.astype(int).diff().fillna(0)
    starts = ret.index[changes == 1].tolist()
    ends   = ret.index[changes == -1].tolist()
    if mask.iloc[0]:
        starts = [ret.index[0]] + starts
    if mask.iloc[-1]:
        ends = ends + [ret.index[-1]]
    for st, en in zip(starts, ends):
        figC.add_vrect(
            x0=st, x1=en,
            fillcolor=REGIME_COLORS[s], opacity=0.12,
            layer="below", line_width=0,
        )
for s in range(N_REGIMES):
    figC.add_trace(go.Scatter(
        x=[None], y=[None], name=f"Regime {s}: {regime_names[s]}",
        mode="markers", marker=dict(size=10, color=REGIME_COLORS[s]),
    ))
figC.update_yaxes(title="IBB normalised (base 100)")

# ─── FIG D: Backtest — all tickers ────────────────────────────────────────────
figD = make_subplots(rows=1, cols=3,
                     subplot_titles=[
                         f"{t}  Sharpe={summary[t]['metrics']['sharpe']:.2f}  "
                         f"Sortino={summary[t]['metrics']['sortino']:.2f}  "
                         f"MaxDD={summary[t]['metrics']['max_dd']:.1%}"
                         for t in TICKERS
                     ],
                     horizontal_spacing=0.07)
for j, t in enumerate(TICKERS, 1):
    sc = summary[t]["strat_cum"]
    bh = summary[t]["bh_cum"]
    figD.add_trace(go.Scatter(
        x=sc.index, y=sc.values, name=f"{t} Strategy",
        line=dict(color=COLORS[t], width=2.5),
        hovertemplate=f"<b>{t} v2</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}x<extra></extra>",
    ), row=1, col=j)
    figD.add_trace(go.Scatter(
        x=bh.index, y=bh.values, name=f"{t} B&H",
        line=dict(color="#484f58", width=1.5, dash="dot"),
        hovertemplate=f"<b>{t} B&H</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}x<extra></extra>",
    ), row=1, col=j)
    figD.add_hline(y=1, line_dash="dot", line_color="#666", opacity=0.4, row=1, col=j)
figD.update_layout(
    height=480, width=1400,
    title=dict(text="Expanding Walk-Forward Backtest — v2 Strategy vs Buy & Hold",
               font=dict(color=TEXT, size=17, family="Inter, sans-serif"), x=0.5),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    hovermode="x unified",
    legend=dict(bgcolor=PANEL, bordercolor=GRID, borderwidth=1, font=dict(color=TEXT)),
)
figD.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
figD.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
for ann in figD.layout.annotations:
    ann.font.color = TEXT

# ─── FIG E: Rolling accuracy ───────────────────────────────────────────────────
figE = dark_fig("Rolling 30-Day OOS Accuracy — v2 Model  (50% = random)", height=400)
figE.add_hline(y=0.5, line_dash="dot", line_color="#888", opacity=0.5)
for t in TICKERS:
    ra = summary[t]["roll_acc"]
    figE.add_trace(go.Scatter(
        x=ra.index, y=ra.values, name=t,
        line=dict(color=COLORS[t], width=2),
        hovertemplate=f"<b>{t}</b><br>%{{x|%Y-%m-%d}}<br>Acc=%{{y:.1%}}<extra></extra>",
    ))
figE.update_yaxes(title="Accuracy", tickformat=".0%", range=[0.3, 0.75])
figE.update_xaxes(title="Date")

# ─── FIG F: Feature importance ────────────────────────────────────────────────
figF = make_subplots(rows=1, cols=3,
                     subplot_titles=[f"{t} — Top 15 Features" for t in TICKERS],
                     horizontal_spacing=0.1)
for j, t in enumerate(TICKERS, 1):
    imp = pd.Series(importances[t], index=feature_names)
    top = imp.sort_values(ascending=True).tail(15)
    figF.add_trace(go.Bar(
        y=top.index, x=top.values, orientation="h",
        marker_color=COLORS[t], marker_line_color="#0d1117", marker_line_width=0.5,
        hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
    ), row=1, col=j)
figF.update_layout(
    height=560, width=1400,
    title=dict(text="Feature Importance — v2 (GARCH + DCC + HMM + PCA)",
               font=dict(color=TEXT, size=17, family="Inter, sans-serif"), x=0.5),
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, sans-serif"), showlegend=False,
)
figF.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
figF.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=8))
for ann in figF.layout.annotations:
    ann.font.color = TEXT

# ─── TOMORROW CARDS ───────────────────────────────────────────────────────────
def gauge_color(conf):
    if conf > 0.65: return "#00ffd5"
    if conf > 0.55: return "#ffd93d"
    return "#ff6b6b"

tomorrow_html = ""
for t, p in tomorrow_preds.items():
    arrow = "&#8679;" if p["direction"] == "UP" else "&#8681;"
    dc    = "#00ffd5" if p["direction"] == "UP" else "#ff6b6b"
    rc    = REGIME_COLORS[p["regime"]]
    cw    = int(p["conf"] * 100)
    gc    = gauge_color(p["conf"])
    tomorrow_html += f"""
    <div class="pred-card">
      <div class="pred-ticker" style="color:{COLORS[t]}">{t}</div>
      <div class="pred-regime" style="background:{rc}22;color:{rc};border-color:{rc}55">
        {p['regime_name']} Regime
      </div>
      <div class="pred-direction" style="color:{dc}">{arrow} {p['direction']}</div>
      <div class="pred-detail">P(UP) = {p['prob_up']:.1%} &nbsp;·&nbsp; GARCH vol = {p['garch_vol']*100:.2f}%/day</div>
      <div class="conf-label">Calibrated confidence: {p['conf']:.1%}</div>
      <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{cw}%;background:{gc}"></div></div>
    </div>"""

# ─── METRICS TABLE HTML ────────────────────────────────────────────────────────
metrics_rows = ""
for t in TICKERS:
    m = summary[t]["metrics"]
    acc = summary[t]["acc"]
    metrics_rows += f"""
    <tr>
      <td style="color:{COLORS[t]};font-weight:700">{t}</td>
      <td>{acc:.1%}</td>
      <td>{m['sharpe']:.2f}</td>
      <td>{m['sortino']:.2f}</td>
      <td>{m['calmar']:.2f}</td>
      <td style="color:#ff6b6b">{m['max_dd']:.1%}</td>
      <td style="color:#00ffd5">{m['total_ret']:+.1%}</td>
    </tr>"""

# ─── ASSEMBLE HTML ─────────────────────────────────────────────────────────────
figs_order = [figA, figB, figC, figD, figE, figF]
html_chunks = [f.to_html(full_html=False, include_plotlyjs="cdn") for f in figs_order]
last_date = X_all.index[-1].strftime("%Y-%m-%d")

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Quant Signal Model v2 — IBB / MRNA / LLY</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d1117;color:#e6edf3;font-family:'Inter',sans-serif;padding:32px 24px}}
    h1{{text-align:center;font-size:2.4rem;font-weight:800;margin-bottom:6px;
        background:linear-gradient(135deg,#00d4ff 0%,#a29bfe 50%,#ffd93d 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .subtitle{{text-align:center;color:#8b949e;font-size:.88rem;margin-bottom:8px}}
    .version-badge{{text-align:center;margin-bottom:32px}}
    .version-badge span{{display:inline-block;padding:4px 14px;border-radius:999px;font-size:.75rem;
      font-weight:700;background:#7c3aed33;color:#a29bfe;border:1px solid #7c3aed55;margin:3px}}

    /* Method box */
    .method-box{{max-width:920px;margin:0 auto 40px;padding:22px 28px;
      background:#161b22;border:1px solid #21262d;border-radius:12px;
      font-size:.85rem;line-height:1.8;color:#c9d1d9}}
    .method-box h3{{color:#00d4ff;margin-bottom:10px;font-size:1rem;font-weight:700}}
    .method-box ul{{padding-left:20px}}
    .method-box li{{margin-bottom:5px}}
    .tag{{display:inline-block;padding:2px 10px;border-radius:6px;font-size:.72rem;
      font-weight:700;margin-right:6px;background:#7c3aed22;color:#a29bfe;border:1px solid #7c3aed55}}

    /* Metrics table */
    .metrics-wrap{{max-width:820px;margin:0 auto 48px}}
    table{{width:100%;border-collapse:collapse;font-size:.88rem}}
    th{{background:#21262d;color:#8b949e;font-weight:600;padding:10px 14px;
       text-align:right;font-size:.78rem;text-transform:uppercase;letter-spacing:.5px}}
    th:first-child{{text-align:left}}
    td{{padding:10px 14px;border-bottom:1px solid #21262d;text-align:right}}
    td:first-child{{text-align:left}}
    tr:hover td{{background:#1c2128}}

    /* Prediction cards */
    .pred-section{{margin-bottom:52px}}
    .pred-section>h2{{font-size:.95rem;text-transform:uppercase;letter-spacing:1px;
      color:#8b949e;margin-bottom:20px;padding-bottom:8px;border-bottom:1px solid #21262d}}
    .pred-row{{display:flex;gap:20px;justify-content:center;flex-wrap:wrap}}
    .pred-card{{flex:1;min-width:240px;max-width:320px;background:#161b22;border:1px solid #21262d;
      border-radius:14px;padding:26px 22px;text-align:center;transition:transform .2s,box-shadow .2s}}
    .pred-card:hover{{transform:translateY(-4px);box-shadow:0 8px 32px rgba(0,0,0,.5)}}
    .pred-ticker{{font-size:1.7rem;font-weight:800;margin-bottom:10px}}
    .pred-regime{{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.72rem;
      font-weight:700;border:1px solid;margin-bottom:12px}}
    .pred-direction{{font-size:3rem;font-weight:800;margin-bottom:8px}}
    .pred-detail{{font-size:.78rem;color:#8b949e;margin-bottom:12px}}
    .conf-label{{font-size:.75rem;color:#484f58;margin-bottom:6px}}
    .conf-bar-bg{{height:6px;background:#21262d;border-radius:999px;overflow:hidden}}
    .conf-bar-fill{{height:100%;border-radius:999px}}

    /* Charts */
    section{{margin-bottom:56px}}
    section h2{{font-size:1rem;font-weight:600;color:#8b949e;text-transform:uppercase;
      letter-spacing:1px;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #21262d}}
    .chart-wrap{{background:#161b22;border:1px solid #21262d;border-radius:12px;overflow:hidden;padding:8px}}
    footer{{text-align:center;color:#484f58;font-size:.75rem;margin-top:32px}}
  </style>
</head>
<body>
  <h1>Quant Signal Model v2</h1>
  <p class="subtitle">IBB · MRNA · LLY &nbsp;|&nbsp; Feb 2022 → Feb 2026</p>
  <div class="version-badge">
    <span>GARCH(1,1) Conditional Vol</span>
    <span>DCC Dynamic Correlation</span>
    <span>HMM 3-State Regime</span>
    <span>PCA Factor Decomposition</span>
    <span>Expanding Walk-Forward</span>
    <span>Calibrated Probabilities</span>
  </div>

  <div class="method-box">
    <h3>v2 Architecture</h3>
    <ul>
      <li><span class="tag">GARCH(1,1)</span> Fitted per asset. Captures volatility clustering and gives a true conditional sigma_t for tomorrow. Replaces naive rolling std.</li>
      <li><span class="tag">DCC-EWMA</span> Engle (2002) Dynamic Conditional Correlation via GARCH-standardised residuals + RiskMetrics EWMA (lambda=0.94). Predicts tomorrow's correlation matrix.</li>
      <li><span class="tag">HMM</span> 3-state Gaussian Hidden Markov Model on IBB returns + GARCH vol. Detects latent regime: Bear / Sideways / Bull. Used as meta-feature and for regime-aware signals.</li>
      <li><span class="tag">PCA</span> Rolling 63-day PCA to decompose returns into systematic factor (PC1) and per-asset idiosyncratic residual. Idio = alpha signal, PC1 = beta signal.</li>
      <li><span class="tag">Walk-Forward</span> Expanding window: retrain every 21 trading days using all past data. Zero lookahead bias. Mimics production deployment.</li>
      <li><span class="tag">Calibration</span> Platt scaling (sigmoid) wraps GBM. Raw GBM scores are not true probabilities — calibration fixes this so P(UP)=70% really means 70%.</li>
    </ul>
  </div>

  <!-- Metrics table -->
  <div class="metrics-wrap">
    <section>
      <h2>OOS Performance Summary</h2>
      <table>
        <thead><tr>
          <th>Ticker</th><th>Accuracy</th><th>Sharpe</th><th>Sortino</th>
          <th>Calmar</th><th>Max DD</th><th>Total Return</th>
        </tr></thead>
        <tbody>{metrics_rows}</tbody>
      </table>
    </section>
  </div>

  <!-- Tomorrow -->
  <div class="pred-section">
    <h2>Tomorrow's Signal &nbsp;(as of {last_date} · calibrated)</h2>
    <div class="pred-row">{tomorrow_html}</div>
  </div>

  <section><h2>01 — GARCH(1,1) vs Rolling Volatility</h2>
    <div class="chart-wrap">{html_chunks[0]}</div></section>
  <section><h2>02 — DCC Dynamic Conditional Correlation vs Rolling Pearson</h2>
    <div class="chart-wrap">{html_chunks[1]}</div></section>
  <section><h2>03 — HMM Regime Detection</h2>
    <div class="chart-wrap">{html_chunks[2]}</div></section>
  <section><h2>04 — Expanding Walk-Forward Backtest</h2>
    <div class="chart-wrap">{html_chunks[3]}</div></section>
  <section><h2>05 — Rolling 30-Day Prediction Accuracy</h2>
    <div class="chart-wrap">{html_chunks[4]}</div></section>
  <section><h2>06 — Feature Importance (GARCH + DCC + HMM + PCA signals)</h2>
    <div class="chart-wrap">{html_chunks[5]}</div></section>

  <footer>Quant Signal Model v2 · Antigravity · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
  · Disclaimer: educational only, not financial advice</footer>
</body>
</html>"""

with open("model_v2_dashboard.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("\n[OK] model_v2_dashboard.html saved. Open in browser.")
