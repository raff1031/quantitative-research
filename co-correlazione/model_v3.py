"""
=============================================================================
  QUANT SIGNAL MODEL v3 — Free Alternative Data Layer
  IBB | MRNA | LLY  |  Feb 2022 -> Feb 2026
=============================================================================
  New vs v2:
    [6] VIX + VVIX as historical fear-regime features (free, 4-year)
    [7] Earnings proximity flag (yfinance calendar, free)
    [8] LIVE options chain: P/C ratio, ATM-IV, IV skew, term structure
    [9] LIVE news sentiment: VADER on yfinance.news (no API key needed)
   [10] Composite signal: ML prob + options overlay + sentiment overlay
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# VADER for news sentiment — free, no API key
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TICKERS    = ["IBB", "MRNA", "LLY"]
START_DATE = "2022-02-20"
END_DATE   = "2026-02-20"
N_REGIMES  = 3
RETRAIN_FREQ = 21
MIN_TRAIN    = 200

BG    = "#0d1117";  PANEL = "#161b22"
TEXT  = "#e6edf3";  GRID  = "#21262d"
COLORS = {"IBB": "#00d4ff", "MRNA": "#ff6b6b", "LLY": "#ffd93d"}
REGIME_COLORS = ["#ff6b6b", "#ffd93d", "#00ffd5"]

# ── [1] FETCH PRICE DATA ───────────────────────────────────────────────────────
print("[1/10] Fetching price data ...")
raw    = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
prices = raw["Close"][TICKERS].dropna()
ret    = np.log(prices / prices.shift(1)).dropna()
print(f"       {len(ret)} trading days | {ret.index[0].date()} -> {ret.index[-1].date()}")

# ── [2] GARCH(1,1) CONDITIONAL VOLATILITY ─────────────────────────────────────
print("[2/10] GARCH(1,1) per asset ...")
garch_vol = {}; garch_resid = {}
for t in TICKERS:
    am  = arch_model(ret[t] * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    sig = res.conditional_volatility / 100
    sig.index = ret.index
    garch_vol[t]   = sig
    garch_resid[t] = ret[t] / (sig + 1e-8)
    p = res.params
    print(f"       {t:>4}  alpha={p['alpha[1]']:.3f}  beta={p['beta[1]']:.3f}  "
          f"persist={(p['alpha[1]']+p['beta[1]']):.3f}")

garch_vol_df   = pd.DataFrame(garch_vol)
garch_resid_df = pd.DataFrame(garch_resid)

# ── [3] DCC-EWMA DYNAMIC CORRELATION ──────────────────────────────────────────
print("[3/10] DCC dynamic correlations ...")
LAMBDA = 0.94
def ewma_cov(e, lam):
    Q = [e.values[:21].T @ e.values[:21] / 21]
    for i in range(1, len(e)):
        ev = e.values[i].reshape(-1,1)
        Q.append(lam * Q[-1] + (1-lam) * ev @ ev.T)
    return Q

def Q_to_R(Q):
    d = np.sqrt(np.diag(Q)); D = np.diag(1/(d+1e-10))
    return D @ Q @ D

Q_list = ewma_cov(garch_resid_df, LAMBDA)
R_list = [Q_to_R(Q) for Q in Q_list]
pairs  = [("IBB","MRNA"), ("IBB","LLY"), ("MRNA","LLY")]
pidx   = {("IBB","MRNA"):(0,1), ("IBB","LLY"):(0,2), ("MRNA","LLY"):(1,2)}
dcc_df = pd.DataFrame(
    {p: [R_list[t][i,j] for t in range(len(R_list))] for p,(i,j) in pidx.items()},
    index=garch_resid_df.index,
)

# ── [4] HMM REGIME DETECTION ──────────────────────────────────────────────────
print("[4/10] HMM 3-state regime ...")
hmm_feat = np.column_stack([ret["IBB"].values, garch_vol["IBB"].values,
                             garch_vol_df.mean(1).values])
hmm_feat = (hmm_feat - hmm_feat.mean(0)) / (hmm_feat.std(0)+1e-8)
hmm = GaussianHMM(n_components=N_REGIMES, covariance_type="full",
                  n_iter=500, random_state=42)
hmm.fit(hmm_feat)
raw_regime = hmm.predict(hmm_feat)
regime_series = pd.Series(raw_regime, index=ret.index)
mu_by = {s: ret["IBB"][regime_series==s].mean() for s in range(N_REGIMES)}
remap  = {o:n for n,o in enumerate(sorted(mu_by, key=mu_by.get))}
regime = regime_series.map(remap)   # 0=bear, 1=sideways, 2=bull
regime_names = {0:"Bear", 1:"Sideways", 2:"Bull"}
for s in range(N_REGIMES):
    mask = regime==s
    print(f"       Regime {s} [{regime_names[s]:>9}]: {mask.sum():3d} days  "
          f"IBB mu={ret['IBB'][mask].mean()*100:+.3f}%/d")

# ── [5] ROLLING PCA ────────────────────────────────────────────────────────────
print("[5/10] Rolling PCA ...")
rolling_pc1 = pd.Series(index=ret.index, dtype=float)
idio = pd.DataFrame(index=ret.index, columns=TICKERS, dtype=float)
for i in range(63, len(ret)):
    w = ret.iloc[i-63:i]
    X = StandardScaler().fit_transform(w)
    pca = PCA(n_components=1); scores = pca.fit_transform(X)
    ld  = pca.components_[0]
    rolling_pc1.iloc[i] = scores[-1,0]
    for k,t in enumerate(TICKERS):
        idio.loc[ret.index[i], t] = w.iloc[-1][t] - scores[-1,0]*ld[k]*w.std()[t]
rolling_pc1 = rolling_pc1.dropna()
idio = idio.dropna()

# ── [6] VIX + VVIX — FREE HISTORICAL IV PROXY ─────────────────────────────────
print("[6/10] Fetching VIX + VVIX (free historical fear gauges) ...")
vix_raw  = yf.download("^VIX",  start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
vvix_raw = yf.download("^VVIX", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
spy_raw  = yf.download("SPY",   start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

vix  = vix_raw["Close"].squeeze().reindex(ret.index).ffill()
vvix = vvix_raw["Close"].squeeze().reindex(ret.index).ffill()
spy_ret = np.log(spy_raw["Close"].squeeze()).diff().reindex(ret.index).ffill()

# VIX z-score (how extreme is fear vs recent history)
vix_zscore = (vix - vix.rolling(63).mean()) / (vix.rolling(63).std() + 1e-8)
# VVIX/VIX ratio: when VVIX is high relative to VIX -> uncertainty about uncertainty
vvix_vix_ratio = vvix / (vix + 1e-8)

# Per-asset rolling beta vs SPY (used to proxy stock-specific IV from VIX)
asset_beta = {}
asset_iv_proxy = {}
for t in TICKERS:
    cov = ret[t].rolling(63).cov(spy_ret)
    var = spy_ret.rolling(63).var() + 1e-8
    beta = (cov / var).clip(-3, 5)
    asset_beta[t] = beta
    # IV proxy = VIX * |beta| → crude but free historical estimate
    asset_iv_proxy[t] = (vix / 100) * beta.abs()

print(f"       VIX range: {vix.min():.1f} - {vix.max():.1f}  "
      f"(current: {vix.iloc[-1]:.1f})")
print(f"       VVIX range: {vvix.min():.1f} - {vvix.max():.1f}  "
      f"(current: {vvix.iloc[-1]:.1f})")

# ── [7] EARNINGS PROXIMITY FLAG ────────────────────────────────────────────────
print("[7/10] Building earnings proximity flags ...")

def get_earnings_dates(ticker_str):
    """Get historical earnings dates for a ticker. Returns sorted array."""
    try:
        tk = yf.Ticker(ticker_str)
        ed = tk.earnings_dates
        if ed is None or len(ed) == 0:
            return np.array([], dtype="datetime64[ns]")
        dates = pd.DatetimeIndex(ed.index).tz_localize(None)
        return np.sort(dates.values)
    except Exception:
        return np.array([], dtype="datetime64[ns]")

def days_to_nearest_earnings(date_idx: pd.DatetimeIndex, earnings: np.ndarray):
    """Minimum absolute distance in trading days to any earnings date."""
    if len(earnings) == 0:
        return pd.Series(np.nan, index=date_idx)
    result = []
    for d in date_idx:
        diffs = np.abs((earnings - np.datetime64(d)).astype("timedelta64[D]").astype(float))
        result.append(diffs.min())
    return pd.Series(result, index=date_idx)

earnings_flags = {}
for t in TICKERS:
    edates = get_earnings_dates(t)
    dist   = days_to_nearest_earnings(ret.index, edates)
    earnings_flags[t] = {
        "days_to": dist,
        "within5": (dist <= 5).astype(float),
        "within10": (dist <= 10).astype(float),
    }
    n_found = int((dist <= 5).sum())
    print(f"       {t:>4}  earnings events within 5d flag: {n_found} day-flags")

# ── [8] FEATURE MATRIX V3 ─────────────────────────────────────────────────────
print("[8/10] Building v3 feature matrix ...")

SHORT, MED, LONG = 10, 21, 63

def build_features_v3(ret, garch_vol_df, dcc_df, regime, rolling_pc1, idio,
                      vix, vix_zscore, vvix_vix_ratio, asset_iv_proxy,
                      asset_beta, earnings_flags):
    f = pd.DataFrame(index=ret.index)

    for t in TICKERS:
        r = ret[t]
        # Momentum
        for lag in [1,2,3,5]: f[f"{t}_lag{lag}"] = r.shift(lag)
        for w in [5,MED,LONG]: f[f"{t}_mom_{w}d"] = r.shift(1).rolling(w).mean()
        # GARCH
        gv = garch_vol_df[t]
        f[f"{t}_garch_vol"]     = gv.shift(1)
        f[f"{t}_garch_vol_chg"] = gv.diff(1).shift(1)
        f[f"{t}_vol_zscore"]    = ((gv - gv.rolling(LONG).mean())/(gv.rolling(LONG).std()+1e-8)).shift(1)
        # Idiosyncratic PCA residual
        if t in idio.columns:
            f[f"{t}_idio"]      = idio[t].shift(1)
            f[f"{t}_idio_mom5"] = idio[t].shift(1).rolling(5).mean()
        # Beta to SPY
        f[f"{t}_beta"]          = asset_beta[t].shift(1)
        # IV proxy (VIX * beta) vs GARCH: fear premium
        iv_prx = asset_iv_proxy[t]
        f[f"{t}_iv_proxy"]      = iv_prx.shift(1)
        f[f"{t}_fear_premium"]  = (iv_prx - gv).shift(1)  # positive = market more afraid than GARCH
        # Earnings flags
        f[f"{t}_days_earnings"] = earnings_flags[t]["days_to"].shift(1).fillna(999)
        f[f"{t}_pre_earnings5"] = earnings_flags[t]["within5"].shift(1).fillna(0)
        f[f"{t}_pre_earn10"]    = earnings_flags[t]["within10"].shift(1).fillna(0)

    # DCC correlations
    for p in pairs:
        key = f"dcc_{p[0]}_{p[1]}"
        dcc = dcc_df[p].shift(1)
        f[key]               = dcc
        f[f"{key}_chg"]      = dcc.diff(1)
        f[f"{key}_zscore"]   = (dcc - dcc.rolling(LONG).mean())/(dcc.rolling(LONG).std()+1e-8)

    # HMM regime dummies
    for s in range(N_REGIMES):
        f[f"regime_{s}"] = (regime.shift(1)==s).astype(float)
    f["regime_val"] = regime.shift(1)

    # PCA
    f["pc1_score"]     = rolling_pc1.shift(1)
    f["pc1_score_mom"] = rolling_pc1.shift(1).rolling(5).mean()

    # Divergence signals
    for (a,b) in pairs:
        sp = (ret[a]-ret[b]).shift(1)
        f[f"div_{a}_{b}"]      = sp
        f[f"div_corr_{a}_{b}"] = sp * dcc_df[(a,b)].shift(1)

    # ── NEW v3: VIX / VVIX features ─────────────────────────────────────────
    f["vix_level"]       = vix.shift(1)
    f["vix_chg"]         = vix.diff(1).shift(1)
    f["vix_zscore"]      = vix_zscore.shift(1)
    f["vvix_vix_ratio"]  = vvix_vix_ratio.shift(1)
    f["vix_high_regime"] = (vix_zscore > 1).astype(float).shift(1)   # VIX spike flag
    f["vix_low_regime"]  = (vix_zscore < -1).astype(float).shift(1)  # complacency flag

    return f.dropna()

features = build_features_v3(
    ret, garch_vol_df, dcc_df, regime, rolling_pc1, idio,
    vix, vix_zscore, vvix_vix_ratio, asset_iv_proxy,
    asset_beta, earnings_flags,
)
feature_names = features.columns.tolist()
print(f"       {len(feature_names)} features  |  {len(features)} usable days")

# ── [9] EXPANDING WALK-FORWARD ─────────────────────────────────────────────────
print("[9/10] Expanding walk-forward (retrain every 21d) ...")

targets = {t: (ret[t].shift(-1)>0).astype(int) for t in TICKERS}
common_idx = features.index.intersection(
    pd.concat([targets[t].rename(t) for t in TICKERS], axis=1).dropna().index
)
X_all = features.loc[common_idx]
Y_all = pd.concat([targets[t].rename(t) for t in TICKERS], axis=1).loc[common_idx]

oos_results = {t: dict(idx=[], prob=[], pred=[], true=[]) for t in TICKERS}
model_cache = {}
importances = {}
GBM_PARAMS  = dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                   subsample=0.75, min_samples_leaf=8, max_features=0.6,
                   random_state=42)

retrain_pts = list(range(MIN_TRAIN, len(common_idx), RETRAIN_FREQ))
for si, train_end in enumerate(retrain_pts):
    pred_end = retrain_pts[si+1] if si+1 < len(retrain_pts) else len(common_idx)
    tr_X = X_all.iloc[:train_end]; tr_Y = Y_all.iloc[:train_end]
    te_X = X_all.iloc[train_end:pred_end]
    te_idx = common_idx[train_end:pred_end]
    if len(te_X) == 0: continue

    for t in TICKERS:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(tr_X)
        X_te_s = sc.transform(te_X)
        clf = CalibratedClassifierCV(
            GradientBoostingClassifier(**GBM_PARAMS), method="sigmoid", cv=3)
        clf.fit(X_tr_s, tr_Y[t])
        prob = clf.predict_proba(X_te_s)[:,1]
        pred = (prob>0.5).astype(int)
        oos_results[t]["idx"].extend(te_idx.tolist())
        oos_results[t]["prob"].extend(prob.tolist())
        oos_results[t]["pred"].extend(pred.tolist())
        oos_results[t]["true"].extend(Y_all[t].iloc[train_end:pred_end].tolist())
        model_cache[t] = (clf, sc)
        importances[t] = GradientBoostingClassifier(**GBM_PARAMS).fit(
            X_tr_s, tr_Y[t]).feature_importances_

    if si % 5 == 0:
        print(f"       {train_end/len(common_idx)*100:4.0f}%  train={train_end}  pred={len(te_X)}d", end="\r")

print("\n       Walk-forward complete.")

def risk_metrics(sr):
    ann = np.sqrt(252)
    sharpe  = (sr.mean()/(sr.std()+1e-8)) * ann
    sortino = (sr.mean()/(sr[sr<0].std()+1e-8)) * ann
    cum = (1+sr).cumprod(); dd = (cum-cum.cummax())/cum.cummax()
    return dict(sharpe=sharpe, sortino=sortino, max_dd=dd.min(),
                total_ret=cum.iloc[-1]-1)

summary = {}
for t in TICKERS:
    o   = oos_results[t]
    idx = pd.DatetimeIndex(o["idx"])
    prob= np.array(o["prob"]); pred=np.array(o["pred"]); true=np.array(o["true"])
    acc = accuracy_score(true, pred)
    sig = pd.Series(np.where(pred==1, 1, -1), index=idx)
    sr  = (sig * ret[t].reindex(idx)).dropna()
    bh  = ret[t].reindex(idx).dropna()
    m   = risk_metrics(sr)
    summary[t] = dict(acc=acc, metrics=m,
                      strat_cum=(1+sr).cumprod(), bh_cum=(1+bh).cumprod(),
                      roll_acc=pd.Series((pred==true).astype(float), index=idx).rolling(30).mean(),
                      prob=pd.Series(prob, index=idx))
    print(f"       {t:>4}  ACC={acc:.1%}  Sharpe={m['sharpe']:.2f}  "
          f"Sortino={m['sortino']:.2f}  MaxDD={m['max_dd']:.1%}  Return={m['total_ret']:+.1%}")

# ── [10] LIVE SIGNAL ENRICHMENT ────────────────────────────────────────────────
print("[10/10] Live signal enrichment ...")

# ─── 10a: BASE MODEL PREDICTION ──────────────────────────────────────────────
last_X = X_all.iloc[[-1]]
model_preds = {}
for t in TICKERS:
    clf, sc = model_cache[t]
    prob_up = clf.predict_proba(sc.transform(last_X))[0,1]
    model_preds[t] = prob_up

# ─── 10b: LIVE OPTIONS CHAIN ─────────────────────────────────────────────────
print("       Fetching live options chains ...")

def analyse_options(ticker_str: str) -> dict:
    """
    Extract from current options chain:
      - pc_ratio_oi: put/call ratio by open interest (near-term)
      - atm_iv:      ATM implied vol (avg call+put nearest strike)
      - iv_skew:     OTM put IV minus OTM call IV (fear asymmetry)
      - term_slope:  near IV / far IV (>1 = inverted = panic)
    Returns dict with values or None on failure.
    """
    try:
        tk     = yf.Ticker(ticker_str)
        spot   = tk.fast_info.get("lastPrice") or tk.fast_info.get("regularMarketPrice")
        expiries = tk.options
        if not expiries or spot is None:
            return {}

        # Near expiry (closest to 30 days out)
        today = pd.Timestamp.today()
        exp_dates = pd.to_datetime(expiries)
        target_near = today + timedelta(days=30)
        target_far  = today + timedelta(days=90)
        near_exp = expiries[np.argmin(np.abs((exp_dates - target_near).days))]
        far_candidates = [e for e in expiries if pd.to_datetime(e) > target_near + timedelta(days=20)]
        far_exp  = far_candidates[np.argmin(np.abs(
            (pd.to_datetime(far_candidates) - target_far).days
        ))] if far_candidates else None

        chain_near = tk.option_chain(near_exp)
        calls_n = chain_near.calls.dropna(subset=["impliedVolatility","openInterest"])
        puts_n  = chain_near.puts.dropna(subset=["impliedVolatility","openInterest"])

        # P/C ratio by Open Interest
        total_call_oi = calls_n["openInterest"].sum()
        total_put_oi  = puts_n["openInterest"].sum()
        pc_ratio_oi   = total_put_oi / (total_call_oi + 1e-6)

        # ATM IV: option with strike closest to spot
        atm_call = calls_n.iloc[(calls_n["strike"]-spot).abs().argsort()[:1]]
        atm_put  = puts_n.iloc[(puts_n["strike"]-spot).abs().argsort()[:1]]
        atm_iv   = float((atm_call["impliedVolatility"].values[0] +
                          atm_put["impliedVolatility"].values[0]) / 2)

        # IV Skew: OTM put (5% below spot) vs OTM call (5% above spot)
        otm_put_target   = spot * 0.95
        otm_call_target  = spot * 1.05
        otm_put  = puts_n.iloc[(puts_n["strike"]-otm_put_target).abs().argsort()[:1]]
        otm_call = calls_n.iloc[(calls_n["strike"]-otm_call_target).abs().argsort()[:1]]
        iv_skew  = float(otm_put["impliedVolatility"].values[0] -
                         otm_call["impliedVolatility"].values[0])

        # Term structure: near ATM IV vs far ATM IV
        term_slope = None
        if far_exp:
            chain_far = tk.option_chain(far_exp)
            calls_f = chain_far.calls.dropna(subset=["impliedVolatility"])
            puts_f  = chain_far.puts.dropna(subset=["impliedVolatility"])
            if len(calls_f) and len(puts_f):
                atm_call_f = calls_f.iloc[(calls_f["strike"]-spot).abs().argsort()[:1]]
                atm_put_f  = puts_f.iloc[(puts_f["strike"]-spot).abs().argsort()[:1]]
                atm_iv_far = float((atm_call_f["impliedVolatility"].values[0] +
                                    atm_put_f["impliedVolatility"].values[0]) / 2)
                term_slope = atm_iv / (atm_iv_far + 1e-6)

        return dict(pc_ratio_oi=pc_ratio_oi, atm_iv=atm_iv,
                    iv_skew=iv_skew, term_slope=term_slope,
                    near_exp=near_exp, spot=spot)
    except Exception as e:
        print(f"         Options fetch failed for {ticker_str}: {e}")
        return {}

options_data = {}
for t in TICKERS:
    od = analyse_options(t)
    options_data[t] = od
    if od:
        gv = float(garch_vol_df[t].iloc[-1]) * np.sqrt(252)
        fear_prem = od["atm_iv"] - gv if od.get("atm_iv") else None
        print(f"       {t:>4}  ATM_IV={od.get('atm_iv',0):.1%}  "
              f"GARCH_annual={gv:.1%}  "
              f"FearPremium={fear_prem:+.1%}  "
              f"P/C(OI)={od.get('pc_ratio_oi',0):.2f}  "
              f"Skew={od.get('iv_skew',0):+.3f}")

# ─── 10c: NEWS SENTIMENT VIA VADER ───────────────────────────────────────────
print("       Fetching news + computing VADER sentiment ...")
vader = SentimentIntensityAnalyzer()

def get_news_sentiment(ticker_str: str) -> dict:
    """
    Fetch recent articles from yfinance, run VADER on title+summary.
    Returns: mean compound score [-1,1], article count, headline strings.
    """
    try:
        tk      = yf.Ticker(ticker_str)
        news    = tk.news or []
        if not news:
            return dict(score=None, n=0, headlines=[])
        scores  = []
        headlines = []
        for art in news[:15]:
            title   = art.get("title","")
            summary = art.get("summary","")
            text    = f"{title}. {summary}" if summary else title
            if text.strip():
                scores.append(vader.polarity_scores(text)["compound"])
                headlines.append(title[:80])
        if not scores:
            return dict(score=None, n=0, headlines=[])
        return dict(score=float(np.mean(scores)), n=len(scores),
                    max_score=float(max(scores)), min_score=float(min(scores)),
                    headlines=headlines)
    except Exception as e:
        return dict(score=None, n=0, headlines=[])

news_data = {}
for t in TICKERS:
    nd = get_news_sentiment(t)
    news_data[t] = nd
    if nd["score"] is not None:
        sentiment_label = "POSITIVE" if nd["score"]>0.05 else ("NEGATIVE" if nd["score"]<-0.05 else "NEUTRAL")
        print(f"       {t:>4}  Sentiment={nd['score']:+.3f} [{sentiment_label}]  "
              f"Articles={nd['n']}  Range=[{nd.get('min_score',0):+.2f},{nd.get('max_score',0):+.2f}]")

# ─── COMPOSITE SIGNAL ────────────────────────────────────────────────────────
print("\n  COMPOSITE SIGNAL BREAKDOWN (as of today):")
print("  " + "="*70)

composite_preds = {}
for t in TICKERS:
    p_model = model_preds[t]
    adjustment = 0.0
    notes = []

    # Options overlay (if available)
    od = options_data.get(t, {})
    if od:
        gv_ann = float(garch_vol_df[t].iloc[-1]) * np.sqrt(252)
        # Fear premium: IV >> GARCH -> expected vol spike -> reduce confidence
        if od.get("atm_iv") and od["atm_iv"] > gv_ann * 1.3:
            adjustment -= 0.04
            notes.append("IV>>GARCH(-4%)")
        elif od.get("atm_iv") and od["atm_iv"] < gv_ann * 0.8:
            notes.append("IV<<GARCH(complacent)")
        # P/C ratio
        pcr = od.get("pc_ratio_oi", 1.0)
        if pcr > 1.5:
            adjustment -= 0.03
            notes.append(f"HighPutDemand({pcr:.1f}x,-3%)")
        elif pcr < 0.6:
            adjustment += 0.02
            notes.append(f"CallDominated({pcr:.1f}x,+2%)")
        # Skew: large positive skew = market fears downside
        skew = od.get("iv_skew", 0)
        if skew > 0.08:
            adjustment -= 0.03
            notes.append(f"HighSkew({skew:.2f},-3%)")
        # Term structure: inverted (near>far) = near-term panic
        ts = od.get("term_slope")
        if ts and ts > 1.2:
            adjustment -= 0.03
            notes.append(f"InvTermStr({ts:.2f},-3%)")

    # Sentiment overlay
    nd = news_data.get(t, {})
    if nd.get("score") is not None:
        s = nd["score"]
        if s < -0.2:
            adjustment -= 0.04; notes.append(f"NegNews({s:+.2f},-4%)")
        elif s < -0.05:
            adjustment -= 0.02; notes.append(f"SlightNeg({s:+.2f},-2%)")
        elif s > 0.2:
            adjustment += 0.04; notes.append(f"PosNews({s:+.2f},+4%)")
        elif s > 0.05:
            adjustment += 0.02; notes.append(f"SlightPos({s:+.2f},+2%)")

    p_final   = np.clip(p_model + adjustment, 0.05, 0.95)
    direction = "UP" if p_final > 0.5 else "DOWN"
    conf      = max(p_final, 1 - p_final)
    composite_preds[t] = dict(
        p_model=p_model, adjustment=adjustment, p_final=p_final,
        direction=direction, conf=conf, notes=notes,
        sentiment=nd.get("score"), regime=int(regime.iloc[-1]),
        garch_vol_day=float(garch_vol_df[t].iloc[-1]),
        atm_iv=od.get("atm_iv"), pcr=od.get("pc_ratio_oi"),
        skew=od.get("iv_skew"),
    )
    bar = "#"*int(conf*25) + "."*(25-int(conf*25))
    adj_str = f"{adjustment:+.0%}" if adjustment else " 0%"
    print(f"  {t:>4}: {direction:>4}  "
          f"Model={p_model:.1%}  Adj={adj_str}  Final={p_final:.1%}  "
          f"conf=[{bar}] {conf:.0%}")
    if notes: print(f"         Signals: {' | '.join(notes)}")

print("  " + "="*70)

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION — focused dashboard
# ══════════════════════════════════════════════════════════════════════════════
def hex_rgba(h, a=0.15):
    h=h.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def std_layout(fig, title, h=450, w=1400):
    fig.update_layout(
        height=h, width=w,
        title=dict(text=title, font=dict(color=TEXT,size=17,family="Inter,sans-serif"), x=0.5),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter,sans-serif"),
        legend=dict(bgcolor=PANEL, bordercolor=GRID, borderwidth=1, font=dict(color=TEXT)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

# FIG A: VIX + VVIX over time
figA = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=["VIX (S&P500 Fear Gauge)", "VVIX / VIX Ratio (Fear of Fear)"],
                     vertical_spacing=0.1)
figA.add_trace(go.Scatter(x=vix.index, y=vix.values, name="VIX",
    line=dict(color="#ff6b6b",width=1.8), fill="tozeroy",
    fillcolor="rgba(255,107,107,0.08)"), row=1, col=1)
figA.add_trace(go.Scatter(x=vix_zscore.index, y=vix_zscore.values, name="VIX Z-score",
    line=dict(color="#ffd93d",width=1.5), visible="legendonly"), row=1, col=1)
figA.add_trace(go.Scatter(x=vvix_vix_ratio.index, y=vvix_vix_ratio.values,
    name="VVIX/VIX", line=dict(color="#a29bfe",width=1.8)), row=2, col=1)
figA.add_hline(y=0, line_dash="dot", line_color="#666", opacity=0.5, row=1, col=1)
figA.add_hline(y=5, line_dash="dot", line_color="#666", opacity=0.3, row=2, col=1)
std_layout(figA, "Free Historical Fear Gauges: VIX + VVIX (proxy for market IV)", h=500)
for ann in figA.layout.annotations: ann.font.color = TEXT

# FIG B: Asset IV proxy vs GARCH vol
figB = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t} — IV Proxy vs GARCH Vol (annualized)" for t in TICKERS],
    horizontal_spacing=0.07)
for j, t in enumerate(TICKERS, 1):
    figB.add_trace(go.Scatter(
        x=asset_iv_proxy[t].index, y=asset_iv_proxy[t].values,
        name=f"{t} IV proxy", line=dict(color=COLORS[t], width=2),
        hovertemplate=f"<b>IV proxy</b><br>%{{y:.1%}}<extra></extra>"), row=1, col=j)
    figB.add_trace(go.Scatter(
        x=garch_vol_df.index, y=(garch_vol_df[t]*np.sqrt(252)).values,
        name=f"{t} GARCH", line=dict(color="#484f58", width=1.5, dash="dot"),
        hovertemplate=f"<b>GARCH</b><br>%{{y:.1%}}<extra></extra>"), row=1, col=j)
std_layout(figB, "Asset IV Proxy (VIX * beta) vs GARCH Conditional Vol")
figB.update_yaxes(tickformat=".0%")
for ann in figB.layout.annotations: ann.font.color = TEXT

# FIG C: Backtest
figC = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t}  Sharpe={summary[t]['metrics']['sharpe']:.2f}  "
                    f"Sortino={summary[t]['metrics']['sortino']:.2f}  "
                    f"MaxDD={summary[t]['metrics']['max_dd']:.1%}" for t in TICKERS],
    horizontal_spacing=0.07)
for j,t in enumerate(TICKERS,1):
    sc = summary[t]["strat_cum"]; bh = summary[t]["bh_cum"]
    figC.add_trace(go.Scatter(x=sc.index, y=sc.values, name=f"{t} v3",
        line=dict(color=COLORS[t],width=2.5)), row=1,col=j)
    figC.add_trace(go.Scatter(x=bh.index, y=bh.values, name=f"{t} B&H",
        line=dict(color="#484f58",width=1.5,dash="dot")), row=1,col=j)
std_layout(figC, "Walk-Forward Backtest — v3 Strategy vs Buy & Hold")
for ann in figC.layout.annotations: ann.font.color = TEXT

# FIG D: Feature importance
figD = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t} — Top 15 Features" for t in TICKERS],
    horizontal_spacing=0.1)
for j,t in enumerate(TICKERS,1):
    imp = pd.Series(importances[t], index=feature_names).sort_values(ascending=True).tail(15)
    figD.add_trace(go.Bar(y=imp.index, x=imp.values, orientation="h",
        marker_color=COLORS[t], hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>"),
        row=1,col=j)
std_layout(figD,"Feature Importance v3 (incl. VIX, Earnings, IV proxy)", h=560)
figD.update_yaxes(tickfont=dict(size=8))
figD.update_xaxes(gridcolor=GRID)
for ann in figD.layout.annotations: ann.font.color = TEXT

# ── TOMORROW CARDS ─────────────────────────────────────────────────────────────
def gc(c): return "#00ffd5" if c>0.65 else ("#ffd93d" if c>0.55 else "#ff6b6b")

tomorrow_html = ""
for t, p in composite_preds.items():
    arrow = "&#8679;" if p["direction"]=="UP" else "&#8681;"
    dc    = "#00ffd5" if p["direction"]=="UP" else "#ff6b6b"
    rc    = REGIME_COLORS[p["regime"]]
    rn    = regime_names[p["regime"]]
    cw    = int(p["conf"]*100)

    sent_badge = ""
    if p["sentiment"] is not None:
        sc_val = p["sentiment"]
        sc_col = "#00ffd5" if sc_val>0.05 else ("#ff6b6b" if sc_val<-0.05 else "#8b949e")
        sc_lab = "Positive" if sc_val>0.05 else ("Negative" if sc_val<-0.05 else "Neutral")
        sent_badge = f'<div class="opts-row"><span class="opts-label">News</span><span style="color:{sc_col}">{sc_lab} ({sc_val:+.2f})</span></div>'

    opts_html = ""
    if p.get("atm_iv"):
        gv_a = p["garch_vol_day"] * np.sqrt(252)
        fear = p["atm_iv"] - gv_a
        fc   = "#ff6b6b" if fear>0.05 else "#00ffd5"
        opts_html += f'<div class="opts-row"><span class="opts-label">ATM IV</span><span>{p["atm_iv"]:.1%}</span></div>'
        opts_html += f'<div class="opts-row"><span class="opts-label">Fear Premium</span><span style="color:{fc}">{fear:+.1%}</span></div>'
    if p.get("pcr"):
        pc_c = "#ff6b6b" if p["pcr"]>1.3 else "#00ffd5"
        opts_html += f'<div class="opts-row"><span class="opts-label">P/C (OI)</span><span style="color:{pc_c}">{p["pcr"]:.2f}x</span></div>'
    if p.get("skew") is not None:
        sk_c = "#ff6b6b" if p["skew"]>0.05 else "#00ffd5"
        opts_html += f'<div class="opts-row"><span class="opts-label">IV Skew</span><span style="color:{sk_c}">{p["skew"]:+.3f}</span></div>'

    adj_c = "#ff6b6b" if p["adjustment"]<0 else ("#00ffd5" if p["adjustment"]>0 else "#8b949e")

    tomorrow_html += f"""
    <div class="pred-card">
      <div class="pred-ticker" style="color:{COLORS[t]}">{t}</div>
      <div class="pred-regime" style="background:{rc}22;color:{rc};border-color:{rc}55">{rn} Regime</div>
      <div class="pred-direction" style="color:{dc}">{arrow} {p['direction']}</div>
      <div class="prob-row">
        <span class="prob-label">Model</span><span class="prob-val">{p['p_model']:.1%}</span>
        <span class="prob-label">Adj</span><span style="color:{adj_c}">{p['adjustment']:+.0%}</span>
        <span class="prob-label">Final</span><span class="prob-val" style="color:{dc}">{p['p_final']:.1%}</span>
      </div>
      <div class="opts-grid">{opts_html}{sent_badge}</div>
      <div class="conf-label">Confidence {p['conf']:.0%}</div>
      <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{cw}%;background:{gc(p['conf'])}"></div></div>
    </div>"""

# News headlines section
news_html = ""
for t in TICKERS:
    nd = news_data.get(t, {}); hl = nd.get("headlines", [])
    if not hl: continue
    news_html += f'<div class="news-col"><div class="news-ticker" style="color:{COLORS[t]}">{t}</div><ul class="news-list">'
    for h in hl[:5]:
        s = vader.polarity_scores(h)["compound"]
        sc = "#00ffd5" if s>0.05 else ("#ff6b6b" if s<-0.05 else "#8b949e")
        news_html += f'<li><span class="dot" style="background:{sc}"></span>{h}</li>'
    news_html += "</ul></div>"

# Metrics table
metrics_rows = "".join(
    f'<tr><td style="color:{COLORS[t]};font-weight:700">{t}</td>'
    f'<td>{summary[t]["acc"]:.1%}</td>'
    f'<td>{summary[t]["metrics"]["sharpe"]:.2f}</td>'
    f'<td>{summary[t]["metrics"]["sortino"]:.2f}</td>'
    f'<td style="color:#ff6b6b">{summary[t]["metrics"]["max_dd"]:.1%}</td>'
    f'<td style="color:#00ffd5">{summary[t]["metrics"]["total_ret"]:+.1%}</td></tr>'
    for t in TICKERS
)

html_parts = [f.to_html(full_html=False, include_plotlyjs="cdn")
              for f in [figA, figB, figC, figD]]
last_date = X_all.index[-1].strftime("%Y-%m-%d")

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Quant Signal v3 — IBB/MRNA/LLY</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d1117;color:#e6edf3;font-family:'Inter',sans-serif;padding:32px 24px}}
    h1{{text-align:center;font-size:2.3rem;font-weight:800;margin-bottom:6px;
       background:linear-gradient(135deg,#00d4ff,#a29bfe,#ffd93d);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .subtitle{{text-align:center;color:#8b949e;font-size:.88rem;margin-bottom:8px}}
    .badges{{text-align:center;margin-bottom:32px}}
    .badge{{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.73rem;
      font-weight:700;background:#7c3aed22;color:#a29bfe;border:1px solid #7c3aed44;margin:3px}}
    .free{{background:#00ffd522;color:#00ffd5;border-color:#00ffd544}}

    /* Prediction cards */
    .pred-section{{margin-bottom:52px}}
    .pred-section>h2{{font-size:.95rem;text-transform:uppercase;letter-spacing:1px;
      color:#8b949e;margin-bottom:20px;padding-bottom:8px;border-bottom:1px solid #21262d}}
    .pred-row{{display:flex;gap:18px;justify-content:center;flex-wrap:wrap}}
    .pred-card{{flex:1;min-width:260px;max-width:340px;background:#161b22;border:1px solid #21262d;
      border-radius:14px;padding:24px 20px;text-align:center;transition:transform .2s,box-shadow .2s}}
    .pred-card:hover{{transform:translateY(-4px);box-shadow:0 8px 32px rgba(0,0,0,.5)}}
    .pred-ticker{{font-size:1.7rem;font-weight:800;margin-bottom:8px}}
    .pred-regime{{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.72rem;
      font-weight:700;border:1px solid;margin-bottom:10px}}
    .pred-direction{{font-size:2.8rem;font-weight:800;margin-bottom:10px}}
    .prob-row{{display:flex;gap:8px;justify-content:center;align-items:baseline;
      font-size:.82rem;margin-bottom:12px;flex-wrap:wrap}}
    .prob-label{{color:#484f58;font-size:.72rem}}
    .prob-val{{font-weight:700}}
    .opts-grid{{text-align:left;margin-bottom:12px;font-size:.78rem}}
    .opts-row{{display:flex;justify-content:space-between;padding:3px 0;
      border-bottom:1px solid #21262d}}
    .opts-label{{color:#8b949e}}
    .conf-label{{font-size:.73rem;color:#484f58;margin-bottom:5px}}
    .conf-bar-bg{{height:5px;background:#21262d;border-radius:999px;overflow:hidden}}
    .conf-bar-fill{{height:100%;border-radius:999px}}

    /* Metrics table */
    .metrics-wrap{{max-width:720px;margin:0 auto 48px}}
    table{{width:100%;border-collapse:collapse;font-size:.87rem}}
    th{{background:#21262d;color:#8b949e;font-weight:600;padding:9px 14px;
      text-align:right;font-size:.76rem;text-transform:uppercase;letter-spacing:.5px}}
    th:first-child{{text-align:left}}
    td{{padding:9px 14px;border-bottom:1px solid #21262d;text-align:right}}
    td:first-child{{text-align:left}}
    tr:hover td{{background:#1c2128}}

    /* News */
    .news-row{{display:flex;gap:20px;margin-bottom:0;flex-wrap:wrap}}
    .news-col{{flex:1;min-width:280px}}
    .news-ticker{{font-size:1rem;font-weight:700;margin-bottom:8px}}
    .news-list{{list-style:none;font-size:.8rem;color:#8b949e;line-height:1.6}}
    .news-list li{{display:flex;gap:6px;align-items:flex-start;margin-bottom:4px;
      border-bottom:1px solid #21262d;padding-bottom:4px}}
    .dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:5px}}

    section{{margin-bottom:52px}}
    section h2{{font-size:1rem;font-weight:600;color:#8b949e;text-transform:uppercase;
      letter-spacing:1px;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #21262d}}
    .chart-wrap{{background:#161b22;border:1px solid #21262d;border-radius:12px;
      overflow:hidden;padding:8px}}
    footer{{text-align:center;color:#484f58;font-size:.74rem;margin-top:32px}}
  </style>
</head>
<body>
  <h1>Quant Signal Model v3</h1>
  <p class="subtitle">IBB · MRNA · LLY &nbsp;|&nbsp; GARCH + DCC + HMM + VIX + Options + Sentiment</p>
  <div class="badges">
    <span class="badge">GARCH(1,1)</span><span class="badge">DCC-EWMA</span>
    <span class="badge">HMM Regime</span><span class="badge">PCA Factor</span>
    <span class="badge">Expanding Walk-Forward</span><span class="badge">Calibrated Probs</span>
    <span class="badge free">FREE: VIX + VVIX History</span>
    <span class="badge free">FREE: Live Options Chain</span>
    <span class="badge free">FREE: VADER News Sentiment</span>
  </div>

  <div class="metrics-wrap">
    <section><h2>OOS Performance Summary (expanding walk-forward)</h2>
    <table><thead><tr>
      <th>Ticker</th><th>Accuracy</th><th>Sharpe</th><th>Sortino</th>
      <th>Max DD</th><th>Total Return</th>
    </tr></thead><tbody>{metrics_rows}</tbody></table></section>
  </div>

  <div class="pred-section">
    <h2>Tomorrow's Composite Signal (as of {last_date}) — Model + Options Overlay + News Sentiment</h2>
    <div class="pred-row">{tomorrow_html}</div>
  </div>

  <section><h2>Latest News Sentiment (VADER, no API key required)</h2>
    <div class="news-row">{news_html}</div>
  </section>

  <section><h2>01 — VIX + VVIX Fear Gauges (Free Historical IV Proxy)</h2>
    <div class="chart-wrap">{html_parts[0]}</div></section>
  <section><h2>02 — Asset IV Proxy (VIX * Beta) vs GARCH Conditional Vol</h2>
    <div class="chart-wrap">{html_parts[1]}</div></section>
  <section><h2>03 — Walk-Forward Backtest: Strategy vs Buy &amp; Hold</h2>
    <div class="chart-wrap">{html_parts[2]}</div></section>
  <section><h2>04 — Feature Importance (v3: incl. VIX, Earnings flags, IV proxy)</h2>
    <div class="chart-wrap">{html_parts[3]}</div></section>

  <footer>Quant Signal Model v3 · Antigravity · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
  · Disclaimer: educational only, not financial advice</footer>
</body>
</html>"""

with open("model_v3_dashboard.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("\n[OK] model_v3_dashboard.html saved.")
