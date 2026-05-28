"""
=============================================================================
  TRADE ANALYTICS — Post-Trade Analysis on Model v2
  Extracts trade-level metrics, identifies filters for optimisation
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TICKERS    = ["IBB", "MRNA", "LLY"]
START_DATE = "2022-02-20"
END_DATE   = "2026-02-20"
N_REGIMES  = 3
RETRAIN_FREQ = 21
MIN_TRAIN    = 200

BG="#0d1117"; PANEL="#161b22"; TEXT="#e6edf3"; GRID="#21262d"
COLORS = {"IBB":"#00d4ff", "MRNA":"#ff6b6b", "LLY":"#ffd93d"}
REGIME_COLORS = ["#ff6b6b","#ffd93d","#00ffd5"]
regime_names = {0:"Bear",1:"Sideways",2:"Bull"}

# ══════════════════════════════════════════════════════════════════════════════
# RERUN V2 CORE (same as model_v2.py, condensed)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/4] Reproducing v2 model ...")
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
prices = raw["Close"][TICKERS].dropna()
ret = np.log(prices / prices.shift(1)).dropna()

# GARCH
garch_vol_df = pd.DataFrame()
garch_resid_df = pd.DataFrame()
for t in TICKERS:
    am = arch_model(ret[t]*100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    sig = res.conditional_volatility / 100; sig.index = ret.index
    garch_vol_df[t] = sig
    garch_resid_df[t] = ret[t] / (sig + 1e-8)

# DCC
LAMBDA = 0.94
Q = [garch_resid_df.values[:21].T @ garch_resid_df.values[:21] / 21]
for i in range(1, len(garch_resid_df)):
    ev = garch_resid_df.values[i].reshape(-1,1)
    Q.append(LAMBDA*Q[-1] + (1-LAMBDA)*ev@ev.T)
def Q2R(Q_):
    d=np.sqrt(np.diag(Q_)); D=np.diag(1/(d+1e-10)); return D@Q_@D
R = [Q2R(q) for q in Q]
pairs = [("IBB","MRNA"),("IBB","LLY"),("MRNA","LLY")]
pidx = {("IBB","MRNA"):(0,1),("IBB","LLY"):(0,2),("MRNA","LLY"):(1,2)}
dcc_df = pd.DataFrame(
    {p:[R[t][i,j] for t in range(len(R))] for p,(i,j) in pidx.items()},
    index=garch_resid_df.index)

# HMM
hmm_feat = np.column_stack([ret["IBB"].values, garch_vol_df["IBB"].values,
                             garch_vol_df.mean(1).values])
hmm_feat = (hmm_feat - hmm_feat.mean(0))/(hmm_feat.std(0)+1e-8)
hmm = GaussianHMM(n_components=N_REGIMES, covariance_type="full", n_iter=500, random_state=42)
hmm.fit(hmm_feat)
reg_raw = hmm.predict(hmm_feat)
reg_s = pd.Series(reg_raw, index=ret.index)
mu_by = {s:ret["IBB"][reg_s==s].mean() for s in range(N_REGIMES)}
remap = {o:n for n,o in enumerate(sorted(mu_by, key=mu_by.get))}
regime = reg_s.map(remap)

# PCA
rolling_pc1 = pd.Series(index=ret.index, dtype=float)
idio = pd.DataFrame(index=ret.index, columns=TICKERS, dtype=float)
for i in range(63, len(ret)):
    w = ret.iloc[i-63:i]; X=StandardScaler().fit_transform(w)
    pca=PCA(n_components=1); sc=pca.fit_transform(X); ld=pca.components_[0]
    rolling_pc1.iloc[i]=sc[-1,0]
    for k,t in enumerate(TICKERS):
        idio.loc[ret.index[i],t]=w.iloc[-1][t]-sc[-1,0]*ld[k]*w.std()[t]
rolling_pc1=rolling_pc1.dropna(); idio=idio.dropna()

# Features v2
SHORT,MED,LONG=10,21,63
def build_features_v2():
    f=pd.DataFrame(index=ret.index)
    for t in TICKERS:
        r=ret[t]; gv=garch_vol_df[t]
        for lag in [1,2,3,5]: f[f"{t}_lag{lag}"]=r.shift(lag)
        for w in [5,MED,LONG]: f[f"{t}_mom_{w}d"]=r.shift(1).rolling(w).mean()
        f[f"{t}_garch_vol"]=gv.shift(1)
        f[f"{t}_garch_vol_chg"]=gv.diff(1).shift(1)
        f[f"{t}_vol_zscore"]=((gv-gv.rolling(LONG).mean())/(gv.rolling(LONG).std()+1e-8)).shift(1)
        if t in idio.columns:
            f[f"{t}_idio"]=idio[t].shift(1)
            f[f"{t}_idio_mom5"]=idio[t].shift(1).rolling(5).mean()
    for p in pairs:
        dcc=dcc_df[p].shift(1)
        f[f"dcc_{p[0]}_{p[1]}"]=dcc
        f[f"dcc_{p[0]}_{p[1]}_chg"]=dcc.diff(1)
        f[f"dcc_{p[0]}_{p[1]}_zscore"]=(dcc-dcc.rolling(LONG).mean())/(dcc.rolling(LONG).std()+1e-8)
    for s in range(N_REGIMES): f[f"regime_{s}"]=(regime.shift(1)==s).astype(float)
    f["regime_val"]=regime.shift(1)
    f["pc1_score"]=rolling_pc1.shift(1)
    f["pc1_score_mom"]=rolling_pc1.shift(1).rolling(5).mean()
    for (a,b) in pairs:
        sp=(ret[a]-ret[b]).shift(1)
        f[f"div_{a}_{b}"]=sp; f[f"div_corr_{a}_{b}"]=sp*dcc_df[(a,b)].shift(1)
        f[f"div_abs_{a}_{b}"]=sp.abs()
    for t in ["MRNA","LLY"]:
        f[f"vol_vs_ibb_{t}"]=(garch_vol_df[t]-garch_vol_df["IBB"]).shift(1)
    return f.dropna()

features=build_features_v2()
fnames=features.columns.tolist()
targets={t:(ret[t].shift(-1)>0).astype(int) for t in TICKERS}
cidx=features.index.intersection(
    pd.concat([targets[t].rename(t) for t in TICKERS],axis=1).dropna().index)
X_all=features.loc[cidx]; Y_all=pd.concat([targets[t].rename(t) for t in TICKERS],axis=1).loc[cidx]

print(f"       {len(fnames)} features, {len(cidx)} days")

# Walk-forward with PROBABILITY output
GBM_P=dict(n_estimators=300,max_depth=3,learning_rate=0.03,subsample=0.75,
           min_samples_leaf=8,max_features=0.6,random_state=42)
oos={t:dict(idx=[],prob=[],pred=[],true=[]) for t in TICKERS}
rpts=list(range(MIN_TRAIN, len(cidx), RETRAIN_FREQ))

for si,te in enumerate(rpts):
    pe=rpts[si+1] if si+1<len(rpts) else len(cidx)
    trX=X_all.iloc[:te]; trY=Y_all.iloc[:te]
    teX=X_all.iloc[te:pe]; tidx=cidx[te:pe]
    if len(teX)==0: continue
    for t in TICKERS:
        sc=StandardScaler(); Xtr=sc.fit_transform(trX); Xte=sc.transform(teX)
        clf=CalibratedClassifierCV(GradientBoostingClassifier(**GBM_P),method="sigmoid",cv=3)
        clf.fit(Xtr,trY[t])
        prob=clf.predict_proba(Xte)[:,1]
        oos[t]["idx"].extend(tidx.tolist())
        oos[t]["prob"].extend(prob.tolist())
        oos[t]["pred"].extend((prob>0.5).astype(int).tolist())
        oos[t]["true"].extend(Y_all[t].iloc[te:pe].tolist())
    if si%5==0: print(f"       {te/len(cidx)*100:4.0f}%", end="\r")

print("\n       Walk-forward complete.")

# ══════════════════════════════════════════════════════════════════════════════
# TRADE-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("[2/4] Extracting trade-level metrics ...")

all_trade_stats = {}

for t in TICKERS:
    idx = pd.DatetimeIndex(oos[t]["idx"])
    prob = np.array(oos[t]["prob"])
    pred = np.array(oos[t]["pred"])
    true = np.array(oos[t]["true"])
    signal = np.where(pred==1, 1, -1)
    daily_ret = ret[t].reindex(idx).values
    strategy_ret = signal * daily_ret
    reg_oos = regime.reindex(idx).values
    gv_oos  = garch_vol_df[t].reindex(idx).values

    # Confidence = distance from 0.5
    confidence = np.abs(prob - 0.5)

    # ── Build trade list: a "trade" = consecutive days with same signal ──
    trades = []
    trade_start = 0
    for i in range(1, len(signal)):
        if signal[i] != signal[trade_start] or i == len(signal)-1:
            end = i if signal[i] != signal[trade_start] else i+1
            t_slice = slice(trade_start, end)
            trade_pnl = strategy_ret[t_slice].sum()
            trade_dur = end - trade_start
            avg_conf  = confidence[t_slice].mean()
            avg_prob  = prob[t_slice].mean()
            avg_vol   = gv_oos[t_slice].mean() * np.sqrt(252) * 100
            trade_regime = int(pd.Series(reg_oos[t_slice]).mode().iloc[0]) if len(reg_oos[t_slice]) else 1
            direction = "LONG" if signal[trade_start] == 1 else "SHORT"
            day_of_week = idx[trade_start].dayofweek
            month = idx[trade_start].month

            trades.append(dict(
                ticker=t,
                start=idx[trade_start],
                end=idx[min(end-1, len(idx)-1)],
                direction=direction,
                duration=trade_dur,
                pnl=trade_pnl,
                pnl_pct=trade_pnl*100,
                win=1 if trade_pnl > 0 else 0,
                avg_confidence=avg_conf,
                avg_prob=avg_prob,
                avg_vol_ann=avg_vol,
                regime=trade_regime,
                regime_name=regime_names.get(trade_regime, "?"),
                day_of_week=day_of_week,
                month=month,
            ))
            trade_start = i

    df_trades = pd.DataFrame(trades)
    n_trades = len(df_trades)
    wins = df_trades["win"].sum()
    losses = n_trades - wins
    win_rate = wins / n_trades if n_trades else 0
    avg_win  = df_trades[df_trades["win"]==1]["pnl_pct"].mean()
    avg_loss = df_trades[df_trades["win"]==0]["pnl_pct"].mean()
    profit_factor = (df_trades[df_trades["pnl"]>0]["pnl"].sum() /
                     (abs(df_trades[df_trades["pnl"]<0]["pnl"].sum()) + 1e-8))
    avg_dur  = df_trades["duration"].mean()
    med_dur  = df_trades["duration"].median()
    max_dur  = df_trades["duration"].max()

    # Streaks
    streak = 0; max_win_streak = 0; max_loss_streak = 0
    curr_streak = 0; streak_type = None
    for _, tr in df_trades.iterrows():
        if tr["win"] == 1:
            if streak_type == "win": curr_streak += 1
            else: curr_streak = 1; streak_type = "win"
            max_win_streak = max(max_win_streak, curr_streak)
        else:
            if streak_type == "loss": curr_streak += 1
            else: curr_streak = 1; streak_type = "loss"
            max_loss_streak = max(max_loss_streak, curr_streak)

    # By confidence bucket
    df_trades["conf_bucket"] = pd.cut(df_trades["avg_confidence"],
                                       bins=[0, 0.03, 0.08, 0.15, 0.5],
                                       labels=["Low(0-3%)", "Med(3-8%)", "High(8-15%)", "VHigh(15%+)"])
    conf_stats = df_trades.groupby("conf_bucket", observed=True).agg(
        n=("win","count"), wr=("win","mean"), avg_pnl=("pnl_pct","mean"),
        total_pnl=("pnl_pct","sum")).reset_index()

    # By regime
    regime_stats = df_trades.groupby("regime_name").agg(
        n=("win","count"), wr=("win","mean"), avg_pnl=("pnl_pct","mean"),
        total_pnl=("pnl_pct","sum")).reset_index()

    # By direction
    dir_stats = df_trades.groupby("direction").agg(
        n=("win","count"), wr=("win","mean"), avg_pnl=("pnl_pct","mean"),
        total_pnl=("pnl_pct","sum")).reset_index()

    # By day of week
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    df_trades["dow_name"] = df_trades["day_of_week"].map(dow_map)
    dow_stats = df_trades.groupby("dow_name").agg(
        n=("win","count"), wr=("win","mean"), avg_pnl=("pnl_pct","mean"),
        total_pnl=("pnl_pct","sum")).reset_index()

    # By month
    month_stats = df_trades.groupby("month").agg(
        n=("win","count"), wr=("win","mean"), avg_pnl=("pnl_pct","mean"),
        total_pnl=("pnl_pct","sum")).reset_index()

    # Top5 / Bottom5 trades
    top5 = df_trades.nlargest(5, "pnl_pct")[["start","direction","duration","pnl_pct","regime_name","avg_confidence"]]
    bot5 = df_trades.nsmallest(5, "pnl_pct")[["start","direction","duration","pnl_pct","regime_name","avg_confidence"]]

    # Vol bucket
    df_trades["vol_bucket"] = pd.cut(df_trades["avg_vol_ann"],
                                      bins=[0,20,35,50,200],
                                      labels=["Low(<20%)","Med(20-35%)","High(35-50%)","VHigh(50%+)"])
    vol_stats = df_trades.groupby("vol_bucket", observed=True).agg(
        n=("win","count"), wr=("win","mean"), avg_pnl=("pnl_pct","mean"),
        total_pnl=("pnl_pct","sum")).reset_index()

    all_trade_stats[t] = dict(
        df=df_trades, n=n_trades, wins=wins, losses=losses,
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=profit_factor, avg_dur=avg_dur, med_dur=med_dur,
        max_dur=max_dur, max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        conf_stats=conf_stats, regime_stats=regime_stats,
        dir_stats=dir_stats, dow_stats=dow_stats, month_stats=month_stats,
        vol_stats=vol_stats, top5=top5, bot5=bot5,
    )

    print(f"\n  {'='*60}")
    print(f"  {t}  TRADE ANALYTICS")
    print(f"  {'='*60}")
    print(f"  Total trades: {n_trades}  |  W/L: {wins}/{losses}  |  Win rate: {win_rate:.1%}")
    print(f"  Avg win: {avg_win:+.2f}%  |  Avg loss: {avg_loss:+.2f}%  |  Profit factor: {profit_factor:.2f}")
    print(f"  Avg duration: {avg_dur:.1f}d  |  Median: {med_dur:.0f}d  |  Max: {max_dur}d")
    print(f"  Max win streak: {max_win_streak}  |  Max loss streak: {max_loss_streak}")
    print(f"\n  BY CONFIDENCE:")
    for _, row in conf_stats.iterrows():
        print(f"    {row['conf_bucket']:<15} n={row['n']:3.0f}  WR={row['wr']:.1%}  avg={row['avg_pnl']:+.2f}%  total={row['total_pnl']:+.1f}%")
    print(f"\n  BY REGIME:")
    for _, row in regime_stats.iterrows():
        print(f"    {row['regime_name']:<10} n={row['n']:3.0f}  WR={row['wr']:.1%}  avg={row['avg_pnl']:+.2f}%  total={row['total_pnl']:+.1f}%")
    print(f"\n  BY DIRECTION:")
    for _, row in dir_stats.iterrows():
        print(f"    {row['direction']:<6} n={row['n']:3.0f}  WR={row['wr']:.1%}  avg={row['avg_pnl']:+.2f}%  total={row['total_pnl']:+.1f}%")
    print(f"\n  BY VOL REGIME:")
    for _, row in vol_stats.iterrows():
        print(f"    {row['vol_bucket']:<15} n={row['n']:3.0f}  WR={row['wr']:.1%}  avg={row['avg_pnl']:+.2f}%  total={row['total_pnl']:+.1f}%")
    print(f"\n  BY DAY OF WEEK (trade start):")
    for _, row in dow_stats.iterrows():
        print(f"    {row['dow_name']:<4} n={row['n']:3.0f}  WR={row['wr']:.1%}  avg={row['avg_pnl']:+.2f}%")
    print(f"\n  TOP 5 TRADES:")
    for _, tr in top5.iterrows():
        print(f"    {tr['start'].strftime('%Y-%m-%d')}  {tr['direction']:<5}  {tr['duration']}d  "
              f"{tr['pnl_pct']:+.2f}%  {tr['regime_name']}  conf={tr['avg_confidence']:.3f}")
    print(f"\n  WORST 5 TRADES:")
    for _, tr in bot5.iterrows():
        print(f"    {tr['start'].strftime('%Y-%m-%d')}  {tr['direction']:<5}  {tr['duration']}d  "
              f"{tr['pnl_pct']:+.2f}%  {tr['regime_name']}  conf={tr['avg_confidence']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("  OPTIMISATION RECOMMENDATIONS")
print("="*70)

for t in TICKERS:
    s = all_trade_stats[t]
    cs = s["conf_stats"]; rs = s["regime_stats"]; vs = s["vol_stats"]; ds = s["dir_stats"]
    print(f"\n  --- {t} ---")

    # Best confidence bucket
    if len(cs):
        best_conf = cs.loc[cs["avg_pnl"].idxmax()]
        worst_conf = cs.loc[cs["avg_pnl"].idxmin()]
        print(f"  [CONF] Best bucket: {best_conf['conf_bucket']} (WR={best_conf['wr']:.1%}, avg={best_conf['avg_pnl']:+.2f}%)")
        print(f"         Worst bucket: {worst_conf['conf_bucket']} (WR={worst_conf['wr']:.1%}, avg={worst_conf['avg_pnl']:+.2f}%)")
        if worst_conf["avg_pnl"] < 0:
            print(f"         -> FILTER OUT '{worst_conf['conf_bucket']}' trades")

    # Best regime
    if len(rs):
        best_reg = rs.loc[rs["avg_pnl"].idxmax()]
        worst_reg = rs.loc[rs["avg_pnl"].idxmin()]
        print(f"  [REGIME] Best: {best_reg['regime_name']} (WR={best_reg['wr']:.1%}, avg={best_reg['avg_pnl']:+.2f}%)")
        print(f"           Worst: {worst_reg['regime_name']} (WR={worst_reg['wr']:.1%}, avg={worst_reg['avg_pnl']:+.2f}%)")
        if worst_reg["avg_pnl"] < -0.1:
            print(f"           -> AVOID trading in {worst_reg['regime_name']} regime")

    # Direction bias
    if len(ds):
        for _, dr in ds.iterrows():
            if dr["avg_pnl"] < -0.05:
                print(f"  [DIR] {dr['direction']} is losing (avg={dr['avg_pnl']:+.2f}%) -> consider removing")

    # Vol filter
    if len(vs):
        best_vol = vs.loc[vs["avg_pnl"].idxmax()]
        worst_vol = vs.loc[vs["avg_pnl"].idxmin()]
        print(f"  [VOL] Best vol regime: {best_vol['vol_bucket']} (WR={best_vol['wr']:.1%}, avg={best_vol['avg_pnl']:+.2f}%)")
        if worst_vol["avg_pnl"] < 0:
            print(f"         Worst: {worst_vol['vol_bucket']} -> AVOID when vol is {worst_vol['vol_bucket']}")

print("\n" + "="*70)

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
print("[3/4] Building analytics dashboard ...")

def hex_rgba(h,a=0.15):
    h=h.lstrip("#");return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

# FIG 1: Trade PnL distribution per ticker
fig1 = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t} — Trade P&L Distribution" for t in TICKERS],
    horizontal_spacing=0.07)
for j,t in enumerate(TICKERS,1):
    df = all_trade_stats[t]["df"]
    fig1.add_trace(go.Histogram(
        x=df["pnl_pct"], nbinsx=40,
        marker_color=COLORS[t], marker_line_color="#0d1117",
        marker_line_width=0.5, name=t,
        hovertemplate="P&L: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ), row=1, col=j)
    fig1.add_vline(x=0, line_dash="dash", line_color="#888", opacity=0.5, row=1, col=j)
fig1.update_layout(height=400,width=1400, paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT,family="Inter,sans-serif"), showlegend=False,
    title=dict(text="Trade P&L Distribution", font=dict(color=TEXT,size=17), x=0.5))
fig1.update_xaxes(gridcolor=GRID, title_text="P&L %")
fig1.update_yaxes(gridcolor=GRID, title_text="Count")
for a in fig1.layout.annotations: a.font.color=TEXT

# FIG 2: Win rate by confidence bucket
fig2 = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t} — Win Rate by Confidence" for t in TICKERS],
    horizontal_spacing=0.08)
for j,t in enumerate(TICKERS,1):
    cs = all_trade_stats[t]["conf_stats"]
    colors_bar = ["#00ffd5" if w>0.5 else "#ff6b6b" for w in cs["wr"]]
    fig2.add_trace(go.Bar(
        x=cs["conf_bucket"].astype(str), y=cs["wr"],
        marker_color=colors_bar, name=t,
        text=[f"{w:.0%}" for w in cs["wr"]], textposition="auto",
        textfont=dict(color=TEXT,size=11),
        hovertemplate="<b>%{x}</b><br>WR: %{y:.1%}<extra></extra>",
    ), row=1, col=j)
    fig2.add_hline(y=0.5,line_dash="dot",line_color="#888",opacity=0.5,row=1,col=j)
fig2.update_layout(height=400,width=1400,paper_bgcolor=BG,plot_bgcolor=PANEL,
    font=dict(color=TEXT,family="Inter,sans-serif"),showlegend=False,
    title=dict(text="Win Rate by Model Confidence", font=dict(color=TEXT,size=17),x=0.5))
fig2.update_xaxes(gridcolor=GRID)
fig2.update_yaxes(gridcolor=GRID,tickformat=".0%",range=[0,1])
for a in fig2.layout.annotations: a.font.color=TEXT

# FIG 3: Win rate by regime
fig3 = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t} — Performance by Regime" for t in TICKERS],
    horizontal_spacing=0.08)
for j,t in enumerate(TICKERS,1):
    rs = all_trade_stats[t]["regime_stats"]
    for _,row in rs.iterrows():
        rid = {"Bear":0,"Sideways":1,"Bull":2}.get(row["regime_name"],1)
        fig3.add_trace(go.Bar(
            x=[row["regime_name"]], y=[row["avg_pnl"]],
            marker_color=REGIME_COLORS[rid],
            text=[f"WR:{row['wr']:.0%}"], textposition="auto",
            textfont=dict(color=TEXT,size=11),
            showlegend=False,
            hovertemplate=f"<b>{row['regime_name']}</b><br>Avg P&L: {row['avg_pnl']:+.2f}%<br>n={row['n']:.0f}<extra></extra>",
        ), row=1, col=j)
    fig3.add_hline(y=0,line_dash="dot",line_color="#888",opacity=0.5,row=1,col=j)
fig3.update_layout(height=400,width=1400,paper_bgcolor=BG,plot_bgcolor=PANEL,
    font=dict(color=TEXT,family="Inter,sans-serif"),
    title=dict(text="Avg Trade P&L by HMM Regime", font=dict(color=TEXT,size=17),x=0.5))
fig3.update_xaxes(gridcolor=GRID)
fig3.update_yaxes(gridcolor=GRID,title_text="Avg P&L %")
for a in fig3.layout.annotations: a.font.color=TEXT

# FIG 4: Win rate by vol bucket
fig4 = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t} — Performance by GARCH Vol" for t in TICKERS],
    horizontal_spacing=0.08)
for j,t in enumerate(TICKERS,1):
    vs = all_trade_stats[t]["vol_stats"]
    colors_v = ["#00ffd5" if p>0 else "#ff6b6b" for p in vs["avg_pnl"]]
    fig4.add_trace(go.Bar(
        x=vs["vol_bucket"].astype(str), y=vs["avg_pnl"],
        marker_color=colors_v, name=t,
        text=[f"WR:{w:.0%}" for w in vs["wr"]], textposition="auto",
        textfont=dict(color=TEXT,size=10),
        hovertemplate="<b>%{x}</b><br>Avg P&L: %{y:+.2f}%<extra></extra>",
    ), row=1, col=j)
    fig4.add_hline(y=0,line_dash="dot",line_color="#888",opacity=0.5,row=1,col=j)
fig4.update_layout(height=400,width=1400,paper_bgcolor=BG,plot_bgcolor=PANEL,
    font=dict(color=TEXT,family="Inter,sans-serif"),showlegend=False,
    title=dict(text="Avg Trade P&L by GARCH Volatility Regime", font=dict(color=TEXT,size=17),x=0.5))
fig4.update_xaxes(gridcolor=GRID)
fig4.update_yaxes(gridcolor=GRID,title_text="Avg P&L %")
for a in fig4.layout.annotations: a.font.color=TEXT

# FIG 5: Trade duration distribution
fig5 = make_subplots(rows=1, cols=3,
    subplot_titles=[f"{t} — Trade Duration Distribution" for t in TICKERS],
    horizontal_spacing=0.07)
for j,t in enumerate(TICKERS,1):
    df=all_trade_stats[t]["df"]
    fig5.add_trace(go.Histogram(
        x=df["duration"], nbinsx=30,
        marker_color=COLORS[t], marker_line_color="#0d1117",
        hovertemplate="Duration: %{x}d<br>Count: %{y}<extra></extra>",
    ), row=1, col=j)
fig5.update_layout(height=380,width=1400,paper_bgcolor=BG,plot_bgcolor=PANEL,
    font=dict(color=TEXT,family="Inter,sans-serif"),showlegend=False,
    title=dict(text="Trade Duration Distribution (days)", font=dict(color=TEXT,size=17),x=0.5))
fig5.update_xaxes(gridcolor=GRID,title_text="Days")
fig5.update_yaxes(gridcolor=GRID,title_text="Count")
for a in fig5.layout.annotations: a.font.color=TEXT

# ─── SUMMARY TABLE HTML ───────────────────────────────────────────────────────
summary_rows = ""
for t in TICKERS:
    s = all_trade_stats[t]
    summary_rows += f"""<tr>
      <td style="color:{COLORS[t]};font-weight:700">{t}</td>
      <td>{s['n']}</td><td>{s['wins']}/{s['losses']}</td>
      <td style="color:{'#00ffd5' if s['win_rate']>0.5 else '#ff6b6b'}">{s['win_rate']:.1%}</td>
      <td style="color:#00ffd5">{s['avg_win']:+.2f}%</td>
      <td style="color:#ff6b6b">{s['avg_loss']:+.2f}%</td>
      <td>{s['profit_factor']:.2f}</td>
      <td>{s['avg_dur']:.1f}d</td><td>{s['med_dur']:.0f}d</td>
      <td>{s['max_win_streak']}</td><td>{s['max_loss_streak']}</td>
    </tr>"""

# Recommendations HTML
recs_html = ""
for t in TICKERS:
    s = all_trade_stats[t]
    cs=s["conf_stats"]; rs=s["regime_stats"]; vs=s["vol_stats"]
    recs = []
    if len(cs):
        wc=cs.loc[cs["avg_pnl"].idxmin()]
        bc=cs.loc[cs["avg_pnl"].idxmax()]
        if wc["avg_pnl"]<0: recs.append(f"Filter out <b>{wc['conf_bucket']}</b> confidence trades (avg P&L={wc['avg_pnl']:+.2f}%)")
        recs.append(f"Best confidence: <b>{bc['conf_bucket']}</b> (WR={bc['wr']:.0%}, avg={bc['avg_pnl']:+.2f}%)")
    if len(rs):
        wr=rs.loc[rs["avg_pnl"].idxmin()]
        if wr["avg_pnl"]<-0.05: recs.append(f"Avoid <b>{wr['regime_name']}</b> regime (avg P&L={wr['avg_pnl']:+.2f}%)")
    if len(vs):
        wv=vs.loc[vs["avg_pnl"].idxmin()]
        if wv["avg_pnl"]<0: recs.append(f"Reduce size when GARCH vol is <b>{wv['vol_bucket']}</b>")
    rec_li = "".join(f"<li>{r}</li>" for r in recs)
    recs_html += f'<div class="rec-card"><div class="rec-ticker" style="color:{COLORS[t]}">{t}</div><ul>{rec_li}</ul></div>'

# Assemble
html_parts = [f.to_html(full_html=False, include_plotlyjs="cdn")
              for f in [fig1, fig2, fig3, fig4, fig5]]

full_html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Trade Analytics — v2 Model</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Inter',sans-serif;padding:32px 24px}}
h1{{text-align:center;font-size:2.2rem;font-weight:800;margin-bottom:6px;
    background:linear-gradient(135deg,#00d4ff,#a29bfe,#ffd93d);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.subtitle{{text-align:center;color:#8b949e;font-size:.88rem;margin-bottom:32px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem;margin-bottom:8px}}
th{{background:#21262d;color:#8b949e;padding:10px 12px;text-align:right;
    font-size:.75rem;text-transform:uppercase;letter-spacing:.5px}}
th:first-child{{text-align:left}}
td{{padding:10px 12px;border-bottom:1px solid #21262d;text-align:right}}
td:first-child{{text-align:left}}
tr:hover td{{background:#1c2128}}
.metrics-wrap{{max-width:960px;margin:0 auto 48px}}
section{{margin-bottom:52px}}
section h2{{font-size:1rem;font-weight:600;color:#8b949e;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #21262d}}
.chart-wrap{{background:#161b22;border:1px solid #21262d;border-radius:12px;overflow:hidden;padding:8px}}
.rec-row{{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;margin-bottom:48px}}
.rec-card{{flex:1;min-width:280px;max-width:380px;background:#161b22;border:1px solid #21262d;
    border-radius:12px;padding:20px 22px;transition:transform .2s}}
.rec-card:hover{{transform:translateY(-3px);box-shadow:0 6px 24px rgba(0,0,0,.4)}}
.rec-ticker{{font-size:1.4rem;font-weight:800;margin-bottom:10px}}
.rec-card ul{{padding-left:18px;font-size:.82rem;line-height:1.7;color:#c9d1d9}}
footer{{text-align:center;color:#484f58;font-size:.74rem;margin-top:32px}}
</style></head><body>
<h1>Trade Analytics — v2 Model</h1>
<p class="subtitle">Per-trade decomposition: duration, confidence, regime, vol splits, optimisation filters</p>

<div class="metrics-wrap"><section><h2>Trade Summary</h2>
<table><thead><tr>
<th>Ticker</th><th>Trades</th><th>W/L</th><th>Win Rate</th><th>Avg Win</th><th>Avg Loss</th>
<th>Profit Factor</th><th>Avg Dur</th><th>Med Dur</th><th>Win Streak</th><th>Loss Streak</th>
</tr></thead><tbody>{summary_rows}</tbody></table></section></div>

<section><h2>Optimisation Recommendations</h2>
<div class="rec-row">{recs_html}</div></section>

<section><h2>01 — Trade P&L Distribution</h2>
<div class="chart-wrap">{html_parts[0]}</div></section>
<section><h2>02 — Win Rate by Model Confidence</h2>
<div class="chart-wrap">{html_parts[1]}</div></section>
<section><h2>03 — Avg Trade P&L by HMM Regime</h2>
<div class="chart-wrap">{html_parts[2]}</div></section>
<section><h2>04 — Avg Trade P&L by GARCH Vol Level</h2>
<div class="chart-wrap">{html_parts[3]}</div></section>
<section><h2>05 — Trade Duration Distribution</h2>
<div class="chart-wrap">{html_parts[4]}</div></section>

<footer>Trade Analytics · v2 Model · Antigravity · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
· Disclaimer: educational only, not financial advice</footer>
</body></html>"""

with open("trade_analytics.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("[4/4] trade_analytics.html saved.")
