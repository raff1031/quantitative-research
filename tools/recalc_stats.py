"""Recompute headline stats from the NT8 export CSVs on ALL rows, with standard
definitions: daily-P&L Sharpe (weekdays, zero-filled, x sqrt(252)) and max
drawdown in dollars on the chronological cumulative P&L. Read-only, stdlib only.

usage: python tools/recalc_stats.py .   (from the repository root)
"""
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

root = Path(sys.argv[1])


def money_it(s):  # "1.234,56 $" / "-840,68 $"
    s = (s or "").replace("$", "").replace(" ", "").replace(" ", "").strip()
    if not s:
        return 0.0
    neg = s.startswith("(") or s.startswith("-")
    s = s.strip("()-").replace(".", "").replace(",", ".")
    return -float(s) if neg else float(s)


def money_us(s):  # "$2000.00" / "($230.52)" / "$1,234.00"
    s = (s or "").replace("$", "").replace(",", "").strip()
    if not s:
        return 0.0
    neg = s.startswith("(") or s.startswith("-")
    s = s.strip("()-")
    return -float(s) if neg else float(s)


def daily_stats(daily):
    """daily: dict date -> net P&L. Returns Sharpe / maxDD / yearly breakdown on a weekday grid."""
    first, last = min(daily), max(daily)
    days, d = [], first
    while d <= last:
        if d.weekday() < 5 or d in daily:
            days.append(d)
        d += timedelta(days=1)
    vals = [daily.get(x, 0.0) for x in days]
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    sharpe = mean / sd * math.sqrt(252) if sd > 0 else float("nan")
    cum = peak = 0.0
    maxdd = 0.0
    yearly = defaultdict(float)
    for x, v in zip(days, vals):
        cum += v
        peak = max(peak, cum)
        maxdd = min(maxdd, cum - peak)
        yearly[x.year] += v
    return dict(sharpe=sharpe, maxdd=maxdd, pos_years=sum(1 for v in yearly.values() if v > 0),
                years=len(yearly), worst_year=min(yearly.values()), first=first, last=last, daily=daily)


def merge(dicts):
    out = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            out[k] += v
    return dict(out)


def per_period(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline=""), delimiter=";"))
    daily, bad = defaultdict(float), 0
    trades = wins = gp = gl = comm = 0.0
    for r in rows:
        try:
            d = datetime.strptime(r["Period"].strip(), "%d/%m/%Y").date()
        except (ValueError, AttributeError, KeyError):
            bad += 1
            continue
        k = float(r["#"])
        trades += k
        wins += k * float(r["% Win"].replace("%", "").replace(",", ".")) / 100.0
        gp += money_it(r["Gross profit"]); gl += money_it(r["Gross loss"])
        comm += money_it(r["Commission"]) * k
        daily[d] += money_it(r["Net profit"])
    good = [r for r in rows if r.get("Period")]
    st = daily_stats(dict(daily))
    return dict(rows=len(rows) - bad, unparsed=bad, trades=int(trades), net=sum(daily.values()),
                last_cum=money_it(good[-1]["Cum. net profit"]), pf=gp / abs(gl), wr=wins / trades,
                commission=comm, **st)


def per_trade(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    daily, bad, recs, instr, comm = defaultdict(float), 0, [], set(), 0.0
    for r in rows:
        try:
            t = datetime.strptime(r["Exit time"].strip(), "%m/%d/%Y %I:%M:%S %p")
        except (ValueError, AttributeError, KeyError):
            bad += 1
            continue
        p = money_us(r["Profit"])
        recs.append((t, p, money_us(r["Cum. net profit"])))
        instr.add(r["Instrument"]); comm += money_us(r["Commission"])
        daily[t.date()] += p
    recs.sort(key=lambda x: x[0])
    ps = [p for _, p, _ in recs]
    st = daily_stats(dict(daily))
    return dict(rows=len(recs), unparsed=bad, trades=len(recs), net=sum(ps), last_cum=recs[-1][2], profits=ps,
                pf=sum(p for p in ps if p > 0) / abs(sum(p for p in ps if p < 0)),
                wr=sum(1 for p in ps if p > 0) / len(ps), commission=comm,
                instrument=", ".join(sorted(instr)), **st)


def show(label, r):
    print(f"{label}\n   rows={r['rows']} unparsed={r['unparsed']} trades={r['trades']} "
          f"{r['first']}..{r['last']} | net={r['net']:,.0f} (file cum={r['last_cum']:,.0f}) "
          f"PF={r['pf']:.2f} WR={r['wr']*100:.1f}% | Sharpe(daily)={r['sharpe']:.2f} "
          f"maxDD={r['maxdd']:,.0f} net/DD={r['net']/abs(r['maxdd']):.1f} | "
          f"pos years {r['pos_years']}/{r['years']} worst year={r['worst_year']:,.0f} | "
          f"commission total={r['commission']:,.0f}" + (f" | instr={r['instrument']}" if 'instrument' in r else ""))


def show_comb(label, rs):
    c = daily_stats(merge(r["daily"] for r in rs))
    ps = [x for r in rs for x in r.get("profits", [])]
    extra = (f" PF={sum(x for x in ps if x > 0) / abs(sum(x for x in ps if x < 0)):.2f}"
             f" WR={sum(1 for x in ps if x > 0) / len(ps) * 100:.1f}%") if ps else ""
    print(f"   {label}: trades={sum(r['trades'] for r in rs)} net={sum(r['net'] for r in rs):,.0f}{extra} "
          f"Sharpe={c['sharpe']:.2f} maxDD={c['maxdd']:,.0f} pos years {c['pos_years']}/{c['years']} "
          f"worst year={c['worst_year']:,.0f}")


print("== ORB (per-day export) ==")
for f in ["csv orb/nq orb 10x30 bar confirm 3 m5.csv", "csv orb/nq orb 0x20 bar confirm2 m5.csv"]:
    show(f, per_period(root / f))

print("\n== Sweep (per-day export) ==")
for f in ["csv strat sweep/NQ 6-7 tp 5 6 bb 50 2 2-3 sweep.csv", "csv strat sweep/NQ 9-16 tp 6 4 bb 30 2 1-2 sweep.csv"]:
    show(f, per_period(root / f))

print("\n== Pattern 5 (per-trade export) ==")
p5 = []
for f in ["pattern 5 csv/ES pattern 5.csv", "pattern 5 csv/nq pattern 5.csv"]:
    r = per_trade(root / f); p5.append(r); show(f, r)
show_comb("combined ES+NQ", p5)

print("\n== FVG (per-trade export) ==")
fvg = {}
for f in sorted((root / "FVG csv").glob("* m*.csv")):
    fvg[f.name] = per_trade(f); show(f.name, fvg[f.name])
show_comb("combined 7 files", list(fvg.values()))
show_comb("combined w/o mislabelled 'ES' file", [v for k, v in fvg.items() if not k.startswith("ES ")])
