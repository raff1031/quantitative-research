# diffmatrix.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

CSV_IN  = r"C:\Users\sas\Documents\NinjaTrader 8\FFD_scan_NQ DEC25_15Minute.csv"
CSV_OUT = r"C:\Users\sas\Documents\NinjaTrader 8\FFD_scan_results.csv"
ALPHA   = 0.05

def load_ffd_csv(path):
    # 1) tenta con header normale (colonna 'time')
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=['time'])
        if 'time' in df.columns:
            return df
    except Exception:
        pass
    # 2) fallback: file SENZA header -> assegna nomi
    df = pd.read_csv(path, encoding="utf-8-sig", header=None)
    ncols = df.shape[1]
    names = ['time', 'close'] + [f'ffd_{i}' for i in range(1, ncols-1)]
    df.columns = names
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

df = load_ffd_csv(CSV_IN).dropna(subset=['time'])
df = df.sort_values('time').reset_index(drop=True)

# colonne FFD (tutto ciò che inizia con 'ffd_')
ffd_cols = [c for c in df.columns if c.startswith('ffd_')]
if not ffd_cols:
    raise SystemExit("Nessuna colonna FFD trovata (serve almeno ffd_1).")

rows = []
close = df['close'].values.astype(float)

for c in ffd_cols:
    s = df[c].astype(float)
    mask = s.notna()
    s = s[mask].values
    close_aligned = df.loc[mask, 'close'].values.astype(float)

    n = len(s)
    if n < 20:
        rows.append({'column': c, 'pvalue': np.nan, 'passes_adf@5%': False,
                     'corr_with_close': np.nan, 'nobs': n})
        continue

    # maxlag prudente per campioni corti
    maxlag = max(0, min(int(np.sqrt(n)), 12))
    try:
        adf_stat, pval, lags, nobs, crit, _ = adfuller(s, maxlag=maxlag, regression='c', autolag='AIC')
    except ValueError:
        adf_stat, pval, lags, nobs, crit, _ = adfuller(s, maxlag=0, regression='c', autolag=None)

    corr = np.corrcoef(close_aligned, s)[0, 1]

    rows.append({
        'column': c, 'adf_stat': adf_stat, 'pvalue': pval, 'lags': lags, 'nobs': n,
        'crit_1%': crit['1%'], 'crit_5%': crit['5%'], 'crit_10%': crit['10%'],
        'passes_adf@5%': pval < ALPHA, 'corr_with_close': corr
    })

res = pd.DataFrame(rows).sort_values(
    by=['passes_adf@5%', 'pvalue', 'corr_with_close'],
    ascending=[False, True, False]
)
print(res.head(12))
res.to_csv(CSV_OUT, index=False)
print(f"\nSalvato: {CSV_OUT}")
