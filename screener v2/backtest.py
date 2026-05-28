"""
Backtest Engine — Edge Scanner
================================
Two strategies tested against SPY buy-and-hold:

  1. MOMENTUM-QUALITY: Monthly rebalance, buy top-N by technical + RS + quality
  2. CONGRESS-FOLLOW: Buy stocks congress is buying (from GitHub historical data)

Usage:
    python backtest.py                              # momentum-quality, top 5, 2y
    python backtest.py --top 10 --years 3           # top 10, 3 years
    python backtest.py --strategy congress --years 2 # congress-follow strategy
    python backtest.py --weighted                    # score-weighted allocation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os
import ta

warnings.filterwarnings('ignore')


# ============================================================================
# FAST SCORER (price-based only — no API calls to fundamentals)
# ============================================================================

class FastScorer:
    """
    Lightweight scoring for backtesting — uses only price data
    so it can run over historical periods without lookahead bias.
    """

    def __init__(self):
        self._cache = {}

    def _prices(self, ticker, start, end):
        key = f"{ticker}|{start}|{end}"
        if key not in self._cache:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                self._cache[key] = df
            except:
                self._cache[key] = pd.DataFrame()
        return self._cache[key]

    def score(self, ticker, as_of_date, spy_hist=None):
        """
        Score a stock as of a date using only data available up to that date.
        Returns dict with composite score 0-100 or None if insufficient data.
        """
        start = (as_of_date - timedelta(days=300)).strftime('%Y-%m-%d')
        end = as_of_date.strftime('%Y-%m-%d')

        h = self._prices(ticker, start, end)
        if h is None or len(h) < 60:
            return None

        c = h['Close']

        tech   = self._score_technical(h, c)
        rs     = self._score_rs(c, spy_hist, as_of_date) if spy_hist is not None else 50.0
        mom    = self._score_momentum(c)
        qual   = self._score_quality(c)

        composite = tech * 0.25 + rs * 0.25 + mom * 0.30 + qual * 0.20

        return {
            'ticker': ticker, 'date': as_of_date,
            'composite': round(composite, 2),
            'technical': round(tech, 1), 'rs': round(rs, 1),
            'momentum': round(mom, 1), 'quality': round(qual, 1),
        }

    def _score_technical(self, df, close):
        s = 50.0
        rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1]
        if 40 <= rsi <= 70: s += 10
        elif rsi > 80: s -= 15
        elif rsi < 25: s += 5  # deep oversold

        macd_h = ta.trend.MACD(close).macd_diff().iloc[-1]
        s += 10 if macd_h > 0 else -5

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        cur = close.iloc[-1]
        if cur > sma20: s += 5
        if cur > sma50: s += 5
        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            if cur > sma200: s += 10
            if sma50 > sma200: s += 5
            else: s -= 5

        if len(df) >= 14:
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], close, 14).adx().iloc[-1]
            if adx > 25: s += 5

        vol = close.pct_change().std() * np.sqrt(252)
        if vol < 0.20: s += 5
        elif vol > 0.50: s -= 10

        return np.clip(s, 0, 100)

    def _score_rs(self, stock_close, spy_hist, as_of_date):
        if spy_hist is None or spy_hist.empty:
            return 50.0

        start = as_of_date - timedelta(days=260)
        end = as_of_date

        s_mask = (stock_close.index >= start) & (stock_close.index <= end)
        b_mask = (spy_hist.index >= start) & (spy_hist.index <= end)

        sc = stock_close[s_mask]
        bc = spy_hist[b_mask]
        common = sc.index.intersection(bc.index)
        if len(common) < 60:
            return 50.0

        rs = sc[common] / bc[common]
        rs_sma = rs.rolling(50).mean()
        if rs_sma.dropna().empty:
            return 50.0

        mansfield = ((rs.iloc[-1] / rs_sma.iloc[-1]) - 1) * 100

        s = 50.0
        if mansfield > 5: s += 25
        elif mansfield > 2: s += 15
        elif mansfield > 0: s += 5
        elif mansfield > -2: s -= 5
        elif mansfield > -5: s -= 15
        else: s -= 25

        for d in [21, 63]:
            if len(rs) >= d:
                rs_ret = (rs.iloc[-1] / rs.iloc[-d] - 1) * 100
                s += 5 if rs_ret > 3 else (-5 if rs_ret < -3 else 0)

        return np.clip(s, 0, 100)

    def _score_momentum(self, close):
        s = 50.0
        for days, bonus in [(126, 15), (63, 10), (21, 5)]:
            if len(close) >= days:
                ret = (close.iloc[-1] / close.iloc[-days] - 1) * 100
                if ret > 20: s += bonus
                elif ret > 10: s += bonus * 0.6
                elif ret > 0: s += bonus * 0.3
                elif ret > -10: s -= bonus * 0.3
                else: s -= bonus * 0.6

        rets = close.pct_change().dropna()
        if len(rets) > 20 and rets.std() > 0:
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
            if sharpe > 1.5: s += 10
            elif sharpe > 1.0: s += 5
            elif sharpe < 0: s -= 10

        return np.clip(s, 0, 100)

    def _score_quality(self, close):
        s = 50.0
        monthly = close.resample('ME').last().pct_change().dropna()
        if len(monthly) >= 3:
            wr = (monthly > 0).mean()
            if wr > 0.7: s += 15
            elif wr > 0.6: s += 10
            elif wr > 0.5: s += 5
            elif wr < 0.4: s -= 10

        peak = close.cummax()
        dd = ((close - peak) / peak).min() * 100
        if dd > -10: s += 15
        elif dd > -15: s += 10
        elif dd > -20: s += 5
        elif dd > -30: s -= 5
        else: s -= 15

        sma50 = close.rolling(50).mean()
        if sma50.dropna().any():
            pct_above = (close > sma50).mean()
            if pct_above > 0.7: s += 10
            elif pct_above > 0.5: s += 5
            elif pct_above < 0.3: s -= 10

        return np.clip(s, 0, 100)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class Backtester:

    def __init__(self, universe, top_n=5, weighted=False, initial_capital=10000):
        self.universe = universe
        self.top_n = top_n
        self.weighted = weighted
        self.initial = initial_capital
        self.scorer = FastScorer()

    def run(self, start_date, end_date):
        print(f"\n{'='*65}")
        print(f"  BACKTEST: Top-{self.top_n} Multi-Factor, Monthly Rebalance")
        print(f"  Universe: {len(self.universe)} stocks | {'Score-weighted' if self.weighted else 'Equal-weight'}")
        print(f"  Period: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")
        print(f"{'='*65}")

        # Monthly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        if len(dates) < 3:
            print("  Need at least 3 months.")
            return None

        # Download all prices once
        print("\n  📥 Downloading prices...")
        full_start = (start_date - timedelta(days=310)).strftime('%Y-%m-%d')
        full_end = (end_date + timedelta(days=5)).strftime('%Y-%m-%d')

        prices = {}
        for t in self.universe + ['SPY']:
            try:
                df = yf.download(t, start=full_start, end=full_end, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    prices[t] = df
            except:
                pass

        available = [t for t in self.universe if t in prices]
        print(f"  Data for {len(available)}/{len(self.universe)} stocks + SPY\n")

        if 'SPY' not in prices:
            print("  ❌ No SPY data.")
            return None

        spy_close = prices['SPY']['Close']

        # Run month by month
        port_rets, spy_rets, log = [], [], []

        for i in range(len(dates) - 1):
            d = dates[i].to_pydatetime()
            d_next = dates[i + 1].to_pydatetime()

            print(f"  {d:%Y-%m}: ", end="")

            # Score all stocks
            scored = []
            for t in available:
                hist = prices[t]
                # Slice to as-of date
                mask = hist.index <= d
                if mask.sum() < 60:
                    continue
                sub = hist[mask]
                result = self.scorer.score(t, d, spy_close[spy_close.index <= d])
                if result:
                    scored.append(result)

            if len(scored) < self.top_n:
                print(f"only {len(scored)} scores, need {self.top_n}")
                port_rets.append(0)
                spy_rets.append(self._period_return(prices['SPY'], d, d_next))
                continue

            scored.sort(key=lambda x: x['composite'], reverse=True)
            picks = scored[:self.top_n]
            tickers = [p['ticker'] for p in picks]
            print(f"{', '.join(tickers)}")

            # Weights
            if self.weighted:
                total_s = sum(p['composite'] for p in picks)
                w = {p['ticker']: p['composite'] / total_s for p in picks} if total_s > 0 else {p['ticker']: 1/len(picks) for p in picks}
            else:
                w = {p['ticker']: 1 / len(picks) for p in picks}

            # Monthly return
            month_ret = sum(w[t] * self._period_return(prices.get(t), d, d_next) for t in tickers)
            spy_ret = self._period_return(prices['SPY'], d, d_next)

            port_rets.append(month_ret)
            spy_rets.append(spy_ret)

            log.append({
                'month': f"{d:%Y-%m}",
                'picks': tickers,
                'scores': [p['composite'] for p in picks],
                'port_ret': round(month_ret * 100, 2),
                'spy_ret': round(spy_ret * 100, 2),
                'alpha': round((month_ret - spy_ret) * 100, 2),
            })

        # Compute & display
        return self._results(np.array(port_rets), np.array(spy_rets), log)

    def _period_return(self, df, start, end):
        if df is None or df.empty:
            return 0
        mask = (df.index >= start) & (df.index < end)
        p = df[mask]
        if len(p) < 2:
            return 0
        return (p['Close'].iloc[-1] / p['Close'].iloc[0]) - 1

    def _results(self, port, spy, log):
        port_cum = np.cumprod(1 + port)
        spy_cum = np.cumprod(1 + spy)
        n = len(port)
        ny = n / 12

        # Metrics
        total_p = (port_cum[-1] - 1) * 100
        total_s = (spy_cum[-1] - 1) * 100
        cagr_p = (port_cum[-1] ** (1/ny) - 1) * 100 if ny > 0 and port_cum[-1] > 0 else 0
        cagr_s = (spy_cum[-1] ** (1/ny) - 1) * 100 if ny > 0 and spy_cum[-1] > 0 else 0
        sharpe_p = (port.mean() / port.std()) * np.sqrt(12) if port.std() > 0 else 0
        sharpe_s = (spy.mean() / spy.std()) * np.sqrt(12) if spy.std() > 0 else 0
        dd_p = self._max_dd(port_cum)
        dd_s = self._max_dd(spy_cum)
        vol_p = port.std() * np.sqrt(12) * 100
        vol_s = spy.std() * np.sqrt(12) * 100
        down = port[port < 0]
        sortino = (port.mean() / down.std()) * np.sqrt(12) if len(down) > 0 and down.std() > 0 else 0
        calmar = cagr_p / abs(dd_p) if dd_p != 0 else 0
        alpha_arr = port - spy
        win_rate = (alpha_arr > 0).mean() * 100
        avg_alpha = alpha_arr.mean() * 100

        # Display
        print(f"\n{'='*65}")
        print(f"  RESULTS — {n} months ({ny:.1f} years)")
        print(f"{'='*65}")

        print(f"\n  RETURNS")
        print(f"  {'─'*55}")
        print(f"  Strategy  {total_p:+8.2f}%   CAGR {cagr_p:+.2f}%")
        print(f"  SPY B&H   {total_s:+8.2f}%   CAGR {cagr_s:+.2f}%")
        print(f"  Alpha     {total_p - total_s:+8.2f}%")

        print(f"\n  RISK")
        print(f"  {'─'*55}")
        print(f"  {'':22s}  {'Strategy':>10s}  {'SPY':>10s}")
        print(f"  {'Sharpe':22s}  {sharpe_p:10.2f}  {sharpe_s:10.2f}")
        print(f"  {'Sortino':22s}  {sortino:10.2f}")
        print(f"  {'Calmar':22s}  {calmar:10.2f}")
        print(f"  {'Max Drawdown':22s}  {dd_p:9.2f}%  {dd_s:9.2f}%")
        print(f"  {'Volatility (ann.)':22s}  {vol_p:9.2f}%  {vol_s:9.2f}%")

        print(f"\n  ALPHA")
        print(f"  {'─'*55}")
        print(f"  Win rate vs SPY:     {win_rate:.1f}%")
        print(f"  Avg monthly alpha:   {avg_alpha:+.2f}%")
        print(f"  Best month:          {port.max()*100:+.2f}%")
        print(f"  Worst month:         {port.min()*100:+.2f}%")

        # Monthly log
        print(f"\n  MONTHLY LOG")
        print(f"  {'─'*55}")
        print(f"  {'Month':8s} {'Port':>8s} {'SPY':>8s} {'Alpha':>8s}  Holdings")
        for entry in log:
            print(f"  {entry['month']:8s} {entry['port_ret']:+7.2f}% {entry['spy_ret']:+7.2f}% "
                  f"{entry['alpha']:+7.2f}%  {','.join(entry['picks'])}")

        # Equity curve
        print(f"\n  EQUITY CURVE (${self.initial:,} initial)")
        print(f"  {'─'*55}")
        pe = port_cum * self.initial
        se = spy_cum * self.initial
        step = max(1, n // 20)
        for i in range(0, n, step):
            month = log[i]['month'] if i < len(log) else ""
            print(f"  {month:8s}  Strategy: ${pe[i]:>10,.0f}   SPY: ${se[i]:>10,.0f}   "
                  f"{'▲' if pe[i] > se[i] else '▼'} {(pe[i]/se[i]-1)*100:+.1f}%")
        print(f"  {'FINAL':8s}  Strategy: ${pe[-1]:>10,.0f}   SPY: ${se[-1]:>10,.0f}   "
              f"{'▲' if pe[-1] > se[-1] else '▼'} {(pe[-1]/se[-1]-1)*100:+.1f}%")

        # Verdict
        print(f"\n  {'='*55}")
        if sharpe_p > sharpe_s and total_p > total_s:
            print(f"  ✅ OUTPERFORMS on return + risk-adjusted basis")
        elif total_p > total_s:
            print(f"  🟡 Higher return but check Sharpe ratio")
        elif sharpe_p > sharpe_s:
            print(f"  🟡 Better Sharpe but lower absolute return")
        else:
            print(f"  ❌ SPY WINS in this period")
        print(f"  {'='*55}\n")

        # Save trade log
        log_df = pd.DataFrame(log)
        log_path = os.path.join(os.path.dirname(__file__), 'backtest_log.csv')
        log_df.to_csv(log_path, index=False)
        print(f"  📄 Log saved: {log_path}")

        return {
            'total_return': total_p, 'spy_return': total_s,
            'cagr': cagr_p, 'spy_cagr': cagr_s,
            'sharpe': sharpe_p, 'spy_sharpe': sharpe_s,
            'sortino': sortino, 'calmar': calmar,
            'max_dd': dd_p, 'spy_dd': dd_s,
            'win_rate': win_rate, 'avg_alpha': avg_alpha,
            'port_equity': pe, 'spy_equity': se,
            'log': log,
        }

    def _max_dd(self, cum):
        peak = np.maximum.accumulate(cum)
        return round(((cum - peak) / peak).min() * 100, 2)


# ============================================================================
# CONGRESS-FOLLOW BACKTEST
# ============================================================================

class CongressBacktester:
    """
    Strategy: Every month, find stocks that congress bought in the last 45 days.
    Equal-weight top N by trade count. Compare vs SPY.
    Uses GitHub historical senate data.
    """

    def __init__(self, top_n=5, initial_capital=10000):
        self.top_n = top_n
        self.initial = initial_capital

    def run(self, start_date, end_date):
        print(f"\n{'='*65}")
        print(f"  BACKTEST: Congress-Follow Strategy")
        print(f"  Buy top-{self.top_n} stocks congress is purchasing")
        print(f"  Period: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")
        print(f"{'='*65}")

        # Load congress data
        from data_sources import GitHubSenateFetcher
        senate = GitHubSenateFetcher().fetch()
        if senate.empty:
            print("  ❌ No senate data available. Run will use synthetic approach.")
            return None

        print(f"  Loaded {len(senate):,} senate trades")
        senate['transaction_date'] = pd.to_datetime(senate['transaction_date'], errors='coerce')

        # Get monthly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        if len(dates) < 3:
            print("  Need at least 3 months.")
            return None

        # Collect all tickers that appear
        all_tickers = senate['ticker'].dropna().unique().tolist()
        # Filter to only liquid tickers (ones yfinance can download)
        print(f"  📥 Downloading prices for {len(all_tickers)} tickers (filtering illiquid)...")

        full_start = (start_date - timedelta(days=60)).strftime('%Y-%m-%d')
        full_end = (end_date + timedelta(days=5)).strftime('%Y-%m-%d')

        # Download in batches
        prices = {}
        batch_size = 20
        for i in range(0, min(len(all_tickers), 100), batch_size):
            batch = all_tickers[i:i+batch_size]
            tickers_str = ' '.join(batch)
            try:
                data = yf.download(tickers_str, start=full_start, end=full_end,
                                  progress=False, auto_adjust=True, group_by='ticker')
                for t in batch:
                    try:
                        if len(batch) == 1:
                            df = data
                        else:
                            df = data[t] if t in data.columns.get_level_values(0) else pd.DataFrame()
                        if not df.empty and len(df) > 20:
                            prices[t] = df
                    except:
                        pass
            except:
                pass

        # Always need SPY
        if 'SPY' not in prices:
            try:
                spy = yf.download('SPY', start=full_start, end=full_end, progress=False, auto_adjust=True)
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                prices['SPY'] = spy
            except:
                print("  ❌ No SPY data.")
                return None

        print(f"  Usable tickers: {len(prices) - 1}")

        # Run month by month
        port_rets, spy_rets, log = [], [], []

        for i in range(len(dates) - 1):
            d = dates[i].to_pydatetime()
            d_next = dates[i+1].to_pydatetime()

            # Find stocks congress bought in last 60 days before this date
            lookback = d - timedelta(days=60)
            mask = (
                (senate['transaction_date'] >= lookback) &
                (senate['transaction_date'] < d) &
                (senate['type'].str.contains('purchase|buy', case=False, na=False))
            )
            recent_buys = senate[mask]

            if recent_buys.empty:
                port_rets.append(0)
                spy_rets.append(self._ret(prices['SPY'], d, d_next))
                continue

            # Rank by number of buyers
            counts = recent_buys.groupby('ticker').agg(
                n_trades=('member', 'count'),
                n_members=('member', 'nunique'),
            ).sort_values('n_members', ascending=False)

            # Filter to tickers we have price data for
            tradeable = [t for t in counts.index if t in prices][:self.top_n]

            if not tradeable:
                port_rets.append(0)
                spy_rets.append(self._ret(prices['SPY'], d, d_next))
                continue

            # Equal weight
            w = 1 / len(tradeable)
            month_ret = sum(w * self._ret(prices[t], d, d_next) for t in tradeable)
            spy_ret = self._ret(prices['SPY'], d, d_next)

            port_rets.append(month_ret)
            spy_rets.append(spy_ret)

            print(f"  {d:%Y-%m}: {', '.join(tradeable)} | "
                  f"Port: {month_ret*100:+.2f}% SPY: {spy_ret*100:+.2f}%")

            log.append({
                'month': f"{d:%Y-%m}", 'picks': tradeable,
                'port_ret': round(month_ret*100, 2),
                'spy_ret': round(spy_ret*100, 2),
                'alpha': round((month_ret - spy_ret)*100, 2),
            })

        if not port_rets:
            print("  No trades executed.")
            return None

        # Use same results display
        p = np.array(port_rets)
        s = np.array(spy_rets)
        pc = np.cumprod(1 + p)
        sc = np.cumprod(1 + s)
        ny = len(p) / 12

        tp = (pc[-1]-1)*100
        ts = (sc[-1]-1)*100
        sp = (p.mean()/p.std())*np.sqrt(12) if p.std()>0 else 0
        ss = (s.mean()/s.std())*np.sqrt(12) if s.std()>0 else 0

        print(f"\n  {'='*55}")
        print(f"  Congress Strategy: {tp:+.2f}%  (CAGR: {(pc[-1]**(1/ny)-1)*100:+.2f}%)")
        print(f"  SPY Buy & Hold:    {ts:+.2f}%  (CAGR: {(sc[-1]**(1/ny)-1)*100:+.2f}%)")
        print(f"  Alpha:             {tp-ts:+.2f}%")
        print(f"  Sharpe: Strategy {sp:.2f} vs SPY {ss:.2f}")
        print(f"  Win rate vs SPY: {(np.array(port_rets) > np.array(spy_rets)).mean()*100:.1f}%")
        print(f"  Final: ${pc[-1]*self.initial:,.0f} vs ${sc[-1]*self.initial:,.0f}")
        print(f"  {'='*55}\n")

        return {'total': tp, 'spy': ts, 'alpha': tp-ts, 'sharpe': sp, 'log': log}

    def _ret(self, df, start, end):
        if df is None or df.empty: return 0
        mask = (df.index >= start) & (df.index < end)
        p = df[mask]
        if len(p) < 2: return 0
        c = p['Close'] if 'Close' in p.columns else p.iloc[:, 0]
        return (c.iloc[-1] / c.iloc[0]) - 1


# ============================================================================
# CLI
# ============================================================================

def main():
    args = sys.argv[1:]
    top_n = 5
    years = 2
    weighted = False
    strategy = "momentum"

    for i, a in enumerate(args):
        if a == '--top' and i+1 < len(args): top_n = int(args[i+1])
        elif a == '--years' and i+1 < len(args): years = int(args[i+1])
        elif a == '--weighted': weighted = True
        elif a == '--strategy' and i+1 < len(args): strategy = args[i+1]

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    if strategy == "congress":
        bt = CongressBacktester(top_n=top_n)
        bt.run(start, end)
    else:
        universe = [
            'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'AAPL', 'META', 'PLTR', 'CRM',
            'LMT', 'RTX', 'GD', 'NOC', 'BA',
            'JPM', 'GS', 'MS', 'BAC',
            'XOM', 'CVX', 'COP',
            'UNH', 'JNJ', 'PFE', 'ABBV',
            'TSLA', 'HD', 'WMT', 'COST',
            'CAT', 'DE', 'UNP',
        ]
        bt = Backtester(universe=universe, top_n=top_n, weighted=weighted)
        bt.run(start, end)


if __name__ == '__main__':
    main()
