"""
Multi-Factor Stock Screener
============================
Combines congressional trade signals with fundamental analysis, 
technical indicators, and relative strength metrics to score and rank stocks.

Author: Built for Raffaele's quant stack
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import warnings
import json
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

SECTOR_BENCHMARKS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial Services': 'XLF',
    'Energy': 'XLE',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Basic Materials': 'XLB',
    'Communication Services': 'XLC',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
}

DEFAULT_WEIGHTS = {
    'fundamental': 0.30,
    'technical': 0.25,
    'relative_strength': 0.20,
    'congressional': 0.10,
    'quality': 0.15,
}

# ============================================================================
# FUNDAMENTAL ANALYZER
# ============================================================================

class FundamentalAnalyzer:
    """Scores stocks on valuation, growth, profitability, and financial health."""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze(self, ticker_obj: yf.Ticker) -> dict:
        info = ticker_obj.info
        result = {
            'valuation_score': self._valuation_score(info),
            'growth_score': self._growth_score(info, ticker_obj),
            'profitability_score': self._profitability_score(info),
            'health_score': self._health_score(info),
            'raw_metrics': {}
        }
        
        # Store raw metrics for transparency
        result['raw_metrics'] = {
            'pe_trailing': info.get('trailingPE'),
            'pe_forward': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'free_cash_flow': info.get('freeCashflow'),
            'market_cap': info.get('marketCap'),
        }
        
        # Composite fundamental score (0-100)
        weights = {'valuation': 0.25, 'growth': 0.30, 'profitability': 0.25, 'health': 0.20}
        result['composite'] = (
            result['valuation_score'] * weights['valuation'] +
            result['growth_score'] * weights['growth'] +
            result['profitability_score'] * weights['profitability'] +
            result['health_score'] * weights['health']
        )
        
        return result
    
    def _valuation_score(self, info: dict) -> float:
        score = 50.0  # neutral baseline
        
        # Forward P/E (lower is better, but not too low)
        fpe = info.get('forwardPE')
        if fpe is not None and fpe > 0:
            if fpe < 10: score += 15
            elif fpe < 15: score += 20
            elif fpe < 20: score += 10
            elif fpe < 30: score += 0
            elif fpe < 50: score -= 10
            else: score -= 20
        
        # PEG Ratio (< 1 is undervalued relative to growth)
        peg = info.get('pegRatio')
        if peg is not None and peg > 0:
            if peg < 0.5: score += 15
            elif peg < 1.0: score += 10
            elif peg < 1.5: score += 5
            elif peg < 2.0: score -= 5
            else: score -= 10
        
        # EV/EBITDA
        ev_ebitda = info.get('enterpriseToEbitda')
        if ev_ebitda is not None and ev_ebitda > 0:
            if ev_ebitda < 8: score += 10
            elif ev_ebitda < 12: score += 5
            elif ev_ebitda < 20: score += 0
            else: score -= 10
        
        return np.clip(score, 0, 100)
    
    def _growth_score(self, info: dict, ticker_obj: yf.Ticker) -> float:
        score = 50.0
        
        rev_growth = info.get('revenueGrowth')
        if rev_growth is not None:
            if rev_growth > 0.30: score += 20
            elif rev_growth > 0.15: score += 15
            elif rev_growth > 0.05: score += 5
            elif rev_growth > 0: score += 0
            else: score -= 15
        
        earn_growth = info.get('earningsGrowth')
        if earn_growth is not None:
            if earn_growth > 0.30: score += 15
            elif earn_growth > 0.15: score += 10
            elif earn_growth > 0.05: score += 5
            elif earn_growth > 0: score += 0
            else: score -= 10
        
        # Revenue consistency (check quarterly)
        try:
            financials = ticker_obj.quarterly_financials
            if financials is not None and not financials.empty:
                if 'Total Revenue' in financials.index:
                    rev = financials.loc['Total Revenue'].dropna().sort_index()
                    if len(rev) >= 4:
                        qoq_growth = rev.pct_change().dropna()
                        positive_quarters = (qoq_growth > 0).sum()
                        consistency = positive_quarters / len(qoq_growth)
                        score += (consistency - 0.5) * 20
        except:
            pass
        
        return np.clip(score, 0, 100)
    
    def _profitability_score(self, info: dict) -> float:
        score = 50.0
        
        margin = info.get('profitMargins')
        if margin is not None:
            if margin > 0.25: score += 20
            elif margin > 0.15: score += 10
            elif margin > 0.05: score += 5
            elif margin > 0: score -= 5
            else: score -= 15
        
        roe = info.get('returnOnEquity')
        if roe is not None:
            if roe > 0.25: score += 15
            elif roe > 0.15: score += 10
            elif roe > 0.08: score += 5
            elif roe > 0: score += 0
            else: score -= 10
        
        op_margin = info.get('operatingMargins')
        if op_margin is not None:
            if op_margin > 0.25: score += 10
            elif op_margin > 0.15: score += 5
            elif op_margin > 0: score += 0
            else: score -= 10
        
        return np.clip(score, 0, 100)
    
    def _health_score(self, info: dict) -> float:
        score = 50.0
        
        dte = info.get('debtToEquity')
        if dte is not None:
            if dte < 30: score += 15
            elif dte < 60: score += 10
            elif dte < 100: score += 0
            elif dte < 200: score -= 10
            else: score -= 20
        
        cr = info.get('currentRatio')
        if cr is not None:
            if cr > 2.0: score += 10
            elif cr > 1.5: score += 5
            elif cr > 1.0: score += 0
            else: score -= 15
        
        fcf = info.get('freeCashflow')
        if fcf is not None:
            if fcf > 0: score += 10
            else: score -= 15
        
        return np.clip(score, 0, 100)


# ============================================================================
# TECHNICAL ANALYZER
# ============================================================================

class TechnicalAnalyzer:
    """Scores stocks on momentum, trend, volatility, and volume signals."""
    
    def analyze(self, hist: pd.DataFrame) -> dict:
        if hist is None or len(hist) < 50:
            return {'composite': 50.0, 'momentum_score': 50, 'trend_score': 50, 
                    'volatility_score': 50, 'volume_score': 50, 'indicators': {}}
        
        df = hist.copy()
        close = df['Close']
        
        # Calculate indicators
        indicators = self._compute_indicators(df)
        
        result = {
            'momentum_score': self._momentum_score(close, indicators),
            'trend_score': self._trend_score(close, indicators),
            'volatility_score': self._volatility_score(close, indicators),
            'volume_score': self._volume_score(df, indicators),
            'indicators': indicators,
        }
        
        weights = {'momentum': 0.30, 'trend': 0.30, 'volatility': 0.20, 'volume': 0.20}
        result['composite'] = (
            result['momentum_score'] * weights['momentum'] +
            result['trend_score'] * weights['trend'] +
            result['volatility_score'] * weights['volatility'] +
            result['volume_score'] * weights['volume']
        )
        
        return result
    
    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        ind = {}
        
        # RSI
        ind['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(close)
        ind['macd'] = macd.macd().iloc[-1]
        ind['macd_signal'] = macd.macd_signal().iloc[-1]
        ind['macd_hist'] = macd.macd_diff().iloc[-1]
        
        # Moving Averages
        ind['sma_20'] = close.rolling(20).mean().iloc[-1]
        ind['sma_50'] = close.rolling(50).mean().iloc[-1]
        ind['sma_200'] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        ind['ema_12'] = close.ewm(span=12).mean().iloc[-1]
        ind['ema_26'] = close.ewm(span=26).mean().iloc[-1]
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20)
        ind['bb_upper'] = bb.bollinger_hband().iloc[-1]
        ind['bb_lower'] = bb.bollinger_lband().iloc[-1]
        ind['bb_pct'] = bb.bollinger_pband().iloc[-1]
        
        # ATR
        ind['atr_14'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        ind['atr_pct'] = ind['atr_14'] / close.iloc[-1] * 100
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        ind['stoch_k'] = stoch.stoch().iloc[-1]
        ind['stoch_d'] = stoch.stoch_signal().iloc[-1]
        
        # ADX (trend strength)
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        ind['adx'] = adx.adx().iloc[-1]
        
        # Volume
        ind['vol_sma_20'] = volume.rolling(20).mean().iloc[-1]
        ind['vol_ratio'] = volume.iloc[-1] / ind['vol_sma_20'] if ind['vol_sma_20'] > 0 else 1
        
        # OBV trend
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_sma = obv.rolling(20).mean()
        ind['obv_trend'] = 'bullish' if obv.iloc[-1] > obv_sma.iloc[-1] else 'bearish'
        
        # Price vs MAs
        current = close.iloc[-1]
        ind['price'] = current
        ind['above_sma20'] = current > ind['sma_20']
        ind['above_sma50'] = current > ind['sma_50']
        ind['above_sma200'] = current > ind['sma_200'] if ind['sma_200'] is not None else None
        
        # Returns
        ind['return_1w'] = (current / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        ind['return_1m'] = (current / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
        ind['return_3m'] = (current / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
        ind['return_6m'] = (current / close.iloc[-126] - 1) * 100 if len(close) >= 126 else 0
        
        return ind
    
    def _momentum_score(self, close: pd.Series, ind: dict) -> float:
        score = 50.0
        
        # RSI
        rsi = ind.get('rsi_14', 50)
        if 40 <= rsi <= 60: score += 5  # neutral
        elif 30 <= rsi < 40: score += 10  # oversold, potential bounce
        elif rsi < 30: score += 15  # very oversold
        elif 60 < rsi <= 70: score += 5  # bullish momentum
        elif rsi > 80: score -= 15  # overbought danger
        elif rsi > 70: score -= 5
        
        # MACD
        if ind.get('macd_hist', 0) > 0:
            score += 10
            if ind.get('macd', 0) > ind.get('macd_signal', 0):
                score += 5  # bullish crossover
        else:
            score -= 5
        
        # Stochastic
        if ind.get('stoch_k', 50) < 20:
            score += 10  # oversold
        elif ind.get('stoch_k', 50) > 80:
            score -= 5  # overbought
        
        # Short-term momentum
        if ind.get('return_1m', 0) > 5: score += 5
        elif ind.get('return_1m', 0) < -10: score -= 5
        
        return np.clip(score, 0, 100)
    
    def _trend_score(self, close: pd.Series, ind: dict) -> float:
        score = 50.0
        
        # Price vs MAs (bullish alignment)
        if ind.get('above_sma20'): score += 10
        if ind.get('above_sma50'): score += 10
        if ind.get('above_sma200'): score += 10
        
        # MA alignment (golden cross type)
        if ind['sma_20'] > ind['sma_50']:
            score += 5
        else:
            score -= 5
        
        if ind['sma_200'] is not None:
            if ind['sma_50'] > ind['sma_200']:
                score += 10  # golden cross
            else:
                score -= 10  # death cross
        
        # ADX (trend strength)
        adx = ind.get('adx', 20)
        if adx > 25: score += 5  # strong trend
        if adx > 40: score += 5  # very strong
        
        return np.clip(score, 0, 100)
    
    def _volatility_score(self, close: pd.Series, ind: dict) -> float:
        score = 50.0
        
        # ATR % (lower = less volatile = safer)
        atr_pct = ind.get('atr_pct', 2)
        if atr_pct < 1: score += 15
        elif atr_pct < 2: score += 10
        elif atr_pct < 3: score += 0
        elif atr_pct < 5: score -= 10
        else: score -= 20
        
        # Bollinger Band position
        bb_pct = ind.get('bb_pct', 0.5)
        if 0.2 <= bb_pct <= 0.8:
            score += 5  # within normal range
        elif bb_pct > 1.0:
            score -= 10  # above upper band
        elif bb_pct < 0.0:
            score += 5  # below lower band (potential bounce)
        
        # Historical volatility (annualized)
        if len(close) >= 21:
            daily_returns = close.pct_change().dropna()
            hist_vol = daily_returns.std() * np.sqrt(252) * 100
            ind['hist_volatility'] = hist_vol
            if hist_vol < 20: score += 10
            elif hist_vol < 30: score += 5
            elif hist_vol > 50: score -= 15
        
        return np.clip(score, 0, 100)
    
    def _volume_score(self, df: pd.DataFrame, ind: dict) -> float:
        score = 50.0
        
        vol_ratio = ind.get('vol_ratio', 1)
        if vol_ratio > 1.5: score += 10  # above average volume (interest)
        elif vol_ratio > 1.0: score += 5
        elif vol_ratio < 0.5: score -= 10  # very low volume
        
        if ind.get('obv_trend') == 'bullish': score += 10
        else: score -= 5
        
        return np.clip(score, 0, 100)


# ============================================================================
# RELATIVE STRENGTH ANALYZER
# ============================================================================

class RelativeStrengthAnalyzer:
    """Mansfield Relative Strength — stock vs sector benchmark and SPY."""
    
    def analyze(self, stock_hist: pd.DataFrame, bench_hist: pd.DataFrame, 
                spy_hist: pd.DataFrame) -> dict:
        
        result = {'composite': 50.0, 'vs_sector': {}, 'vs_spy': {}}
        
        if stock_hist is None or len(stock_hist) < 50:
            return result
        
        stock_close = stock_hist['Close']
        
        # RS vs Sector Benchmark
        if bench_hist is not None and len(bench_hist) >= 50:
            bench_close = bench_hist['Close']
            result['vs_sector'] = self._compute_rs(stock_close, bench_close)
        
        # RS vs SPY
        if spy_hist is not None and len(spy_hist) >= 50:
            spy_close = spy_hist['Close']
            result['vs_spy'] = self._compute_rs(stock_close, spy_close)
        
        # Composite: overweight sector RS slightly
        sector_score = result['vs_sector'].get('rs_score', 50)
        spy_score = result['vs_spy'].get('rs_score', 50)
        result['composite'] = sector_score * 0.55 + spy_score * 0.45
        
        return result
    
    def _compute_rs(self, stock: pd.Series, bench: pd.Series) -> dict:
        # Align dates
        common = stock.index.intersection(bench.index)
        if len(common) < 50:
            return {'rs_score': 50}
        
        s = stock.loc[common]
        b = bench.loc[common]
        
        # Relative Strength line
        rs_line = s / b
        
        # Mansfield RS: normalized to SMA
        rs_sma = rs_line.rolling(52).mean()  # ~52 day SMA
        
        current_rs = rs_line.iloc[-1]
        if len(rs_sma.dropna()) > 0:
            mansfield = ((current_rs / rs_sma.iloc[-1]) - 1) * 100
        else:
            mansfield = 0
        
        # RS trend (is RS line rising?)
        rs_returns = {}
        for period, days in [('1m', 21), ('3m', 63), ('6m', 126)]:
            if len(rs_line) >= days:
                rs_returns[period] = (rs_line.iloc[-1] / rs_line.iloc[-days] - 1) * 100
        
        # Score
        score = 50.0
        
        # Mansfield RS positive = outperforming
        if mansfield > 5: score += 20
        elif mansfield > 2: score += 10
        elif mansfield > 0: score += 5
        elif mansfield > -2: score -= 5
        elif mansfield > -5: score -= 10
        else: score -= 20
        
        # RS trend
        for period in ['1m', '3m', '6m']:
            if period in rs_returns:
                if rs_returns[period] > 3: score += 5
                elif rs_returns[period] < -3: score -= 5
        
        return {
            'rs_score': np.clip(score, 0, 100),
            'mansfield_rs': round(mansfield, 2),
            'rs_returns': rs_returns,
            'current_rs_ratio': round(current_rs, 4),
        }


# ============================================================================
# CONGRESSIONAL TRADE ANALYZER
# ============================================================================

class CongressionalAnalyzer:
    """Scores based on congressional trading activity.
    
    Uses a built-in dataset of known high-conviction congressional traders
    and their recent activity. In production, connect to QuiverQuant API 
    or scrape House Clerk / Senate EFD directly.
    """
    
    # Top congressional traders by historical alpha (based on research)
    NOTABLE_TRADERS = {
        'Nancy Pelosi': {'party': 'D', 'alpha_score': 9, 'role': 'leadership'},
        'Dan Crenshaw': {'party': 'R', 'alpha_score': 7, 'role': 'committee'},
        'Marjorie Taylor Greene': {'party': 'R', 'alpha_score': 6, 'role': 'rank'},
        'Josh Gottheimer': {'party': 'D', 'alpha_score': 7, 'role': 'committee'},
        'Tommy Tuberville': {'party': 'R', 'alpha_score': 7, 'role': 'committee'},
        'Mark Green': {'party': 'R', 'alpha_score': 6, 'role': 'committee'},
        'Ro Khanna': {'party': 'D', 'alpha_score': 5, 'role': 'rank'},
        'Michael McCaul': {'party': 'R', 'alpha_score': 7, 'role': 'committee_chair'},
    }
    
    def analyze(self, ticker: str, congressional_trades: list = None) -> dict:
        """
        congressional_trades: list of dicts with keys:
            - member: str (name)
            - type: 'purchase' | 'sale'
            - amount: str (e.g. '$50,001 - $100,000')
            - date: str (disclosure date)
            - instrument: 'stock' | 'option_call' | 'option_put'
        """
        result = {
            'composite': 50.0,
            'trade_count': 0,
            'buy_count': 0,
            'sell_count': 0,
            'notable_traders': [],
            'conviction_score': 0,
            'net_sentiment': 'neutral',
            'details': [],
        }
        
        if not congressional_trades:
            return result
        
        trades = congressional_trades
        result['trade_count'] = len(trades)
        result['buy_count'] = sum(1 for t in trades if t.get('type') == 'purchase')
        result['sell_count'] = sum(1 for t in trades if t.get('type') == 'sale')
        
        score = 50.0
        
        # Net sentiment
        if result['buy_count'] > result['sell_count']:
            result['net_sentiment'] = 'bullish'
            score += 10
        elif result['sell_count'] > result['buy_count']:
            result['net_sentiment'] = 'bearish'
            score -= 10
        
        # Notable trader involvement
        for trade in trades:
            member = trade.get('member', '')
            if member in self.NOTABLE_TRADERS:
                trader_info = self.NOTABLE_TRADERS[member]
                result['notable_traders'].append(member)
                
                # Leadership trades carry more weight
                role_mult = {'leadership': 3, 'committee_chair': 2.5, 'committee': 2, 'rank': 1}
                mult = role_mult.get(trader_info['role'], 1)
                
                if trade.get('type') == 'purchase':
                    score += 5 * mult
                elif trade.get('type') == 'sale':
                    score -= 3 * mult
                
                # Options signal higher conviction
                if trade.get('instrument', '').startswith('option'):
                    score += 5 if trade['type'] == 'purchase' else -5
        
        # Conviction: multiple members buying = stronger signal
        if result['buy_count'] >= 3: score += 10
        if result['buy_count'] >= 5: score += 5
        
        # Trade recency (newer = more relevant)
        recent_count = 0
        for trade in trades:
            try:
                trade_date = datetime.strptime(trade.get('date', ''), '%Y-%m-%d')
                if (datetime.now() - trade_date).days <= 60:
                    recent_count += 1
            except:
                pass
        
        if recent_count > 0: score += 5
        if recent_count >= 3: score += 5
        
        result['conviction_score'] = min(recent_count * 2 + result['buy_count'], 10)
        result['composite'] = np.clip(score, 0, 100)
        result['details'] = trades
        
        return result


# ============================================================================
# QUALITY ANALYZER (Piotroski-inspired + extras)
# ============================================================================

class QualityAnalyzer:
    """Piotroski F-Score inspired quality assessment + earnings stability."""
    
    def analyze(self, ticker_obj: yf.Ticker) -> dict:
        info = ticker_obj.info
        
        f_score = 0
        checks = {}
        
        # 1. Positive ROA
        roa = info.get('returnOnAssets')
        if roa is not None and roa > 0:
            f_score += 1
            checks['positive_roa'] = True
        else:
            checks['positive_roa'] = False
        
        # 2. Positive operating cash flow
        ocf = info.get('operatingCashflow')
        if ocf is not None and ocf > 0:
            f_score += 1
            checks['positive_ocf'] = True
        else:
            checks['positive_ocf'] = False
        
        # 3. Cash flow > Net income (earnings quality)
        net_income = info.get('netIncomeToCommon')
        if ocf is not None and net_income is not None and ocf > net_income:
            f_score += 1
            checks['cf_gt_income'] = True
        else:
            checks['cf_gt_income'] = False
        
        # 4. Low debt/equity
        dte = info.get('debtToEquity')
        if dte is not None and dte < 100:
            f_score += 1
            checks['low_leverage'] = True
        else:
            checks['low_leverage'] = False
        
        # 5. Positive profit margin
        margin = info.get('profitMargins')
        if margin is not None and margin > 0:
            f_score += 1
            checks['profitable'] = True
        else:
            checks['profitable'] = False
        
        # 6. Revenue growth
        rev_growth = info.get('revenueGrowth')
        if rev_growth is not None and rev_growth > 0:
            f_score += 1
            checks['revenue_growing'] = True
        else:
            checks['revenue_growing'] = False
        
        # 7. Positive free cash flow
        fcf = info.get('freeCashflow')
        if fcf is not None and fcf > 0:
            f_score += 1
            checks['positive_fcf'] = True
        else:
            checks['positive_fcf'] = False
        
        # 8. ROE > 15%
        roe = info.get('returnOnEquity')
        if roe is not None and roe > 0.15:
            f_score += 1
            checks['high_roe'] = True
        else:
            checks['high_roe'] = False
        
        # 9. Current ratio > 1
        cr = info.get('currentRatio')
        if cr is not None and cr > 1.0:
            f_score += 1
            checks['adequate_liquidity'] = True
        else:
            checks['adequate_liquidity'] = False
        
        # Score (0-9 mapped to 0-100)
        composite = (f_score / 9) * 100
        
        return {
            'composite': composite,
            'f_score': f_score,
            'max_score': 9,
            'checks': checks,
        }


# ============================================================================
# MAIN SCREENER ENGINE
# ============================================================================

class MultiFactorScreener:
    """Orchestrates all analyzers and produces final composite scores."""
    
    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.fundamental = FundamentalAnalyzer()
        self.technical = TechnicalAnalyzer()
        self.relative_strength = RelativeStrengthAnalyzer()
        self.congressional = CongressionalAnalyzer()
        self.quality = QualityAnalyzer()
        
        # Cache benchmark data
        self._bench_cache = {}
    
    def _get_history(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            if hist.empty:
                return None
            return hist
        except:
            return None
    
    def _get_benchmark_hist(self, sector: str) -> pd.DataFrame:
        bench_ticker = SECTOR_BENCHMARKS.get(sector, 'SPY')
        if bench_ticker not in self._bench_cache:
            self._bench_cache[bench_ticker] = self._get_history(bench_ticker)
        return self._bench_cache[bench_ticker]
    
    def analyze_stock(self, ticker: str, congressional_trades: list = None) -> dict:
        """Full multi-factor analysis for a single stock."""
        print(f"  Analyzing {ticker}...")
        
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            if not info or info.get('regularMarketPrice') is None:
                return {'ticker': ticker, 'error': 'No data available'}
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}
        
        # Get price history
        hist = self._get_history(ticker)
        
        # Determine sector & get benchmarks
        sector = info.get('sector', 'Unknown')
        bench_hist = self._get_benchmark_hist(sector)
        spy_hist = self._get_history('SPY') if 'SPY' not in self._bench_cache else self._bench_cache['SPY']
        self._bench_cache['SPY'] = spy_hist
        
        # Run all analyzers
        fund_result = self.fundamental.analyze(t)
        tech_result = self.technical.analyze(hist)
        rs_result = self.relative_strength.analyze(hist, bench_hist, spy_hist)
        congress_result = self.congressional.analyze(ticker, congressional_trades)
        quality_result = self.quality.analyze(t)
        
        # Composite score
        composite = (
            fund_result['composite'] * self.weights['fundamental'] +
            tech_result['composite'] * self.weights['technical'] +
            rs_result['composite'] * self.weights['relative_strength'] +
            congress_result['composite'] * self.weights['congressional'] +
            quality_result['composite'] * self.weights['quality']
        )
        
        # Signal classification
        if composite >= 75: signal = 'STRONG BUY'
        elif composite >= 65: signal = 'BUY'
        elif composite >= 55: signal = 'HOLD+'
        elif composite >= 45: signal = 'HOLD'
        elif composite >= 35: signal = 'HOLD-'
        elif composite >= 25: signal = 'SELL'
        else: signal = 'STRONG SELL'
        
        return {
            'ticker': ticker,
            'name': info.get('shortName', ticker),
            'sector': sector,
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap'),
            'price': info.get('regularMarketPrice') or info.get('currentPrice'),
            'composite_score': round(composite, 1),
            'signal': signal,
            'scores': {
                'fundamental': round(fund_result['composite'], 1),
                'technical': round(tech_result['composite'], 1),
                'relative_strength': round(rs_result['composite'], 1),
                'congressional': round(congress_result['composite'], 1),
                'quality': round(quality_result['composite'], 1),
            },
            'fundamental': fund_result,
            'technical': tech_result,
            'relative_strength': rs_result,
            'congressional': congress_result,
            'quality': quality_result,
        }
    
    def screen(self, tickers: list, congressional_data: dict = None) -> pd.DataFrame:
        """Screen multiple stocks and return ranked DataFrame."""
        results = []
        
        for ticker in tickers:
            congress_trades = congressional_data.get(ticker, []) if congressional_data else []
            result = self.analyze_stock(ticker, congress_trades)
            if 'error' not in result:
                results.append(result)
            else:
                print(f"  ⚠ Skipping {ticker}: {result['error']}")
        
        if not results:
            return pd.DataFrame()
        
        # Build summary DataFrame
        rows = []
        for r in results:
            rows.append({
                'Ticker': r['ticker'],
                'Name': r['name'],
                'Sector': r['sector'],
                'Price': r['price'],
                'Market Cap': r['market_cap'],
                'COMPOSITE': r['composite_score'],
                'Signal': r['signal'],
                'Fundamental': r['scores']['fundamental'],
                'Technical': r['scores']['technical'],
                'Rel. Strength': r['scores']['relative_strength'],
                'Congressional': r['scores']['congressional'],
                'Quality': r['scores']['quality'],
                'F-Score': r['quality']['f_score'],
                'RSI': round(r['technical']['indicators'].get('rsi_14', 0), 1),
                'Mansfield RS': r['relative_strength']['vs_spy'].get('mansfield_rs', 0),
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('COMPOSITE', ascending=False).reset_index(drop=True)
        df.index += 1  # 1-indexed ranking
        
        return df, results


# ============================================================================
# DEMO / CLI
# ============================================================================

def format_market_cap(cap):
    if cap is None: return 'N/A'
    if cap >= 1e12: return f"${cap/1e12:.1f}T"
    if cap >= 1e9: return f"${cap/1e9:.1f}B"
    if cap >= 1e6: return f"${cap/1e6:.0f}M"
    return f"${cap:,.0f}"


def run_demo():
    """Demo with a mix of popular congressional-traded stocks."""
    
    print("=" * 70)
    print("  MULTI-FACTOR STOCK SCREENER")
    print("  Congressional Trades × Fundamentals × Technicals × Rel. Strength")
    print("=" * 70)
    print()
    
    # Stocks frequently traded by congress + some controls
    tickers = [
        'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'AAPL',  # mega-cap tech (congress favorites)
        'TSLA', 'META',                                # high-profile names
        'LMT', 'RTX',                                  # defense (committee trades)
        'JPM', 'GS',                                   # financials
        'XOM', 'CVX',                                  # energy
        'PLTR', 'SMCI',                                # high-momentum
        'JNJ', 'UNH',                                  # healthcare
    ]
    
    # Simulated congressional trade data (in production, scrape from House Clerk / QuiverQuant)
    congressional_data = {
        'NVDA': [
            {'member': 'Nancy Pelosi', 'type': 'purchase', 'amount': '$1,000,001 - $5,000,000', 
             'date': '2025-01-15', 'instrument': 'option_call'},
            {'member': 'Josh Gottheimer', 'type': 'purchase', 'amount': '$100,001 - $250,000',
             'date': '2025-01-20', 'instrument': 'stock'},
            {'member': 'Dan Crenshaw', 'type': 'purchase', 'amount': '$15,001 - $50,000',
             'date': '2025-01-22', 'instrument': 'stock'},
        ],
        'GOOGL': [
            {'member': 'Nancy Pelosi', 'type': 'purchase', 'amount': '$250,001 - $500,000',
             'date': '2025-01-15', 'instrument': 'option_call'},
            {'member': 'Ro Khanna', 'type': 'sale', 'amount': '$50,001 - $100,000',
             'date': '2025-01-10', 'instrument': 'stock'},
        ],
        'TSLA': [
            {'member': 'Marjorie Taylor Greene', 'type': 'purchase', 'amount': '$100,001 - $250,000',
             'date': '2024-12-15', 'instrument': 'stock'},
        ],
        'LMT': [
            {'member': 'Michael McCaul', 'type': 'purchase', 'amount': '$50,001 - $100,000',
             'date': '2025-01-05', 'instrument': 'stock'},
            {'member': 'Mark Green', 'type': 'purchase', 'amount': '$15,001 - $50,000',
             'date': '2025-01-08', 'instrument': 'stock'},
        ],
        'PLTR': [
            {'member': 'Tommy Tuberville', 'type': 'purchase', 'amount': '$50,001 - $100,000',
             'date': '2025-01-18', 'instrument': 'stock'},
        ],
        'AMZN': [
            {'member': 'Nancy Pelosi', 'type': 'purchase', 'amount': '$500,001 - $1,000,000',
             'date': '2025-01-15', 'instrument': 'option_call'},
        ],
    }
    
    print(f"Screening {len(tickers)} stocks...\n")
    
    screener = MultiFactorScreener()
    df, full_results = screener.screen(tickers, congressional_data)
    
    if df.empty:
        print("No results. Check your internet connection.")
        return
    
    # Format display
    display_df = df.copy()
    display_df['Market Cap'] = display_df['Market Cap'].apply(format_market_cap)
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.2f}" if x else 'N/A')
    
    print("\n" + "=" * 70)
    print("  RANKING")
    print("=" * 70)
    
    cols = ['Ticker', 'Name', 'Price', 'COMPOSITE', 'Signal', 
            'Fundamental', 'Technical', 'Rel. Strength', 'Quality', 'F-Score', 'RSI']
    print(display_df[cols].to_string())
    
    print("\n" + "=" * 70)
    print("  TOP 5 DETAILED BREAKDOWN")
    print("=" * 70)
    
    for r in full_results[:5]:
        if 'error' in r:
            continue
        # Find rank
        rank = df[df['Ticker'] == r['ticker']].index[0]
        
        print(f"\n{'─' * 60}")
        print(f"  #{rank} {r['ticker']} — {r['name']}")
        print(f"  {r['sector']} | {r['industry']}")
        print(f"  Price: ${r['price']:,.2f} | Cap: {format_market_cap(r['market_cap'])}")
        print(f"  ★ COMPOSITE: {r['composite_score']}/100 → {r['signal']}")
        print(f"{'─' * 60}")
        
        # Factor scores
        print(f"  Fundamental:  {r['scores']['fundamental']:5.1f}/100  │  Technical:     {r['scores']['technical']:5.1f}/100")
        print(f"  Rel Strength: {r['scores']['relative_strength']:5.1f}/100  │  Quality:       {r['scores']['quality']:5.1f}/100")
        print(f"  Congressional:{r['scores']['congressional']:5.1f}/100  │  F-Score:       {r['quality']['f_score']}/9")
        
        # Key metrics
        fm = r['fundamental']['raw_metrics']
        ti = r['technical']['indicators']
        
        print(f"\n  Key Fundamentals:")
        print(f"    P/E (fwd): {fm.get('pe_forward', 'N/A')}  |  PEG: {fm.get('peg_ratio', 'N/A')}")
        print(f"    Revenue Growth: {fm.get('revenue_growth', 'N/A')}  |  Profit Margin: {fm.get('profit_margin', 'N/A')}")
        print(f"    ROE: {fm.get('roe', 'N/A')}  |  D/E: {fm.get('debt_to_equity', 'N/A')}")
        
        print(f"\n  Key Technicals:")
        print(f"    RSI(14): {ti.get('rsi_14', 0):.1f}  |  MACD Hist: {ti.get('macd_hist', 0):.3f}")
        print(f"    Above SMA20: {ti.get('above_sma20')}  |  SMA50: {ti.get('above_sma50')}  |  SMA200: {ti.get('above_sma200')}")
        print(f"    ATR%: {ti.get('atr_pct', 0):.2f}%  |  ADX: {ti.get('adx', 0):.1f}")
        print(f"    Returns: 1w={ti.get('return_1w', 0):+.1f}%  1m={ti.get('return_1m', 0):+.1f}%  3m={ti.get('return_3m', 0):+.1f}%  6m={ti.get('return_6m', 0):+.1f}%")
        
        # RS
        rs_spy = r['relative_strength']['vs_spy']
        rs_sec = r['relative_strength']['vs_sector']
        print(f"\n  Relative Strength:")
        print(f"    vs SPY - Mansfield: {rs_spy.get('mansfield_rs', 'N/A')}")
        print(f"    vs Sector - Mansfield: {rs_sec.get('mansfield_rs', 'N/A')}")
        
        # Congressional
        cong = r['congressional']
        if cong['trade_count'] > 0:
            print(f"\n  Congressional Activity:")
            print(f"    Trades: {cong['trade_count']} (Buy: {cong['buy_count']}, Sell: {cong['sell_count']})")
            print(f"    Net Sentiment: {cong['net_sentiment'].upper()}")
            if cong['notable_traders']:
                print(f"    Notable: {', '.join(cong['notable_traders'])}")
    
    # Sort full_results by composite score for saving
    full_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    
    print(f"\n{'=' * 70}")
    print(f"  Weights: Fund={screener.weights['fundamental']:.0%} | "
          f"Tech={screener.weights['technical']:.0%} | "
          f"RS={screener.weights['relative_strength']:.0%} | "
          f"Congress={screener.weights['congressional']:.0%} | "
          f"Quality={screener.weights['quality']:.0%}")
    print(f"{'=' * 70}")
    
    return df, full_results


if __name__ == '__main__':
    df, results = run_demo()
