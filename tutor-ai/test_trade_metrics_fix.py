"""
Quick test to verify that calculate_trade_metrics() fix works correctly
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_trade_metrics_old(positions, close_prices, high_prices, low_prices, actual_returns, commission_rate):
    """OLD VERSION - with bug that creates duplicate trades"""
    trades = []
    current_position = 0
    entry_date = None
    entry_price = None

    aligned_idx = positions.index.intersection(close_prices.index).intersection(high_prices.index).intersection(low_prices.index).intersection(actual_returns.index)
    positions = positions.loc[aligned_idx]
    close_prices = close_prices.loc[aligned_idx]
    high_prices = high_prices.loc[aligned_idx]
    low_prices = low_prices.loc[aligned_idx]
    actual_returns = actual_returns.loc[aligned_idx]

    strategy_returns = positions * actual_returns
    position_changes = positions.diff().fillna(0)
    commission_costs = position_changes.abs() * commission_rate
    strategy_returns = strategy_returns - commission_costs

    for date in positions.index:
        new_position = positions.loc[date]

        if new_position != current_position:
            if current_position != 0 and entry_date is not None:
                exit_price = close_prices.loc[date]
                exit_date = date

                if current_position == 1:
                    pnl_gross = (exit_price - entry_price) / entry_price
                else:
                    pnl_gross = (entry_price - exit_price) / entry_price

                commission_total = 2 * commission_rate
                pnl_net = pnl_gross - commission_total
                holding_time = (exit_date - entry_date).days

                trade_period = positions.index[(positions.index >= entry_date) & (positions.index <= exit_date)]
                if len(trade_period) > 0:
                    highs_in_trade = high_prices.loc[trade_period]
                    lows_in_trade = low_prices.loc[trade_period]

                    if current_position == 1:
                        mfe = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                        mae = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                    else:
                        mfe = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                        mae = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                else:
                    mfe = mae = 0.0

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': 'LONG' if current_position == 1 else 'SHORT',
                    'pnl_gross': pnl_gross * 100,
                    'commission': commission_total * 100,
                    'pnl_net': pnl_net * 100,
                    'holding_time': holding_time,
                    'mfe': mfe,
                    'mae': mae
                })

            # OLD BUG: only update current_position if new_position != 0
            if new_position != 0:
                entry_date = date
                entry_price = close_prices.loc[date]
                current_position = new_position

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, strategy_returns

def calculate_trade_metrics_new(positions, close_prices, high_prices, low_prices, actual_returns, commission_rate):
    """NEW VERSION - with fix"""
    trades = []
    current_position = 0
    entry_date = None
    entry_price = None

    aligned_idx = positions.index.intersection(close_prices.index).intersection(high_prices.index).intersection(low_prices.index).intersection(actual_returns.index)
    positions = positions.loc[aligned_idx]
    close_prices = close_prices.loc[aligned_idx]
    high_prices = high_prices.loc[aligned_idx]
    low_prices = low_prices.loc[aligned_idx]
    actual_returns = actual_returns.loc[aligned_idx]

    strategy_returns = positions * actual_returns
    position_changes = positions.diff().fillna(0)
    commission_costs = position_changes.abs() * commission_rate
    strategy_returns = strategy_returns - commission_costs

    for date in positions.index:
        new_position = positions.loc[date]

        if new_position != current_position:
            if current_position != 0 and entry_date is not None:
                exit_price = close_prices.loc[date]
                exit_date = date

                if current_position == 1:
                    pnl_gross = (exit_price - entry_price) / entry_price
                else:
                    pnl_gross = (entry_price - exit_price) / entry_price

                commission_total = 2 * commission_rate
                pnl_net = pnl_gross - commission_total
                holding_time = (exit_date - entry_date).days

                trade_period = positions.index[(positions.index >= entry_date) & (positions.index <= exit_date)]
                if len(trade_period) > 0:
                    highs_in_trade = high_prices.loc[trade_period]
                    lows_in_trade = low_prices.loc[trade_period]

                    if current_position == 1:
                        mfe = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                        mae = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                    else:
                        mfe = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                        mae = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                else:
                    mfe = mae = 0.0

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': 'LONG' if current_position == 1 else 'SHORT',
                    'pnl_gross': pnl_gross * 100,
                    'commission': commission_total * 100,
                    'pnl_net': pnl_net * 100,
                    'holding_time': holding_time,
                    'mfe': mfe,
                    'mae': mae
                })

            # FIX: always update current_position and reset entry tracking when going flat
            if new_position != 0:
                entry_date = date
                entry_price = close_prices.loc[date]
            else:
                entry_date = None
                entry_price = None

            current_position = new_position

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, strategy_returns


# Create test data that mimics the SOL-USD case
dates = pd.date_range('2025-05-01', periods=30, freq='D')

# Create a positions series that goes:
# LONG on day 7 (May 8), holds for 8 days, then goes FLAT on day 15 (May 16), stays FLAT for many days
positions = pd.Series(0, index=dates)
positions.iloc[7:15] = 1  # LONG from May 8 to May 15 (exits on May 16)
# Remains FLAT from May 16 onwards

# Create mock price data
close_prices = pd.Series(100 + np.random.randn(30).cumsum(), index=dates)
high_prices = close_prices + np.abs(np.random.randn(30))
low_prices = close_prices - np.abs(np.random.randn(30))
actual_returns = close_prices.pct_change().fillna(0)

commission_rate = 0.001

print("=" * 80)
print("TEST: calculate_trade_metrics() fix")
print("=" * 80)
print()

print("Position series:")
print(positions)
print()

print("=" * 80)
print("OLD VERSION (with bug):")
print("=" * 80)
trades_old, _ = calculate_trade_metrics_old(positions, close_prices, high_prices, low_prices, actual_returns, commission_rate)
print(f"Number of trades: {len(trades_old)}")
print()
if not trades_old.empty:
    print(trades_old[['entry_date', 'exit_date', 'position', 'pnl_net']])
    print()
    print(f"Sum of PnL: {trades_old['pnl_net'].sum():.2f}%")
print()

print("=" * 80)
print("NEW VERSION (with fix):")
print("=" * 80)
trades_new, _ = calculate_trade_metrics_new(positions, close_prices, high_prices, low_prices, actual_returns, commission_rate)
print(f"Number of trades: {len(trades_new)}")
print()
if not trades_new.empty:
    print(trades_new[['entry_date', 'exit_date', 'position', 'pnl_net']])
    print()
    print(f"Sum of PnL: {trades_new['pnl_net'].sum():.2f}%")
print()

print("=" * 80)
print("RESULT:")
print("=" * 80)
if len(trades_old) > len(trades_new):
    print(f"[SUCCESS] Fix works! Reduced from {len(trades_old)} trades to {len(trades_new)} trades")
else:
    print(f"[FAIL] Fix did not reduce duplicate trades")
print()
