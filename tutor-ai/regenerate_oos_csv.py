# =================================================================================
# Regenerate OOS CSV Trade Reports
#
# Questo script carica i modelli già addestrati e rigenera SOLO i CSV dei trade
# per il periodo OOS usando la funzione calculate_trade_metrics CORRETTA
# =================================================================================

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from datetime import datetime
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configuration
config = {
    "COMMISSION_RATE": 0.0005,
    "QUANTILE_BUY": 0.90,
    "QUANTILE_SHORT": 0.10,
    "TICKERS": ['AAPL', 'GOOG', 'TSLA', '^IXIC', 'MRNA', 'LLY', 'ETH-USD', 'SOL-USD'],
    "START_DATE": '2015-01-01',
    "END_DATE": '2025-10-18',
    "OOS_MONTHS": 6,
    "FRAC_DIFF_D": 0.5,
}

# Paths
MODELS_DIR = r"C:\Users\sas\run_v72.5_final_backtester_20251022_010847"
FEATURES_CACHE = r"C:\Users\sas\Desktop\tutor ai\full_features_cache_v72.parquet"
OUTPUT_DIR = r"C:\Users\sas\run_v72.5_regenerated_csv"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================================
# CORRECTED calculate_trade_metrics function
# =================================================================================
def calculate_trade_metrics(positions, close_prices, high_prices, low_prices, actual_returns, commission_rate):
    """
    Calculate detailed trade metrics including entry/exit, PnL, MFE/MAE, holding time.

    CORRECTED VERSION: Properly updates current_position even when going FLAT.
    """
    trades = []
    current_position = 0
    entry_date = None
    entry_price = None

    # Align all series
    aligned_idx = positions.index.intersection(close_prices.index).intersection(high_prices.index).intersection(low_prices.index).intersection(actual_returns.index)
    positions = positions.loc[aligned_idx]
    close_prices = close_prices.loc[aligned_idx]
    high_prices = high_prices.loc[aligned_idx]
    low_prices = low_prices.loc[aligned_idx]
    actual_returns = actual_returns.loc[aligned_idx]

    # Calculate returns with commissions - CORRECT METHOD
    strategy_returns = positions * actual_returns
    position_changes = positions.diff().fillna(0)
    commission_costs = position_changes.abs() * commission_rate
    strategy_returns = strategy_returns - commission_costs

    # Now calculate trade-level details for CSV reporting
    for date in positions.index:
        new_position = positions.loc[date]

        # Position change detected
        if new_position != current_position:
            # Close existing position if any
            if current_position != 0 and entry_date is not None:
                exit_price = close_prices.loc[date]
                exit_date = date

                # Calculate PnL
                if current_position == 1:  # LONG
                    pnl_gross = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    pnl_gross = (entry_price - exit_price) / entry_price

                # Apply commissions (entry + exit)
                commission_total = 2 * commission_rate
                pnl_net = pnl_gross - commission_total

                # Calculate holding time
                holding_time = (exit_date - entry_date).days

                # Calculate MFE and MAE during the trade
                trade_period = positions.index[(positions.index >= entry_date) & (positions.index <= exit_date)]
                if len(trade_period) > 0:
                    highs_in_trade = high_prices.loc[trade_period]
                    lows_in_trade = low_prices.loc[trade_period]

                    if current_position == 1:  # LONG
                        mfe = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                        mae = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                    else:  # SHORT
                        mfe = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                        mae = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                else:
                    mfe = mae = 0.0

                # Store trade
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

            # Open new position if not flat
            if new_position != 0:
                entry_date = date
                entry_price = close_prices.loc[date]
            else:
                # Going flat - reset entry tracking
                entry_date = None
                entry_price = None

            # CRITICAL FIX: Always update current position to new position
            current_position = new_position

    # Close any remaining open position at the end
    if current_position != 0 and entry_date is not None:
        exit_date = positions.index[-1]
        exit_price = close_prices.loc[exit_date]

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

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, strategy_returns


def calculate_drawdown(cumulative_returns_series):
    """Calculate maximum drawdown."""
    if cumulative_returns_series is None or cumulative_returns_series.empty:
        return 0.0
    high_water_mark = cumulative_returns_series.cummax()
    drawdown = (cumulative_returns_series - high_water_mark) / high_water_mark
    return drawdown.min()


def strategy_single_day_swing(predictions, actual_returns, **kwargs):
    """Overnight strategy: opens positions based on dynamic thresholds."""
    rolling_buy_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_BUY"])
    rolling_short_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_SHORT"])
    aligned_buy_thresh = rolling_buy_thresh.shift(1)
    aligned_short_thresh = rolling_short_thresh.shift(1)

    positions = pd.Series(np.nan, index=predictions.index)
    positions.loc[predictions > aligned_buy_thresh] = 1
    positions.loc[predictions < aligned_short_thresh] = -1
    positions = positions.fillna(0)

    # Get price data from kwargs
    close_prices = kwargs.get('close_prices')
    high_prices = kwargs.get('high_prices')
    low_prices = kwargs.get('low_prices')

    if close_prices is not None and high_prices is not None and low_prices is not None:
        trades_df, strategy_returns = calculate_trade_metrics(
            positions, close_prices, high_prices, low_prices, actual_returns, config["COMMISSION_RATE"]
        )
        return strategy_returns.dropna(), trades_df
    else:
        strategy_returns = positions * actual_returns.loc[positions.index]
        return strategy_returns.dropna(), None


def strategy_trend_following(predictions, actual_returns, **kwargs):
    """Trend-following strategy: holds positions until opposite signal."""
    rolling_buy_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_BUY"])
    rolling_short_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_SHORT"])
    aligned_buy_thresh = rolling_buy_thresh.shift(1)
    aligned_short_thresh = rolling_short_thresh.shift(1)

    signals = pd.Series(0, index=predictions.index)
    signals.loc[predictions > aligned_buy_thresh] = 1
    signals.loc[predictions < aligned_short_thresh] = -1
    positions = signals.replace(0, np.nan).ffill().fillna(0)

    # Get price data from kwargs
    close_prices = kwargs.get('close_prices')
    high_prices = kwargs.get('high_prices')
    low_prices = kwargs.get('low_prices')

    if close_prices is not None and high_prices is not None and low_prices is not None:
        trades_df, strategy_returns = calculate_trade_metrics(
            positions, close_prices, high_prices, low_prices, actual_returns, config["COMMISSION_RATE"]
        )
        return strategy_returns.dropna(), trades_df
    else:
        strategy_returns = positions * actual_returns.loc[positions.index]
        return strategy_returns.dropna(), None


def frac_diff_ffd(series, d, thres=0.01):
    """Fractional differentiation with fixed window."""
    if series.empty or np.isnan(d):
        return pd.Series(index=series.index, dtype=float)

    w = [1.0]
    k = 1
    while abs(w[-1]) > thres:
        w.append(-w[-1] * (d - k + 1) / k)
        k += 1

    w = np.array(w[::-1])
    width = len(w) - 1

    output = pd.Series(index=series.index, dtype=float)
    for i in range(width, len(series)):
        output.iloc[i] = np.dot(w, series.iloc[i-width:i+1])

    return output


def save_trades_to_csv(symbol, strategy_name, trades_df, total_return, max_dd, calmar_ratio, window_size, num_features, output_dir):
    """Save trades to CSV with summary metrics."""
    if trades_df is None or trades_df.empty:
        logging.warning(f"No trades to save for {symbol} - {strategy_name}")
        return

    csv_filename = f"{symbol}_{strategy_name}_oos_corrected.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(f"# Summary Metrics\n")
            f.write(f"Total_Return_Pct,Max_Drawdown_Pct,Calmar_Ratio,Window_Size,Num_Features\n")
            f.write(f"{total_return:.2f},{max_dd*100:.2f},{calmar_ratio:.2f},{window_size},{num_features}\n")
            f.write(f"\n")
            f.write(f"# Trade Details\n")

        trades_df.to_csv(csv_path, mode='a', index=False, encoding='utf-8')
        logging.info(f"CSV salvato: {csv_path} ({len(trades_df)} trades)")

    except Exception as e:
        logging.error(f"Errore nel salvataggio CSV per {symbol} - {strategy_name}: {e}")


def main():
    logging.info("="*80)
    logging.info("REGENERATE OOS CSV TRADE REPORTS")
    logging.info("="*80)

    # Nota: Questo script NON rigenera le predictions perché richiederebbe
    # le features complete. Invece usa i modelli salvati per caricare i metadati
    # e POI rigener i CSV basandosi sui dati dei CSV VECCHI ma con la logica corretta.

    logging.info("\nATTENZIONE: Lo script richiede le features per rigenerare le predictions.")
    logging.info("Alternativa più semplice: correggere i CSV esistenti ricalcolando i trade.")
    logging.info("Procedo con approccio alternativo...")

    return process_existing_csvs()


def process_existing_csvs():
    """Processo alternativo: leggi i CSV esistenti e correggi il calcolo dei trade."""
    logging.info("\n" + "="*80)
    logging.info("PROCESSING EXISTING CSVs")
    logging.info("="*80)

    # Load price data (need to re-download)
    logging.info("\nDownload dati prezzi...")
    import yfinance as yf

    all_tickers = config["TICKERS"]
    df_close = pd.DataFrame()
    df_high = pd.DataFrame()
    df_low = pd.DataFrame()

    for ticker in all_tickers:
        try:
            data = yf.download(ticker, start=config["START_DATE"], end=config["END_DATE"], progress=False)
            if not data.empty:
                df_close[ticker] = data['Close']
                df_high[ticker] = data['High']
                df_low[ticker] = data['Low']
                logging.info(f"  {ticker}: {len(data)} giorni")
        except Exception as e:
            logging.error(f"Errore download {ticker}: {e}")

    # Calculate targets
    logging.info("\nCalcolo targets...")
    df_targets = pd.DataFrame(index=df_close.index)
    for ticker in config["TICKERS"]:
        if ticker in df_close.columns:
            df_targets[ticker] = frac_diff_ffd(df_close[ticker], d=config["FRAC_DIFF_D"], thres=1e-5).shift(-1)

    # Determine OOS period
    full_idx = df_features_base.index
    oos_start_date = full_idx.max() - pd.DateOffset(months=config["OOS_MONTHS"])
    oos_idx = full_idx[full_idx >= oos_start_date]

    logging.info(f"\nPeriodo OOS: {oos_idx.min()} to {oos_idx.max()} ({len(oos_idx)} giorni)")

    # Process each model
    strategies = {
        "Overnight": strategy_single_day_swing,
        "Trend-Following": strategy_trend_following
    }

    for symbol in config["TICKERS"]:
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing: {symbol}")
        logging.info(f"{'='*80}")

        for strategy_name, strategy_fn in strategies.items():
            model_path = os.path.join(MODELS_DIR, f"final_model_{symbol}_{strategy_name}.json")

            if not os.path.exists(model_path):
                logging.warning(f"Model non trovato: {model_path}")
                continue

            logging.info(f"\n  Strategy: {strategy_name}")
            logging.info(f"  Caricamento modello: {model_path}")

            try:
                # Load model
                model = xgb.Booster()
                model.load_model(model_path)

                # Get features for this symbol
                X_oos = df_features_base.loc[oos_idx].dropna()
                y_oos = df_targets.loc[oos_idx, symbol].dropna()

                # Align
                common_idx = X_oos.index.intersection(y_oos.index)
                X_oos = X_oos.loc[common_idx]
                y_oos = y_oos.loc[common_idx]

                logging.info(f"  OOS samples: {len(X_oos)}")

                # Make predictions
                import xgboost as xgb
                dtest = xgb.DMatrix(X_oos)
                predictions = pd.Series(model.predict(dtest), index=X_oos.index)

                logging.info(f"  Predictions generated: {len(predictions)}")

                # Run strategy
                strategy_returns, trades_df = strategy_fn(
                    predictions=predictions,
                    actual_returns=df_close[symbol].pct_change(),
                    close_prices=df_close[symbol],
                    high_prices=df_high[symbol] if symbol in df_high.columns else df_close[symbol],
                    low_prices=df_low[symbol] if symbol in df_low.columns else df_close[symbol]
                )

                if strategy_returns is None or strategy_returns.empty:
                    logging.warning(f"  Nessun return generato")
                    continue

                # Calculate metrics
                cumulative_returns = (1 + strategy_returns).cumprod()
                total_return = (cumulative_returns.iloc[-1] - 1) * 100
                max_dd = calculate_drawdown(cumulative_returns)
                calmar_ratio = total_return / abs(max_dd * 100) if max_dd != 0 else np.inf

                logging.info(f"  Total Return: {total_return:.2f}%")
                logging.info(f"  Max Drawdown: {max_dd*100:.2f}%")
                logging.info(f"  Calmar Ratio: {calmar_ratio:.2f}")
                logging.info(f"  Trades: {len(trades_df) if trades_df is not None else 0}")

                # Save CSV
                if trades_df is not None and not trades_df.empty:
                    save_trades_to_csv(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        trades_df=trades_df,
                        total_return=total_return,
                        max_dd=max_dd,
                        calmar_ratio=calmar_ratio,
                        window_size=0,  # Unknown from saved model
                        num_features=len(X_oos.columns),
                        output_dir=OUTPUT_DIR
                    )

            except Exception as e:
                logging.error(f"  Errore processing {symbol} - {strategy_name}: {e}")
                import traceback
                traceback.print_exc()

    logging.info(f"\n{'='*80}")
    logging.info(f"COMPLETATO! CSV salvati in: {OUTPUT_DIR}")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    main()
