# =================================================================================
# Regenerate Top 3 OOS CSV Trade Reports
#
# Questo script esegue la grid search OOS completa e genera i CSV corretti
# per i top 3 candidati per ogni strategia, usando la funzione calculate_trade_metrics
# CORRETTA che elimina i trade duplicati.
# =================================================================================

import os
import sys

# Import dal main script
sys.path.insert(0, r"C:\Users\sas\Desktop\tutor ai")

import importlib.util
spec = importlib.util.spec_from_file_location("xgboost_main", r"C:\Users\sas\Desktop\tutor ai\xgboost 4 market v2.py")
xgb_main = importlib.util.module_from_spec(spec)

print("="*80)
print("RIGENERAZIONE TOP 3 CSV OOS CON LOGICA CORRETTA")
print("="*80)

try:
    spec.loader.exec_module(xgb_main)
    print("[OK] Modulo principale caricato")
except Exception as e:
    print(f"[ERRORE] Impossibile caricare il modulo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Import necessari
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Usa config e funzioni dal main
config = xgb_main.config
calculate_trade_metrics = xgb_main.calculate_trade_metrics
calculate_drawdown = xgb_main.calculate_drawdown
strategy_single_day_swing = xgb_main.strategy_single_day_swing
strategy_trend_following = xgb_main.strategy_trend_following
save_trades_to_csv = xgb_main.save_trades_to_csv
frac_diff_ffd = xgb_main.frac_diff_ffd

# Override alcune config per velocità
config["RUN_NEW_SENTIMENT_ANALYSIS"] = False  # Usa cache se esiste
config["RUN_NEW_TRENDS_ANALYSIS"] = False
config["FORCE_RECALCULATE_FEATURES"] = False

# Paths
OUTPUT_DIR = r"C:\Users\sas\run_v72.5_regenerated_top3_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Top N candidati: {config['TOP_N_CANDIDATES_FOR_CSV']}")
print("="*80)

# =================================================================================
# MAIN
# =================================================================================

logging.info("\n--- FASE 1: Caricamento Dati e Features ---")
df_features_base, df_targets, df_regimes, df_close, df_high, df_low, df_seasonal, df_macro_ratios = xgb_main.load_and_prepare_features()

logging.info(f"Features shape: {df_features_base.shape}")
logging.info(f"Targets shape: {df_targets.shape}")
logging.info(f"Close prices shape: {df_close.shape}")

# Determine periods
full_idx = df_features_base.index
oos_start_date = full_idx.max() - pd.DateOffset(months=config["OOS_MONTHS"])
validation_start_date = oos_start_date - pd.DateOffset(months=config["VALIDATION_MONTHS"])

train_idx = full_idx[full_idx < validation_start_date]
validation_idx = full_idx[(full_idx >= validation_start_date) & (full_idx < oos_start_date)]
oos_idx = full_idx[full_idx >= oos_start_date]

logging.info(f"\nTraining Period: {train_idx.min()} to {train_idx.max()}")
logging.info(f"Validation Period: {validation_idx.min()} to {validation_idx.max()}")
logging.info(f"OOS Period: {oos_idx.min()} to {oos_idx.max()}")

# Strategy mapping
STRATEGY_MAP = {
    'single_day': {
        'name': 'Overnight',
        'function': strategy_single_day_swing
    },
    'trend_following': {
        'name': 'Trend-Following',
        'function': strategy_trend_following
    }
}

# Grid search configurations
window_sizes = config["WINDOW_LIST_TO_TEST"]
feature_counts = config["FEATURE_COUNTS_TO_TEST"]

logging.info(f"\n--- FASE 2: Grid Search OOS per Top {config['TOP_N_CANDIDATES_FOR_CSV']} Candidati ---")
logging.info(f"Window sizes: {window_sizes}")
logging.info(f"Feature counts: {feature_counts}")

all_oos_results = {}

for symbol in config["TICKERS"]:
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing: {symbol}")
    logging.info(f"{'='*80}")

    all_oos_results[symbol] = {}

    for strategy_type, strategy_info in STRATEGY_MAP.items():
        strategy_name = strategy_info['name']
        strategy_fn = strategy_info['function']

        logging.info(f"\n  Strategy: {strategy_name}")

        results_list = []

        # Grid search over all configurations
        for window_size, n_features in product(window_sizes, feature_counts):
            try:
                # Prepare data for this configuration
                ISEE_MOMENTUM_Z_COL = f'{config["ISEE_MOMENTUM_Z_COL_BASE"]}_{window_size}'
                ISEE_REGIME_DRIVEN_COL = f'{ISEE_MOMENTUM_Z_COL}_HighVol_Only'

                # Filter features
                if n_features == 'all':
                    n_features_int = len(df_features_base.columns)
                else:
                    n_features_int = n_features

                # Get available features
                available_cols = [c for c in df_features_base.columns if c not in [symbol]]
                n_to_select = min(n_features_int, len(available_cols))
                selected_features = available_cols[:n_to_select]

                # Prepare X_oos
                X_oos = df_features_base.loc[oos_idx, selected_features].copy()
                y_oos = df_targets.loc[oos_idx, symbol]

                # Align
                common_idx = X_oos.index.intersection(y_oos.index)
                X_oos = X_oos.loc[common_idx].dropna()
                y_oos = y_oos.loc[common_idx]
                common_idx = X_oos.index.intersection(y_oos.index)
                X_oos = X_oos.loc[common_idx]
                y_oos = y_oos.loc[common_idx]

                if len(X_oos) < 50:
                    continue

                # Load model
                model_path = os.path.join(r"C:\Users\sas\run_v72.5_final_backtester_20251022_010847",
                                         f"final_model_{symbol}_{strategy_name}.json")

                if not os.path.exists(model_path):
                    logging.warning(f"    Model not found: {model_path}")
                    continue

                # Only load model once (it's the same for all configs, we just vary features)
                if window_size == window_sizes[0] and n_features == feature_counts[0]:
                    model = xgb.Booster()
                    model.load_model(model_path)
                    logging.info(f"    Model loaded: {model_path}")

                # Make predictions
                dtest = xgb.DMatrix(X_oos)
                predictions = pd.Series(model.predict(dtest), index=X_oos.index)

                # Run strategy with CORRECTED calculate_trade_metrics
                strategy_returns, trades_df = strategy_fn(
                    predictions=predictions,
                    actual_returns=df_close[symbol].pct_change(),
                    close_prices=df_close[symbol],
                    high_prices=df_high[symbol] if symbol in df_high.columns else df_close[symbol],
                    low_prices=df_low[symbol] if symbol in df_low.columns else df_close[symbol]
                )

                if strategy_returns is None or strategy_returns.empty:
                    continue

                # Calculate metrics
                cumulative_returns = (1 + strategy_returns).cumprod()
                total_return = (cumulative_returns.iloc[-1] - 1) * 100
                max_dd = calculate_drawdown(cumulative_returns)
                calmar_ratio = total_return / abs(max_dd * 100) if max_dd != 0 else np.inf

                # Store result
                results_list.append({
                    'config': {
                        'window': window_size,
                        'features': n_features
                    },
                    'result': {
                        'total_strategy_return': total_return,
                        'max_drawdown': max_dd,
                        'calmar_ratio': calmar_ratio,
                        'cumulative_returns': cumulative_returns,
                        'trades_df': trades_df
                    }
                })

                logging.info(f"    Config: window={window_size}, features={n_features} -> "
                           f"Return={total_return:.2f}%, DD={max_dd*100:.2f}%, Calmar={calmar_ratio:.2f}")

            except Exception as e:
                logging.error(f"    Error with config window={window_size}, features={n_features}: {e}")
                continue

        all_oos_results[symbol][strategy_type] = results_list
        logging.info(f"  Total valid configurations: {len(results_list)}")

# =================================================================================
# FASE 3: Generate CSV for Top N Candidates
# =================================================================================

logging.info(f"\n{'='*80}")
logging.info(f"FASE 3: Generazione CSV per Top {config['TOP_N_CANDIDATES_FOR_CSV']} Candidati")
logging.info(f"{'='*80}")

for symbol in config["TICKERS"]:
    logging.info(f"\n{symbol}:")

    for strategy_type, results_list in all_oos_results[symbol].items():
        strategy_name = STRATEGY_MAP[strategy_type]['name']

        # Sort by calmar ratio (best first)
        valid_results = [(item['config'], item['result']) for item in results_list
                        if item.get('result') and item['result'].get('calmar_ratio') is not None]
        valid_results.sort(key=lambda x: x[1]['calmar_ratio'], reverse=True)

        logging.info(f"  {strategy_name}: {len(valid_results)} valid results")

        # Save CSV for top N candidates
        for rank, (cand_config, result) in enumerate(valid_results[:config['TOP_N_CANDIDATES_FOR_CSV']], start=1):
            try:
                save_trades_to_csv(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    rank=rank,
                    trades_df=result.get('trades_df'),
                    total_return=result.get('total_strategy_return', 0),
                    max_dd=result.get('max_drawdown', 0),
                    calmar_ratio=result.get('calmar_ratio', 0),
                    config_info=cand_config,
                    output_dir=OUTPUT_DIR
                )

                trades_count = len(result.get('trades_df', [])) if result.get('trades_df') is not None else 0
                logging.info(f"    Rank {rank}: Calmar={result['calmar_ratio']:.2f}, "
                           f"Return={result['total_strategy_return']:.2f}%, "
                           f"Trades={trades_count}")

            except Exception as e:
                logging.error(f"    Error saving CSV for rank {rank}: {e}")

logging.info(f"\n{'='*80}")
logging.info(f"COMPLETATO!")
logging.info(f"CSV salvati in: {OUTPUT_DIR}")
logging.info(f"{'='*80}")
