# =================================================================================
# Regenerate OOS CSV - Simplified Standalone Version
#
# Questo script carica i modelli salvati e rigenera i CSV OOS con la logica corretta.
# Usa UN APPROCCIO SEMPLIFICATO: carica solo le funzioni necessarie dal main script.
# =================================================================================

import os
import sys

# Add parent directory to path per importare dal main script
sys.path.insert(0, r"C:\Users\sas\Desktop\tutor ai")

# Import SOLO le funzioni necessarie dal main script
import importlib.util
spec = importlib.util.spec_from_file_location("xgboost_main", r"C:\Users\sas\Desktop\tutor ai\xgboost 4 market v2.py")
xgb_main = importlib.util.module_from_spec(spec)

print("="*80)
print("CARICAMENTO MODULO PRINCIPALE...")
print("="*80)

try:
    spec.loader.exec_module(xgb_main)
    print("[OK] Modulo caricato con successo")
except Exception as e:
    print(f"[ERRORE] Impossibile caricare il modulo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Ora posso usare le funzioni dal modulo principale
config = xgb_main.config
calculate_trade_metrics = xgb_main.calculate_trade_metrics
calculate_drawdown = xgb_main.calculate_drawdown
strategy_single_day_swing = xgb_main.strategy_single_day_swing
strategy_trend_following = xgb_main.strategy_trend_following
save_trades_to_csv = xgb_main.save_trades_to_csv

import pandas as pd
import numpy as np
import xgboost as xgb
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurazione
MODELS_DIR = r"C:\Users\sas\run_v72.5_final_backtester_20251022_010847"
OUTPUT_DIR = r"C:\Users\sas\run_v72.5_regenerated_csv_corrected"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("REGENERATE OOS CSV - VERSIONE SEMPLIFICATA")
print("="*80)
print(f"\nModelli input: {MODELS_DIR}")
print(f"Output CSV: {OUTPUT_DIR}")
print("\nQuesta versione usa il CODICE CORRETTO dal main script.")
print("="*80)

# Carica tutte le funzioni e dati dal main module
logging.info("\nCaricamento features e dati...")

try:
    # Esegui la funzione di loading del main
    df_features_base, df_targets, df_regimes, df_close, df_high, df_low, df_seasonal, df_macro_ratios = xgb_main.load_and_prepare_features()
    logging.info(f"Features caricate: {df_features_base.shape}")
    logging.info(f"Prezzi caricati: {df_close.shape}")
except Exception as e:
    logging.error(f"Errore nel caricamento features: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Determine OOS period
full_idx = df_features_base.index
oos_start_date = full_idx.max() - pd.DateOffset(months=config["OOS_MONTHS"])
oos_idx = full_idx[full_idx >= oos_start_date]

logging.info(f"\nPeriodo OOS: {oos_idx.min()} to {oos_idx.max()} ({len(oos_idx)} giorni)")

# Strategies
STRATEGY_MAP = {
    'Overnight': ('Overnight', strategy_single_day_swing),
    'Trend-Following': ('Trend-Following', strategy_trend_following)
}

# Process each ticker and strategy
for symbol in config["TICKERS"]:
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing: {symbol}")
    logging.info(f"{'='*80}")

    for strategy_key, (strategy_name, strategy_fn) in STRATEGY_MAP.items():
        model_path = os.path.join(MODELS_DIR, f"final_model_{symbol}_{strategy_name}.json")

        if not os.path.exists(model_path):
            logging.warning(f"  Model non trovato: {model_path}")
            continue

        logging.info(f"\n  Strategy: {strategy_name}")

        try:
            # Load model
            model = xgb.Booster()
            model.load_model(model_path)
            logging.info(f"  Model caricato: {model_path}")

            # Prepare OOS data
            X_oos = df_features_base.loc[oos_idx]
            y_oos = df_targets.loc[oos_idx, symbol]

            # Align
            common_idx = X_oos.index.intersection(y_oos.index)
            X_oos = X_oos.loc[common_idx].dropna()
            y_oos = y_oos.loc[common_idx]

            # Re-align
            common_idx = X_oos.index.intersection(y_oos.index)
            X_oos = X_oos.loc[common_idx]
            y_oos = y_oos.loc[common_idx]

            logging.info(f"  OOS samples: {len(X_oos)}")

            # Make predictions
            dtest = xgb.DMatrix(X_oos)
            predictions = pd.Series(model.predict(dtest), index=X_oos.index)
            logging.info(f"  Predictions: {len(predictions)}")

            # Run strategy with CORRECTED calculate_trade_metrics
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

            if trades_df is not None:
                logging.info(f"  Trades generati: {len(trades_df)}")
                logging.info(f"  Primo trade: {trades_df.iloc[0]['entry_date']} to {trades_df.iloc[0]['exit_date']}")
                logging.info(f"  Ultimo trade: {trades_df.iloc[-1]['entry_date']} to {trades_df.iloc[-1]['exit_date']}")

                # Save CSV
                save_trades_to_csv(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    rank=1,
                    trades_df=trades_df,
                    total_return=total_return,
                    max_dd=max_dd,
                    calmar_ratio=calmar_ratio,
                    config_info={'window': 'unknown', 'features': len(X_oos.columns)},
                    output_dir=OUTPUT_DIR
                )
            else:
                logging.warning(f"  Nessun trade generato")

        except Exception as e:
            logging.error(f"  Errore: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "="*80)
print(f"COMPLETATO!")
print(f"CSV rigenerati in: {OUTPUT_DIR}")
print("="*80)
