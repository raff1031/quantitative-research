# ==============================================================================
# SCRIPT DI ANALISI FEATURE AUTONOMO (per v73)
#
# DESCRIZIONE:
# Questo script carica i dati elaborati dalla cache (generata dallo script 
# principale v73) ed esegue un'analisi diagnostica delle feature per 
# asset specifici (AAPL e TSLA).
#
# COSA FA:
# 1. Carica 'market_data_cache.parquet' e 'full_features_cache.parquet'.
# 2. Ricalcola 'df_targets' (rendimenti fraz. diff.) dai prezzi di chiusura.
# 3. Definisce il set di training (escludendo validazione e OOS).
# 4. Per AAPL e TSLA, calcola e plotta:
#    - Correlazione Feature-Target (Top 50).
#    - Elenco delle Feature Altamente Correlate (Multicollinearità).
#    - Analisi Univariata (Scatter, Time Series, Distribuzione) per le 
#      Top 2 feature più correlate.
# 5. Salva tutti i grafici nella cartella 'feature_analysis_results'.
#
# COME USARLO:
# 1. Assicurati di avere 'pandas', 'seaborn', 'matplotlib', 'pyarrow', 'tqdm'
#    installati (es. 'pip install pandas seaborn matplotlib pyarrow tqdm').
# 2. Salva questo codice come 'run_feature_analysis.py'.
# 3. Posiziona questo file NELLA STESSA CARTELLA che CONTIENE la 
#    tua cartella 'cache_v73_backtester'.
# 4. Esegui lo script (es. 'python run_feature_analysis.py').
# ==============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from tqdm.auto import tqdm
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore') # Ignora warning temporanei per pulizia output

# --- CONFIGURAZIONE ---
# (Copiata dallo script v72.7/v73 per coerenza)
config = {
    "OOS_MONTHS": 6,
    "VALIDATION_MONTHS": 12,
    "SEED": 42,
    "FRAC_DIFF_D": 0.5,
    "TICKERS": ['AAPL', 'GOOG', 'TSLA', '^IXIC', 'MRNA', 'LLY', 'ETH-USD', 'SOL-USD'],
    # Aggiungi altri parametri se necessari, ma questi sono i minimi
}

# --- IMPOSTAZIONI ANALISI ---
CACHE_DIR = "cache_v72.7_backtester" # Nome della tua cartella cache
ASSETS_TO_ANALYZE = ['AAPL', 'TSLA']
ANALYSIS_OUTPUT_DIR = "feature_analysis_results"
NUM_TOP_FEATURES_FOR_UNIVARIATE = 2 # Analizza in dettaglio le top N feature

# File di cache necessari (devono esistere in CACHE_DIR)
MARKET_DATA_PATH = os.path.join(CACHE_DIR, 'market_data_cache.parquet')
FEATURES_PATH = os.path.join(CACHE_DIR, 'full_features_cache.parquet')


# ==============================================================================
# --- FUNZIONI HELPER (copiate dallo script principale) ---
# ==============================================================================

def get_weights_ffd(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres: break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    w = get_weights_ffd(d, thres)
    width = len(w)
    series_padded = series.fillna(method='ffill').dropna()
    if len(series_padded) < width: return pd.Series(index=series.index, dtype='float64')
    series_windows = sliding_window_view(series_padded.values, window_shape=width)
    frac_diff_values = (series_windows @ w).flatten()
    index_out = series_padded.index[width-1:]
    return pd.Series(frac_diff_values, index=index_out)

# ==============================================================================
# --- FUNZIONI DI ANALISI (definite qui per riutilizzarle nel loop) ---
# ==============================================================================

def analyze_feature_target_correlation(features_df, target_series, symbol, output_dir):
    """Calcola, stampa e plotta la correlazione tra feature e target."""
    print(f"\n[{symbol}] --- 1. Analisi Correlazione Feature-Target ---")
    if features_df.empty or target_series.empty:
        print(f"[{symbol}] Dati vuoti, impossibile calcolare correlazione.")
        return None

    try:
        # Assicurati che tutti i dati siano numerici
        features_df = features_df.apply(pd.to_numeric, errors='coerce')
        target_series = pd.to_numeric(target_series, errors='coerce')
        
        # Allinea e rimuovi NaN
        aligned_data = features_df.join(target_series).dropna()
        if aligned_data.empty:
            print(f"[{symbol}] Dati vuoti dopo l'allineamento. Salto.")
            return None
            
        features_df_aligned = aligned_data.drop(columns=[target_series.name])
        target_series_aligned = aligned_data[target_series.name]

        feature_target_corr = features_df_aligned.corrwith(target_series_aligned)
        feature_target_corr.dropna(inplace=True) # Rimuovi eventuali NaN

        if feature_target_corr.empty:
            print(f"[{symbol}] Nessuna correlazione calcolabile.")
            return None

        feature_target_corr_sorted = feature_target_corr.abs().sort_values(ascending=False)

        print(f"[{symbol}] Correlazione Feature-Target (Top 20 Assolute):")
        print(feature_target_corr_sorted.head(20))

        # Plotta le top 50
        plt.figure(figsize=(12, 7))
        feature_target_corr_sorted.head(50).plot(kind='bar')
        plt.title(f'[{symbol}] Correlazione Assoluta Feature-Target (Top 50)')
        plt.ylabel('Correlazione di Pearson (Assoluta)')
        plt.xlabel('Feature')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'{symbol}_feature_target_correlation.png')
        plt.savefig(plot_filename)
        plt.close() # Chiudi la figura per liberare memoria
        print(f"[{symbol}] Grafico correlazione feature-target salvato in: {plot_filename}")

        low_corr_threshold = 0.01
        num_low_corr = (feature_target_corr.abs() < low_corr_threshold).sum()
        print(f"[{symbol}] Numero di feature con correlazione assoluta < {low_corr_threshold}: {num_low_corr} su {len(features_df_aligned.columns)}")

        return feature_target_corr_sorted # Ritorna le correlazioni ordinate

    except Exception as e:
        print(f"[{symbol}] Errore durante l'analisi correlazione feature-target: {e}")
        return None

def analyze_inter_feature_correlation(features_df, symbol, output_dir, threshold=0.90):
    """Identifica e stampa coppie di feature altamente correlate."""
    print(f"\n[{symbol}] --- 2. Analisi Correlazione Tra Feature (Multicollinearità > {threshold}) ---")
    if features_df.empty:
        print(f"[{symbol}] Dati vuoti, impossibile calcolare correlazione tra feature.")
        return

    try:
        features_df = features_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1) # Rimuovi colonne non numeriche
        correlation_matrix = features_df.corr()
        highly_correlated_pairs = []
        cols = correlation_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    highly_correlated_pairs.append((cols[i], cols[j], corr_value))

        if highly_correlated_pairs:
            print(f"[{symbol}] Trovate {len(highly_correlated_pairs)} coppie con correlazione > {threshold} (mostro max 20):")
            for f1, f2, val in highly_correlated_pairs[:20]:
                print(f"  - {f1}  <-->  {f2}: {val:.3f}")
        else:
            print(f"[{symbol}] Nessuna coppia trovata con correlazione superiore alla soglia.")

    except Exception as e:
        print(f"[{symbol}] Errore durante l'analisi correlazione tra feature: {e}")


def perform_univariate_analysis(features_df, target_series, feature_name, symbol, output_dir):
    """Genera i plot per l'analisi univariata di una specifica feature."""
    print(f"\n[{symbol}] --- 3. Analisi Univariata per Feature: {feature_name} ---")
    if features_df.empty or target_series.empty or feature_name not in features_df.columns:
        print(f"[{symbol}] Dati o feature '{feature_name}' mancanti.")
        return

    try:
        # a) Scatter plot Feature vs Target
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=features_df[feature_name], y=target_series, alpha=0.5)
        plt.title(f'[{symbol}] Scatter Plot: {feature_name} vs Target')
        plt.xlabel(feature_name)
        plt.ylabel('Target (Rendimento Fraz. Diff.)')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        scatter_filename = os.path.join(output_dir, f'{symbol}_scatter_{feature_name}_vs_target.png')
        plt.savefig(scatter_filename)
        plt.close()
        print(f"[{symbol}] Grafico scatter '{feature_name}' salvato in: {scatter_filename}")

        # b) Plot Temporale (Feature vs Target)
        plt.figure(figsize=(14, 7))
        ax1 = plt.gca()
        line1, = ax1.plot(features_df.index, features_df[feature_name], label=feature_name, color='blue', alpha=0.7)
        ax1.set_xlabel('Data')
        ax1.set_ylabel(feature_name, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        line2, = ax2.plot(target_series.index, target_series, label='Target', color='red', alpha=0.6)
        ax2.set_ylabel('Target', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title(f'[{symbol}] Andamento Temporale: {feature_name} vs Target')
        ax1.legend(handles=[line1, line2], loc='upper left')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        ts_filename = os.path.join(output_dir, f'{symbol}_timeseries_{feature_name}_vs_target.png')
        plt.savefig(ts_filename)
        plt.close()
        print(f"[{symbol}] Grafico time series '{feature_name}' salvato in: {ts_filename}")

        # c) Distribuzione della Feature (Istogramma)
        plt.figure(figsize=(8, 5))
        sns.histplot(features_df[feature_name], kde=True, bins=50)
        plt.title(f'[{symbol}] Distribuzione della Feature: {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Frequenza')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        dist_filename = os.path.join(output_dir, f'{symbol}_distribution_{feature_name}.png')
        plt.savefig(dist_filename)
        plt.close()
        print(f"[{symbol}] Grafico distribuzione '{feature_name}' salvato in: {dist_filename}")

    except Exception as e:
        print(f"[{symbol}] Errore durante l'analisi univariata per '{feature_name}': {e}")

# ==============================================================================
# --- FUNZIONE MAIN DELLO SCRIPT DI ANALISI ---
# ==============================================================================

def main_analysis():
    """Funzione principale per caricare i dati e avviare l'analisi."""
    
    print(f"===== INIZIO ANALISI FEATURE PER {ASSETS_TO_ANALYZE} =====")
    print(f"Cartella cache: {CACHE_DIR}")
    
    # Crea la cartella di output per i grafici
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    print(f"I risultati verranno salvati in: {ANALYSIS_OUTPUT_DIR}")

    # --- 1. Caricamento Dati ---
    print("\nCaricamento dati grezzi (market_data_cache.parquet)...")
    try:
        df_full_market_data = pd.read_parquet(MARKET_DATA_PATH)
        print("Dati grezzi caricati con successo.")
    except FileNotFoundError:
        print(f"ERRORE: File '{MARKET_DATA_PATH}' non trovato.")
        print("Assicurati di eseguire lo script principale (v73) prima, per generare la cache.")
        return
    except Exception as e:
        print(f"Errore nel caricamento di '{MARKET_DATA_PATH}': {e}")
        return

    print("Caricamento feature (full_features_cache.parquet)...")
    try:
        df_features_base = pd.read_parquet(FEATURES_PATH)
        print("Feature caricate con successo.")
    except FileNotFoundError:
        print(f"ERRORE: File '{FEATURES_PATH}' non trovato.")
        print("Assicurati di eseguire lo script principale (v73) prima, per generare la cache.")
        return
    except Exception as e:
        print(f"Errore nel caricamento di '{FEATURES_PATH}': {e}")
        return

    # --- 2. Ricalcolo 'df_targets' ---
    print("\nRicalcolo 'df_targets' dai prezzi di chiusura...")
    df_close = df_full_market_data.loc[:, (slice(None), 'Close')].ffill()
    df_close.columns = df_close.columns.get_level_values(0)
    
    df_targets = pd.DataFrame(index=df_close.index)
    for ticker in tqdm(config["TICKERS"], desc="Ricalcolo Target Fraz. Diff."):
        if ticker in df_close.columns:
            df_targets[ticker] = frac_diff_ffd(df_close[ticker], d=config["FRAC_DIFF_D"], thres=1e-5).shift(-1)
    
    print("'df_targets' ricalcolato.")

    # --- 3. Definizione Indice di Training ---
    full_idx = df_features_base.index
    try:
        oos_start_date = full_idx.max() - pd.DateOffset(months=config["OOS_MONTHS"])
        validation_start_date = oos_start_date - pd.DateOffset(months=config["VALIDATION_MONTHS"])
        train_idx = full_idx[full_idx < validation_start_date]
        print(f"\nIndice di training definito: da {train_idx.min().date()} a {train_idx.max().date()}")
    except Exception as e:
        print(f"Errore nella definizione dell'indice di training: {e}")
        return

    # --- 4. Esecuzione Analisi per Asset ---
    for symbol in ASSETS_TO_ANALYZE:
        print(f"\n{'='*25} INIZIO ANALISI PER: {symbol} {'='*25}")

        # Prepara i dati specifici per l'asset
        features_train = df_features_base.loc[train_idx]
        if symbol not in df_targets.columns:
            print(f"Target per {symbol} non trovato. Salto.")
            continue
        
        target_train = df_targets.loc[train_idx, symbol].rename('target')

        # Unisci per allineare e rimuovere NaN nel target
        data_train = features_train.join(target_train).dropna(subset=['target'])
        
        if data_train.empty:
            print(f"Nessun dato di training valido per {symbol} dopo il join/dropna. Salto.")
            continue
            
        features_train_clean = data_train.drop(columns=['target'])
        target_train_clean = data_train['target']

        # 1. Correlazione Feature-Target
        correlations_sorted = analyze_feature_target_correlation(features_train_clean, target_train_clean, symbol, ANALYSIS_OUTPUT_DIR)

        # 2. Correlazione Tra Feature
        analyze_inter_feature_correlation(features_train_clean, symbol, ANALYSIS_OUTPUT_DIR)

        # 3. Analisi Univariata (per le top N feature)
        if correlations_sorted is not None and not correlations_sorted.empty:
            top_features = correlations_sorted.head(NUM_TOP_FEATURES_FOR_UNIVARIATE).index.tolist()
            print(f"\n[{symbol}] Eseguo Analisi Univariata per le top {NUM_TOP_FEATURES_FOR_UNIVARIATE} feature: {top_features}")
            for feature_name in top_features:
                perform_univariate_analysis(features_train_clean, target_train_clean, feature_name, symbol, ANALYSIS_OUTPUT_DIR)
        else:
            print(f"[{symbol}] Salto Analisi Univariata a causa di errori precedenti o nessuna correlazione trovata.")
        
        print(f"\n{'='*25} FINE ANALISI PER: {symbol} {'='*25}")

    print(f"\n===== ANALISI FEATURE COMPLETATA =====")
    print(f"Controlla la cartella '{ANALYSIS_OUTPUT_DIR}' per i grafici.")

# --- Avvia lo script ---
if __name__ == "__main__":
    main_analysis()