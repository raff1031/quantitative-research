# =================================================================================
# XGBoost Stock Prediction - v77 (Hyperparameter Tuning for Quantiles)
#
# DESCRIZIONE:
# Questa versione integra l'apprendimento chiave delle v75/v76:
# i quantili non sono fissi, ma sono un iperparametro da ottimizzare.
#
# - ⚠️ v77: Aggiunto un ciclo 'QUANTILE_PAIRS_TO_TEST' nella Fase 3 per 
#   trovare la coppia di quantili ottimale per ogni asset/configurazione.
# - ⚠️ v77: Riattivata la selezione delle feature [20, 40, ..., 'all'].
# - ⚠️ v77: Test completo riattivato su tutti gli 8 asset.
# - Mantiene il fix v74 (feature stazionarie .diff()).
# - Mantiene il fix v73 (lookahead bias .shift(1)).
# =================================================================================

# --- Imports ---
import time
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
import torch
import pickle
import random
import logging
from sklearn.model_selection import RandomizedSearchCV
from transformers import pipeline
import matplotlib.pyplot as plt
import warnings
from tqdm.auto import tqdm
import os
from datetime import datetime
from hmmlearn import hmm
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from arch import arch_model
from sklearn.model_selection._split import _BaseKFold
from pytrends.request import TrendReq
from numpy.lib.stride_tricks import sliding_window_view


# --- DEADLOCK FIX: Force single-threaded execution for underlying libraries ---
os.environ['OMP_NUM_THREADS'] = '-1'
os.environ['MKL_NUM_THREADS'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


# --- Configuration ---
config = {
    # --- Execution Control ---
    "RUN_NEW_SENTIMENT_ANALYSIS": True,
    "RUN_NEW_TRENDS_ANALYSIS": True,
    "RUN_NEW_HMM_ANALYSIS": True,
    "FORCE_RECALCULATE_FEATURES": True,

    # --- Robust Validation Parameters ---
    "OOS_MONTHS": 6,
    "VALIDATION_MONTHS": 12,
    "N_CANDIDATE_STRATEGIES": 10,

    # --- Model & Feature Parameters ---
    "SEED": 42,
    "FRAC_DIFF_D": 0.5,
    "WINDOW_LIST_TO_TEST": [15, 30, 45, 60, 75, 90],
    "FEATURE_COUNTS_TO_TEST": [20, 40, 60, 80, 100, 200, 'all'],
    "TRENDS_KEYWORDS": [
        'recession', 'inflation', 'investing', 'stock market crash', 'Tesla stock',
        'Apple stock', 'Moderna stock', 'Eli Lilly stock', 'bitcoin price', 'crypto winter', 'buy crypto'
    ],
    "TRENDS_Z_SCORE_WINDOW": 90,
    "ISEE_MOMENTUM_Z_COL_BASE": 'ISEE_Equity_Momentum_Z',
    "WALK_FORWARD_VALIDATION_MONTHS": 3,
    "REOPTIMIZATION_FREQUENCY_MONTHS": 3,
    "PURGE_DAYS": 10,
    
    # ⚠️ v77: NUOVO - Elenco di quantili da testare
    "QUANTILE_PAIRS_TO_TEST": [ (0.90, 0.10), (0.80, 0.20) ],
    # Questi verranno sovrascritti dal ciclo di test
    "QUANTILE_BUY": 0.90,
    "QUANTILE_SHORT": 0.10,

    # --- Asset & File Paths ---
    # ⚠️ v77: Test completo riattivato
    "TICKERS": ['AAPL', 'GOOG', 'TSLA', '^IXIC', 'MRNA', 'LLY', 'ETH-USD', 'SOL-USD'],
    
    "STABLE_CRYPTO_ASSETS": ['AAPL', '^IXIC', 'ETH-USD', 'SOL-USD', 'MRNA'],
    "NEWS_FILES": {
        "TECH_NEWS": "apple_news.xlsx", "MRNA": "mrna_news.xlsx",
        "LLY": "lly_news.xlsx", "GOOG": "goog_news.xlsx",
        "TSLA": "tsla_news.xlsx",
    },
    "ISEE_FILE_PATH": "ISEE.csv",
    "COMPUTED_SENTIMENT_PATH": "Dual_Sentiment_Analysis_Results.csv",
    "ADDITIONAL_MACRO_TICKERS": {'Gold': 'GC=F', 'Copper': 'HG=F', 'Silver': 'SI=F', 'VIX': '^VIX', 'VXV': '^VXV'},
    "TECH_ASSETS": ['AAPL', 'GOOG', 'TSLA', '^IXIC'],
    "BIOTECH_ASSETS": ['MRNA', 'LLY'],
    "CRYPTO_ASSETS": ['ETH-USD', 'SOL-USD'],
    "NO_OHE_ASSETS": ['SOL-USD', 'AAPL', 'ETH-USD', 'TSLA'],
    "MARKET_DIRECTIONAL_INDICATORS": {'spy_dir': 'SPY', 'nasdaq_dir': '^IXIC'},
    "CRYPTO_DIRECTIONAL_INDICATORS": {'btc_dir': 'BTC-USD'},
    "BIOTECH_DIRECTIONAL_INDICATORS": {'biotech_dir': 'XBI'},
    "GENERAL_DIRECTIONAL_INDICATORS": {'gbpjpy_dir': 'GBPJPY=X', 'gold_dir':'GC=F', 'xle_dir': 'XLE'},
    "CRYPTO_INDICATORS": {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Solana': 'SOL-USD'},
    "MACRO_TICKERS": {'10Y_Treasury': '^TNX', 'Dollar_Index': 'DX-Y.NYB', 'Yen_Index': 'JPY=X', 'Brent_Oil': 'BZ=F', 'Fed_Rate_Proxy': '^IRX'},

    # --- Trading & Commission Config ---
    "COMMISSION_RATE": 0.0005,  # 0.10% per trade (IBKR style)
    "TOP_N_CANDIDATES_FOR_CSV": 3, 

    # --- Google Trends API Config (NUOVO v3: chunking temporale) ---
    "GOOGLE_TRENDS_MIN_DELAY": 2.5,
    "GOOGLE_TRENDS_MAX_DELAY": 4.0,
    "GOOGLE_TRENDS_MAX_RETRIES": 5,
    "GOOGLE_TRENDS_CHUNK_MONTHS": 12,
    "GOOGLE_TRENDS_CHUNK_DELAY": 1.5,

    # --- GPU Config ---
    "GPU_MAX_BIN": 256,
    "GPU_GROW_POLICY": 'depthwise',

    # --- Date & System Config ---
    "START_DATE": '2015-01-01',
    "END_DATE": '2025-10-18',
    "MAX_CPU_CORES": max(1, os.cpu_count() - 2 if os.cpu_count() else 1)
}

# --- Setup ---
CACHE_DIR = "cache_v77_backtester" # ⚠️ v77
os.makedirs(CACHE_DIR, exist_ok=True)
output_dir = f"run_v77_quantile_tuning_{datetime.today().strftime('%Y%m%d_%H%M%S')}" # ⚠️ v77
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, 'backtest_v77.log') # ⚠️ v77
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
warnings.filterwarnings('ignore', category=ConvergenceWarning); warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR); logging.getLogger("yfinance").setLevel(logging.ERROR)

DATA_CACHE_PATH = os.path.join(CACHE_DIR, 'market_data_cache.parquet')
TECH_INDICATORS_CACHE_PATH = os.path.join(CACHE_DIR, 'tech_indicators_cache.parquet')
GARCH_FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, 'garch_features_cache.parquet')
TRENDS_CACHE_PATH = os.path.join(CACHE_DIR, 'google_trends_cache.parquet')
HMM_REGIMES_CACHE_PATH = os.path.join(CACHE_DIR, 'hmm_regimes_cache.parquet')
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'checkpoint_validation.pkl')
OOS_CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'checkpoint_oos.pkl')
FULL_FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, 'full_features_cache.parquet')

def download_data_with_fallback(symbols, start, end):
    symbols = list(set(symbols))
    if os.path.exists(DATA_CACHE_PATH):
        try:
            df_full = pd.read_parquet(DATA_CACHE_PATH)
            last_cached_date = df_full.index.max().normalize()
            requested_end_date = pd.to_datetime(end).normalize()
            if last_cached_date >= requested_end_date and set(symbols).issubset(set(df_full.columns.get_level_values(0).unique())):
                logging.info("La cache dati di mercato è aggiornata. Utilizzo dei dati locali.")
                df_close = df_full.loc[:, (slice(None), 'Close')].ffill(); df_close.columns = df_close.columns.get_level_values(0)
                df_open = df_full.loc[:, (slice(None), 'Open')].ffill(); df_open.columns = df_open.columns.get_level_values(0)
                df_high = df_full.loc[:, (slice(None), 'High')].ffill(); df_high.columns = df_high.columns.get_level_values(0)
                df_low = df_full.loc[:, (slice(None), 'Low')].ffill(); df_low.columns = df_low.columns.get_level_values(0)
                df_volume = df_full.loc[:, (slice(None), 'Volume')].ffill(); df_volume.columns = df_volume.columns.get_level_values(0)
                return df_close, df_open, df_high, df_low, df_volume
        except Exception as e:
            logging.warning(f"Errore lettura cache dati: {e}. Riscrivo la cache.")
    logging.info("La cache dati di mercato non è valida. Scaricamento di nuovi dati...")
    for path in [TECH_INDICATORS_CACHE_PATH, GARCH_FEATURES_CACHE_PATH, TRENDS_CACHE_PATH, HMM_REGIMES_CACHE_PATH, FULL_FEATURES_CACHE_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f"Impossibile rimuovere il file cache {path}: {e}")
                
    all_data = {}
    for symbol in tqdm(symbols, desc="Scaricamento dati di mercato"):
        try:
            data = yf.download(symbol, start=start, end=end, progress=False, timeout=20)
            if not data.empty: all_data[symbol] = data
        except Exception as e: logging.error(f"Errore download {symbol}: {e}")
    if not all_data: raise SystemExit("Download dati fallito. Nessun dato scaricato.")
    df_full = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)
    df_full.to_parquet(DATA_CACHE_PATH)
    df_close = df_full.loc[:, (slice(None), 'Close')].ffill(); df_close.columns = df_close.columns.get_level_values(0)
    df_open = df_full.loc[:, (slice(None), 'Open')].ffill(); df_open.columns = df_open.columns.get_level_values(0)
    df_high = df_full.loc[:, (slice(None), 'High')].ffill(); df_high.columns = df_high.columns.get_level_values(0)
    df_low = df_full.loc[:, (slice(None), 'Low')].ffill(); df_low.columns = df_low.columns.get_level_values(0)
    df_volume = df_full.loc[:, (slice(None), 'Volume')].ffill(); df_volume.columns = df_volume.columns.get_level_values(0)
    return df_close, df_open, df_high, df_low, df_volume

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

def create_seasonality_features(index):
    logging.info("Creazione delle feature di stagionalità avanzate...")
    df_seasonal = pd.DataFrame(index=index)
    df_seasonal['day_of_week'] = index.dayofweek
    df_seasonal['month_of_year'] = index.month
    df_seasonal['day_of_month'] = index.day
    df_seasonal['month_sin'] = np.sin(2 * np.pi * df_seasonal['month_of_year'] / 12)
    df_seasonal['month_cos'] = np.cos(2 * np.pi * df_seasonal['month_of_year'] / 12)
    df_seasonal['day_sin'] = np.sin(2 * np.pi * df_seasonal['day_of_month'] / 31)
    df_seasonal['day_cos'] = np.cos(2 * np.pi * df_seasonal['day_of_month'] / 31)
    df_seasonal['weekday_sin'] = np.sin(2 * np.pi * df_seasonal['day_of_week'] / 7)
    df_seasonal['weekday_cos'] = np.cos(2 * np.pi * df_seasonal['day_of_week'] / 7)
    df_seasonal['is_first_week'] = (df_seasonal['day_of_month'] <= 7).astype(int)
    df_seasonal['is_last_week'] = (index.days_in_month - index.day < 7).astype(int)
    columns_to_drop = ['day_of_week', 'month_of_year', 'day_of_month']
    return df_seasonal.drop(columns=columns_to_drop)

def get_sentiment_scores(symbols, news_files, index, run_new_analysis=False):
    """
    Analizza il sentiment delle news usando:
    - FinBERT per sentiment formale (financial news)
    - cardiffnlp/twitter-roberta per sentiment social

    IMPORTANTE: Applica shift(1) per evitare data leakage!
    """
    if not run_new_analysis and os.path.exists(config["COMPUTED_SENTIMENT_PATH"]):
        logging.info(f"Caricamento sentiment pre-calcolato da cache: {config['COMPUTED_SENTIMENT_PATH']}")
        sentiment_source_df = pd.read_csv(config["COMPUTED_SENTIMENT_PATH"], encoding='latin1')
    elif not run_new_analysis:
        logging.info("Sentiment analysis disabilitata e nessuna cache trovata. Ritorno DataFrame vuoto.")
        return pd.DataFrame(index=index)
    else:
        logging.info("\n" + "="*80)
        logging.info("ESECUZIONE NUOVA ANALISI SENTIMENT NLP")
        logging.info("="*80)

        # Importa transformers solo quando necessario
        from transformers import pipeline, logging as transformers_logging
        transformers_logging.set_verbosity_error()

        # Configura device (GPU se disponibile)
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            logging.info("[OK] CUDA GPU trovata. Analisi sentiment su GPU.")
        else:
            logging.info("[OK] CPU utilizzata per analisi sentiment.")

        # Inizializza pipeline NLP
        logging.info("Inizializzazione pipeline FinBERT (sentiment formale)...")
        pipe_formal = pipeline('sentiment-analysis', model='ProsusAI/finbert', device=device)

        logging.info("Inizializzazione pipeline Social Sentiment (cardiffnlp)...")
        pipe_social = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=device)

        # Carica e processa news
        all_news_data = []
        for file_key, file_path in news_files.items():
            if not os.path.exists(file_path):
                logging.warning(f"File news non trovato: {file_path}")
                continue

            try:
                logging.info(f"Processando file: {file_path}")

                # Leggi file
                if file_path.endswith('.xlsx'):
                    news_df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    news_df = pd.read_csv(file_path, encoding='latin1')

                news_df.columns = [str(c).strip().lower() for c in news_df.columns]

                # Formato 1: Eikon format (date/time, identifier(s), headline)
                if 'date/time' in news_df.columns and 'identifier(s)' in news_df.columns:
                    temp_df = news_df[['date/time', 'identifier(s)', 'headline']].copy()
                    temp_df.dropna(subset=['identifier(s)', 'headline'], inplace=True)
                    temp_df.rename(columns={'date/time': 'date', 'headline': 'title'}, inplace=True)

                    # Espandi multiple identifiers
                    temp_df['stock_list'] = temp_df['identifier(s)'].astype(str).str.split(',')
                    temp_df = temp_df.explode('stock_list')
                    temp_df['stock'] = temp_df['stock_list'].str.replace('-US', '').str.strip()
                    temp_df = temp_df[temp_df['stock'].isin(symbols)]
                    all_news_data.append(temp_df[['date', 'title', 'stock']])

                # Formato 2: Formato semplice (date, headline)
                else:
                    date_col = next((c for c in news_df.columns if 'date' in c), None)
                    headline_col = next((c for c in news_df.columns if 'head' in c or 'title' in c), None)

                    if not (date_col and headline_col):
                        logging.warning(f"Colonne date/headline non trovate in {file_path}")
                        continue

                    temp_df = news_df[[date_col, headline_col]].copy()
                    temp_df.rename(columns={date_col: 'date', headline_col: 'title'}, inplace=True)
                    temp_df['stock'] = file_key
                    all_news_data.append(temp_df)

            except Exception as e:
                logging.error(f"Errore lettura file {file_path}: {e}")

        if not all_news_data:
            logging.warning("Nessun dato news trovato. Ritorno DataFrame vuoto.")
            return pd.DataFrame(index=index)

        # Concatena tutti i dati
        sentiment_source_df = pd.concat(all_news_data, ignore_index=True)
        sentiment_source_df['date'] = pd.to_datetime(sentiment_source_df['date'])
        sentiment_source_df.dropna(subset=['title'], inplace=True)
        sentiment_source_df = sentiment_source_df[sentiment_source_df['title'].str.strip() != '']

        logging.info(f"Totale news da analizzare: {len(sentiment_source_df)}")

        # Calcola sentiment FORMALE (FinBERT)
        titles = sentiment_source_df['title'].tolist()
        logging.info("Calcolo sentiment FORMALE con FinBERT...")
        scores_formal = [res for res in tqdm(pipe_formal(titles, truncation=True, max_length=512), total=len(titles), desc="FinBERT")]
        sentiment_source_df['sentiment_formal'] = [
            (1 if r.get('label') == 'positive' else -1 if r.get('label') == 'negative' else 0) * r.get('score', 0)
            for r in scores_formal
        ]

        # Calcola sentiment SOCIAL (cardiffnlp)
        logging.info("Calcolo sentiment SOCIAL con cardiffnlp...")
        scores_social = []
        for res in tqdm(pipe_social(titles, truncation=True, max_length=512), total=len(titles), desc="Social"):
            label = res.get('label', 'neutral').lower()
            score = res.get('score', 0.0)
            if 'positive' in label or label == 'label_2':
                sentiment = 1 * score
            elif 'negative' in label or label == 'label_0':
                sentiment = -1 * score
            else:
                sentiment = 0
            scores_social.append(sentiment)
        sentiment_source_df['sentiment_social'] = scores_social

        # Salva risultati
        logging.info(f"Salvataggio sentiment in: {config['COMPUTED_SENTIMENT_PATH']}")
        sentiment_source_df.to_csv(config["COMPUTED_SENTIMENT_PATH"], index=False)
        logging.info("="*80 + "\n")

    # Processa e aggrega sentiment
    sentiment_source_df['date'] = pd.to_datetime(sentiment_source_df['date']).dt.normalize()
    pivot_formal = sentiment_source_df.pivot_table(index='date', columns='stock', values='sentiment_formal', aggfunc='mean')
    pivot_social = sentiment_source_df.pivot_table(index='date', columns='stock', values='sentiment_social', aggfunc='mean')

    # ⚠️ CRITICO: SHIFT(1) PER EVITARE DATA LEAKAGE!
    # Le news del giorno T sono disponibili solo il giorno T+1
    logging.info("⚠️  Applicazione shift(1) per prevenire DATA LEAKAGE...")
    df_formal = pivot_formal.reindex(index).shift(1).ffill().fillna(0.0)
    df_social = pivot_social.reindex(index).shift(1).ffill().fillna(0.0)

    # Crea DataFrame finale
    final_df = df_formal.add_suffix('_Sentiment_Formal').join(df_social.add_suffix('_Sentiment_Social'))

    # Assicurati che tutte le colonne esistano
    for s in symbols:
        if f"{s}_Sentiment_Formal" not in final_df.columns:
            final_df[f"{s}_Sentiment_Formal"] = 0.0
        if f"{s}_Sentiment_Social" not in final_df.columns:
            final_df[f"{s}_Sentiment_Social"] = 0.0

    return final_df

def get_google_trends_data(keywords, index, z_score_window, run_new_analysis=False, min_delay=2.5, max_delay=4.0, max_retries=5, chunk_months=12, chunk_delay=1.5):
    """
    NUOVO v3: Scarica dati Google Trends con chunking temporale per evitare errore 429.
    IMPORTANTE: Mantiene shift(1) per evitare data leakage!
    """
    if not run_new_analysis and os.path.exists(TRENDS_CACHE_PATH):
        logging.info(f"Caricamento dati Google Trends dalla cache: {TRENDS_CACHE_PATH}")
        try: return pd.read_parquet(TRENDS_CACHE_PATH)
        except Exception as e: logging.warning(f"Errore lettura cache Trends: {e}. Ricalcolo.")

    logging.info(f"Scaricamento di nuovi dati da Google Trends con CHUNKING TEMPORALE")
    logging.info(f"Configurazione: chunk_size={chunk_months} mesi, delay={min_delay}-{max_delay}s, chunk_delay={chunk_delay}s, max_retries={max_retries}")

    pytrends = TrendReq(hl='en-US', tz=360)

    # Crea chunk temporali
    start_date = index.min()
    end_date = index.max()
    date_chunks = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + pd.DateOffset(months=chunk_months), end_date)
        date_chunks.append((current, chunk_end))
        current = chunk_end + pd.DateOffset(days=1) # Evita overlap

    logging.info(f"Periodo totale suddiviso in {len(date_chunks)} chunk temporali:")
    for i, (chunk_start, chunk_end) in enumerate(date_chunks):
        logging.info(f"  Chunk {i+1}: {chunk_start.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}")

    df_trends_full = pd.DataFrame()

    # Itera su ogni keyword
    for keyword in tqdm(keywords, desc="Analisi Google Trends per keyword"):
        keyword_data_chunks = []

        # Per ogni chunk temporale
        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks):
            timeframe = f'{chunk_start.strftime("%Y-%m-%d")} {chunk_end.strftime("%Y-%m-%d")}'

            # Retry logic per questo chunk
            for attempt in range(max_retries):
                try:
                    # Delay PRIMA di ogni richiesta
                    time.sleep(random.uniform(min_delay, max_delay))

                    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
                    interest_over_time_df = pytrends.interest_over_time()

                    if not interest_over_time_df.empty and keyword in interest_over_time_df.columns:
                        keyword_data_chunks.append(interest_over_time_df[[keyword]])
                        logging.info(f"  [OK] '{keyword}' - Chunk {chunk_idx+1}/{len(date_chunks)} scaricato")
                        break  # Successo, passa al prossimo chunk
                    elif not interest_over_time_df.empty and keyword not in interest_over_time_df.columns:
                        logging.info(f"  [OK] '{keyword}' - Chunk {chunk_idx+1}/{len(date_chunks)} - Nessun dato (vuoto).")
                        break
                    else:
                        logging.info(f"  [OK] '{keyword}' - Chunk {chunk_idx+1}/{len(date_chunks)} - Nessun dato (vuoto).")
                        break

                except Exception as e:
                    # Backoff esponenziale per rate limiting
                    wait_time = max_delay + (2 ** attempt) + random.uniform(1, 3)
                    logging.warning(f"  [FAIL] '{keyword}' - Chunk {chunk_idx+1}/{len(date_chunks)} - Tentativo {attempt+1} fallito. Errore: {e}. Attendo {wait_time:.1f}s.")

                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        logging.error(f"  [FAIL] '{keyword}' - Chunk {chunk_idx+1}/{len(date_chunks)} - Impossibile scaricare dopo {max_retries} tentativi.")

            # Delay extra tra chunk per ridurre rate limiting
            if chunk_idx < len(date_chunks) - 1:
                time.sleep(chunk_delay)

        # Concatena tutti i chunk per questa keyword
        if keyword_data_chunks:
            keyword_full = pd.concat(keyword_data_chunks).sort_index()
            # Rimuovi duplicati se presenti (per overlap tra chunk)
            keyword_full = keyword_full[~keyword_full.index.duplicated(keep='first')]

            if df_trends_full.empty:
                df_trends_full = keyword_full
            else:
                df_trends_full = df_trends_full.join(keyword_full, how='outer')

            logging.info(f"'{keyword}' - Completato: {len(keyword_full)} data points")
        else:
            logging.warning(f"[WARN] '{keyword}' - Nessun dato scaricato")

    if df_trends_full.empty:
        logging.warning("Nessun dato Google Trends scaricato. Ritorno DataFrame vuoto.")
        return pd.DataFrame(index=index)

    # Preprocessing
    df_trends_full.columns = [f"GT_{c.replace(' ', '_')}" for c in df_trends_full.columns]

    # ⚠️ CRITICO: SHIFT(1) PER EVITARE DATA LEAKAGE!
    # I dati di Trends del giorno T sono disponibili solo il giorno T+1
    df_trends_shifted = df_trends_full.reindex(index).shift(1).ffill().fillna(0)

    logging.info("Calcolo Z-Score per i dati di Google Trends...")
    df_trends_z = pd.DataFrame(index=df_trends_shifted.index)
    for col in df_trends_shifted.columns:
        rolling_mean = df_trends_shifted[col].rolling(window=z_score_window).mean()
        rolling_std = df_trends_shifted[col].rolling(window=z_score_window).std().replace(0, 1)
        df_trends_z[f"{col}_Z"] = (df_trends_shifted[col] - rolling_mean) / rolling_std
    df_trends_z.fillna(0, inplace=True)
    df_trends_z.to_parquet(TRENDS_CACHE_PATH)

    logging.info(f"[WARN] Google Trends completato: {len(df_trends_z.columns)} feature create")
    return df_trends_z

def create_trends_ratios(df_trends_z):
    logging.info("Creazione di feature ratio da Google Trends...")
    df_ratios = pd.DataFrame(index=df_trends_z.index)
    epsilon = 1e-6
    fear_col, greed_col = 'GT_stock_market_crash_Z', 'GT_investing_Z'
    if fear_col in df_trends_z.columns and greed_col in df_trends_z.columns:
        df_ratios['GT_FearGreed_Ratio'] = df_trends_z[fear_col] / (df_trends_z[greed_col] + epsilon)
    hype_col, winter_col = 'GT_buy_crypto_Z', 'GT_crypto_winter_Z'
    if hype_col in df_trends_z.columns and winter_col in df_trends_z.columns:
        df_ratios['GT_CryptoHype_Ratio'] = df_trends_z[hype_col] / (df_trends_z[winter_col] + epsilon)
    return df_ratios.fillna(0)

def get_normalized_market_momentum(open_prices, close_prices, indicators, window_size=90):
    logging.info("Calcolo del momentum normalizzato...")
    momentum_df = pd.DataFrame(index=open_prices.index)
    for name, ticker in indicators.items():
        if ticker in open_prices.columns and ticker in close_prices.columns:
            intraday_return = (close_prices[ticker] - open_prices[ticker]) / open_prices[ticker]
            rolling_mean = intraday_return.rolling(window=window_size).mean()
            rolling_std = intraday_return.rolling(window=window_size).std()
            normalized_momentum = (intraday_return - rolling_mean) / rolling_std
            momentum_df[name] = normalized_momentum
    momentum_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return momentum_df.fillna(0)

# ==============================================================================
# --- ⚠️ FUNZIONE MODIFICATA (v74) ---
# ==============================================================================
def calculate_features_for_ticker(s, df_open, df_high, df_low, df_close, df_volume):
    """
    Calcola indicatori tecnici STAZIONARI (usando .diff()) per un singolo ticker.
    Questo previene la correlazione spuria.
    """
    import pandas as pd
    import pandas_ta as ta
    from numpy import nan, sqrt, std, subtract, log, polyfit
    
    def _get_hurst_exponent(series, max_lag=100):
        if len(series) < max_lag: return nan
        try:
            lags = range(2, max_lag)
            tau = [sqrt(std(subtract(series[lag:], series[:-lag]))) for lag in lags]
            tau_filtered = [t for t in tau if t > 0]
            lags_filtered = [lags[i] for i, t in enumerate(tau) if t > 0]
            if len(tau_filtered) < 2: return nan
            poly = polyfit(log(lags_filtered), log(tau_filtered), 1)
            return poly[0] * 2.0
        except Exception: return nan

    # DataFrame di input con i dati grezzi
    df_ta_input = pd.DataFrame({
        'Open': df_open[s], 
        'High': df_high[s], 
        'Low': df_low[s], 
        'Close': df_close[s], 
        'Volume': df_volume.get(s, pd.Series(0, index=df_close.index))
    }).dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    if df_ta_input.empty: 
        return None

    # --- ⚠️ v74: Nuovo DataFrame per le feature STAZIONARIE ---
    df_ta = pd.DataFrame(index=df_ta_input.index)

    # Calcola e trasforma indicatori (usando .diff() per stazionarizzare)

    # Oscillatori (già relativamente stazionari, ma .diff() misura il 'cambiamento')
    df_ta['RSI_14_diff'] = ta.rsi(df_ta_input['Close'], length=14).diff()
    df_ta['RSI_21_diff'] = ta.rsi(df_ta_input['Close'], length=21).diff()
    df_ta['MOM_10_diff'] = ta.mom(df_ta_input['Close'], length=10).diff()
    df_ta['ROC_12_diff'] = ta.roc(df_ta_input['Close'], length=12).diff()
    df_ta['CG_10_diff'] = ta.cg(df_ta_input['Close'], length=10).diff()

    stoch = ta.stoch(df_ta_input['High'], df_ta_input['Low'], df_ta_input['Close'], k=14, d=3)
    if stoch is not None and not stoch.empty:
        df_ta['STOCHk_14_3_3_diff'] = stoch.iloc[:, 0].diff()
        df_ta['STOCHd_14_3_3_diff'] = stoch.iloc[:, 1].diff()

    # Trend/Medie Mobili (NON stazionari, .diff() è FONDAMENTALE)
    df_ta['SMA_20_diff'] = ta.sma(df_ta_input['Close'], length=20).diff()
    df_ta['EMA_20_diff'] = ta.ema(df_ta_input['Close'], length=20).diff()
    df_ta['EMA_50_diff'] = ta.ema(df_ta_input['Close'], length=50).diff()

    macd = ta.macd(df_ta_input['Close'], fast=12, slow=26)
    if macd is not None and not macd.empty:
        df_ta['MACD_12_26_9_diff'] = macd.iloc[:, 0].diff() # Linea MACD
        df_ta['MACDh_12_26_9_diff'] = macd.iloc[:, 1].diff() # Istogramma
        df_ta['MACDs_12_26_9_diff'] = macd.iloc[:, 2].diff() # Linea di segnale

    adx = ta.adx(df_ta_input['High'], df_ta_input['Low'], df_ta_input['Close'], length=14)
    if adx is not None and not adx.empty:
        df_ta['ADX_14_diff'] = adx.iloc[:, 0].diff()
        df_ta['DMP_14_diff'] = adx.iloc[:, 1].diff()
        df_ta['DMN_14_diff'] = adx.iloc[:, 2].diff()

    vortex = ta.vortex(df_ta_input['High'], df_ta_input['Low'], df_ta_input['Close'])
    if vortex is not None and not vortex.empty:
        df_ta['VTXP_14_diff'] = vortex.iloc[:, 0].diff()
        df_ta['VTXM_14_diff'] = vortex.iloc[:, 1].diff()

    bbands = ta.bbands(df_ta_input['Close'], length=20)
    if bbands is not None and not bbands.empty:
        df_ta['BBL_20_2.0_diff'] = bbands.iloc[:, 0].diff()
        df_ta['BBM_20_2.0_diff'] = bbands.iloc[:, 1].diff()
        df_ta['BBU_20_2.0_diff'] = bbands.iloc[:, 2].diff()
        df_ta['BBB_20_2.0'] = bbands.iloc[:, 3] # Bandwidth è già un ratio, OK
        df_ta['BBP_20_2.0'] = bbands.iloc[:, 4] # Percentuale è già un ratio, OK

    donchian = ta.donchian(df_ta_input['High'], df_ta_input['Low'], lower_length=20, upper_length=20)
    if donchian is not None and not donchian.empty:
        df_ta['DCL_20_20_diff'] = donchian.iloc[:, 0].diff()
        df_ta['DCM_20_20_diff'] = donchian.iloc[:, 1].diff()
        df_ta['DCU_20_20_diff'] = donchian.iloc[:, 2].diff()

    # Volatilità (ATR è già una misura di 'range', ma .diff() ne misura il cambiamento)
    df_ta['ATR_14_diff'] = ta.atr(df_ta_input['High'], df_ta_input['Low'], df_ta_input['Close'], length=14).diff()

    # Volume (OBV, EFI sono NON stazionari, CMF è un oscillatore)
    if 'Volume' in df_ta_input.columns and df_ta_input['Volume'].sum() > 0:
        df_ta['OBV_diff'] = ta.obv(df_ta_input['Close'], df_ta_input['Volume']).diff()
        df_ta['CMF_20_diff'] = ta.cmf(df_ta_input['High'], df_ta_input['Low'], df_ta_input['Close'], df_ta_input['Volume'], length=20).diff()
        df_ta['EFI_13_diff'] = ta.efi(df_ta_input['Close'], df_ta_input['Volume'], length=13).diff()

    # Hurst (già un esponente, stazionario)
    df_ta['HURST_100D'] = df_ta_input['Close'].rolling(window=100).apply(_get_hurst_exponent, args=(100,), raw=False)

    # Riempi i NaN creati da .diff() e rolling()
    df_ta = df_ta.fillna(0.0)
    
    # Rinomina tutte le colonne
    df_ta = df_ta.rename(columns=lambda x: f"{s}_{x}")
    return df_ta
# ==============================================================================
# --- (Fine della funzione modificata v74) ---
# ==============================================================================


def get_advanced_technical_indicators_parallel(df_close, df_high, df_low, df_open, df_volume):
    logging.info("Calcolo indicatori tecnici STAZIONARI (v74, CPU-Parallelized)...") # ⚠️ v74
    results = Parallel(n_jobs=config["MAX_CPU_CORES"], backend='loky')(delayed(calculate_features_for_ticker)(s, df_open, df_high, df_low, df_close, df_volume) for s in tqdm(df_close.columns, desc="Calcolo indicatori per ticker"))
    valid_results = [res for res in results if res is not None]
    if not valid_results: return pd.DataFrame()
    return pd.concat(valid_results, axis=1)

def get_or_calculate_technical_indicators(df_close, df_high, df_low, df_open, df_volume):
    if os.path.exists(TECH_INDICATORS_CACHE_PATH) and not config["FORCE_RECALCULATE_FEATURES"]:
        logging.info(f"Caricamento indicatori tecnici dalla cache: {TECH_INDICATORS_CACHE_PATH}")
        try: return pd.read_parquet(TECH_INDICATORS_CACHE_PATH)
        except Exception as e: logging.warning(f"Errore lettura cache indicatori: {e}. Ricalcolo.")
    tech_df = get_advanced_technical_indicators_parallel(df_close, df_high, df_low, df_open, df_volume)
    tech_df.to_parquet(TECH_INDICATORS_CACHE_PATH)
    return tech_df

def get_garch_features(df_close):
    if os.path.exists(GARCH_FEATURES_CACHE_PATH):
        logging.info(f"Caricamento feature GARCH dalla cache: {GARCH_FEATURES_CACHE_PATH}")
        try: return pd.read_parquet(GARCH_FEATURES_CACHE_PATH)
        except Exception as e: logging.warning(f"Errore lettura cache GARCH: {e}. Ricalcolo.")
    logging.info("Pre-calcolo delle feature di volatilità GARCH...")
    def calculate_garch_for_ticker(s):
        returns = df_close[s].pct_change().dropna() * 100
        returns = returns.asfreq('B').fillna(0.0)
        if len(returns) < 100: return None
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
            res = model.fit(update_freq=0, disp='off', show_warning=False)
            return pd.Series(res.conditional_volatility / 100, name=f'{s}_GARCH_forecast')
        except Exception: return None
    results = Parallel(n_jobs=config["MAX_CPU_CORES"])(delayed(calculate_garch_for_ticker)(s) for s in tqdm(df_close.columns, desc="Previsioni GARCH"))
    garch_df = pd.concat([res for res in results if res is not None], axis=1)
    garch_df = garch_df.shift(1).fillna(0)
    garch_df.to_parquet(GARCH_FEATURES_CACHE_PATH)
    return garch_df

def fit_predict_single_hmm(train_data, predict_data, n_regimes=3):
    if len(train_data) < n_regimes * 20 or train_data['returns'].std() < 1e-6 or train_data['volatility'].std() < 1e-6:
        return pd.Series(0, index=predict_data.index)
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=config["SEED"], min_covar=1e-3)
    try:
        model.fit(train_data.values)
        regime_stats = pd.DataFrame(model.means_, columns=['returns', 'volatility']).sort_values(by='volatility')
        regime_map = {old_label: new_label for new_label, old_label in enumerate(regime_stats.index)}
        if predict_data.empty: return pd.Series(dtype='int64')
        hidden_states = model.predict(predict_data.values)
        mapped_states = pd.Series(hidden_states, index=predict_data.index).map(regime_map)
        return mapped_states
    except (ValueError, LinAlgError):
        return pd.Series(0, index=predict_data.index)

def calculate_all_walk_forward_hmm_regimes(df_close, market_indices_map, run_new_analysis=False):
    if not run_new_analysis and os.path.exists(HMM_REGIMES_CACHE_PATH):
        logging.info(f"Caricamento dei regimi HMM pre-calcolati (walk-forward) da: {HMM_REGIMES_CACHE_PATH}")
        return pd.read_parquet(HMM_REGIMES_CACHE_PATH)
    logging.info("Inizio calcolo regimi HMM con approccio Walk-Forward (potrebbe richiedere molto tempo)...")
    all_regimes, full_index = {}, df_close.index
    for market_index, regime_name in tqdm(market_indices_map.items(), desc="Calcolo Regimi HMM per Indice"):
        if market_index not in df_close.columns:
            logging.warning(f"Indice HMM '{market_index}' non trovato nei dati. Salto.")
            continue
        price_series = df_close[market_index]
        returns, volatility = price_series.pct_change(), price_series.pct_change().rolling(window=21).std()
        hmm_data = pd.concat([returns, volatility], axis=1).dropna(); hmm_data.columns = ['returns', 'volatility']
        monthly_groups, monthly_regimes = list(hmm_data.groupby(pd.Grouper(freq='M'))), []
        for i in tqdm(range(1, len(monthly_groups)), desc=f"WF per {regime_name}", leave=False):
            train_end_date, predict_indices = monthly_groups[i-1][0], monthly_groups[i][1].index
            train_data, predict_data = hmm_data.loc[:train_end_date], hmm_data.loc[predict_indices]
            if not train_data.empty and not predict_data.empty:
                monthly_regimes.append(fit_predict_single_hmm(train_data, predict_data))
        if monthly_regimes: all_regimes[f"{regime_name}_regime"] = pd.concat(monthly_regimes)
    regimes_df = pd.DataFrame(all_regimes, index=full_index).ffill().fillna(0).astype(int)
    logging.info("Salvataggio dei regimi HMM calcolati su disco...")
    regimes_df.to_parquet(HMM_REGIMES_CACHE_PATH)
    return regimes_df

def get_regime_info(symbol):
    market_map = {**{s: 'crypto' for s in config["CRYPTO_ASSETS"]}, **{s: 'biotech' for s in config["BIOTECH_ASSETS"]}, **{s: 'trad' for s in config["TECH_ASSETS"]}}
    regime_name = market_map.get(symbol, 'trad')
    directional_features_map = {'crypto': list(config["CRYPTO_DIRECTIONAL_INDICATORS"].keys()), 'biotech': list(config["BIOTECH_DIRECTIONAL_INDICATORS"].keys()), 'trad': list(config["MARKET_DIRECTIONAL_INDICATORS"].keys())}
    return f"{regime_name}", directional_features_map[regime_name]

def calculate_drawdown(cumulative_returns_series):
    if cumulative_returns_series is None or cumulative_returns_series.empty: return 0.0
    high_water_mark = cumulative_returns_series.cummax()
    drawdown = (cumulative_returns_series - high_water_mark) / high_water_mark
    return drawdown.min()

def load_isee_data(index, window_size, file_path):
    logging.info(f"Caricamento ISEE e calcolo Z-score con finestra di {window_size} giorni...")
    col_name = f'{config["ISEE_MOMENTUM_Z_COL_BASE"]}_{window_size}'
    if not os.path.exists(file_path):
        logging.warning(f"File dati ISEE '{file_path}' non trovato.")
        return pd.DataFrame({col_name: 0.0}, index=index)
    try:
        df = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')
        if 'ALL ETFS' not in df.columns:
            logging.warning("Colonna 'ALL ETFS' non trovata nel file ISEE.")
            return pd.DataFrame({col_name: 0.0}, index=index)
        momentum = df['ALL ETFS'].ewm(span=20, adjust=False).mean() - df['ALL ETFS'].ewm(span=50, adjust=False).mean()
        rolling_mean, rolling_std = momentum.rolling(window=window_size).mean(), momentum.rolling(window=window_size).std().replace(0, 1)
        z_score = (momentum - rolling_mean) / rolling_std
        final_df = z_score.to_frame(name=col_name).reindex(index).shift(1).fillna(0)
        return final_df
    except Exception as e:
        logging.critical(f"Errore CRITICO durante il processamento dei dati ISEE: {e}")
        return pd.DataFrame({col_name: 0.0}, index=index)

def create_macro_ratios(df_close):
    logging.info("Creazione dei meta-indicatori macroeconomici...")
    df_ratios = pd.DataFrame(index=df_close.index)
    if 'GC=F' in df_close.columns and 'HG=F' in df_close.columns:
        ratio = df_close['GC=F'] / df_close['HG=F'].replace(0, np.nan)
        mean, std = ratio.rolling(window=90).mean(), ratio.rolling(window=90).std().replace(0, np.nan)
        df_ratios['GCR_Z'] = (ratio - mean) / std
    if 'GC=F' in df_close.columns and 'SI=F' in df_close.columns:
        ratio = df_close['GC=F'] / df_close['SI=F'].replace(0, np.nan)
        mean, std = ratio.rolling(window=90).mean(), ratio.rolling(window=90).std().replace(0, np.nan)
        df_ratios['GSR_Z'] = (ratio - mean) / std
    if '^VIX' in df_close.columns and '^VXV' in df_close.columns:
        ratio = df_close['^VIX'] / df_close['^VXV'].replace(0, np.nan)
        mean, std = ratio.rolling(window=10).mean(), ratio.rolling(window=10).std().replace(0, np.nan)
        df_ratios['VIX_VXV_Z'] = (ratio - mean) / std
    return df_ratios.ffill().shift(1).fillna(0)

def create_trend_momentum_divergence(df_close, df_tech_indicators):
    logging.info("Creazione della feature di divergenza Trend/Momentum...")
    df_div = pd.DataFrame(index=df_close.index)
    sma200 = df_close.rolling(window=200, min_periods=200).mean()
    # ⚠️ v74: Cerca i nomi delle feature modificate (es. _MACDh_..._diff)
    macd_h_cols = {col: col.split('_')[0] for col in df_tech_indicators.columns if "MACDh" in col and "diff" in col}
    for col, asset in macd_h_cols.items():
        if asset not in sma200.columns: continue
        trend_state = np.sign(df_close[asset] - sma200[asset]).shift(1).fillna(0)
        momentum_state = np.sign(df_tech_indicators[col]).shift(1).fillna(0) # .diff() non cambia il segno
        df_div[f"{asset}_trend_mom_div"] = np.where(trend_state * momentum_state < 0, -1, 0)
    return df_div.fillna(0)

def create_ts_momentum_percentile(df_close, window=250):
    logging.info(f"Creazione della feature di Time-Series Momentum Percentile (window={window})...")
    ts_pct = df_close.pct_change().shift(1).rolling(window, min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    ts_pct.columns = [f"{col}_ts_mom_pct" for col in ts_pct.columns]
    return ts_pct.fillna(0)

def create_realized_volatility(df_close, window=21):
    logging.info(f"Creazione della feature di Realized Volatility (window={window})...")
    rv_shifted = df_close.pct_change().rolling(window, min_periods=window).std().shift(1)
    rv_shifted.columns = [f"{col}_rv{window}" for col in rv_shifted.columns]
    return rv_shifted.fillna(0)

class PurgedTimeSeriesSplit(_BaseKFold):
    def __init__(self, n_splits=3, purge_length=10):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.purge_length = purge_length
    def split(self, X, y=None, groups=None):
        n_samples, indices = len(X), np.arange(len(X))
        n_folds = self.n_splits + 1
        fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
        fold_sizes[:n_samples % n_folds] += 1
        current = 0
        for i in range(self.n_splits):
            test_start_idx = current + fold_sizes[i]
            purged_train_end_idx = test_start_idx - self.purge_length
            test_end_idx = test_start_idx + fold_sizes[i + 1]
            if purged_train_end_idx > 0:
                train_indices, test_indices = indices[0:purged_train_end_idx], indices[test_start_idx:test_end_idx]
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
            current = test_start_idx

def prepare_model_data(symbol, df_features_base, df_targets, df_regimes, indices_to_use, isee_z_col, isee_regime_col):
    regime_col_name, _ = get_regime_info(symbol)
    use_ohe = symbol not in config["NO_OHE_ASSETS"]
    X_data = df_features_base.loc[indices_to_use].join(df_regimes)
    if f"{regime_col_name}_regime" in X_data.columns:
         X_data.rename(columns={f"{regime_col_name}_regime": 'market_regime'}, inplace=True)
    else:
        X_data['market_regime'] = 0
    if isee_z_col in X_data.columns and 'market_regime' in X_data.columns:
        X_data[isee_regime_col] = (X_data[isee_z_col] * (X_data['market_regime'] == 2)).astype(float)
    else:
        X_data[isee_regime_col] = 0.0
    if use_ohe:
        X_data['market_regime'] = pd.Categorical(X_data['market_regime'], categories=[0.0, 1.0, 2.0])
        X_data = pd.get_dummies(X_data, columns=['market_regime'], prefix='regime', dtype=float)
    X_data.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X_data.columns]
    y_series = df_targets[symbol].loc[X_data.index]
    combined_data = X_data.join(y_series.to_frame(name='target')).dropna(subset=['target'])
    X_clean, y_clean = combined_data.drop(columns=['target']), combined_data['target']
    return X_clean, y_clean

def calculate_trade_metrics(positions, close_prices, high_prices, low_prices, actual_returns, commission_rate):
    """
    Calcola metriche di trade dettagliate.
    IMPORTANTE: Questa funzione riceve 'positions' che sono GIÀ SHIFTATE (v73+).
    """
    trades = []
    current_position = 0
    entry_date = None
    entry_price = None

    # Allinea tutti gli indici
    aligned_idx = positions.index.intersection(close_prices.index).intersection(high_prices.index).intersection(low_prices.index).intersection(actual_returns.index)
    positions = positions.loc[aligned_idx]
    close_prices = close_prices.loc[aligned_idx]
    high_prices = high_prices.loc[aligned_idx]
    low_prices = low_prices.loc[aligned_idx]
    actual_returns = actual_returns.loc[aligned_idx]

    # --- Calcolo Ritorno Giornaliero (per il PnL cumulativo) ---
    # 'positions' è già shiftato, quindi questa è la posizione del Giorno T 
    # basata sui dati del Giorno T-1, moltiplicata per il ritorno del Giorno T.
    strategy_returns = positions * actual_returns

    # Applica commissioni sui cambi di posizione
    position_changes = positions.diff().fillna(0)
    commission_costs = position_changes.abs() * commission_rate
    strategy_returns = strategy_returns - commission_costs

    # --- Calcolo Trade-per-Trade (per il CSV) ---
    for date in positions.index:
        new_position = positions.loc[date]

        # Rilevato cambio di posizione
        if new_position != current_position:
            # 1. Chiudi la posizione esistente (se ce n'è una)
            if current_position != 0 and entry_date is not None:
                # Esci al prezzo di chiusura del giorno 'date'
                exit_price = close_prices.loc[date]
                exit_date = date

                # Calcola PnL
                if current_position == 1:  # LONG
                    pnl_gross = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    pnl_gross = (entry_price - exit_price) / entry_price

                # Commissioni totali (entrata + uscita)
                commission_total = 2 * commission_rate
                pnl_net = pnl_gross - commission_total

                holding_time = (exit_date - entry_date).days

                # Calcola MFE e MAE durante il trade
                trade_period_idx = positions.index[(positions.index >= entry_date) & (positions.index <= exit_date)]
                if not trade_period_idx.empty:
                    highs_in_trade = high_prices.loc[trade_period_idx]
                    lows_in_trade = low_prices.loc[trade_period_idx]

                    if current_position == 1:  # LONG
                        mfe = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                        mae = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                    else:  # SHORT
                        mfe = ((entry_price - lows_in_trade.min()) / entry_price) * 100
                        mae = ((highs_in_trade.max() - entry_price) / entry_price) * 100
                else:
                    mfe = mae = 0.0

                # Salva il trade
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

            # 2. Apri una nuova posizione (se non si sta andando a 'flat')
            if new_position != 0:
                entry_date = date
                entry_price = close_prices.loc[date] # Entra al prezzo di chiusura 'date'
            else:
                # Reset se si va a 'flat'
                entry_date = None
                entry_price = None

            # Aggiorna la posizione corrente
            current_position = new_position

    # Gestisci l'ultimo trade se è ancora aperto alla fine
    if current_position != 0 and entry_date is not None:
        exit_date = positions.index[-1]
        exit_price = close_prices.loc[exit_date]
        
        if (exit_date > entry_date): # Assicurati che non sia un trade di 0 giorni non valido
            if current_position == 1:
                pnl_gross = (exit_price - entry_price) / entry_price
            else:
                pnl_gross = (entry_price - exit_price) / entry_price

            commission_total = 2 * commission_rate
            pnl_net = pnl_gross - commission_total
            holding_time = (exit_date - entry_date).days

            trade_period_idx = positions.index[(positions.index >= entry_date) & (positions.index <= exit_date)]
            if not trade_period_idx.empty:
                highs_in_trade = high_prices.loc[trade_period_idx]
                lows_in_trade = low_prices.loc[trade_period_idx]

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

    # Ritorna sia i ritorni giornalieri (per il grafico/calmar) sia i trade (per il CSV)
    return trades_df, strategy_returns

# =============================================================================
# --- FUNZIONI STRATEGIA CORRETTE (FIX v73) ---
# =============================================================================

def strategy_single_day_swing(predictions, actual_returns, **kwargs):
    """Day trading strategy: opens and closes positions daily."""
    # ⚠️ v77: I quantili ora sono letti dalla config che viene aggiornata dinamicamente
    rolling_buy_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_BUY"])
    rolling_short_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_SHORT"])
    
    aligned_buy_thresh = rolling_buy_thresh.shift(1)
    aligned_short_thresh = rolling_short_thresh.shift(1)
    positions = pd.Series(np.nan, index=predictions.index)
    positions.loc[predictions > aligned_buy_thresh] = 1
    positions.loc[predictions < aligned_short_thresh] = -1
    positions = positions.fillna(0)

    # ⚠️ FIX v73: SHIFT POSITIONS BY 1 to prevent lookahead bias
    shifted_positions = positions.shift(1).fillna(0)

    # Get price data from kwargs if available
    close_prices = kwargs.get('close_prices')
    high_prices = kwargs.get('high_prices')
    low_prices = kwargs.get('low_prices')

    if close_prices is not None and high_prices is not None and low_prices is not None:
        trades_df, strategy_returns = calculate_trade_metrics(
            shifted_positions,  # ⚠️ USA LE POSIZIONI SPOSTATE
            close_prices, high_prices, low_prices, actual_returns, config["COMMISSION_RATE"]
        )
        return strategy_returns.dropna(), trades_df
    else:
        # Fallback:
        strategy_returns = shifted_positions * actual_returns.loc[shifted_positions.index] # ⚠️ USA LE POSIZIONI SPOSTATE
        return strategy_returns.dropna(), None

def strategy_trend_following(predictions, actual_returns, **kwargs):
    """Momentum/Trend-following strategy: holds positions until opposite signal."""
    # ⚠️ v77: I quantili ora sono letti dalla config che viene aggiornata dinamicamente
    rolling_buy_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_BUY"])
    rolling_short_thresh = predictions.rolling(window=60, min_periods=20).quantile(config["QUANTILE_SHORT"])
    
    aligned_buy_thresh = rolling_buy_thresh.shift(1)
    aligned_short_thresh = rolling_short_thresh.shift(1)
    signals = pd.Series(0, index=predictions.index)
    signals.loc[predictions > aligned_buy_thresh] = 1
    signals.loc[predictions < aligned_short_thresh] = -1
    positions = signals.replace(0, np.nan).ffill().fillna(0)

    # ⚠️ FIX v73: SHIFT POSITIONS BY 1 to prevent lookahead bias
    shifted_positions = positions.shift(1).fillna(0)

    # Get price data from kwargs if available
    close_prices = kwargs.get('close_prices')
    high_prices = kwargs.get('high_prices')
    low_prices = kwargs.get('low_prices')

    if close_prices is not None and high_prices is not None and low_prices is not None:
        trades_df, strategy_returns = calculate_trade_metrics(
            shifted_positions,  # ⚠️ USA LE POSIZIONI SPOSTATE
            close_prices, high_prices, low_prices, actual_returns, config["COMMISSION_RATE"]
        )
        return strategy_returns.dropna(), trades_df
    else:
        # Fallback:
        strategy_returns = shifted_positions * actual_returns.loc[shifted_positions.index] # ⚠️ USA LE POSIZIONI SPOSTATE
        return strategy_returns.dropna(), None

# =============================================================================
# --- ⚠️ FUNZIONE MODIFICATA (v76) ---
# =============================================================================

def run_backtest_evaluation(strategy_fn, symbol, df_features_base, df_targets, df_regimes, df_close, df_high, df_low, X_main_idx, X_unseen_idx, isee_z_col, isee_regime_col, n_features, df_seasonal, df_macro_ratios, strategy_name="Overnight"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xgb_params = {
        'objective': 'reg:squarederror',
        'random_state': config["SEED"],
        'device': device,
        'tree_method': 'hist'
    }
    if device == "cuda":
        xgb_params['max_bin'] = config["GPU_MAX_BIN"]
        xgb_params['grow_policy'] = config["GPU_GROW_POLICY"]

    logging.info(f"[{symbol} | {strategy_name}] Preparazione dati di training...")
    X_train_initial, y_train_initial = prepare_model_data(symbol, df_features_base, df_targets, df_regimes, X_main_idx, isee_z_col, isee_regime_col)
    
    if X_train_initial.empty:
        logging.error(f"Impossibile preparare i dati per {symbol}, dati iniziali vuoti.")
        return None

    # Pulisci colonne con varianza zero
    non_constant_cols = X_train_initial.columns[X_train_initial.nunique() > 1]
    X_train_initial = X_train_initial[non_constant_cols]

    final_feature_list = []

    # ⚠️ v76: Nuova logica di feature selection "curated_lite"
    if n_features == 'curated_lite':
        logging.info(f"[{symbol} | {strategy_name}] Applicazione del feature set 'curated_lite'...")
        
        # 1. Feature Tecniche Stazionarie (poche e buone)
        curated_tech_features = [
            f'{symbol}_RSI_14_diff',
            f'{symbol}_MACDh_12_26_9_diff',
            f'{symbol}_OBV_diff',
            f'{symbol}_CMF_20_diff',
            f'{symbol}_ATR_14_diff',
            f'{symbol}_BBP_20_2.0', # Già stazionario
        ]
        
        # 2. Feature di Regime (HMM)
        regime_features = ['market_regime']
        if symbol not in config["NO_OHE_ASSETS"]:
             regime_features = ['regime_0.0', 'regime_1.0', 'regime_2.0']

        # 3. Feature di Sentiment & News (come da tua richiesta)
        sentiment_features = [
            f'{symbol}_Sentiment_Formal',
            f'{symbol}_Sentiment_Social',
            'GT_FearGreed_Ratio',
        ]
        
        # 4. Feature di Momentum di Mercato (Specifiche per Asset)
        market_momentum_features = []
        if symbol in config["TECH_ASSETS"]:
            market_momentum_features.append('nasdaq_dir')
        if symbol in config["BIOTECH_ASSETS"]:
            market_momentum_features.append('biotech_dir')
        if symbol in config["CRYPTO_ASSETS"]:
             market_momentum_features.append('btc_dir')
        
        # 5. Feature Globali & Stagionali
        global_features = [
            'VIX_VXV_Z',
            'weekday_sin',
            'month_sin',
        ]
        
        # Unisci tutte le liste
        final_feature_list = (
            curated_tech_features + 
            regime_features + 
            sentiment_features + 
            market_momentum_features + 
            global_features
        )
        
        # Pulisci la lista: rimuovi duplicati e feature che non esistono nei dati
        final_feature_list = list(dict.fromkeys(final_feature_list))
        final_feature_list = [f for f in final_feature_list if f in X_train_initial.columns]
        
    else:
        # Logica precedente (v74) per 'all' o numeri [20, 40, ...]
        logging.info(f"[{symbol} | {strategy_name}] Esecuzione selezione feature XGBoost per {n_features} feature...")
        prelim_model = xgb.XGBRegressor(**xgb_params).fit(X_train_initial, y_train_initial)
        ranked_features = pd.Series(prelim_model.feature_importances_, index=prelim_model.feature_names_in_).sort_values(ascending=False).index.tolist()
        
        _, directional_features = get_regime_info(symbol)
        base_global_features = list(config["GENERAL_DIRECTIONAL_INDICATORS"].keys()) + list(config["MACRO_TICKERS"].keys()) + directional_features + list(df_macro_ratios.columns)
        base_global_features.extend([c for c in df_features_base.columns if 'ISEE_' in c] + [f"{symbol}_trend_mom_div", isee_regime_col] + [c for c in df_features_base.columns if 'GT_' in c] + list(df_seasonal.columns))
        if symbol not in config["NO_OHE_ASSETS"]: base_global_features.extend(['regime_0.0', 'regime_1.0', 'regime_2.0'])
        else: base_global_features.append('market_regime')
        if symbol in config["CRYPTO_ASSETS"]: base_global_features.extend([name for name, ticker in config["CRYPTO_INDICATORS"].items() if ticker != symbol])
        
        cleaned_ranked_features = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in ranked_features]
        
        symbol_specific_keywords_to_exclude = [
            'regime', 'dir', 'Treasury', 'Index', 'Oil', 'Rate', 'Bitcoin', 'Ethereum', 'Solana', 'Close', 
            'market_regime', 'ISEE_', 'GCR_', 'GSR_', 'VIX_VXV_', '_GARCH_', 'GT_', 
            'month_', 'day_', 'week_', 'is_first', 'is_last', 'trend_mom_div'
        ]
        if symbol in config["STABLE_CRYPTO_ASSETS"]: symbol_specific_keywords_to_exclude.extend(['_ts_mom_pct', '_rv21'])
        
        symbol_specific_ranked = [f for f in cleaned_ranked_features if not any(keyword in f for keyword in symbol_specific_keywords_to_exclude)]
        
        selected_features_base = (symbol_specific_ranked[:n_features] if n_features != 'all' else symbol_specific_ranked) + base_global_features
        final_feature_list = list(dict.fromkeys(selected_features_base))
        final_feature_list = [f for f in final_feature_list if f in X_train_initial.columns]

    
    logging.info(f"[{symbol} | {strategy_name}] Selezionate {len(final_feature_list)} feature per il backtest walk-forward.")
    
    unseen_chunks_indices = [g.index for _, g in X_unseen_idx.to_frame().groupby(pd.Grouper(freq='M')) if not g.empty]
    current_train_idx = X_main_idx
    all_predictions = pd.Series(dtype='float64')
    final_model = None
    
    for i, chunk_idx in enumerate(tqdm(unseen_chunks_indices, desc=f"WF Predictions for {symbol}", leave=False, position=1)):
        if i % config["REOPTIMIZATION_FREQUENCY_MONTHS"] == 0:
            logging.info(f"[{symbol} | {strategy_name}] Ri-ottimizzazione parametri modello per il chunk {i+1}/{len(unseen_chunks_indices)}...")
            X_train_wf, y_train_wf = prepare_model_data(symbol, df_features_base, df_targets, df_regimes, current_train_idx, isee_z_col, isee_regime_col)
            
            # Applica la lista di feature curata/selezionata
            X_train_wf_selected = X_train_wf.reindex(columns=final_feature_list, fill_value=0)
            
            if X_train_wf_selected.empty:
                logging.warning(f"Skipping re-optimization for {symbol}; empty training data.")
                continue
                
            random_search = RandomizedSearchCV(
                estimator=xgb.XGBRegressor(**xgb_params),
                param_distributions={'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [500, 700]},
                n_iter=6, cv=PurgedTimeSeriesSplit(n_splits=3, purge_length=config["PURGE_DAYS"]),
                scoring='neg_root_mean_squared_error', n_jobs=1, verbose=0, random_state=config["SEED"]
            )
            random_search.fit(X_train_wf_selected, y_train_wf.loc[X_train_wf_selected.index])
            best_model_params = random_search.best_params_
            final_model = xgb.XGBRegressor(**best_model_params, **xgb_params).fit(X_train_wf_selected, y_train_wf.loc[X_train_wf_selected.index])
        
        if final_model:
            X_chunk_test, _ = prepare_model_data(symbol, df_features_base, df_targets, df_regimes, chunk_idx, isee_z_col, isee_regime_col)
            X_chunk_test_selected = X_chunk_test.reindex(columns=final_model.feature_names_in_, fill_value=0)
            
            if not X_chunk_test_selected.empty:
                chunk_predictions = pd.Series(final_model.predict(X_chunk_test_selected), index=X_chunk_test_selected.index)
                all_predictions = pd.concat([all_predictions, chunk_predictions])
        else:
             logging.warning(f"[{symbol} | {strategy_name}] No model available, cannot make predictions for chunk {i+1}.")
        current_train_idx = current_train_idx.union(chunk_idx)
        
    if all_predictions.empty:
        logging.warning(f"[{symbol} | {strategy_name}] No predictions were generated during the walk-forward backtest.")
        return None

    # Chiama la funzione di strategia (che ora ha il fix v73 per lo shift)
    strategy_returns, trades_df = strategy_fn(
        predictions=all_predictions,
        actual_returns=df_close[symbol].pct_change(),
        close_prices=df_close[symbol],
        high_prices=df_high[symbol] if symbol in df_high.columns else df_close[symbol],
        low_prices=df_low[symbol] if symbol in df_low.columns else df_close[symbol]
    )
    if strategy_returns is None or strategy_returns.empty:
        return None
        
    all_strategy_daily_returns = [strategy_returns]
    if final_model:
        model_save_path = os.path.join(output_dir, f"final_model_{symbol}_{strategy_name}.json")
        try:
            final_model.save_model(model_save_path)
            logging.info(f"[{symbol} | {strategy_name}] Modello finale salvato in: {model_save_path}")
        except Exception as e:
            logging.error(f"[{symbol} | {strategy_name}] Errore nel salvataggio del modello finale: {e}")
            
    final_strategy_daily_returns = pd.concat(all_strategy_daily_returns).dropna()
    cumulative_strategy_returns = (1 + final_strategy_daily_returns).cumprod()
    total_strategy_return = (cumulative_strategy_returns.iloc[-1] - 1) * 100 if not cumulative_strategy_returns.empty else 0
    max_dd = calculate_drawdown(cumulative_strategy_returns)
    calmar_ratio = total_strategy_return / abs(max_dd * 100) if max_dd != 0 else (np.inf if total_strategy_return > 0 else 0)
    
    return {
        "total_strategy_return": total_strategy_return,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar_ratio,
        "cumulative_returns": cumulative_strategy_returns,
        "trades_df": trades_df  # Aggiunge il DataFrame dei trade ai risultati
    }

def run_backtest_for_symbol(symbol, df_features_for_run, df_targets, df_regimes, df_close, df_high, df_low, n_features, window_size, train_idx, test_idx, df_seasonal, df_macro_ratios):
    logging.info(f"Starting backtest for {symbol} (Win={window_size}, Feats={n_features}, Q={config['QUANTILE_BUY']})") # ⚠️ v77: Log dei quantili
    ISEE_MOMENTUM_Z_COL = f'{config["ISEE_MOMENTUM_Z_COL_BASE"]}_{window_size}'
    ISEE_REGIME_DRIVEN_COL = f'{ISEE_MOMENTUM_Z_COL}_HighVol_Only'
    common_args = {
        "symbol": symbol, "df_features_base": df_features_for_run, "df_targets": df_targets,
        "df_regimes": df_regimes, "df_close": df_close, "df_high": df_high, "df_low": df_low,
        "X_main_idx": train_idx, "X_unseen_idx": test_idx, "isee_z_col": ISEE_MOMENTUM_Z_COL,
        "isee_regime_col": ISEE_REGIME_DRIVEN_COL, "n_features": n_features,
        "df_seasonal": df_seasonal, "df_macro_ratios": df_macro_ratios
    }
    result_sds = run_backtest_evaluation(strategy_fn=strategy_single_day_swing, **{**common_args, "strategy_name": "Overnight"})
    result_tf = run_backtest_evaluation(strategy_fn=strategy_trend_following, **{**common_args, "strategy_name": "Trend-Following"})
    return {symbol: {'single_day': result_sds, 'trend_following': result_tf}}

def load_and_prepare_features():
    # ⚠️ v77: Assicurati che TUTTI i ticker necessari siano scaricati
    all_tickers_in_config = list(set(
        config["TICKERS"] + 
        list(config["ADDITIONAL_MACRO_TICKERS"].values()) + 
        list(v for v in {**config["MARKET_DIRECTIONAL_INDICATORS"], **config["CRYPTO_DIRECTIONAL_INDICATORS"], **config["BIOTECH_DIRECTIONAL_INDICATORS"], **config["GENERAL_DIRECTIONAL_INDICATORS"]}.values()) + 
        list(config["CRYPTO_INDICATORS"].values()) + 
        list(config["MACRO_TICKERS"].values())
    ))

    if os.path.exists(FULL_FEATURES_CACHE_PATH) and not config["FORCE_RECALCULATE_FEATURES"]:
        logging.info(f"Caricamento del DataFrame delle feature pre-calcolate da: {FULL_FEATURES_CACHE_PATH}")
        df_features_base = pd.read_parquet(FULL_FEATURES_CACHE_PATH)
        for col in df_features_base.columns:
            if df_features_base[col].dtype == 'float64': df_features_base[col] = df_features_base[col].astype('float32')
        df_close, df_open, df_high, df_low, df_volume = download_data_with_fallback(all_tickers_in_config, config["START_DATE"], config["END_DATE"])
        df_seasonal = create_seasonality_features(df_close.index)
        df_macro_ratios = create_macro_ratios(df_close)
    else:
        logging.info("--- Fase 1: Preparazione Dati Base ---")
        df_close, df_open, df_high, df_low, df_volume = download_data_with_fallback(all_tickers_in_config, config["START_DATE"], config["END_DATE"])
        logging.info("\n--- Fase 2: Calcolo Feature ---")
        df_seasonal = create_seasonality_features(df_close.index)
        # ⚠️ v77: Passa l'elenco completo dei ticker a get_sentiment_scores
        all_sentiment_tickers = list(set(config["TICKERS"] + config["TECH_ASSETS"] + config["BIOTECH_ASSETS"]))
        df_sentiment = get_sentiment_scores(all_sentiment_tickers, config["NEWS_FILES"], df_close.index, run_new_analysis=config["RUN_NEW_SENTIMENT_ANALYSIS"])
        df_trends_z = get_google_trends_data(
            config["TRENDS_KEYWORDS"],
            df_close.index,
            config["TRENDS_Z_SCORE_WINDOW"],
            run_new_analysis=config["RUN_NEW_TRENDS_ANALYSIS"],
            min_delay=config["GOOGLE_TRENDS_MIN_DELAY"],
            max_delay=config["GOOGLE_TRENDS_MAX_DELAY"],
            max_retries=config["GOOGLE_TRENDS_MAX_RETRIES"],
            chunk_months=config["GOOGLE_TRENDS_CHUNK_MONTHS"],
            chunk_delay=config["GOOGLE_TRENDS_CHUNK_DELAY"]
        )
        df_trends_ratios = create_trends_ratios(df_trends_z)
        df_market_dir = get_normalized_market_momentum(df_open, df_close, {**config["MARKET_DIRECTIONAL_INDICATORS"], **config["CRYPTO_DIRECTIONAL_INDICATORS"], **config["BIOTECH_DIRECTIONAL_INDICATORS"], **config["GENERAL_DIRECTIONAL_INDICATORS"]})
        df_advanced_tech = get_or_calculate_technical_indicators(df_close, df_high, df_low, df_open, df_volume) # ⚠️ v74: Feature stazionarie
        df_divergence = create_trend_momentum_divergence(df_close, df_advanced_tech)
        df_garch = get_garch_features(df_close)
        df_crypto_prices = df_close[list(config["CRYPTO_INDICATORS"].values())].rename(columns={v: k for k, v in config["CRYPTO_INDICATORS"].items()})
        df_macro_base = df_close[[v for k,v in config["MACRO_TICKERS"].items()]].rename(columns={v:k for k,v in config["MACRO_TICKERS"].items()})
        df_macro_ratios = create_macro_ratios(df_close)
        df_macro_combined = df_macro_base.join(df_macro_ratios)
        # ⚠️ v77: Assicurati di calcolare ts_mom e rv per tutti i ticker rilevanti
        all_base_tickers = list(set(config["TICKERS"] + config["TECH_ASSETS"] + config["BIOTECH_ASSETS"] + config["CRYPTO_ASSETS"]))
        df_ts_mom = create_ts_momentum_percentile(df_close[all_base_tickers])
        df_rv = create_realized_volatility(df_close[all_base_tickers])
        
        feature_dfs = [df_seasonal, df_sentiment, df_advanced_tech, df_divergence, df_garch, df_market_dir, df_crypto_prices, df_macro_combined, df_trends_z, df_trends_ratios, df_ts_mom, df_rv]
        df_features_base = pd.DataFrame(index=df_close.index)
        for f_df in tqdm(feature_dfs, desc="Unione Feature Base"):
            if f_df is not None and not f_df.empty: df_features_base = df_features_base.join(f_df, how='left')
        df_features_base.ffill(inplace=True); df_features_base.fillna(0, inplace=True)
        for col in df_features_base.columns:
            if df_features_base[col].dtype == 'float64': df_features_base[col] = df_features_base[col].astype('float32')
        logging.info(f"Salvataggio del DataFrame delle feature complete in: {FULL_FEATURES_CACHE_PATH}")
        df_features_base.to_parquet(FULL_FEATURES_CACHE_PATH)

    df_targets = pd.DataFrame(index=df_close.index)
    # ⚠️ v77: Calcola i target per *tutti* gli asset, non solo quelli dello smoke test
    all_target_tickers = list(set(config["TICKERS"] + config["TECH_ASSETS"] + config["BIOTECH_ASSETS"] + config["CRYPTO_ASSETS"]))
    for ticker in tqdm(all_target_tickers, desc="Differenziazione Frazionaria"):
        if ticker in df_close.columns:
            df_targets[ticker] = frac_diff_ffd(df_close[ticker], d=config["FRAC_DIFF_D"], thres=1e-5).shift(-1)
        else:
            logging.warning(f"Ticker {ticker} non trovato in df_close per il calcolo del target.")
            
    # Assicurati che gli indici HMM esistano prima di provare a calcolarli
    market_indices_map_hmm = {'^IXIC': 'trad', 'BTC-USD': 'crypto'}
    if 'XBI' in df_close.columns:
        market_indices_map_hmm['XBI'] = 'biotech'
    else:
        logging.warning("Ticker XBI non trovato, impossibile calcolare regime HMM 'biotech'.")
        
    df_regimes = calculate_all_walk_forward_hmm_regimes(df_close, market_indices_map_hmm, run_new_analysis=config["RUN_NEW_HMM_ANALYSIS"])

    return df_features_base, df_targets, df_regimes, df_close, df_high, df_low, df_seasonal, df_macro_ratios

def save_trades_to_csv(symbol, strategy_name, rank, trades_df, total_return, max_dd, calmar_ratio, config_info, output_dir):
    """
    Salva i dettagli dei trade su CSV con metriche di riepilogo nell'intestazione.
    """
    if trades_df is None or trades_df.empty:
        logging.warning(f"Nessun trade da salvare per {symbol} - {strategy_name} - Top {rank}")
        return

    # Crea nome file
    # ⚠️ v77: Aggiungi i quantili al nome del file per distinguerli
    q_buy = config_info.get('quantile_buy', 'N/A')
    csv_filename = f"{symbol}_{strategy_name}_top{rank}_q{q_buy}_oos.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Scrivi intestazione con metriche di riepilogo
            f.write(f"# Summary Metrics\n")
            f.write(f"Total_Return_Pct,Max_Drawdown_Pct,Calmar_Ratio,Window_Size,Num_Features,Quantile_Buy,Quantile_Short\n")
            f.write(f"{total_return:.2f},{max_dd*100:.2f},{calmar_ratio:.2f},{config_info['window']},{config_info['features']},{config_info['quantile_buy']},{config_info['quantile_short']}\n")
            f.write(f"\n")
            f.write(f"# Trade Details\n")

        # Aggiungi i dati dei trade
        trades_df.to_csv(csv_path, mode='a', index=False, encoding='utf-8')
        logging.info(f"CSV salvato: {csv_path} ({len(trades_df)} trades)")

    except Exception as e:
        logging.error(f"Errore nel salvataggio CSV per {symbol} - {strategy_name} - Top {rank}: {e}")

def main():
    np.random.seed(config["SEED"]); random.seed(config["SEED"]); torch.manual_seed(config["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["SEED"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logging.info(f"***** CUDA (GPU) disponibile: {gpu_name} ({gpu_memory:.1f} GB) *****")
        logging.info(f"***** XGBoost userà GPU con tensor cores (device='cuda', tree_method='hist', max_bin=256) *****")
    else:
        logging.info("***** CUDA (GPU) non disponibile. Lo script utilizzerà la CPU per l'addestramento di XGBoost. *****")

    df_features_base, df_targets, df_regimes, df_close, df_high, df_low, df_seasonal, df_macro_ratios = load_and_prepare_features()

    full_idx = df_features_base.index
    oos_start_date = full_idx.max() - pd.DateOffset(months=config["OOS_MONTHS"])
    validation_start_date = oos_start_date - pd.DateOffset(months=config["VALIDATION_MONTHS"])

    train_idx = full_idx[full_idx < validation_start_date]
    validation_idx = full_idx[(full_idx >= validation_start_date) & (full_idx < oos_start_date)]
    oos_idx = full_idx[full_idx >= oos_start_date]

    logging.info(f"Training Period: {train_idx.min()} to {train_idx.max()}")
    logging.info(f"Validation Period: {validation_idx.min()} to {validation_idx.max()}")
    logging.info(f"Out-of-Sample Period: {oos_idx.min()} to {oos_idx.max()}")

    logging.info("\n--- Fase 3: Ottimizzazione e Backtest su Set di Validazione ---")
    all_validation_results = {}
    if os.path.exists(CHECKPOINT_PATH):
        logging.info(f"Trovato file di checkpoint di validazione. Caricamento...: {CHECKPOINT_PATH}")
        try:
            with open(CHECKPOINT_PATH, 'rb') as f: all_validation_results = pickle.load(f)
        except Exception as e:
            logging.warning(f"Errore caricamento checkpoint: {e}. Ricalcolo.")
            all_validation_results = {}

    # ⚠️ v77: Aggiunto ciclo per i quantili
    for q_buy, q_short in config["QUANTILE_PAIRS_TO_TEST"]:
        # Aggiorna la config globale per questo ciclo
        config["QUANTILE_BUY"] = q_buy
        config["QUANTILE_SHORT"] = q_short
        quantile_label = f"q_{q_buy}_{q_short}"
        
        logging.info(f"\n{'='*25} INIZIO TEST QUANTILI: {quantile_label} {'='*25}\n")

        for window_size_val in config["WINDOW_LIST_TO_TEST"]:
            df_isee = load_isee_data(df_features_base.index, window_size=window_size_val, file_path=config["ISEE_FILE_PATH"])
            df_features_for_run = df_features_base.join(df_isee)
            df_features_for_run.ffill(inplace=True); df_features_for_run.fillna(0, inplace=True)

            for n_features in config["FEATURE_COUNTS_TO_TEST"]:
                n_features_label = 'all' if isinstance(n_features, str) else n_features
                
                # ⚠️ v77: Il checkpoint ora include i quantili
                if quantile_label in all_validation_results.get(window_size_val, {}).get(n_features_label, {}):
                    logging.info(f"RISULTATI VALIDAZIONE GIÀ PRESENTI PER: Q={quantile_label}, ISEE={window_size_val}, Feats={n_features_label}. SALTO.")
                    continue
                logging.info(f"\n{'='*20} INIZIO RUN VALIDAZIONE: Q={quantile_label}, ISEE={window_size_val}, Feats={n_features_label} {'='*20}\n")

                is_gpu_run = torch.cuda.is_available()
                n_jobs_validation = 1 if is_gpu_run else config["MAX_CPU_CORES"]
                if n_jobs_validation == 1 and is_gpu_run:
                    logging.info("GPU in uso: Esecuzione della convalida dei simboli in modo sequenziale.")
                else:
                     logging.info(f"CPU in uso: Esecuzione della convalida in parallelo su {n_jobs_validation} core.")

                results_for_this_config = {}

                parallel_results = Parallel(n_jobs=n_jobs_validation, backend='loky')(
                    delayed(run_backtest_for_symbol)(
                        symbol, df_features_for_run, df_targets, df_regimes, df_close, df_high, df_low, n_features, window_size_val,
                        train_idx, validation_idx, df_seasonal, df_macro_ratios
                    ) for symbol in config["TICKERS"] # ⚠️ v77: Esegue su tutti gli 8 asset
                )
                for result in parallel_results:
                    if result: results_for_this_config.update(result)

                # ⚠️ v77: Salva i risultati sotto la chiave dei quantili
                all_validation_results.setdefault(window_size_val, {}).setdefault(n_features_label, {})[quantile_label] = results_for_this_config
                
                try:
                    with open(CHECKPOINT_PATH + ".tmp", 'wb') as f: pickle.dump(all_validation_results, f)
                    os.replace(CHECKPOINT_PATH + ".tmp", CHECKPOINT_PATH)
                except Exception as e: logging.error(f"ERRORE salvataggio checkpoint di validazione: {e}")

    logging.info("\n--- Fase 4: Selezione delle Strategie Candidate ---")
    candidate_strategies = {}
    for symbol in config["TICKERS"]:
        best_sds_configs, best_tf_configs = [], []
        
        # ⚠️ v77: Itera su tutte le combinazioni, inclusi i quantili
        for q_buy, q_short in config["QUANTILE_PAIRS_TO_TEST"]:
            quantile_label = f"q_{q_buy}_{q_short}"
            for window in config["WINDOW_LIST_TO_TEST"]:
                for n_features in config["FEATURE_COUNTS_TO_TEST"]:
                    n_features_label = 'all' if isinstance(n_features, str) else n_features
                    
                    if (window in all_validation_results and 
                        n_features_label in all_validation_results[window] and 
                        quantile_label in all_validation_results[window][n_features_label] and
                        symbol in all_validation_results[window][n_features_label][quantile_label]):
                        
                        results = all_validation_results[window][n_features_label][quantile_label][symbol]
                        
                        # Salva la configurazione COMPLETA
                        config_dict = {
                            'window': window, 
                            'features': n_features_label, 
                            'quantile_buy': q_buy, 
                            'quantile_short': q_short
                        }
                        
                        if results['single_day'] and results['single_day'].get('calmar_ratio') is not None:
                            best_sds_configs.append({'calmar': results['single_day']['calmar_ratio'], 'config': config_dict})
                        if results['trend_following'] and results['trend_following'].get('calmar_ratio') is not None:
                            best_tf_configs.append({'calmar': results['trend_following']['calmar_ratio'], 'config': config_dict})

        best_sds_configs.sort(key=lambda x: x['calmar'], reverse=True)
        best_tf_configs.sort(key=lambda x: x['calmar'], reverse=True)
        candidate_strategies[symbol] = {
            'single_day': [item['config'] for item in best_sds_configs[:config["N_CANDIDATE_STRATEGIES"]]],
            'trend_following': [item['config'] for item in best_tf_configs[:config["N_CANDIDATE_STRATEGIES"]]]
        }
        logging.info(f"Top {config['N_CANDIDATE_STRATEGIES']} candidati 'Overnight' per {symbol}:")
        for cfg in candidate_strategies[symbol]['single_day']:
            logging.info(f"  - {cfg}")
        logging.info(f"Top {config['N_CANDIDATE_STRATEGIES']} candidati 'Trend-Following' per {symbol}:")
        for cfg in candidate_strategies[symbol]['trend_following']:
            logging.info(f"  - {cfg}")


    STRATEGY_MAP = {
        'single_day': {'fn': strategy_single_day_swing, 'name': 'Overnight'},
        'trend_following': {'fn': strategy_trend_following, 'name': 'Trend-Following'}
    }

    logging.info("\n--- Fase 5: Test Finale su Dati Out-of-Sample ---")
    all_oos_results = {}
    if os.path.exists(OOS_CHECKPOINT_PATH):
        logging.info(f"Trovato file di checkpoint OOS. Caricamento...: {OOS_CHECKPOINT_PATH}")
        try:
            with open(OOS_CHECKPOINT_PATH, 'rb') as f:
                all_oos_results = pickle.load(f)
        except Exception as e:
            logging.error(f"Errore caricamento checkpoint OOS: {e}. Riavvio da zero.")
            all_oos_results = {}

    for symbol in config["TICKERS"]:
        all_oos_results.setdefault(symbol, {'single_day': [], 'trend_following': []})
        logging.info(f"\n--- Inizio Test OOS per {symbol} ---")
        for strategy_type, candidates in candidate_strategies[symbol].items():
            existing_configs = [item['config'] for item in all_oos_results[symbol].get(strategy_type, [])]

            for i, cand_config in enumerate(candidates):
                if cand_config in existing_configs:
                    logging.info(f"RISULTATO OOS GIÀ PRESENTE PER: {symbol}, {strategy_type}, {cand_config}. SALTO.")
                    continue

                logging.info(f"Test OOS {strategy_type} per {symbol} - Candidato {i+1}/{len(candidates)}: {cand_config}")
                
                # ⚠️ v77: Imposta la config con i quantili corretti per questo test OOS
                window_size, n_features = cand_config['window'], cand_config['features']
                config["QUANTILE_BUY"] = cand_config['quantile_buy']
                config["QUANTILE_SHORT"] = cand_config['quantile_short']

                df_isee = load_isee_data(df_features_base.index, window_size=window_size, file_path=config["ISEE_FILE_PATH"])
                df_features_for_oos_run = df_features_base.join(df_isee)
                df_features_for_oos_run.ffill(inplace=True); df_features_for_oos_run.fillna(0, inplace=True)

                train_val_idx = train_idx.union(validation_idx)

                strategy_info = STRATEGY_MAP[strategy_type]
                isee_z_col = f'{config["ISEE_MOMENTUM_Z_COL_BASE"]}_{window_size}'
                isee_regime_col = f'{isee_z_col}_HighVol_Only'

                oos_result = run_backtest_evaluation(
                    strategy_fn=strategy_info['fn'],
                    strategy_name=strategy_info['name'],
                    symbol=symbol,
                    df_features_base=df_features_for_oos_run,
                    df_targets=df_targets,
                    df_regimes=df_regimes,
                    df_close=df_close,
                    df_high=df_high,
                    df_low=df_low,
                    X_main_idx=train_val_idx,
                    X_unseen_idx=oos_idx,
                    isee_z_col=isee_z_col,
                    isee_regime_col=isee_regime_col,
                    n_features=n_features,
                    df_seasonal=df_seasonal,
                    df_macro_ratios=df_macro_ratios
                )

                all_oos_results[symbol].setdefault(strategy_type, []).append({'config': cand_config, 'result': oos_result})

                try:
                    with open(OOS_CHECKPOINT_PATH + ".tmp", 'wb') as f:
                        pickle.dump(all_oos_results, f)
                    os.replace(OOS_CHECKPOINT_PATH + ".tmp", OOS_CHECKPOINT_PATH)
                    logging.info(f"Checkpoint OOS salvato per {symbol}, {strategy_type}, {cand_config}")
                except Exception as e:
                    logging.error(f"ERRORE salvataggio checkpoint OOS: {e}")

    # --- Generate CSV reports for top N candidates ---
    logging.info(f"\n--- Fase 6: Generazione CSV Trade Reports per Top {config['TOP_N_CANDIDATES_FOR_CSV']} Candidati ---")
    for symbol in config["TICKERS"]:
        for strategy_type, results_list in all_oos_results[symbol].items():
            # Sort by calmar ratio (best first)
            valid_results = [(item['config'], item['result']) for item in results_list if item.get('result') and item['result'].get('calmar_ratio') is not None]
            valid_results.sort(key=lambda x: x[1]['calmar_ratio'], reverse=True)

            # Save CSV for top N candidates
            strategy_name_clean = STRATEGY_MAP[strategy_type]['name']
            for rank, (cand_config, result) in enumerate(valid_results[:config['TOP_N_CANDIDATES_FOR_CSV']], start=1):
                save_trades_to_csv(
                    symbol=symbol,
                    strategy_name=strategy_name_clean,
                    rank=rank,
                    trades_df=result.get('trades_df'),
                    total_return=result.get('total_strategy_return', 0),
                    max_dd=result.get('max_drawdown', 0),
                    calmar_ratio=result.get('calmar_ratio', 0),
                    config_info=cand_config,
                    output_dir=output_dir
                )

    logging.info(f"\n{'='*30} REPORT FINALE OUT-OF-SAMPLE (OOS) {'='*30}\n")
    final_report_path = os.path.join(output_dir, "oos_final_report.txt")
    with open(final_report_path, 'w', encoding='utf-8') as f:
        for symbol, strategies in all_oos_results.items():
            f.write(f"\n{'='*20} Risultati OOS per: {symbol} {'='*20}\n")
            bh_returns = (1 + df_close.loc[oos_idx, symbol].pct_change()).cumprod()
            total_buy_and_hold_return = (bh_returns.iloc[-1] - 1) * 100 if not bh_returns.empty else 0
            f.write(f"Buy & Hold Return nel periodo OOS: {total_buy_and_hold_return:.2f}%\n")

            for strategy_type, results_list in strategies.items():
                f.write(f"\n--- Strategia: {strategy_type.replace('_', ' ').title()} ---\n")
                if not results_list:
                    f.write("Nessun risultato per questa strategia.\n")
                    continue

                flat_results = []
                for res_item in results_list:
                    if res_item and res_item.get('result'):
                        flat_results.append(res_item['result'])

                calmars = [res['calmar_ratio'] for res in flat_results if res and res.get('calmar_ratio') is not None]
                returns = [res['total_strategy_return'] for res in flat_results if res and res.get('total_strategy_return') is not None]

                if not calmars:
                    f.write("Nessun risultato valido per questa strategia.\n")
                    continue

                f.write(f"  - Numero di candidati testati: {len(calmars)}\n")
                f.write(f"  - Performance dei candidati (Calmar Ratio):\n")
                f.write(f"    - Media: {np.mean(calmars):.2f}\n")
                f.write(f"    - Mediana: {np.median(calmars):.2f}\n")
                f.write(f"    - Min: {np.min(calmars):.2f}\n")
                f.write(f"    - Max: {np.max(calmars):.2f}\n")
                f.write(f"    - Dev. Standard: {np.std(calmars):.2f}\n")
                f.write(f"  - Performance dei candidati (Ritorno %):\n")
                f.write(f"    - Media: {np.mean(returns):.2f}%\n")
                f.write(f"    - Mediana: {np.median(returns):.2f}%\n")
    logging.info(f"Report finale OOS salvato in: {final_report_path}")

if __name__ == "__main__":
    main()