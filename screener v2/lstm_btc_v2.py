import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAZIONE
# ============================================================
TICKERS = ["AAPL", "GOOG", "NFLX", "AMZN", "GC=F", "MRNA", "BTC-USD"]
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
WINDOW_SIZE = 16          # settimane di lookback
EPOCHS = 60
BATCH_SIZE = 32
LR = 5e-4
TRAIN_RATIO = 0.80
COSTO_TRANSAZIONE = 0.001
SIGNAL_THRESHOLD = 0.002  # soglia per entrare Long/Short
CONFIDENCE_SCALE = True   # position sizing by confidence

# --- FORZA CUDA ---
assert torch.cuda.is_available(), "CUDA non disponibile! Installa PyTorch con CUDA."
DEVICE = torch.device("cuda")
print(f"Device: {DEVICE} - {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# FEATURE ENGINEERING — 10 feature, tutte calcolate da dati passati
# ============================================================
def compute_rsi(series, period=14):
    """RSI classico, senza lookahead."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_features(df_raw):
    """
    Riceve un DataFrame con colonne Close, Volume già resampleato settimanale.
    Ritorna un DataFrame con feature + target, senza lookahead.
    """
    df = pd.DataFrame(index=df_raw.index)
    
    # Rendimento settimanale
    df['log_ret'] = np.log(df_raw['Close'] / df_raw['Close'].shift(1))
    
    # 1. Skewness 12 settimane (come la strategia rule-based)
    df['feat_skew_12'] = df['log_ret'].rolling(12).skew().shift(1)
    
    # 2. Volume force (come la strategia rule-based)
    df['feat_vol_force'] = df_raw['Volume'].pct_change().rolling(4).mean().shift(1)
    
    # 3. Volatilità realizzata 12 settimane
    df['feat_volat_12'] = df['log_ret'].rolling(12).std().shift(1)
    
    # 4. Momentum 4 settimane
    df['feat_mom_4'] = df['log_ret'].rolling(4).sum().shift(1)
    
    # 5. Momentum 12 settimane
    df['feat_mom_12'] = df['log_ret'].rolling(12).sum().shift(1)
    
    # 6. RSI 14 settimane  
    df['feat_rsi_14'] = compute_rsi(df_raw['Close'], 14).shift(1)
    
    # 7. MA ratio: Close / SMA20
    sma20 = df_raw['Close'].rolling(20).mean()
    df['feat_ma_ratio'] = (df_raw['Close'] / sma20).shift(1)
    
    # 8. Volume z-score
    vol_mean = df_raw['Volume'].rolling(20).mean()
    vol_std = df_raw['Volume'].rolling(20).std()
    df['feat_vol_zscore'] = ((df_raw['Volume'] - vol_mean) / vol_std).shift(1)
    
    # 9. Kurtosis 12 settimane (fat tails detector)
    df['feat_kurt_12'] = df['log_ret'].rolling(12).kurt().shift(1)
    
    # 10. Volatility ratio (short/long vol = regime change detector)
    vol_short = df['log_ret'].rolling(4).std()
    vol_long = df['log_ret'].rolling(12).std()
    df['feat_vol_ratio'] = (vol_short / vol_long).shift(1)
    
    # TARGET: rendimento della settimana SUCCESSIVA
    df['target'] = df['log_ret'].shift(-1)
    
    return df


FEATURES = [
    'feat_skew_12', 'feat_vol_force', 'feat_volat_12',
    'feat_mom_4', 'feat_mom_12', 'feat_rsi_14',
    'feat_ma_ratio', 'feat_vol_zscore', 'feat_kurt_12', 'feat_vol_ratio'
]

# ============================================================
# MODELLO LSTM POTENZIATO
# ============================================================
class LSTMStrategy(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.norm(last_hidden)
        out = self.dropout(out)
        out = self.gelu(self.fc1(out))
        out = self.gelu(self.fc2(out))
        out = self.fc3(out)
        return out.squeeze(-1)


# ============================================================
# FUNZIONE SEQUENZE
# ============================================================
def create_sequences(scaled_features, target_series, window):
    X, y = [], []
    for i in range(len(scaled_features) - window):
        X.append(scaled_features[i : i + window])
        y.append(target_series.iloc[i + window])
    return np.array(X), np.array(y)


# ============================================================
# TRAINING LOOP PER UN SINGOLO TICKER
# ============================================================
def train_and_backtest(ticker):
    print(f"\n{'='*60}")
    print(f"  LSTM v2 - {ticker}")
    print(f"{'='*60}")
    
    # --- Download ---
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # --- Resampling settimanale (come la strategia rule-based) ---
    df_weekly = data.resample('W-MON').apply({'Close': 'last', 'Volume': 'sum'}).dropna()
    
    # --- Feature engineering ---
    df = build_features(df_weekly)
    df.dropna(inplace=True)
    
    if len(df) < 100:
        print(f"  ⚠ Dati insufficienti per {ticker}, skip.")
        return None
    
    # --- Split temporale ---
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"  Dataset: {len(df)} settimane")
    print(f"  Train: {len(train_df)} ({train_df.index[0].date()} - {train_df.index[-1].date()})")
    print(f"  Test:  {len(test_df)}  ({test_df.index[0].date()} - {test_df.index[-1].date()})")
    
    # --- Scaling rigoroso (fit solo su train) ---
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURES])
    test_scaled = scaler.transform(test_df[FEATURES])
    
    # --- Sequenze ---
    X_train, y_train = create_sequences(train_scaled, train_df['target'], WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, test_df['target'], WINDOW_SIZE)
    
    if len(X_train) < 50 or len(X_test) < 10:
        print(f"  ⚠ Sequenze insufficienti per {ticker}, skip.")
        return None
    
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # --- Tensori su CUDA ---
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)
    
    # --- Validation split (ultimo 15% del train) ---
    val_split = int(len(X_train_t) * 0.85)
    X_val = X_train_t[val_split:]
    y_val = y_train_t[val_split:]
    
    train_subset = TensorDataset(X_train_t[:val_split], y_train_t[:val_split])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Modello ---
    model = LSTMStrategy(input_size=len(FEATURES)).to(DEVICE)
    criterion = nn.HuberLoss(delta=0.01)  # Huber: meno sensibile agli outlier del MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri: {n_params:,}")
    
    # --- Training ---
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_subset)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.1e}")
        
        if patience_counter >= PATIENCE:
            print(f"  Early stopping @ epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    
    # ============================================================
    # BACKTEST — Long / Short / Flat
    # ============================================================
    model.eval()
    
    # CORREZIONE LOOKAHEAD BIAS: Calcola std dalle predizioni del validation set
    # (dati passati rispetto al test set) invece che dal test set stesso
    with torch.no_grad():
        val_predictions_full = model(X_val).cpu().numpy()
        val_std = np.std(val_predictions_full)
        print(f"  Validation predictions std: {val_std:.6f}")
        
        predictions = model(X_test_t).cpu().numpy()
    
    test_results = pd.DataFrame(index=test_df.index[WINDOW_SIZE:])
    test_results['real_ret'] = y_test
    test_results['pred_ret'] = predictions
    
    # --- SEGNALE LONG/SHORT/FLAT ---
    if CONFIDENCE_SCALE:
        # Position sizing basato sulla confidenza della previsione
        # Usa val_std (dati passati) NON np.std(predictions) (include futuro)
        raw_pos = predictions / max(val_std * 2, 1e-6)
        raw_pos = np.clip(raw_pos, -1.0, 1.0)
        # Dead zone: se la previsione è troppo debole → flat
        raw_pos[np.abs(predictions) < SIGNAL_THRESHOLD] = 0.0
        test_results['position'] = raw_pos
    else:
        conditions = [
            predictions > SIGNAL_THRESHOLD,
            predictions < -SIGNAL_THRESHOLD
        ]
        choices = [1.0, -1.0]
        test_results['position'] = np.select(conditions, choices, default=0.0)
    
    # --- Costi ---
    test_results['trade'] = test_results['position'].diff().abs().fillna(0)
    test_results['costs'] = test_results['trade'] * COSTO_TRANSAZIONE
    
    # --- Rendimenti ---
    test_results['strat_ret_gross'] = test_results['position'].shift(1).fillna(0) * test_results['real_ret']
    test_results['strat_ret_net'] = test_results['strat_ret_gross'] - test_results['costs']
    
    # --- Equity ---
    test_results['equity_bh'] = (1 + test_results['real_ret']).cumprod()
    test_results['equity_lstm'] = (1 + test_results['strat_ret_net']).cumprod()
    
    # ============================================================
    # RULE-BASED STRATEGY (per confronto diretto)
    # ============================================================
    rb_df = df_weekly.copy()
    rb_df['log_ret'] = np.log(rb_df['Close'] / rb_df['Close'].shift(1))
    rb_df['skew'] = rb_df['log_ret'].rolling(window=12).skew().shift(1)
    rb_df['vol_force'] = rb_df['Volume'].pct_change().rolling(window=4).mean().shift(1)
    
    positions_rb = []
    current_pos = 0
    for i in range(len(rb_df)):
        skew = rb_df['skew'].iloc[i]
        vol = rb_df['vol_force'].iloc[i]
        if pd.isna(skew) or pd.isna(vol):
            positions_rb.append(0)
            continue
        is_long = (skew > 0.2 and vol > 0.05) or (skew > 0.6) or (skew < -0.8 and vol < -0.1)
        is_short = (-0.5 < skew < -0.1 and vol < 0.05)
        if is_long:
            current_pos = 1
        elif is_short:
            current_pos = -1
        positions_rb.append(current_pos)
    
    rb_df['position'] = positions_rb
    rb_df['strat_ret'] = rb_df['position'].shift(1) * rb_df['log_ret']
    trades_rb = rb_df['position'].diff().abs()
    rb_df['strat_ret_net'] = rb_df['strat_ret'] - (trades_rb * COSTO_TRANSAZIONE)
    
    # Allinea rule-based al periodo di test
    test_start = test_results.index[0]
    test_end = test_results.index[-1]
    rb_test = rb_df.loc[test_start:test_end].copy()
    if len(rb_test) > 0:
        rb_test['equity_rb'] = (1 + rb_test['strat_ret_net'].fillna(0)).cumprod()
    
    # ============================================================
    # METRICHE
    # ============================================================
    def calc_metrics(returns, name):
        total_ret = (1 + returns).prod() - 1
        n_periods = len(returns)
        ann_factor = 52 / n_periods  # settimanale
        ann_ret = (1 + total_ret) ** (ann_factor) - 1 if n_periods > 0 else 0
        ann_vol = returns.std() * np.sqrt(52)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        active = returns[returns != 0]
        win_rate = (active > 0).mean() if len(active) > 0 else 0
        n_trades = int(test_results['trade'].sum()) if 'LSTM' in name else 0
        
        print(f"\n  {'-'*38}")
        print(f"  {name}")
        print(f"  {'-'*38}")
        print(f"  Return totale:    {total_ret:>8.2%}")
        print(f"  Return annualiz:  {ann_ret:>8.2%}")
        print(f"  Volatilità ann:   {ann_vol:>8.2%}")
        print(f"  Sharpe Ratio:     {sharpe:>8.2f}")
        print(f"  Max Drawdown:     {max_dd:>8.2%}")
        print(f"  Win Rate:         {win_rate:>8.2%}")
        
        return {
            'name': name, 'total_ret': total_ret, 'ann_ret': ann_ret,
            'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': win_rate
        }
    
    m_bh = calc_metrics(test_results['real_ret'], "Buy & Hold")
    m_lstm = calc_metrics(test_results['strat_ret_net'], "LSTM v2 Strategy")
    
    m_rb = None
    if len(rb_test) > 0:
        m_rb = calc_metrics(rb_test['strat_ret_net'].fillna(0), "Rule-Based L/S")
    
    # ============================================================
    # SCOREBOARD
    # ============================================================
    print(f"\n  {'='*38}")
    print(f"  SCOREBOARD - {ticker}")
    print(f"  {'='*38}")
    strategies = [m_bh, m_lstm]
    if m_rb:
        strategies.append(m_rb)
    
    best = max(strategies, key=lambda x: x['sharpe'])
    for s in strategies:
        flag = " [BEST]" if s == best else ""
        print(f"  {s['name']:.<25s} Sharpe: {s['sharpe']:.2f} | Ret: {s['total_ret']:.2%}{flag}")
    
    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # --- Equity ---
    ax1 = axes[0]
    ax1.plot(test_results['equity_bh'], label='Buy & Hold', color='grey', alpha=0.5, linewidth=1.5)
    ax1.plot(test_results['equity_lstm'], label='LSTM v2 (Long/Short)', color='dodgerblue', linewidth=2.5)
    if len(rb_test) > 0 and 'equity_rb' in rb_test.columns:
        ax1.plot(rb_test['equity_rb'], label='Rule-Based L/S', color='darkorange', linewidth=1.5, linestyle='--')
    
    # Evidenzia zone SHORT dell'LSTM
    pos_series = test_results['position']
    ax1.fill_between(test_results.index, test_results['equity_lstm'].min() * 0.9,
                     test_results['equity_lstm'].max() * 1.1,
                     where=(pos_series < -0.1), color='red', alpha=0.08, label='LSTM SHORT zone')
    
    ax1.set_title(f"LSTM v2 vs Rule-Based vs Buy&Hold — {ticker}", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    ax1.set_ylabel('Equity (log)')
    ax1.grid(True, alpha=0.3)
    
    # --- Posizioni LSTM ---
    ax2 = axes[1]
    ax2.fill_between(test_results.index, test_results['position'], 0,
                     where=(test_results['position'] > 0), color='green', alpha=0.4, label='LONG')
    ax2.fill_between(test_results.index, test_results['position'], 0,
                     where=(test_results['position'] < 0), color='red', alpha=0.4, label='SHORT')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Posizione LSTM')
    ax2.set_ylim(-1.2, 1.2)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Drawdown ---
    cumulative = (1 + test_results['strat_ret_net']).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    ax3 = axes[2]
    ax3.fill_between(test_results.index, drawdown, alpha=0.5, color='crimson', label='Drawdown LSTM v2')
    ax3.set_ylabel('Drawdown')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"lstm_v2_{ticker.replace('=','').replace('-','')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'ticker': ticker,
        'lstm_ret': m_lstm['total_ret'],
        'lstm_sharpe': m_lstm['sharpe'],
        'bh_ret': m_bh['total_ret'],
        'bh_sharpe': m_bh['sharpe'],
        'rb_ret': m_rb['total_ret'] if m_rb else None,
        'rb_sharpe': m_rb['sharpe'] if m_rb else None,
    }


# ============================================================
# MAIN — LOOP SU TUTTI I TICKER
# ============================================================
if __name__ == "__main__":
    all_results = []
    
    for ticker in TICKERS:
        result = train_and_backtest(ticker)
        if result:
            all_results.append(result)
    
    # --- SUMMARY TABLE ---
    print("\n\n" + "="*70)
    print("  RIEPILOGO FINALE - LSTM v2 vs Rule-Based vs Buy & Hold")
    print("="*70)
    print(f"  {'Ticker':<10} {'LSTM Ret':>10} {'LSTM Sh':>8} {'RB Ret':>10} {'RB Sh':>8} {'B&H Ret':>10} {'B&H Sh':>8}  Best")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}  {'-'*6}")
    
    lstm_wins = 0
    for r in all_results:
        rb_ret_str = f"{r['rb_ret']:.2%}" if r['rb_ret'] is not None else "N/A"
        rb_sh_str = f"{r['rb_sharpe']:.2f}" if r['rb_sharpe'] is not None else "N/A"
        
        sharpes = {'LSTM': r['lstm_sharpe'], 'B&H': r['bh_sharpe']}
        if r['rb_sharpe'] is not None:
            sharpes['RB'] = r['rb_sharpe']
        best = max(sharpes, key=sharpes.get)
        if best == 'LSTM':
            lstm_wins += 1
        
        print(f"  {r['ticker']:<10} {r['lstm_ret']:>10.2%} {r['lstm_sharpe']:>8.2f} {rb_ret_str:>10} {rb_sh_str:>8} {r['bh_ret']:>10.2%} {r['bh_sharpe']:>8.2f}  [BEST: {best}]")
    
    print(f"\n  LSTM v2 vince su {lstm_wins}/{len(all_results)} ticker (by Sharpe Ratio)")
    print("="*70)
