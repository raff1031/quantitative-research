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
WINDOW_SIZE = 5        # settimane di lookback LSTM
EPOCHS = 60
BATCH_SIZE = 16
LR = 5e-4
COSTO_TRANSAZIONE = 0.001
SIGNAL_THRESHOLD = 0.002
CONFIDENCE_SCALE = True

# Walk-forward params
MIN_TRAIN_WEEKS = 10    # minimo settimane per il primo training
RETRAIN_EVERY = 3        # retrain ogni 13 settimane (~trimestre)
VAL_RATIO = 0.15          # 15% del train per validation

# --- FORZA CUDA ---
assert torch.cuda.is_available(), "CUDA non disponibile!"
DEVICE = torch.device("cuda")
print(f"Device: {DEVICE} - {torch.cuda.get_device_name(0)}")

# ============================================================
# FEATURE ENGINEERING (identico a v2)
# ============================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_features(df_raw):
    df = pd.DataFrame(index=df_raw.index)
    df['log_ret'] = np.log(df_raw['Close'] / df_raw['Close'].shift(1))
    df['feat_skew_12'] = df['log_ret'].rolling(12).skew().shift(1)
    df['feat_vol_force'] = df_raw['Volume'].pct_change().rolling(4).mean().shift(1)
    df['feat_volat_12'] = df['log_ret'].rolling(12).std().shift(1)
    df['feat_mom_4'] = df['log_ret'].rolling(4).sum().shift(1)
    df['feat_mom_12'] = df['log_ret'].rolling(12).sum().shift(1)
    df['feat_rsi_14'] = compute_rsi(df_raw['Close'], 14).shift(1)
    sma20 = df_raw['Close'].rolling(20).mean()
    df['feat_ma_ratio'] = (df_raw['Close'] / sma20).shift(1)
    vol_mean = df_raw['Volume'].rolling(20).mean()
    vol_std = df_raw['Volume'].rolling(20).std()
    df['feat_vol_zscore'] = ((df_raw['Volume'] - vol_mean) / vol_std).shift(1)
    df['feat_kurt_12'] = df['log_ret'].rolling(12).kurt().shift(1)
    vol_short = df['log_ret'].rolling(4).std()
    vol_long = df['log_ret'].rolling(12).std()
    df['feat_vol_ratio'] = (vol_short / vol_long).shift(1)
    df['target'] = df['log_ret'].shift(-1)
    return df


FEATURES = [
    'feat_skew_12', 'feat_vol_force', 'feat_volat_12',
    'feat_mom_4', 'feat_mom_12', 'feat_rsi_14',
    'feat_ma_ratio', 'feat_vol_zscore', 'feat_kurt_12', 'feat_vol_ratio'
]

# ============================================================
# MODELLO LSTM (identico a v2)
# ============================================================
class LSTMStrategy(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
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


def create_sequences(scaled_features, target_series, window):
    X, y = [], []
    for i in range(len(scaled_features) - window):
        X.append(scaled_features[i : i + window])
        y.append(target_series.iloc[i + window])
    return np.array(X), np.array(y)


# ============================================================
# TRAIN ONE FOLD
# ============================================================
def train_fold(X_train, y_train, fold_num, total_folds):
    """Allena il modello su un fold, restituisce modello + val_std."""
    val_split = int(len(X_train) * (1 - VAL_RATIO))
    
    X_tr_t = torch.FloatTensor(X_train[:val_split]).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_train[:val_split]).to(DEVICE)
    X_val_t = torch.FloatTensor(X_train[val_split:]).to(DEVICE)
    y_val_t = torch.FloatTensor(y_train[val_split:]).to(DEVICE)
    
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LSTMStrategy(input_size=len(FEATURES)).to(DEVICE)
    criterion = nn.HuberLoss(delta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_ds)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Calcola val_std per position sizing (zero lookahead)
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_t).cpu().numpy()
        val_std = np.std(val_predictions)
    
    print(f"    Fold {fold_num}/{total_folds} | "
          f"Train: {len(X_tr_t)} seq | Val: {len(X_val_t)} seq | "
          f"Best val_loss: {best_val_loss:.6f} | val_std: {val_std:.6f}")
    
    return model, val_std


# ============================================================
# WALK-FORWARD BACKTEST PER UN SINGOLO TICKER
# ============================================================
def walkforward_backtest(ticker):
    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD LSTM -- {ticker}")
    print(f"{'='*65}")
    
    # --- Download ---
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df_weekly = data.resample('W-MON').apply({'Close': 'last', 'Volume': 'sum'}).dropna()
    df = build_features(df_weekly)
    df.dropna(inplace=True)
    
    if len(df) < MIN_TRAIN_WEEKS + RETRAIN_EVERY + WINDOW_SIZE:
        print(f"  ⚠ Dati insufficienti per {ticker}, skip.")
        return None
    
    print(f"  Dataset totale: {len(df)} settimane "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    
    # --- Definisci i fold ---
    # Primo fold: allena su [0 : MIN_TRAIN_WEEKS], testa su [MIN_TRAIN: MIN_TRAIN+RETRAIN]
    # Secondo fold: allena su [0 : MIN_TRAIN+RETRAIN], testa su [MIN_TRAIN+RETRAIN : +RETRAIN]
    # ...expanding window
    
    folds = []
    train_end = MIN_TRAIN_WEEKS
    while train_end + WINDOW_SIZE < len(df):
        test_end = min(train_end + RETRAIN_EVERY, len(df))
        folds.append((train_end, test_end))
        train_end = test_end
    
    if not folds:
        print(f"  ⚠ Nessun fold possibile per {ticker}, skip.")
        return None
    
    total_test_weeks = sum(te - ts for ts, te in folds)
    print(f"  Folds: {len(folds)} | Test weeks totali: {total_test_weeks}")
    print(f"  Primo test:  {df.index[folds[0][0]].date()}")
    print(f"  Ultimo test: {df.index[min(folds[-1][1]-1, len(df)-1)].date()}")
    
    # --- Walk-forward loop ---
    all_predictions = []
    all_real_returns = []
    all_positions = []
    all_dates = []
    
    for fold_idx, (train_end_idx, test_end_idx) in enumerate(folds):
        # Expanding window: sempre dal primo dato
        train_data = df.iloc[:train_end_idx]
        test_data = df.iloc[train_end_idx:test_end_idx]
        
        if len(test_data) == 0:
            continue
        
        # Scaling (fit solo su train di questo fold)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[FEATURES])
        test_scaled = scaler.transform(test_data[FEATURES])
        
        # Sequenze
        X_train, y_train = create_sequences(train_scaled, train_data['target'], WINDOW_SIZE)
        
        # Per il test: servono WINDOW_SIZE settimane di contesto PRIMA del test
        # Concatena le ultime WINDOW_SIZE settimane del train con il test
        bridge_scaled = np.vstack([train_scaled[-WINDOW_SIZE:], test_scaled])
        bridge_target = pd.concat([train_data['target'].iloc[-WINDOW_SIZE:], test_data['target']])
        X_test, y_test = create_sequences(bridge_scaled, bridge_target, WINDOW_SIZE)
        
        if len(X_train) < 50 or len(X_test) == 0:
            continue
        
        # Train
        model, val_std = train_fold(X_train, y_train, fold_idx + 1, len(folds))
        
        # Predict
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(DEVICE)
            preds = model(X_test_t).cpu().numpy()
        
        # Position sizing con val_std del fold corrente (zero lookahead)
        if CONFIDENCE_SCALE:
            raw_pos = preds / max(val_std * 2, 1e-6)
            raw_pos = np.clip(raw_pos, -1.0, 1.0)
            raw_pos[np.abs(preds) < SIGNAL_THRESHOLD] = 0.0
        else:
            conditions = [preds > SIGNAL_THRESHOLD, preds < -SIGNAL_THRESHOLD]
            raw_pos = np.select(conditions, [1.0, -1.0], default=0.0)
        
        # Salva risultati
        test_dates = test_data.index[:len(preds)]
        all_dates.extend(test_dates)
        all_predictions.extend(preds)
        all_real_returns.extend(y_test[:len(test_dates)])
        all_positions.extend(raw_pos[:len(test_dates)])
        
        # Cleanup GPU
        del model, X_test_t
        torch.cuda.empty_cache()
    
    if not all_dates:
        print(f"  ⚠ Nessuna predizione generata per {ticker}.")
        return None
    
    # --- Assembla risultati ---
    results = pd.DataFrame({
        'real_ret': all_real_returns,
        'pred_ret': all_predictions,
        'position': all_positions,
    }, index=all_dates)
    
    # Costi di transazione
    results['trade'] = results['position'].diff().abs().fillna(0)
    results['costs'] = results['trade'] * COSTO_TRANSAZIONE
    
    # Rendimenti strategia
    results['strat_ret_gross'] = results['position'].shift(1).fillna(0) * results['real_ret']
    results['strat_ret_net'] = results['strat_ret_gross'] - results['costs']
    
    # Equity curves
    results['equity_bh'] = (1 + results['real_ret']).cumprod()
    results['equity_wf'] = (1 + results['strat_ret_net']).cumprod()
    
    # --- Rule-Based (stessa finestra temporale) ---
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
    
    rb_test = rb_df.loc[results.index[0]:results.index[-1]].copy()
    if len(rb_test) > 0:
        rb_test['equity_rb'] = (1 + rb_test['strat_ret_net'].fillna(0)).cumprod()
    
    # ============================================================
    # METRICHE
    # ============================================================
    def calc_metrics(returns, name):
        total_ret = (1 + returns).prod() - 1
        n_periods = len(returns)
        ann_factor = 52 / n_periods if n_periods > 0 else 1
        ann_ret = (1 + total_ret) ** ann_factor - 1 if n_periods > 0 else 0
        ann_vol = returns.std() * np.sqrt(52)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        active = returns[returns != 0]
        win_rate = (active > 0).mean() if len(active) > 0 else 0
        
        print(f"\n  {'-'*40}")
        print(f"  {name}")
        print(f"  {'-'*40}")
        print(f"  Return totale:    {total_ret:>8.2%}")
        print(f"  Return annualiz:  {ann_ret:>8.2%}")
        print(f"  Volatilità ann:   {ann_vol:>8.2%}")
        print(f"  Sharpe Ratio:     {sharpe:>8.2f}")
        print(f"  Max Drawdown:     {max_dd:>8.2%}")
        print(f"  Win Rate:         {win_rate:>8.2%}")
        
        return {'name': name, 'total_ret': total_ret, 'ann_ret': ann_ret,
                'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': win_rate}
    
    # Posizioni attive
    n_long = (results['position'] > 0.1).sum()
    n_short = (results['position'] < -0.1).sum()
    n_flat = ((results['position'] >= -0.1) & (results['position'] <= 0.1)).sum()
    print(f"\n  Posizioni WF: LONG {n_long} | SHORT {n_short} | FLAT {n_flat}")
    
    m_bh = calc_metrics(results['real_ret'], "Buy & Hold")
    m_wf = calc_metrics(results['strat_ret_net'], "WF-LSTM Strategy")
    m_rb = None
    if len(rb_test) > 0:
        m_rb = calc_metrics(rb_test['strat_ret_net'].fillna(0), "Rule-Based L/S")
    
    # --- Scoreboard ---
    print(f"\n  {'='*40}")
    print(f"  SCOREBOARD -- {ticker}")
    print(f"  {'='*40}")
    strategies = [m_bh, m_wf]
    if m_rb:
        strategies.append(m_rb)
    best = max(strategies, key=lambda x: x['sharpe'])
    for s in strategies:
        flag = " [BEST]" if s == best else ""
        print(f"  {s['name']:.<27s} Sharpe: {s['sharpe']:.2f} | Ret: {s['total_ret']:.2%}{flag}")
    
    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    
    ax1 = axes[0]
    ax1.plot(results['equity_bh'], label='Buy & Hold', color='grey', alpha=0.5, lw=1.5)
    ax1.plot(results['equity_wf'], label='WF-LSTM (Long/Short)', color='dodgerblue', lw=2.5)
    if len(rb_test) > 0 and 'equity_rb' in rb_test.columns:
        ax1.plot(rb_test['equity_rb'], label='Rule-Based L/S',
                 color='darkorange', lw=1.5, ls='--')
    
    # Retrain points
    for ts, _ in folds[1:]:
        ax1.axvline(x=df.index[ts], color='green', alpha=0.2, ls=':', lw=0.8)
    
    pos_s = results['position']
    ax1.fill_between(results.index,
                     results['equity_wf'].min() * 0.9,
                     results['equity_wf'].max() * 1.1,
                     where=(pos_s < -0.1), color='red', alpha=0.08,
                     label='WF-LSTM SHORT zone')
    
    ax1.set_title(f"Walk-Forward LSTM vs Rule-Based vs Buy&Hold -- {ticker}",
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    ax1.set_ylabel('Equity (log)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.fill_between(results.index, results['position'], 0,
                     where=(results['position'] > 0), color='green', alpha=0.4, label='LONG')
    ax2.fill_between(results.index, results['position'], 0,
                     where=(results['position'] < 0), color='red', alpha=0.4, label='SHORT')
    ax2.axhline(y=0, color='black', lw=0.5)
    ax2.set_ylabel('Posizione WF-LSTM')
    ax2.set_ylim(-1.2, 1.2)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    cumulative = (1 + results['strat_ret_net']).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    ax3 = axes[2]
    ax3.fill_between(results.index, drawdown, alpha=0.5, color='crimson',
                     label='Drawdown WF-LSTM')
    ax3.set_ylabel('Drawdown')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"wf_lstm_{ticker.replace('=','').replace('-','')}.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'ticker': ticker,
        'wf_ret': m_wf['total_ret'], 'wf_sharpe': m_wf['sharpe'],
        'wf_maxdd': m_wf['max_dd'],
        'bh_ret': m_bh['total_ret'], 'bh_sharpe': m_bh['sharpe'],
        'rb_ret': m_rb['total_ret'] if m_rb else None,
        'rb_sharpe': m_rb['sharpe'] if m_rb else None,
        'n_folds': len(folds),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_results = []
    
    for ticker in TICKERS:
        result = walkforward_backtest(ticker)
        if result:
            all_results.append(result)
    
    # --- SUMMARY TABLE ---
    print("\n\n" + "=" * 75)
    print("  RIEPILOGO FINALE -- Walk-Forward LSTM vs Rule-Based vs Buy & Hold")
    print("=" * 75)
    print(f"  {'Ticker':<10} {'WF Ret':>10} {'WF Sh':>8} {'RB Ret':>10} "
          f"{'RB Sh':>8} {'B&H Ret':>10} {'B&H Sh':>8}  {'Folds':>5}  Best")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10} "
          f"{'-'*8} {'-'*10} {'-'*8}  {'-'*5}  {'-'*6}")
    
    wf_wins = 0
    for r in all_results:
        rb_ret_s = f"{r['rb_ret']:.2%}" if r['rb_ret'] is not None else "N/A"
        rb_sh_s = f"{r['rb_sharpe']:.2f}" if r['rb_sharpe'] is not None else "N/A"
        
        sharpes = {'WF-LSTM': r['wf_sharpe'], 'B&H': r['bh_sharpe']}
        if r['rb_sharpe'] is not None:
            sharpes['RB'] = r['rb_sharpe']
        best = max(sharpes, key=sharpes.get)
        if best == 'WF-LSTM':
            wf_wins += 1
        
        print(f"  {r['ticker']:<10} {r['wf_ret']:>10.2%} {r['wf_sharpe']:>8.2f} "
              f"{rb_ret_s:>10} {rb_sh_s:>8} {r['bh_ret']:>10.2%} {r['bh_sharpe']:>8.2f}  "
              f"{r['n_folds']:>5}  [BEST: {best}]")
    
    print(f"\n  WF-LSTM vince su {wf_wins}/{len(all_results)} ticker (by Sharpe Ratio)")
    print("=" * 75)
