import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIGURAZIONE
# ============================================================
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
WINDOW_SIZE = 20
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PARAMETRI STRATEGIA MIGLIORATI
SIGNAL_MODE = "percentile"  # "fixed", "percentile", o "long_short"
SIGNAL_THRESHOLD = 0.0005     # soglia fissa (usata solo se SIGNAL_MODE = "fixed")
LONG_PERCENTILE = 60          # percentile sopra il quale andare long
SHORT_PERCENTILE = 40         # percentile sotto il quale andare short

print(f"Device: {DEVICE}")

# ============================================================
# 1. DOWNLOAD E PREPARAZIONE DATI
# ============================================================
data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)

# Pulizia MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Feature Engineering — calcolate SOLO da dati passati (nessun lookahead)
df = pd.DataFrame(index=data.index)
df['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
df['feat_skew'] = df['log_ret'].rolling(60).skew()          # skewness 60gg (usa solo dati passati)
df['feat_vol_force'] = data['Volume'].pct_change().rolling(5).mean()  # media variaz. volume 5gg
df['feat_volat'] = df['log_ret'].rolling(20).std()           # volatilità realizzata 20gg

# TARGET: rendimento del giorno SUCCESSIVO
# shift(-1) = il rendimento che avverrà domani. Lo conosciamo solo a fine giornata domani.
df['target'] = df['log_ret'].shift(-1)

# Dropna: eliminiamo le righe dove rolling non ha abbastanza storia
df.dropna(inplace=True)

features = ['feat_skew', 'feat_vol_force', 'feat_volat']

print(f"Dataset: {len(df)} righe, da {df.index[0].date()} a {df.index[-1].date()}")

# ============================================================
# 2. SPLIT TEMPORALE CRONOLOGICO (NO LOOKAHEAD)
# ============================================================
split_idx = int(len(df) * TRAIN_RATIO)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Train: {len(train_df)} righe ({train_df.index[0].date()} -> {train_df.index[-1].date()})")
print(f"Test:  {len(test_df)} righe  ({test_df.index[0].date()} -> {test_df.index[-1].date()})")

# ============================================================
# 3. SCALING RIGOROSO
# ============================================================
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

# ============================================================
# 4. CREAZIONE SEQUENZE (FINESTRA MOBILE)
# ============================================================
def create_sequences(scaled_features, target_series, window):
    X, y = [], []
    for i in range(len(scaled_features) - window):
        X.append(scaled_features[i : i + window])
        y.append(target_series.iloc[i + window])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, train_df['target'], WINDOW_SIZE)
X_test, y_test = create_sequences(test_scaled, test_df['target'], WINDOW_SIZE)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# 5. CONVERSIONE A TENSORI PYTORCH
# ============================================================
X_train_t = torch.FloatTensor(X_train).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).to(DEVICE)
X_test_t = torch.FloatTensor(X_test).to(DEVICE)
y_test_t = torch.FloatTensor(y_test).to(DEVICE)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 6. MODELLO LSTM IN PYTORCH
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


model = LSTMModel(input_size=len(features)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nParametri modello: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 7. TRAINING
# ============================================================
val_split = int(len(X_train_t) * 0.9)
X_val = X_train_t[val_split:]
y_val = y_train_t[val_split:]

train_subset = TensorDataset(X_train_t[:val_split], y_train_t[:val_split])
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)

best_val_loss = float('inf')
patience_counter = 0
PATIENCE = 5

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(train_subset)

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val).item()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0 or patience_counter >= PATIENCE:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping a epoch {epoch+1}")
        break

model.load_state_dict(best_state)

# ============================================================
# 8. BACKTEST CON STRATEGIA MIGLIORATA
# ============================================================
model.eval()
with torch.no_grad():
    predictions = model(X_test_t).cpu().numpy()

test_results = pd.DataFrame(index=test_df.index[WINDOW_SIZE:])
test_results['real_ret'] = y_test
test_results['pred_ret'] = predictions

print(f"\n{'='*60}")
print("ANALISI PREDIZIONI")
print(f"{'='*60}")
print(f"Predizioni - Media: {predictions.mean():.6f}, Std: {predictions.std():.6f}")
print(f"Predizioni - Min: {predictions.min():.6f}, Max: {predictions.max():.6f}")
print(f"% predizioni > 0: {(predictions > 0).mean():.2%}")
print(f"% predizioni < 0: {(predictions < 0).mean():.2%}")
print(f"Correlazione pred vs real: {np.corrcoef(predictions, y_test)[0, 1]:.4f}")

# ============================================================
# GENERAZIONE SEGNALI - TRE MODALITA'
# ============================================================
if SIGNAL_MODE == "fixed":
    # Modalità 1: Soglia fissa (originale)
    test_results['position'] = np.where(test_results['pred_ret'] > SIGNAL_THRESHOLD, 1, 0)
    print(f"\nModalità: Soglia fissa > {SIGNAL_THRESHOLD}")
    
elif SIGNAL_MODE == "percentile":
    # Modalità 2: Basata su percentili (relativa alle predizioni)
    long_threshold = np.percentile(predictions, LONG_PERCENTILE)
    short_threshold = np.percentile(predictions, SHORT_PERCENTILE)
    
    conditions = [
        predictions >= long_threshold,
        predictions <= short_threshold
    ]
    choices = [1, -1]
    test_results['position'] = np.select(conditions, choices, default=0)
    
    print(f"\nModalità: Percentili")
    print(f"  Long quando pred >= {long_threshold:.6f} (percentile {LONG_PERCENTILE})")
    print(f"  Short quando pred <= {short_threshold:.6f} (percentile {SHORT_PERCENTILE})")
    
elif SIGNAL_MODE == "long_short":
    # Modalità 3: Long/Short semplice basato sul segno
    test_results['position'] = np.where(predictions > 0, 1, -1)
    print(f"\nModalità: Long/Short semplice (segno predizione)")

# Calcolo turnovers (cambi di posizione) per costi di transazione
test_results['trade'] = test_results['position'].diff().abs().fillna(0)

# Costi di transazione
TRANSACTION_COST = 0.001
test_results['costs'] = test_results['trade'] * TRANSACTION_COST

# Rendimento strategia
test_results['strat_ret_gross'] = test_results['position'] * test_results['real_ret']
test_results['strat_ret_net'] = test_results['strat_ret_gross'] - test_results['costs']

# Equity curves
test_results['equity_bh'] = (1 + test_results['real_ret']).cumprod()
test_results['equity_lstm_gross'] = (1 + test_results['strat_ret_gross']).cumprod()
test_results['equity_lstm_net'] = (1 + test_results['strat_ret_net']).cumprod()

# ============================================================
# 9. METRICHE DI PERFORMANCE
# ============================================================
def calc_metrics(returns, name, trades_series=None):
    """Calcola metriche standard di backtest."""
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Win rate
    active = returns[returns != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    
    # Numero trades
    n_trades = int(trades_series.sum()) if trades_series is not None else 0
    
    # Exposure (% tempo a mercato)
    if trades_series is not None:
        positions = test_results['position']
        exposure = (positions != 0).mean()
    else:
        exposure = 1.0

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Return totale:    {total_ret:>8.2%}")
    print(f"  Return annualiz:  {ann_ret:>8.2%}")
    print(f"  Volatilità ann:   {ann_vol:>8.2%}")
    print(f"  Sharpe Ratio:     {sharpe:>8.2f}")
    print(f"  Max Drawdown:     {max_dd:>8.2%}")
    print(f"  Win Rate:         {win_rate:>8.2%}")
    print(f"  N. trades:        {n_trades:>8d}")
    print(f"  Exposure:         {exposure:>8.2%}")

calc_metrics(test_results['real_ret'], "Buy & Hold")
calc_metrics(test_results['strat_ret_net'], "LSTM Strategy (netto costi)", test_results['trade'])

# Statistiche posizioni
print(f"\n{'='*40}")
print(f"  STATISTICHE POSIZIONI")
print(f"{'='*40}")
print(f"  % Long:   {(test_results['position'] > 0).mean():>8.2%}")
print(f"  % Short:  {(test_results['position'] < 0).mean():>8.2%}")
print(f"  % Flat:   {(test_results['position'] == 0).mean():>8.2%}")

# ============================================================
# 10. VISUALIZZAZIONE
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 1, 1, 1]})

# --- Equity Curve ---
ax1 = axes[0]
ax1.plot(test_results['equity_bh'], label='Bitcoin Buy & Hold', color='grey', alpha=0.6)
ax1.plot(test_results['equity_lstm_gross'], label='LSTM (lordo)', color='dodgerblue', alpha=0.4, linestyle='--')
ax1.plot(test_results['equity_lstm_net'], label='LSTM (netto costi)', color='blue', linewidth=2)
ax1.set_title(f"Backtest LSTM su Dati Mai Visti — {TICKER} — Modalità: {SIGNAL_MODE}", fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_yscale('log')
ax1.set_ylabel('Equity (log)')
ax1.grid(True, alpha=0.3)

# --- Posizioni ---
ax2 = axes[1]
ax2.fill_between(test_results.index, test_results['position'], 0, 
                 where=(test_results['position'] > 0), alpha=0.4, color='green', label='Long')
ax2.fill_between(test_results.index, test_results['position'], 0,
                 where=(test_results['position'] < 0), alpha=0.4, color='red', label='Short')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('Posizione')
ax2.set_ylim(-1.2, 1.2)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Predizioni nel tempo ---
ax3 = axes[2]
ax3.plot(test_results.index, predictions, label='Predizioni LSTM', alpha=0.7, linewidth=1, color='purple')
ax3.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero')
if SIGNAL_MODE == "percentile":
    ax3.axhline(long_threshold, color='green', linestyle=':', linewidth=1, label=f'Long threshold (p{LONG_PERCENTILE})')
    ax3.axhline(short_threshold, color='red', linestyle=':', linewidth=1, label=f'Short threshold (p{SHORT_PERCENTILE})')
ax3.set_ylabel('Predicted Return')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- Drawdown ---
cumulative = (1 + test_results['strat_ret_net']).cumprod()
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
ax4 = axes[3]
ax4.fill_between(test_results.index, drawdown, alpha=0.5, color='red', label='Drawdown LSTM')
ax4.set_ylabel('Drawdown')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
