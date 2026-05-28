import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Stesso setup del modello originale
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
WINDOW_SIZE = 20
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# Download e preparazione dati
data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df = pd.DataFrame(index=data.index)
df['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
df['feat_skew'] = df['log_ret'].rolling(60).skew()
df['feat_vol_force'] = data['Volume'].pct_change().rolling(5).mean()
df['feat_volat'] = df['log_ret'].rolling(20).std()
df['target'] = df['log_ret'].shift(-1)
df.dropna(inplace=True)

features = ['feat_skew', 'feat_vol_force', 'feat_volat']

split_idx = int(len(df) * TRAIN_RATIO)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

def create_sequences(scaled_features, target_series, window):
    X, y = [], []
    for i in range(len(scaled_features) - window):
        X.append(scaled_features[i : i + window])
        y.append(target_series.iloc[i + window])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, train_df['target'], WINDOW_SIZE)
X_test, y_test = create_sequences(test_scaled, test_df['target'], WINDOW_SIZE)

X_train_t = torch.FloatTensor(X_train).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).to(DEVICE)
X_test_t = torch.FloatTensor(X_test).to(DEVICE)
y_test_t = torch.FloatTensor(y_test).to(DEVICE)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Training veloce
val_split = int(len(X_train_t) * 0.9)
X_val = X_train_t[val_split:]
y_val = y_train_t[val_split:]
train_subset = TensorDataset(X_train_t[:val_split], y_train_t[:val_split])
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)

best_val_loss = float('inf')
patience_counter = 0
PATIENCE = 5

for epoch in range(EPOCHS):
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

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        break

model.load_state_dict(best_state)

# Analisi predizioni
model.eval()
with torch.no_grad():
    predictions = model(X_test_t).cpu().numpy()

test_results = pd.DataFrame(index=test_df.index[WINDOW_SIZE:])
test_results['real_ret'] = y_test
test_results['pred_ret'] = predictions

print("\n" + "="*60)
print("ANALISI PREDIZIONI LSTM")
print("="*60)
print(f"\nStatistiche predizioni:")
print(f"  Media:        {predictions.mean():.6f}")
print(f"  Mediana:      {np.median(predictions):.6f}")
print(f"  Std Dev:      {predictions.std():.6f}")
print(f"  Min:          {predictions.min():.6f}")
print(f"  Max:          {predictions.max():.6f}")
print(f"  % > 0:        {(predictions > 0).mean():.2%}")
print(f"  % < 0:        {(predictions < 0).mean():.2%}")
print(f"  % == 0:       {(predictions == 0).mean():.2%}")

print(f"\nStatistiche target reali:")
print(f"  Media:        {y_test.mean():.6f}")
print(f"  Mediana:      {np.median(y_test):.6f}")
print(f"  Std Dev:      {y_test.std():.6f}")
print(f"  Min:          {y_test.min():.6f}")
print(f"  Max:          {y_test.max():.6f}")
print(f"  % > 0:        {(y_test > 0).mean():.2%}")

# Correlazione
corr = np.corrcoef(predictions, y_test)[0, 1]
print(f"\nCorrelazione pred vs real: {corr:.4f}")

# Grafico distribuzione
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram predizioni
axes[0, 0].hist(predictions, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero threshold')
axes[0, 0].set_title('Distribuzione Predizioni LSTM')
axes[0, 0].set_xlabel('Predicted Return')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histogram target
axes[0, 1].hist(y_test, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_title('Distribuzione Target Reali')
axes[0, 1].set_xlabel('Real Return')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Scatter pred vs real
axes[1, 0].scatter(predictions, y_test, alpha=0.3, s=10)
axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title(f'Predizioni vs Realtà (corr={corr:.4f})')
axes[1, 0].set_xlabel('Predicted Return')
axes[1, 0].set_ylabel('Real Return')
axes[1, 0].grid(True, alpha=0.3)

# Time series predizioni
axes[1, 1].plot(test_results.index, predictions, label='Predizioni', alpha=0.7, linewidth=1)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2, label='Zero threshold')
axes[1, 1].set_title('Serie Temporale Predizioni')
axes[1, 1].set_xlabel('Data')
axes[1, 1].set_ylabel('Predicted Return')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mostra prime 20 predizioni
print("\nPrime 20 predizioni:")
print(test_results[['real_ret', 'pred_ret']].head(20))
