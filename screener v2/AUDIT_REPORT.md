# 🔍 AUDIT REPORT - LSTM v2 Strategy

## Data: 2026-02-14
## File: lstm_btc_v2.py

---

## ✅ COMPONENTI CORRETTI

### 1. Feature Engineering (Linee 56-88)
```python
df['feat_skew_12'] = df['log_ret'].rolling(12).skew().shift(1)
df['feat_vol_force'] = df_raw['Volume'].pct_change().rolling(4).mean().shift(1)
# ... tutte le altre feature hanno .shift(1)
```
**VERDICT: ✅ CORRETTO** - Tutte le feature sono shift(1), quindi al tempo t usi solo dati fino a t-1.

### 2. Target Construction (Linea 91)
```python
df['target'] = df['log_ret'].shift(-1)
```
**VERDICT: ✅ CORRETTO** - Il target è il rendimento futuro (t+1), disponibile solo dopo che t+1 è passato.

### 3. Train/Test Split (Linee 169-171)
```python
split_idx = int(len(df) * TRAIN_RATIO)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()
```
**VERDICT: ✅ CORRETTO** - Split cronologico, nessuna contaminazione.

### 4. Scaling (Linee 178-180)
```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[FEATURES])
test_scaled = scaler.transform(test_df[FEATURES])
```
**VERDICT: ✅ CORRETTO** - Lo scaler è fittato solo sul training set.

### 5. Sequence Creation (Linee 136-141)
```python
def create_sequences(scaled_features, target_series, window):
    X, y = [], []
    for i in range(len(scaled_features) - window):
        X.append(scaled_features[i : i + window])
        y.append(target_series.iloc[i + window])
    return np.array(X), np.array(y)
```
**VERDICT: ✅ CORRETTO** - La finestra usa solo dati passati [i : i+window].

### 6. Position Shift (Linea 292)
```python
test_results['strat_ret_gross'] = test_results['position'].shift(1).fillna(0) * test_results['real_ret']
```
**VERDICT: ✅ CORRETTO** - La posizione al tempo t-1 determina il rendimento al tempo t.

### 7. Rule-Based Implementation (Linea 324)
```python
rb_df['strat_ret'] = rb_df['position'].shift(1) * rb_df['log_ret']
```
**VERDICT: ✅ CORRETTO** - Anche la rule-based shifta correttamente.

---

## 🚨 PROBLEMI CRITICI TROVATI

### **PROBLEMA #1: LOOKAHEAD BIAS NEL POSITION SIZING (Linea 274)**

```python
# LINEA 274 - CODICE ATTUALE (ERRATO)
raw_pos = predictions / max(np.std(predictions) * 2, 1e-6)
```

**DESCRIZIONE DEL BIAS:**
- `np.std(predictions)` viene calcolato su TUTTE le predizioni del test set
- Quando decidi la posizione al tempo t, stai usando la standard deviation che include predizioni fino alla fine del test set
- Stai usando informazioni dal FUTURO per normalizzare le posizioni

**IMPATTO:**
- Il modello sa implicitamente quanto sarà "volatile" nei prossimi mesi
- Può scalare le posizioni in modo più aggressivo o conservativo basandosi su dati futuri
- **Questo gonfia artificialmente le performance**

**SEVERITY: 🔴 CRITICO**

---

### **PROBLEMA #2: POSITION PERSISTENCE NELLA RULE-BASED (Linee 307-321)**

```python
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
    positions_rb.append(current_pos)  # ← Mantiene la posizione precedente se non long/short
```

**DESCRIZIONE:**
- La strategia rule-based mantiene la posizione precedente (`current_pos`) se i segnali non sono né long né short
- Questo crea una **strategia trend-following sticky**
- È un comportamento intentionale o un bug?

**IMPATTO:**
- La rule-based può rimanere in posizione long/short per lunghi periodi anche senza nuovi segnali
- Questo NON è comparabile con l'LSTM che ricalcola la posizione ogni periodo
- **Confronto unfair**

**SEVERITY: 🟡 MEDIO** (dipende dall'intenzione)

---

## 🔧 SOLUZIONI PROPOSTE

### Soluzione per Problema #1 (Lookahead Bias)

**Opzione A: Usa std dal validation set (RACCOMANDATO)**
```python
# Calcola std dalle predizioni del validation set (dati passati)
model.eval()
with torch.no_grad():
    val_predictions = model(X_val).cpu().numpy()
    val_std = np.std(val_predictions)

# Usa questa std fissa per tutto il test set
raw_pos = predictions / max(val_std * 2, 1e-6)
raw_pos = np.clip(raw_pos, -1.0, 1.0)
```

**Opzione B: Expanding window std (più rigoroso ma complesso)**
```python
# Calcola std in modo expanding
positions = []
for i in range(len(predictions)):
    if i < 20:  # Minimo 20 osservazioni
        scale = 1e-6
    else:
        scale = max(np.std(predictions[:i]) * 2, 1e-6)  # Usa solo dati fino a i
    positions.append(np.clip(predictions[i] / scale, -1.0, 1.0))
test_results['position'] = positions
```

**Opzione C: Soglia fissa senza normalizzazione (più semplice)**
```python
# Non normalizzare, usa soglie fisse
conditions = [
    predictions > SIGNAL_THRESHOLD,
    predictions < -SIGNAL_THRESHOLD
]
choices = [1.0, -1.0]
test_results['position'] = np.select(conditions, choices, default=0.0)
```

### Soluzione per Problema #2 (Rule-Based Unfair)

**Modifica la rule-based per essere stateless:**
```python
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
    else:
        current_pos = 0  # ← CAMBIA QUESTO: flat se nessun segnale
    positions_rb.append(current_pos)
```

---

## 📊 CONCLUSIONE

**Il modello LSTM v2 HA BARATO attraverso lookahead bias nel position sizing.**

Se i risultati sono:
- LSTM vince su 4/7 ticker
- Con Sharpe medio ~1.5

Dopo la correzione del bias, **mi aspetto**:
- Performance LSTM ridotte del 10-30%
- Sharpe ratio più bassi
- Possibilmente meno vittorie vs Rule-Based

**RACCOMANDAZIONE: Applica la Soluzione A (usa std dal validation set) e ri-esegui il backtest.**

---

## ✅ COSA MANTENERE

1. L'architettura LSTM è solida (3 layer, LayerNorm, Huber loss)
2. Le 10 feature sono ben costruite senza lookahead
3. Il train/test split è rigoroso
4. Lo scaling è corretto
5. La strategia long/short è concettualmente valida

**Il concetto è buono, solo il position sizing ha un bug.**
