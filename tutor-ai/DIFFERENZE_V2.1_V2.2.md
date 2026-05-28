# Differenze tra v2.1 (ORIGINAL) e v2.2 (CORRECTED)

## File Creati

1. **`xgboost 4 market v2.1 ORIGINAL.py`** - Versione originale con il bug
2. **`xgboost 4 market v2.2 CORRECTED.py`** - Versione corretta
3. **`xgboost 4 market v2.py`** - Versione corretta (copia di v2.2)

---

## Il Bug e La Fix

### Localizzazione
**File**: `calculate_trade_metrics()` function
**Righe interessate**: 735-745

### Codice ORIGINALE (v2.1) - CON BUG

```python
# Righe 735-739 in v2.1
# Open new position if not flat
if new_position != 0:
    entry_date = date
    entry_price = close_prices.loc[date]
    current_position = new_position  # BUG: solo aggiornato quando != 0!
```

**Problema:**
- `current_position` viene aggiornato SOLO quando `new_position != 0`
- Quando la posizione va FLAT (`new_position = 0`), `current_position` rimane al valore precedente (1 o -1)
- Nei giorni successivi, `new_position (0) != current_position (1)` è sempre TRUE
- Questo genera un nuovo record di "chiusura trade" per OGNI giorno che rimane FLAT

### Codice CORRETTO (v2.2) - FIX APPLICATA

```python
# Righe 735-745 in v2.2
# Open new position if not flat
if new_position != 0:
    entry_date = date
    entry_price = close_prices.loc[date]
else:
    # Going flat - reset entry tracking
    entry_date = None
    entry_price = None

# Always update current position to new position
current_position = new_position  # FIX: sempre aggiornato!
```

**Fix:**
- `current_position = new_position` è SEMPRE eseguito (fuori dall'if)
- Quando va FLAT, resettiamo anche `entry_date` e `entry_price` a `None`
- Previene la generazione di trade duplicati

---

## Impatto sui CSV Trade

### Esempio Pratico: SOL-USD Overnight

#### v2.1 (ORIGINAL - con bug)
```
Entry 2025-05-08:
  - Exit 2025-05-16: PnL 1.96%, holding 8 giorni
  - Exit 2025-05-17: PnL 1.10%, holding 9 giorni    ← DUPLICATO
  - Exit 2025-05-19: PnL 1.67%, holding 11 giorni   ← DUPLICATO
  - Exit 2025-05-20: PnL 2.66%, holding 12 giorni   ← DUPLICATO
  ... (altre 15 righe duplicate)
  - Exit 2025-06-05: PnL -11.94%, holding 28 giorni ← DUPLICATO

Entry 2025-06-05:
  - Exit 2025-06-06: PnL -2.43%, holding 1 giorno
  - Exit 2025-06-07: PnL -3.97%, holding 2 giorni   ← DUPLICATO
  ... (altre 22 righe duplicate)

Entry 2025-07-10:
  ... (altre 40+ righe duplicate)

TOTALE: 114 righe
```

**Caratteristiche del bug:**
- Stesso `entry_date` per molte righe
- Stesso `entry_price` per tutte le righe con lo stesso entry
- Stesso `MFE` e `MAE` (calcolati sul periodo completo)
- `exit_date` diverse ma consecutive/vicine
- `holding_time` incrementale

#### v2.2 (CORRECTED - fix applicata)
```
Entry 2025-05-08:
  - Exit 2025-06-05: PnL -11.94%, holding 28 giorni  ← UNA SOLA RIGA

Entry 2025-06-05:
  - Exit 2025-07-10: PnL -13.60%, holding 35 giorni  ← UNA SOLA RIGA

Entry 2025-07-10:
  - Exit 2025-08-26: PnL 19.31%, holding 47 giorni   ← UNA SOLA RIGA

... (altri ~3 trade)

TOTALE: ~6 righe
```

**Caratteristiche corrette:**
- Un `entry_date` = UNA riga nel CSV
- Ogni trade registrato UNA SOLA volta
- Exit date corrisponde al momento effettivo di chiusura
- Holding time corretto

---

## Dettagli Tecnici

### Scenario di Esecuzione che Causa il Bug

**Positions Series (esempio):**
```
2025-05-08: 1 (LONG)
2025-05-09: 1 (LONG)
...
2025-05-15: 1 (LONG)
2025-05-16: 0 (FLAT)  ← Primo cambio
2025-05-17: 0 (FLAT)  ← Bug si manifesta qui
2025-05-19: 0 (FLAT)  ← E qui
...
2025-06-05: -1 (SHORT) ← Nuovo trade
```

**Con il BUG (v2.1):**
```
Giorno 2025-05-08:
  - new_position = 1, current_position = 0
  - Cambio detected: apre LONG
  - current_position = 1 ✓

Giorni 2025-05-09 a 2025-05-15:
  - new_position = 1, current_position = 1
  - Nessun cambio, OK ✓

Giorno 2025-05-16:
  - new_position = 0, current_position = 1
  - Cambio detected: chiude LONG ✓
  - new_position = 0, quindi NON entra nell'if
  - current_position rimane 1 ✗

Giorno 2025-05-17:
  - new_position = 0, current_position = 1 (BUG!)
  - new_position != current_position → TRUE
  - Chiude LONG di nuovo! ✗
  - Salva trade duplicato ✗
  - current_position rimane 1 ✗

Giorno 2025-05-19:
  - Stesso problema, altro duplicato ✗

... continua fino a 2025-06-05
```

**Con la FIX (v2.2):**
```
Giorno 2025-05-16:
  - new_position = 0, current_position = 1
  - Cambio detected: chiude LONG ✓
  - new_position = 0, quindi entra nell'else
  - entry_date = None, entry_price = None ✓
  - current_position = 0 ✓ (SEMPRE aggiornato)

Giorno 2025-05-17:
  - new_position = 0, current_position = 0
  - new_position == current_position → FALSE
  - Nessun trade salvato ✓

Giorno 2025-05-19:
  - Nessun cambio, nessun trade ✓
```

---

## Metriche di Performance

**IMPORTANTE:** Le metriche di performance (Total Return, Max Drawdown, Calmar Ratio) sono **SEMPRE STATE CORRETTE** in entrambe le versioni.

### Perché?

Il calcolo dei returns è fatto SEPARATAMENTE dal tracking dei trade:

```python
# Linee 672-679 - Calcolo returns (CORRETTO in entrambe le versioni)
strategy_returns = positions * actual_returns
position_changes = positions.diff().fillna(0)
commission_costs = position_changes.abs() * commission_rate
strategy_returns = strategy_returns - commission_costs

# Linee 924-928 - Calcolo metriche finali
cumulative_strategy_returns = (1 + final_strategy_daily_returns).cumprod()
total_strategy_return = (cumulative_strategy_returns.iloc[-1] - 1) * 100
max_dd = calculate_drawdown(cumulative_strategy_returns)
calmar_ratio = total_strategy_return / abs(max_dd * 100)
```

**Il bug affettava SOLO il CSV dei trade**, non il calcolo delle performance.

---

## Verifica dei CSV

### Come Identificare CSV Buggati

1. **Conta le righe**: Se hai molte più righe del previsto (100+ per pochi mesi)
2. **Cerca duplicati di entry_date**: Usa `df.groupby('entry_date').size()`
3. **Verifica entry_price identici**: Righe con stesso entry hanno stesso price/MFE/MAE?
4. **Pattern di exit consecutive**: Exit dates molto ravvicinate per stesso entry?

### Comando Python per Verificare

```python
import pandas as pd

csv_path = "SOL-USD_Overnight_top1_oos.csv"
df = pd.read_csv(csv_path, skiprows=4)

print(f"Totale righe: {len(df)}")
print(f"Entry uniche: {df['entry_date'].nunique()}")
print(f"\nRighe per entry:")
print(df.groupby('entry_date').size())

# Se "Righe per entry" mostra valori > 1, è buggato
```

---

## Raccomandazioni

1. **Usa v2.2 (CORRECTED)** per nuovi backtest
2. **Rigenera i CSV** esistenti usando `regenerate_oos_only.py`
3. **Verifica i nuovi CSV** con lo script sopra
4. **Conserva v2.1** solo per riferimento storico

---

## File di Test Creati

- `test_trade_metrics_fix.py` - Dimostra il bug e la fix
- `verify_total_return.py` - Verifica che Total Return è sempre corretto
- `regenerate_oos_only.py` - Rigenera CSV corretti dai modelli salvati

---

**Data Fix**: 2025-10-23
**Severity**: ALTA (CSV trade completamente inaffidabili)
**Impact on Performance**: NESSUNO (metriche corrette)
**Impact on Trade Analysis**: CRITICO (trade duplicati/errati)
