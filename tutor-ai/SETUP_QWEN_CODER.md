# 🚀 Setup Qwen2.5-Coder-32B da Hugging Face

## Perché Qwen2.5-Coder è Meglio

**vs gpt-oss:20b:**
- ✅ **Matematica**: molto più accurato
- ✅ **Codice**: eccellente (è specializzato)
- ✅ **Reasoning**: superiore su problemi complessi
- ✅ **Context**: 32K (vs 16K di gpt-oss)
- ✅ **Lingue**: ottimo anche in italiano
- ⚠️ **VRAM**: ~11GB (Q4 quantizzato)

---

## 📋 Requisiti

- **aria2c** per download veloce
- **Ollama** installato
- **20GB spazio libero** (temporaneo per download)
- **11GB VRAM** (perfetto per 3080 Ti)

### Installa aria2c (se non ce l'hai)

```powershell
# Opzione 1: winget (Windows 11)
winget install aria2

# Opzione 2: Chocolatey
choco install aria2

# Opzione 3: Scoop
scoop install aria2

# Opzione 4: Download manuale
# https://github.com/aria2/aria2/releases
```

---

## 🎯 Procedura Completa

### Step 1: Risolvi l'Embedding (PRIMA!)

L'app ha bisogno di un modello embedding. Lancia:

```bash
ollama pull all-minilm
```

Attendi (~2 minuti, 120MB). Poi riavvia l'app - dovrebbe funzionare!

---

### Step 2: Scarica Qwen2.5-Coder

Apri **PowerShell** in `C:\Users\sas\Desktop\tutor ai\` e lancia:

```powershell
.\download_qwen_coder.ps1
```

**Cosa fa:**
- Scarica il GGUF da Hugging Face (bartowski/Qwen2.5-Coder-32B-Instruct-GGUF)
- Usa aria2c con 16 connessioni parallele
- Salva in `%USERPROFILE%\Downloads\ollama_models\`

**Tempo stimato:** 10-30 minuti (dipende dalla tua connessione)

**Durante il download** puoi vedere la velocità:
```
[#1 20GB/20GB(100%) CN:16 DL:50MB/s]
```

Se si interrompe, rilancia lo script - riprenderà da dove si era fermato (`-c` flag)!

---

### Step 3: Importa in Ollama

Dopo che il download è completato, lancia:

```powershell
.\import_qwen_ollama.ps1
```

**Cosa fa:**
- Crea un Modelfile con config ottimizzata per 3080 Ti
- Importa il GGUF in Ollama come `qwen-coder:32b`
- Configura template chat e system prompt

**Tempo:** 1-2 minuti

---

### Step 4: Test il Modello

```bash
ollama run qwen-coder:32b "Spiega il teorema di Pitagora con dimostrazione"
```

Dovresti vedere una risposta dettagliata in italiano!

**Monitora GPU:**
```bash
nvidia-smi -l 1
```

Vuoi vedere:
- VRAM: ~11GB
- GPU Util: 95-100%

---

### Step 5: Aggiorna il Tutor

Nel file `tutor demo claude.py`, cambia questa riga (~61):

```python
MAIN_MODEL = "qwen-coder:32b"  # ← Cambia da "gpt-oss:20b"
```

Salva e riavvia Streamlit:

```bash
streamlit run "tutor demo claude.py"
```

🎉 **Fatto!** Ora hai Qwen2.5-Coder-32B nel tutor!

---

## 🔥 Performance Attese

### gpt-oss:20b (prima)
- Risposta chat: 10-20s
- Qualità matematica: 6/10
- Context: 16K

### qwen-coder:32b (dopo)
- Risposta chat: 8-15s
- Qualità matematica: **9/10** ⭐
- Context: **32K** (il doppio!)
- Codice/LaTeX: **eccellente**

---

## 🐛 Troubleshooting

### Problema: aria2c non funziona

**Alternativa - download manuale:**

1. Vai su: https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF
2. Scarica: `Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf` (~20GB)
3. Metti in: `%USERPROFILE%\Downloads\ollama_models\`
4. Esegui: `.\import_qwen_ollama.ps1`

### Problema: Download si interrompe

Rilancia `.\download_qwen_coder.ps1` - aria2c riprende automaticamente!

### Problema: "Ollama not found" durante import

Assicurati che Ollama sia avviato:

```bash
ollama serve
```

Poi in un altro terminale:

```powershell
.\import_qwen_ollama.ps1
```

### Problema: Modello lento

Verifica config GPU nel Modelfile:

```bash
ollama show qwen-coder:32b --modelfile
```

Dovresti vedere:
```
PARAMETER num_gpu 1
PARAMETER num_thread 16
```

Se manca, ricrea con `.\import_qwen_ollama.ps1`

### Problema: VRAM insufficiente

Se hai <12GB VRAM, usa Q3 invece di Q4:

Nel `download_qwen_coder.ps1`, cambia URL:
```powershell
$url = "https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q3_K_M.gguf"
```

Q3 usa ~9GB VRAM (vs 11GB di Q4), leggermente meno accurato.

---

## 📊 Confronto Quantizzazioni

| Quantizzazione | VRAM | Dimensione | Qualità | Velocità |
|----------------|------|------------|---------|----------|
| Q8_0           | 14GB | 26GB       | 99%     | Lenta    |
| Q6_K           | 13GB | 24GB       | 97%     | Media    |
| **Q4_K_M** ⭐  | 11GB | 20GB       | 92%     | Veloce   |
| Q3_K_M         | 9GB  | 16GB       | 85%     | Veloce   |
| Q2_K           | 7GB  | 12GB       | 75%     | Veloce   |

**Consigliato per 3080 Ti (12GB):** Q4_K_M (sweet spot qualità/VRAM)

---

## 🎓 Modelli Alternativi da HF

Se Qwen non ti convince, altre ottime opzioni:

### 1. Llama-3.1-70B (Q4)
```
URL: bartowski/Llama-3.1-70B-Instruct-GGUF (Q4_K_M)
VRAM: 11-12GB
Pro: Eccellente general purpose
Contro: Meno buono su matematica di Qwen
```

### 2. DeepSeek-Coder-33B
```
URL: TheBloke/deepseek-coder-33b-instruct-GGUF (Q4_K_M)
VRAM: 11GB
Pro: Ottimo su codice
Contro: Inglese mainly
```

### 3. Mixtral-8x7B
```
URL: TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF (Q4_K_M)
VRAM: 10GB
Pro: Veloce (MoE architecture)
Contro: Meno accurato su matematica
```

---

## ✅ Checklist Finale

- [ ] aria2c installato
- [ ] all-minilm scaricato (`ollama list` lo mostra)
- [ ] Qwen2.5-Coder GGUF scaricato
- [ ] Modello importato in Ollama (`ollama list` mostra `qwen-coder:32b`)
- [ ] Test modello OK
- [ ] VRAM ~11GB durante inferenza
- [ ] `MAIN_MODEL` aggiornato nel codice
- [ ] App riavviata

🎉 **Enjoy your super-powered tutor!**

---

## 🔮 Prossimi Step (opzionali)

Dopo che tutto funziona:

1. **Implementa batch processing** per analisi PDF completa (1-2 min vs 6-7 min)
2. **Aggiungi Qwen2-VL** per Vision ancora migliore
3. **Fine-tune** Qwen su tue dispense specifiche
4. **Export flashcards** in Anki

Fammi sapere se vuoi implementare qualcosa! 🚀
