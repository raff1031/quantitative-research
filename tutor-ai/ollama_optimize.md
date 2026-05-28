# 🚀 Ottimizzazione Ollama per GPU Offload Massimo

## RTX 3080 Ti + Ryzen 9950X3D Setup

### 1. Configura Variabili Ambiente Windows

Apri **PowerShell come Amministratore** e lancia:

```powershell
# Massimo GPU offload
[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_GPU", "1", "User")

# Massimo layers su GPU (per gpt-oss:20b su 12GB VRAM)
[System.Environment]::SetEnvironmentVariable("OLLAMA_GPU_LAYERS", "55", "User")

# Thread CPU (metà dei core per lasciare spazio)
[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_THREAD", "16", "User")

# Batch size (aumenta throughput)
[System.Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")

# Context size default
[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_CTX", "16384", "User")

# VRAM headroom (lascia margine per Streamlit)
[System.Environment]::SetEnvironmentVariable("OLLAMA_VRAM_FRACTION", "0.95", "User")
```

**Riavvia il terminale** dopo aver impostato queste variabili!

---

### 2. Verifica CUDA e cuBLAS

Ollama usa CUDA per GPU offload. Verifica:

```bash
nvidia-smi
```

Dovresti vedere:
- Driver: 535+ (o più recente)
- CUDA Version: 12.x
- GPU: RTX 3080 Ti (12GB)

---

### 3. Test Performance

Dopo aver configurato, testa la velocità:

```bash
# Pull modello se non l'hai già
ollama pull gpt-oss:20b

# Test velocità
ollama run gpt-oss:20b "Spiega il teorema di Pitagora in 50 parole"
```

**Performance attese con GPU offload:**
- Caricamento modello: 2-5 secondi
- Primo token: 1-2 secondi
- Token/sec: 25-40 (dipende da context)

**Senza GPU offload** (solo CPU):
- Primo token: 5-10 secondi
- Token/sec: 5-10

---

### 4. Monitora VRAM Usage

Durante l'uso, apri un altro terminale e lancia:

```bash
nvidia-smi -l 1
```

Con gpt-oss:20b dovresti vedere:
- **VRAM usata**: 10-11GB (~90% di 12GB)
- **GPU Usage**: 95-100% durante inferenza
- **Power**: ~250-320W

Se VRAM < 8GB → Ollama sta usando CPU! Aumenta `OLLAMA_GPU_LAYERS`.

---

### 5. Ottimizzazioni Specifiche per Modelli

#### gpt-oss:20b (attuale)
```bash
# Massimo offload (55 layers)
OLLAMA_GPU_LAYERS=55 ollama run gpt-oss:20b
```

#### llama3.2-vision:11b
```bash
# Vision usa meno VRAM, può andare tutto su GPU
OLLAMA_GPU_LAYERS=999 ollama run llama3.2-vision:11b
```

#### Se vuoi usare Qwen2.5-Coder:32b in futuro
```bash
# 32B richiede quantizzazione Q4 per stare in 12GB
ollama pull qwen2.5-coder:32b-q4_K_M

# Layers ridotti (non stanno tutti)
OLLAMA_GPU_LAYERS=35 ollama run qwen2.5-coder:32b-q4_K_M
```

---

### 6. Troubleshooting Velocità

#### Problema: Ollama lento anche con config GPU
**Soluzioni:**

1. **Verifica che Ollama usi GPU:**
```bash
ollama ps
```
Dovresti vedere il modello caricato con memoria allocata.

2. **Restart servizio Ollama:**
```bash
# Windows
taskkill /F /IM ollama.exe
ollama serve
```

3. **Svuota cache:**
```bash
# Windows
del %USERPROFILE%\.ollama\cache\*
```

4. **Disabilita Windows Defender (temporaneo test):**
Windows Defender può rallentare l'inferenza. Prova a disabilitarlo temporaneamente per vedere se migliora.

---

### 7. Benchmark Your Setup

Crea un file `benchmark.py`:

```python
import time
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.3,
    num_ctx=16384,
    num_gpu=1,
    num_thread=16
)

prompt = "Spiega la derivata di x^2 in dettaglio con 200 parole"

print("🔥 Benchmark iniziato...")
start = time.time()
response = llm.invoke(prompt)
end = time.time()

content = getattr(response, "content", str(response))
tokens = len(content.split())  # approssimazione
elapsed = end - start

print(f"\n✅ Completato in {elapsed:.2f}s")
print(f"📊 Tokens generati: ~{tokens}")
print(f"🚀 Velocità: ~{tokens/elapsed:.1f} tokens/sec")
print(f"\n💬 Risposta:\n{content[:200]}...")
```

Lancia:
```bash
python benchmark.py
```

**Target performance:**
- Tempo: 10-20s per 200 parole
- Velocità: 20-40 tokens/sec

---

### 8. Modalità "Turbo" per PDF Processing

Nel tutor, usa sempre **⚡ Modalità Veloce**:
- ✅ Checkbox attivo
- ❌ Disattiva "analisi struttura completa"

Questo salta le chiamate LLM per ogni pagina e fa solo:
1. Estrazione testo (veloce)
2. Chunking (veloce)
3. Embedding (veloce)

**Tempo PDF (50 pagine):**
- Modalità veloce: 10-30 secondi
- Modalità completa: 5-10 minuti (con analisi LLM)

---

### 9. Config Ottimale Finale

Nel file `.py`, le impostazioni sono già ottimizzate:

```python
GPU_LAYERS = 50  # Max layers su GPU
NUM_GPU = 1      # Una GPU
NUM_THREAD = 16  # Metà core CPU
```

**Se vuoi ancora più velocità**, prova:
```python
GPU_LAYERS = 55  # Praticamente tutto su GPU
NUM_THREAD = 12  # Meno thread CPU = più spazio GPU
```

---

### 10. Alternative se ancora Lento

Se con tutte queste ottimizzazioni è ancora lento, considera:

1. **Modello più piccolo:**
```bash
ollama pull llama3.1:8b  # Molto più veloce, ~40-60 tok/s
```

2. **Quantizzazione ridotta:**
```bash
ollama pull gpt-oss:20b-q3_K_M  # Più veloce ma meno accurato
```

3. **Disabilita Vision temporaneamente:**
Nel codice, commenta il blocco Vision per PDF scannerizzati se non serve.

---

## ✅ Checklist Finale

- [ ] Variabili ambiente configurate
- [ ] Ollama riavviato
- [ ] `nvidia-smi` mostra >10GB VRAM usata
- [ ] Benchmark >20 tok/s
- [ ] Modalità veloce attiva nel tutor
- [ ] PDF processa in <30s (50 pagine)

Se tutto ok → **Goditi il tuo tutor super veloce!** 🚀
