# 🚀 Setup Tutor AI Enhanced - Guida Completa

## 📋 Requisiti Sistema

- **CPU**: Ryzen 9950X3D (o simile high-end)
- **RAM**: 96GB (minimo 32GB)
- **GPU**: RTX 3080 Ti 12GB (o superiore)
- **Storage**: ~80GB liberi per modelli
- **OS**: Windows/Linux/MacOS

---

## 🔧 Installazione

### 1. Installa Ollama

**Windows/Linux/Mac:**
```bash
# Scarica da: https://ollama.ai/download
# Oppure su Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Scarica i Modelli Richiesti

Apri un terminale e lancia questi comandi:

```bash
# Modello principale (32B params, ~20GB)
ollama pull qwen2.5-coder:32b

# Modello Vision per PDF scannerizzati (11B params, ~7GB)
ollama pull llama3.2-vision:11b

# Modello embeddings per RAG (~1GB)
ollama pull mxbai-embed-large
```

**Tempo stimato download**: 30-60 minuti (dipende dalla connessione)

### 3. Verifica installazione

```bash
ollama list
```

Dovresti vedere:
```
NAME                    SIZE
qwen2.5-coder:32b      20GB
llama3.2-vision:11b    7.0GB
mxbai-embed-large      669MB
```

### 4. Installa dipendenze Python

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-ollama>=0.1.0
langchain-community>=0.1.0
chromadb>=0.4.0
PyMuPDF>=1.23.0  # Per PDF avanzati
PyPDF2>=3.0.0    # Fallback PDF
Pillow>=10.0.0   # Per Vision
pandas>=2.0.0
plotly>=5.17.0
networkx>=3.0    # Per knowledge graph (opzionale)
```

---

## 🎮 Avvio Applicazione

```bash
cd "c:\Users\sas\Desktop\tutor ai"
streamlit run "tutor demo claude.py"
```

L'app si aprirà su `http://localhost:8501`

---

## 🌟 Nuove Feature Implementate

### 1. **Parser PDF Intelligente**
- ✅ Estrazione strutturata (titoli, sezioni, definizioni, teoremi)
- ✅ Rilevamento automatico formule LaTeX
- ✅ Fallback Vision per PDF scannerizzati
- ✅ Metadata arricchiti per ogni chunk

**Come funziona:**
1. Carica PDF dalla sidebar (sezione Chat)
2. Clicca "Processa 🚀"
3. Il sistema:
   - Analizza ogni pagina
   - Se scannerizzata → usa Llama Vision (OCR intelligente)
   - Estrae struttura gerarchica con Qwen2.5-Coder
   - Identifica formule matematiche
   - Crea chunks context-aware

### 2. **Llama Vision per PDF Scannerizzati**
Quando il testo estratto è < 50 caratteri:
- Converte pagina in immagine ad alta risoluzione
- Llama Vision legge l'immagine
- Estrae testo + formule LaTeX + descrizione diagrammi
- Risultato: Markdown strutturato con formule

### 3. **Chat Migliorata**
- Modello principale: **Qwen2.5-Coder:32b** (molto più intelligente)
- Context window: 32K tokens
- Prompt ottimizzato per didattica
- Supporto LaTeX in risposta
- Citazioni precise con pagina

### 4. **Flashcards Auto-Generate** 🃏
**Menu → Flashcards**
- Genera automaticamente carte di studio dal materiale
- Varia difficoltà (1-5)
- Modalità flip front/back
- Tags per argomento

**Come usare:**
1. Carica PDF
2. Vai su Flashcards
3. Scegli quante carte (5-30)
4. Clicca "Genera"
5. Studia con navigazione ◀▶

### 5. **Quiz Interattivi** 🎯
**Menu → Quiz**
- Quiz a risposta multipla generati dal materiale
- Difficoltà personalizzabile
- Spiegazioni dettagliate
- Punteggio salvato nel profilo

**Come funziona:**
1. Seleziona n° domande e difficoltà
2. Clicca "Genera Quiz"
3. Rispondi alle domande
4. Consegna → vedi risultati + spiegazioni
5. Guadagni punti nel profilo

### 6. **Estrazione Struttura**
Il sistema ora capisce:
- **Teoremi** (riconoscimento pattern)
- **Definizioni** (evidenziate in grassetto)
- **Formule** (estratte in LaTeX: $...$, $$...$$)
- **Sezioni** (gerarchia # ## ###)
- **Concetti chiave** (per knowledge graph futuro)

---

## ⚙️ Configurazione Avanzata

### Modifica modelli (nel file .py)

Cerca queste righe (~riga 60):
```python
MAIN_MODEL = "qwen2.5-coder:32b"
VISION_MODEL = "llama3.2-vision:11b"
EMBEDDING_MODEL = "mxbai-embed-large"
```

**Alternative:**
- `MAIN_MODEL = "llama3.1:70b"` (se hai più VRAM, migliore reasoning)
- `MAIN_MODEL = "qwen2.5:32b"` (versione standard, non coder)
- `VISION_MODEL = "llava:13b"` (alternativa a llama vision)

### Ottimizzazione GPU

Per RTX 3080 Ti (12GB VRAM):
- **32B quantizzato Q4**: perfetto (usa ~10GB)
- **70B quantizzato Q3**: possibile ma lento
- **13B full precision**: velocissimo

Ollama gestisce automaticamente quantizzazione e offload CPU/GPU.

### Aumentare context window

Nel codice, riga ~955:
```python
llm = ChatOllama(model=MAIN_MODEL, temperature=0.3, num_ctx=32768)
```

Puoi aumentare `num_ctx` fino a 128K (per Qwen2.5):
```python
num_ctx=128000  # Attenzione: serve più RAM!
```

---

## 🐛 Troubleshooting

### Problema: "Model not found"
**Soluzione:**
```bash
ollama list  # Verifica modelli installati
ollama pull qwen2.5-coder:32b  # Riscarica se mancante
```

### Problema: Chat lenta/blocca
**Cause:**
1. Modello troppo grande per GPU
2. Troppi PDF indicizzati

**Soluzioni:**
- Usa modello più piccolo (es. qwen2.5:14b)
- Indicizza meno PDF alla volta
- Riduci `num_ctx` a 16384

### Problema: Vision non funziona
**Verifica:**
```bash
pip install Pillow PyMuPDF
ollama pull llama3.2-vision:11b
```

### Problema: Formule LaTeX non visualizzate
Streamlit supporta LaTeX markdown nativo:
- Inline: `$E = mc^2$`
- Blocco: `$$\int_0^\infty e^{-x} dx = 1$$`

Se non vedi le formule, aggiorna Streamlit:
```bash
pip install --upgrade streamlit
```

---

## 📊 Performance Attese

**Setup: Ryzen 9950X3D + 96GB RAM + RTX 3080 Ti**

| Operazione | Tempo | Note |
|------------|-------|------|
| Carica 1 PDF (50 pag) | 30-90s | Con Vision: più lento |
| Risposta chat | 5-15s | Dipende da context |
| Genera 10 flashcards | 20-40s | |
| Genera quiz 5 domande | 30-60s | |
| Indicizzazione 100 chunks | 10-20s | |

**Ottimizzazioni:**
- **SSD NVMe**: migliora caricamento modelli
- **RAM speed**: 5600MHz+ riduce latency
- **GPU cooling**: mantieni <75°C per max performance

---

## 🎓 Best Practices

### Per migliori risultati:

1. **PDF di qualità**
   - Preferisci PDF testuali (non scansioni)
   - Se scansioni: alta risoluzione (300+ DPI)

2. **Chunking intelligente**
   - Il sistema ora usa chunks da 1200 caratteri (più context)
   - Overlap 300 caratteri (continuità)

3. **Prompt efficaci**
   - Specifica: "Spiega il Teorema X passo-passo"
   - Cita: "Secondo la definizione a pag. Y..."
   - Collega: "Qual è la relazione tra X e Y?"

4. **Studio progressivo**
   - Inizia con Flashcards (ripasso passivo)
   - Poi Chat (approfondimento attivo)
   - Infine Quiz (verifica conoscenza)

---

## 🔮 Prossime Feature (Roadmap)

- [ ] Knowledge Graph visualizzabile (NetworkX + Plotly)
- [ ] Spaced Repetition per flashcards (algoritmo SM-2)
- [ ] Export flashcards in Anki
- [ ] Sintesi automatica capitoli
- [ ] Riconoscimento diagrammi con descrizione IA
- [ ] Mode "Socratic tutor" (domande per guidare ragionamento)
- [ ] Integration con Notion/Obsidian

---

## 📞 Supporto

Per problemi o domande:
1. Controlla questa guida
2. Verifica log Ollama: `ollama logs`
3. Controlla console Streamlit per errori

---

## 🎉 Enjoy Your Super-Powered AI Tutor!

Con Qwen2.5-Coder 32B + Llama Vision hai un sistema che:
- Capisce matematica avanzata
- Legge PDF scannerizzati come un umano
- Genera contenuti di studio personalizzati
- Si adatta al tuo livello

**Buono studio! 📚🚀**
