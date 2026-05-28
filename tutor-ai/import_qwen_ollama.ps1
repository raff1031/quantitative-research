# Script per importare Qwen2.5-Coder-32B in Ollama da file GGUF

Write-Host "🔧 Importazione Qwen2.5-Coder-32B in Ollama" -ForegroundColor Green
Write-Host ""

# Path al file GGUF scaricato
$ggufPath = "$env:USERPROFILE\Downloads\ollama_models\Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"

# Controlla se il file esiste
if (-not (Test-Path $ggufPath)) {
    Write-Host "❌ File GGUF non trovato: $ggufPath" -ForegroundColor Red
    Write-Host "📥 Prima esegui: .\download_qwen_coder.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ File trovato: $ggufPath" -ForegroundColor Green
$fileSize = (Get-Item $ggufPath).Length / 1GB
Write-Host "📦 Dimensione: $([math]::Round($fileSize, 2)) GB" -ForegroundColor Cyan
Write-Host ""

# Crea Modelfile per Ollama
$modelfilePath = "$env:USERPROFILE\Downloads\ollama_models\Modelfile_qwen_coder"
$modelfileContent = @"
# Qwen2.5-Coder-32B-Instruct Q4_K_M
FROM $ggufPath

# Template per chat (formato Qwen)
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

# Parametri ottimizzati per RTX 3080 Ti
PARAMETER num_ctx 32768
PARAMETER num_gpu 1
PARAMETER num_thread 16
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# System prompt per tutor
SYSTEM """Sei un tutor esperto che spiega concetti complessi in modo chiaro e didattico. Rispondi sempre in italiano."""
"@

Set-Content -Path $modelfilePath -Value $modelfileContent -Encoding UTF8

Write-Host "📝 Modelfile creato: $modelfilePath" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Importazione in Ollama (può richiedere 1-2 minuti)..." -ForegroundColor Yellow
Write-Host ""

# Importa in Ollama
ollama create qwen-coder:32b -f $modelfilePath

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Modello importato con successo!" -ForegroundColor Green
    Write-Host "🏷️  Nome: qwen-coder:32b" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🧪 Test rapido:" -ForegroundColor Yellow
    Write-Host '   ollama run qwen-coder:32b "Spiega cosa sono i puntatori in C"' -ForegroundColor White
    Write-Host ""
    Write-Host "🎓 Per usarlo nel tutor, aggiorna il codice:" -ForegroundColor Yellow
    Write-Host '   MAIN_MODEL = "qwen-coder:32b"' -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Importazione fallita!" -ForegroundColor Red
    Write-Host "🔍 Controlla che Ollama sia in esecuzione: ollama serve" -ForegroundColor Yellow
}
