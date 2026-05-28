# Script per scaricare Qwen2.5-Coder-32B da Hugging Face
# Usa aria2c per download veloce multi-thread

Write-Host "🚀 Download Qwen2.5-Coder-32B-Instruct (Q4_K_M quantizzato)" -ForegroundColor Green
Write-Host "📦 Dimensione: ~20GB" -ForegroundColor Yellow
Write-Host ""

# Controlla se aria2c è installato
$aria2cPath = Get-Command aria2c -ErrorAction SilentlyContinue
if (-not $aria2cPath) {
    Write-Host "❌ aria2c non trovato!" -ForegroundColor Red
    Write-Host "📥 Installalo con: winget install aria2" -ForegroundColor Yellow
    Write-Host "   oppure scarica da: https://github.com/aria2/aria2/releases" -ForegroundColor Yellow
    exit 1
}

# Directory download
$downloadDir = "$env:USERPROFILE\Downloads\ollama_models"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

Write-Host "📂 Download in: $downloadDir" -ForegroundColor Cyan
Write-Host ""

# URL Hugging Face (bartowski ha ottimi quantizzati GGUF)
$url = "https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
$outputFile = "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"

Write-Host "🌐 URL: $url" -ForegroundColor Cyan
Write-Host ""

# Download con aria2c (16 connessioni parallele)
aria2c -x16 -s16 -k1M -c $url -d $downloadDir -o $outputFile

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Download completato!" -ForegroundColor Green
    Write-Host "📁 File: $downloadDir\$outputFile" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔧 Prossimo step: importa in Ollama" -ForegroundColor Yellow
    Write-Host "   Esegui: .\import_qwen_ollama.ps1" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "❌ Download fallito!" -ForegroundColor Red
    Write-Host "🔄 Riprova o scarica manualmente da:" -ForegroundColor Yellow
    Write-Host "   https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF" -ForegroundColor Yellow
}
