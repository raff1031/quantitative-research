@echo off
echo ============================================
echo  GitHub Repository Setup — Quant Research
echo ============================================
echo.

cd /d C:\Users\roman\Desktop\quant

git init
git add .
git commit -m "Initial commit: quantitative research and trading strategies"

echo.
echo ============================================
echo  Ora vai su https://github.com/new
echo  Crea un repo chiamato: quantitative-research
echo  NON aggiungere README (ne abbiamo gia' uno)
echo  Poi torna qui e premi INVIO
echo ============================================
pause

echo.
set /p GITHUB_USER=Inserisci il tuo username GitHub:
git remote add origin https://github.com/%GITHUB_USER%/quantitative-research.git
git branch -M main
git push -u origin main

echo.
echo ============================================
echo  DONE! Repo pubblicato su:
echo  https://github.com/%GITHUB_USER%/quantitative-research
echo ============================================
pause
