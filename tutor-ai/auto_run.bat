@echo off
:: =============================================================================
:: AUTO_RUN.BAT — Esecuzione automatica del mega-portfolio pipeline
::
:: Configurato per Windows Task Scheduler.
:: Lancia run_pipeline.py in modalità --silent (nessuna conferma utente).
::
:: SETUP TASK SCHEDULER (una volta sola):
::   schtasks /create /tn "MegaPortfolio" /tr "C:\Users\sas\Desktop\tutor ai\auto_run.bat" /sc weekly /d MON /st 15:25 /f
::   (esegue ogni lunedì alle 15:25 = 09:25 ET, prima dell'apertura US)
::
:: Per rimuovere il task:
::   schtasks /delete /tn "MegaPortfolio" /f
::
:: Per eseguire manualmente:
::   auto_run.bat
:: =============================================================================

setlocal

:: ── Configurazione ────────────────────────────────────────────────────────────
set SCRIPT_DIR=C:\Users\sas\Desktop\tutor ai
set EQUITY=10000
set MODE=paper
:: Per connettere a TWS invece di Gateway: aggiungere --use-tws
set EXTRA_ARGS=

:: ── Avvio ─────────────────────────────────────────────────────────────────────
cd /d "%SCRIPT_DIR%"

echo.
echo [%DATE% %TIME%] === AUTO_RUN MEGA-PORTFOLIO START ===
echo Script dir: %SCRIPT_DIR%
echo Equity:     %EQUITY%
echo Mode:       %MODE%
echo.

:: Lancia il pipeline in modalità silent (nessuna conferma, live diretto)
python "%SCRIPT_DIR%\run_pipeline.py" ^
    --equity %EQUITY% ^
    --mode %MODE% ^
    --silent ^
    %EXTRA_ARGS%

set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
    echo [%DATE% %TIME%] === AUTO_RUN COMPLETATO OK (exit 0) ===
) else (
    echo [%DATE% %TIME%] === AUTO_RUN TERMINATO CON ERRORI (exit %RC%) ===
    echo Controlla logs\pipeline_log_*.txt per i dettagli.
)
echo.

endlocal
exit /b %RC%
