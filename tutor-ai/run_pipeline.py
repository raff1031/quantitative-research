#!/usr/bin/env python3
# =================================================================================
# RUN PIPELINE — Runner completo con log per debugging e modalità silent-auto
#
# Esegue in sequenza:
#   1. generate_signals.py   (aggiorna i segnali dalle quotazioni recenti)
#   2. tws_executor.py       (dry-run prima, poi opzionalmente live)
#
# Salva log dettagliati e registra ogni esecuzione in auto_run_history.csv.
#
# USO NORMALE:
#   python run_pipeline.py                          # dry-run con equity=10000
#   python run_pipeline.py --equity 25000           # dry-run con equity diversa
#   python run_pipeline.py --equity 10000 --live    # dry-run → conferma → live
#   python run_pipeline.py --skip-signals           # salta generate_signals
#   python run_pipeline.py --use-tws                # connette a TWS invece di Gateway
#
# MODALITÀ SILENT (per Task Scheduler — nessun input utente):
#   python run_pipeline.py --silent                 # genera segnali + live diretto
#   python run_pipeline.py --silent --skip-signals  # solo live (segnali già aggiornati)
#
# OUTPUT:
#   logs/pipeline_log_YYYYMMDD_HHMMSS.txt — log completo di ogni run
#   auto_run_history.csv                  — storico successi/errori (append)
# =================================================================================

import os
import sys
import csv
import subprocess
import argparse
import platform
from datetime import datetime, date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR   = os.path.join(SCRIPT_DIR, "logs")
HISTORY_CSV = os.path.join(SCRIPT_DIR, "auto_run_history.csv")


def _ensure_logs_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def run_step(cmd: list, log_lines: list, step_name: str,
             timeout: int = 300) -> int:
    """
    Ranna un sottoprocesso, stampa l'output in tempo reale e lo aggiunge al log.
    Ritorna il return code.
    """
    separator = "=" * 72
    header = (f"\n{separator}\nSTEP: {step_name}\n"
              f"CMD:  {' '.join(cmd)}\n{separator}\n")
    print(header)
    log_lines.append(header)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=SCRIPT_DIR,
            timeout=timeout,
        )
        combined = result.stdout
        if result.stderr:
            combined += result.stderr

        print(combined)
        log_lines.append(combined)

        if result.returncode != 0:
            msg = f"\n⚠️  '{step_name}' terminato con codice {result.returncode}\n"
            print(msg)
            log_lines.append(msg)

        return result.returncode

    except subprocess.TimeoutExpired:
        msg = f"\n❌ TIMEOUT: '{step_name}' ha superato {timeout}s\n"
        print(msg)
        log_lines.append(msg)
        return 1
    except FileNotFoundError as e:
        msg = f"\n❌ Errore: file non trovato — {e}\n"
        print(msg)
        log_lines.append(msg)
        return 1
    except Exception as e:
        msg = f"\n❌ Errore inatteso in '{step_name}': {e}\n"
        print(msg)
        log_lines.append(msg)
        return 1


def collect_system_info() -> str:
    """Raccoglie info di sistema per diagnostica."""
    lines = []
    lines.append(f"Python     : {sys.version}")
    lines.append(f"OS         : {platform.system()} {platform.release()}")
    lines.append(f"Script dir : {SCRIPT_DIR}")

    for pkg in ["yfinance", "ib_insync", "pandas", "numpy"]:
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "?")
            lines.append(f"  {pkg:<12}: {ver}")
        except ImportError:
            lines.append(f"  {pkg:<12}: ❌ NON INSTALLATO")

    signal_dirs = {
        "Bio Cheap" : "stat_arb_results_bio_cheap",
        "Tech"      : "stat_arb_results_tech",
        "Macro"     : "stat_arb_results_macro",
        "Commodity" : "stat_arb_results_commodity",
    }
    lines.append("\nFile segnali (current_signals.csv):")
    for leg, subdir in signal_dirs.items():
        path = os.path.join(SCRIPT_DIR, subdir, "current_signals.csv")
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            stale = " ⚠️ VECCHIO" if age_h > 26 else ""
            lines.append(f"  {leg:<12}: ✅ {mtime.strftime('%Y-%m-%d %H:%M')} "
                         f"({age_h:.0f}h fa){stale}")
        else:
            lines.append(f"  {leg:<12}: ❌ NON TROVATO")

    return "\n".join(lines)


def append_history(run_ts: str, equity: float, mode: str,
                   silent: bool, errors: list, log_path: str):
    """Aggiunge una riga al CSV storico delle esecuzioni automatiche."""
    file_exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "equity", "mode", "silent",
                             "status", "errors", "log_file"])
        status = "OK" if not errors else "ERRORS"
        err_str = "; ".join(errors) if errors else ""
        writer.writerow([run_ts, equity, mode, silent, status,
                         err_str, os.path.basename(log_path)])


def is_trading_day() -> bool:
    """Controlla se oggi è un giorno feriale (Mon–Fri)."""
    return date.today().weekday() < 5   # 0=Mon … 4=Fri


def main():
    parser = argparse.ArgumentParser(
        description="Runner pipeline mega-portfolio con log automatico"
    )
    parser.add_argument(
        "--equity", type=float, default=10_000,
        help="Equity totale in USD (default: 10000)"
    )
    parser.add_argument(
        "--live", action="store_true", default=False,
        help="Dopo il dry-run chiede conferma e ranna gli ordini reali"
    )
    parser.add_argument(
        "--silent", action="store_true", default=False,
        help=("Modalità automatica: salta dry-run e conferma, "
              "esegue live direttamente. Usare con Task Scheduler.")
    )
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Modalità IB: paper (default) o live"
    )
    parser.add_argument(
        "--skip-signals", action="store_true", default=False,
        help="Salta generate_signals.py, usa i CSV di segnali già esistenti"
    )
    parser.add_argument(
        "--use-tws", action="store_true", default=False,
        help="Connette a TWS (porta 7497) invece di IB Gateway (porta 4002)"
    )
    parser.add_argument(
        "--force-weekday", action="store_true", default=False,
        help="In modalità --silent, forza l'esecuzione anche nel weekend"
    )
    args = parser.parse_args()

    # ── Controllo giorno operativo (solo in silent mode) ──────────────────────
    if args.silent and not args.force_weekday and not is_trading_day():
        print(f"⏸️  Oggi è {date.today().strftime('%A %d/%m/%Y')} — "
              f"mercati chiusi. Aggiungi --force-weekday per forzare.")
        sys.exit(0)

    _ensure_logs_dir()
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"pipeline_log_{run_ts}.txt")

    log_lines = []
    errors    = []

    # ── INTESTAZIONE LOG ──────────────────────────────────────────────────────
    mode_label = "SILENT AUTO-RUN 🤖" if args.silent else ("LIVE" if args.live else "DRY-RUN")
    sysinfo = collect_system_info()
    header = f"""\
================================================================================
MEGA-PORTFOLIO PIPELINE LOG  [{mode_label}]
================================================================================
Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Equity       : ${args.equity:,.0f}
Mode         : {args.mode.upper()}
Silent       : {args.silent}
Live orders  : {args.live or args.silent}
Skip signals : {args.skip_signals}
Use TWS      : {args.use_tws}

{sysinfo}
================================================================================
"""
    print(header)
    log_lines.append(header)

    py = sys.executable

    # ── STEP 1: generate_signals.py ───────────────────────────────────────────
    if not args.skip_signals:
        rc = run_step(
            [py, "generate_signals.py", "--equity", str(int(args.equity))],
            log_lines,
            "generate_signals.py",
            timeout=120,
        )
        if rc != 0:
            msg = "⚠️  generate_signals.py ha avuto errori — continuo con segnali precedenti\n"
            print(msg)
            log_lines.append(msg)
            errors.append(f"generate_signals.py: exit {rc}")
    else:
        skip_msg = "\n[SKIP] generate_signals.py — uso segnali da esecuzioni precedenti\n"
        print(skip_msg)
        log_lines.append(skip_msg)

    # ── MODALITÀ SILENT: esegue live direttamente ─────────────────────────────
    if args.silent:
        live_msg = "\n🤖 SILENT MODE: invio ordini live senza conferma...\n"
        print(live_msg)
        log_lines.append(live_msg)

        live_cmd = [
            py, "tws_executor.py",
            "--mode",   args.mode,
            "--equity", str(int(args.equity)),
        ]
        if args.use_tws:
            live_cmd.append("--use-tws")

        rc = run_step(live_cmd, log_lines, "tws_executor.py [LIVE AUTO]",
                      timeout=180)
        if rc != 0:
            errors.append(f"tws_executor.py [live-auto]: exit {rc}")

    # ── MODALITÀ NORMALE: dry-run + conferma opzionale ────────────────────────
    else:
        exec_cmd = [
            py, "tws_executor.py",
            "--mode",   args.mode,
            "--equity", str(int(args.equity)),
            "--dry-run",
        ]
        if args.use_tws:
            exec_cmd.append("--use-tws")

        rc = run_step(exec_cmd, log_lines, "tws_executor.py [DRY RUN]",
                      timeout=180)
        if rc != 0:
            errors.append(f"tws_executor.py [dry-run]: exit {rc}")

        if args.live:
            print("\n" + "=" * 72)
            print("✅ Il dry-run è completato.")
            print(f"   Modalità: {args.mode.upper()} | Equity: ${args.equity:,.0f}")
            confirm = input(
                "   Digita 'SI' per inviare gli ordini REALI, altro per annullare: "
            )
            print()

            if confirm.strip().upper() == "SI":
                log_lines.append("\n[UTENTE HA CONFERMATO] Invio ordini reali...\n")
                live_cmd = [
                    py, "tws_executor.py",
                    "--mode",   args.mode,
                    "--equity", str(int(args.equity)),
                ]
                if args.use_tws:
                    live_cmd.append("--use-tws")

                rc = run_step(live_cmd, log_lines,
                              "tws_executor.py [LIVE ORDERS]", timeout=180)
                if rc != 0:
                    errors.append(f"tws_executor.py [live]: exit {rc}")
            else:
                cancel_msg = "\n[ANNULLATO] Utente ha scelto di non inviare ordini.\n"
                print(cancel_msg)
                log_lines.append(cancel_msg)

    # ── FOOTER E SALVATAGGIO LOG ──────────────────────────────────────────────
    status_icon = "✅" if not errors else "⚠️ "
    footer = f"""
================================================================================
PIPELINE COMPLETATA {status_icon} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    if errors:
        footer += "Errori riscontrati:\n"
        for e in errors:
            footer += f"  ⚠️  {e}\n"
    else:
        footer += "  Nessun errore critico.\n"
    footer += "================================================================================\n"

    print(footer)
    log_lines.append(footer)

    log_content = "".join(log_lines)
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)
        print(f"📄 LOG SALVATO: logs/{os.path.basename(log_file)}")
    except Exception as e:
        print(f"❌ Impossibile salvare il log: {e}")

    # Storico esecuzioni (utile per monitorare gli auto-run nel tempo)
    try:
        append_history(run_ts, args.equity, args.mode,
                       args.silent, errors, log_file)
    except Exception:
        pass


if __name__ == "__main__":
    main()
