#!/usr/bin/env python3
# =================================================================================
# PNL TRACKER — Confronta P&L paper account vs backtest teorico
#
# Registra giornalmente il NAV del paper account e lo confronta con la curva
# teorica del backtest walk-forward. Risponde a domande come:
#   "Il sistema live sta performando come il backtest?"
#   "Ci sono divergenze significative?"
#
# PREREQUISITI:
#   pip install ib_insync pandas matplotlib
#   IB Gateway in esecuzione (porta 4002 paper)
#
# USO:
#   # Registra NAV di oggi (da schedulare con run_pipeline.py o Task Scheduler)
#   python pnl_tracker.py --log
#
#   # Mostra grafico e statistiche P&L paper vs teorico
#   python pnl_tracker.py --report
#
#   # Entrambi (log + report)
#   python pnl_tracker.py --log --report
#
# OUTPUT:
#   pnl_history.csv       — storico NAV paper (date, nav, daily_return, ...)
#   pnl_report.html       — grafico interattivo (apri nel browser)
#   pnl_report_latest.txt — summary testuale dell'ultimo report
# =================================================================================

import os
import sys
import logging
import argparse
import csv
from datetime import datetime, date, timedelta
from typing import Optional, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PNL_HISTORY_CSV = os.path.join(SCRIPT_DIR, "pnl_history.csv")
PNL_REPORT_HTML = os.path.join(SCRIPT_DIR, "pnl_report.html")
PNL_REPORT_TXT  = os.path.join(SCRIPT_DIR, "pnl_report_latest.txt")

# Directory con i returns teorici del backtest (stesso path di tws_executor.py)
BACKTEST_DIRS = {
    "Bio":       "stat_arb_results_bio_cheap",
    "Tech":      "stat_arb_results_tech",
    "Macro":     "stat_arb_results_macro",
    "Commodity": "stat_arb_results_commodity",
}
BACKTEST_FILES = {
    "Bio":       "Combined_Dynamic_returns.csv",
    "Tech":      "Combined_Dynamic_returns.csv",
    "Macro":     "Macro_Blend_returns.csv",
    "Commodity": "Commodity_TSMOM_returns.csv",
}
FALLBACK_WEIGHTS = {"Bio": 0.11, "Tech": 0.13, "Macro": 0.65, "Commodity": 0.11}


# =================================================================================
# SEZIONE 1 — LETTURA NAV DA IB
# =================================================================================
def fetch_paper_nav(host: str = "127.0.0.1", port: int = 4002,
                    client_id: int = 11) -> Optional[Dict]:
    """
    Connette a IB e legge il NAV corrente dell'account paper.
    Ritorna dict con: nav, cash, unrealized_pnl, realized_pnl, account_id
    """
    try:
        from ib_insync import IB
    except ImportError:
        logger.error("ib_insync non installato — impossibile leggere NAV")
        return None

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        logger.info(f"Connesso a IB paper (porta {port})")

        result = {
            "nav": None, "cash": None,
            "unrealized_pnl": None, "realized_pnl": None,
            "account_id": None,
        }

        for av in ib.accountValues():
            tag = av.tag
            cur = av.currency
            val = av.value
            if tag == "NetLiquidation"   and cur == "USD":
                result["nav"] = float(val)
            elif tag == "CashBalance"    and cur == "USD":
                result["cash"] = float(val)
            elif tag == "UnrealizedPnL"  and cur == "USD":
                result["unrealized_pnl"] = float(val)
            elif tag == "RealizedPnL"    and cur == "USD":
                result["realized_pnl"] = float(val)
            elif tag == "AccountType":
                result["account_id"] = str(val)

        ib.disconnect()
        logger.info(f"NAV letto: ${result['nav']:,.2f}")
        return result

    except Exception as e:
        logger.error(f"Errore lettura NAV: {e}")
        if ib.isConnected():
            ib.disconnect()
        return None


# =================================================================================
# SEZIONE 2 — LOG NAV SU CSV
# =================================================================================
def log_nav(nav_data: dict, initial_equity: float = 10_000.0):
    """
    Appende una riga al CSV storico pnl_history.csv.
    Calcola anche il daily return e il cumulative return dall'inizio.
    """
    today_str = date.today().isoformat()
    nav       = nav_data.get("nav", 0.0)

    # Leggi storico per calcolare daily return
    prev_nav = initial_equity
    if os.path.exists(PNL_HISTORY_CSV):
        try:
            hist = pd.read_csv(PNL_HISTORY_CSV, index_col=0, parse_dates=True)
            if not hist.empty and "nav" in hist.columns:
                prev_nav = float(hist["nav"].iloc[-1])
                # Non duplicare la data di oggi
                if hist.index[-1].date() == date.today():
                    logger.info("NAV già registrato per oggi — aggiorno.")
                    hist.loc[hist.index[-1], "nav"]          = nav
                    hist.loc[hist.index[-1], "unrealized_pnl"] = nav_data.get("unrealized_pnl", 0)
                    hist.loc[hist.index[-1], "realized_pnl"]   = nav_data.get("realized_pnl", 0)
                    hist.loc[hist.index[-1], "cash"]            = nav_data.get("cash", 0)
                    hist.to_csv(PNL_HISTORY_CSV)
                    logger.info(f"✅ pnl_history.csv aggiornato (NAV={nav:,.2f})")
                    return
        except Exception:
            pass

    daily_ret  = (nav - prev_nav) / prev_nav if prev_nav else 0.0
    cum_ret    = (nav - initial_equity) / initial_equity if initial_equity else 0.0

    file_exists = os.path.exists(PNL_HISTORY_CSV)
    with open(PNL_HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "date", "nav", "daily_return", "cumulative_return",
                "unrealized_pnl", "realized_pnl", "cash", "account_id"
            ])
        writer.writerow([
            today_str,
            round(nav, 2),
            round(daily_ret, 6),
            round(cum_ret, 6),
            round(nav_data.get("unrealized_pnl") or 0, 2),
            round(nav_data.get("realized_pnl") or 0, 2),
            round(nav_data.get("cash") or 0, 2),
            nav_data.get("account_id", ""),
        ])

    logger.info(f"✅ pnl_history.csv aggiornato: "
                f"NAV={nav:,.2f} daily={daily_ret:+.2%} cum={cum_ret:+.2%}")


# =================================================================================
# SEZIONE 3 — CURVA TEORICA DAL BACKTEST
# =================================================================================
def load_theoretical_returns(start_date: Optional[str] = None) -> pd.Series:
    """
    Legge i returns teorici del backtest e li combina in un portafoglio
    usando i pesi N_MinVar (o FALLBACK_WEIGHTS se non calcolabili).
    Ritorna una Series di returns giornalieri dal start_date.
    """
    legs = {}
    for leg, subdir in BACKTEST_DIRS.items():
        fname = BACKTEST_FILES.get(leg, "")
        path  = os.path.join(SCRIPT_DIR, subdir, fname)
        if os.path.exists(path):
            try:
                df   = pd.read_csv(path, index_col=0, parse_dates=True)
                rets = df.iloc[:, 0].dropna()
                legs[leg] = rets
                logger.info(f"  Loaded {leg}: {len(rets)} giorni")
            except Exception as e:
                logger.warning(f"  {leg}: errore lettura ({e})")

    if not legs:
        logger.warning("Nessun file backtest trovato — curva teorica non disponibile")
        return pd.Series(dtype=float)

    # Calcola pesi N_MinVar (semplificato: usa fallback se troppo poco storico)
    weights = FALLBACK_WEIGHTS
    if len(legs) >= 2:
        try:
            common_idx = list(legs.values())[0].index
            for s in legs.values():
                common_idx = common_idx.intersection(s.index)
            lookback = min(63, len(common_idx))
            if lookback >= 10:
                df_ret = pd.DataFrame({k: v.loc[common_idx] for k, v in legs.items()})
                recent = df_ret.iloc[-lookback:]
                cov    = recent.cov().values
                n      = len(legs)
                inv_c  = np.linalg.inv(cov + np.eye(n) * 1e-8)
                ones   = np.ones(n)
                raw_w  = inv_c @ ones / (ones @ inv_c @ ones + 1e-10)
                raw_w  = np.clip(raw_w, 0.05, 0.80)
                raw_w  = raw_w / raw_w.sum()
                weights = {list(legs.keys())[i]: raw_w[i] for i in range(n)}
                logger.info(f"  Pesi N_MinVar: { {k: f'{v:.1%}' for k,v in weights.items()} }")
        except Exception:
            pass

    # Combina i returns pesati
    all_dates = sorted(set(
        d for s in legs.values() for d in s.index
    ))
    portfolio_rets = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
    for leg, rets in legs.items():
        w = weights.get(leg, 0.0)
        portfolio_rets = portfolio_rets.add(rets * w, fill_value=0.0)

    if start_date:
        portfolio_rets = portfolio_rets.loc[start_date:]

    return portfolio_rets.sort_index()


# =================================================================================
# SEZIONE 4 — REPORT P&L PAPER VS TEORICO
# =================================================================================
def generate_report(initial_equity: float = 10_000.0, days_back: int = 252):
    """
    Genera il report P&L: confronta curva paper con curva teorica backtest.
    Salva pnl_report.html e pnl_report_latest.txt.
    """
    # ── Carica storico paper ─────────────────────────────────────────────────
    if not os.path.exists(PNL_HISTORY_CSV):
        logger.warning("pnl_history.csv non trovato — esegui prima --log")
        return

    paper_hist = pd.read_csv(PNL_HISTORY_CSV, index_col=0, parse_dates=True)
    if paper_hist.empty:
        logger.warning("pnl_history.csv vuoto")
        return

    paper_nav  = paper_hist["nav"].dropna()
    start_date = paper_nav.index[0].strftime("%Y-%m-%d")

    # ── Carica curva teorica ─────────────────────────────────────────────────
    theo_rets  = load_theoretical_returns(start_date=start_date)

    # Converte returns teorici in curva NAV (base = initial_equity)
    theo_nav   = pd.Series(dtype=float)
    if not theo_rets.empty:
        theo_cum   = (1 + theo_rets).cumprod()
        theo_nav   = initial_equity * theo_cum
        theo_nav   = theo_nav.loc[start_date:]

    # ── Calcola statistiche ──────────────────────────────────────────────────
    paper_ret_series = paper_nav.pct_change().dropna()

    def _stats(nav_series: pd.Series, initial: float) -> dict:
        if nav_series.empty:
            return {}
        rets  = nav_series.pct_change().dropna()
        ann   = 252
        total_ret   = (nav_series.iloc[-1] - initial) / initial
        ann_ret     = (1 + total_ret) ** (ann / max(len(nav_series), 1)) - 1
        vol         = rets.std() * np.sqrt(ann) if len(rets) > 1 else 0.0
        sharpe      = ann_ret / vol if vol > 0 else 0.0
        roll_max    = nav_series.cummax()
        drawdown    = (nav_series - roll_max) / roll_max
        max_dd      = drawdown.min()
        return {
            "total_return":  total_ret,
            "ann_return":    ann_ret,
            "ann_vol":       vol,
            "sharpe":        sharpe,
            "max_drawdown":  max_dd,
            "current_nav":   nav_series.iloc[-1],
            "n_days":        len(nav_series),
        }

    paper_stats = _stats(paper_nav, initial_equity)
    theo_stats  = _stats(theo_nav,  initial_equity) if not theo_nav.empty else {}

    # ── Testo del report ─────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append(f"P&L REPORT  — generato {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)
    lines.append(f"Equity iniziale : ${initial_equity:,.0f}")
    lines.append(f"Dati dal        : {start_date}")
    lines.append(f"Giorni          : {paper_stats.get('n_days', 0)}")
    lines.append("")
    lines.append(f"{'Metrica':<22} {'Paper':>12} {'Teorico':>12}")
    lines.append("-" * 48)

    metrics_map = [
        ("NAV corrente",    "current_nav",   "${:,.2f}"),
        ("Return totale",   "total_return",  "{:+.2%}"),
        ("Return ann.",     "ann_return",    "{:+.2%}"),
        ("Volatilità ann.", "ann_vol",       "{:.2%}"),
        ("Sharpe ratio",    "sharpe",        "{:.2f}"),
        ("Max drawdown",    "max_drawdown",  "{:.2%}"),
    ]

    for label, key, fmt in metrics_map:
        p_val = paper_stats.get(key)
        t_val = theo_stats.get(key)
        p_str = fmt.format(p_val) if p_val is not None else "N/A"
        t_str = fmt.format(t_val) if t_val is not None else "N/A"
        lines.append(f"{label:<22} {p_str:>12} {t_str:>12}")

    # Divergenza Paper vs Teorico
    if paper_stats and theo_stats:
        p_cum = paper_stats["total_return"]
        t_cum = theo_stats["total_return"]
        div   = p_cum - t_cum
        lines.append("")
        lines.append(f"{'Divergenza (P-T)':<22} {div:>+12.2%}")
        if abs(div) > 0.05:
            lines.append("⚠️  Divergenza >5%: verifica se il sistema live è allineato al backtest")
        else:
            lines.append("✅ Divergenza contenuta (<5%): sistema allineato")

    lines.append("=" * 60)
    report_txt = "\n".join(lines)
    print(report_txt)

    # Salva il testo
    with open(PNL_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_txt)
    logger.info(f"✅ Report testuale salvato: {PNL_REPORT_TXT}")

    # ── HTML report con grafico ──────────────────────────────────────────────
    _generate_html_report(paper_nav, theo_nav, paper_stats, theo_stats,
                          initial_equity, report_txt)


def _generate_html_report(paper_nav: pd.Series, theo_nav: pd.Series,
                           paper_stats: dict, theo_stats: dict,
                           initial_equity: float, summary_txt: str):
    """Genera un file HTML con grafico interattivo (usa Chart.js, no dipendenze)."""

    def _series_to_js(s: pd.Series, name: str) -> str:
        """Converte una Series pandas in array JS."""
        points = [f"{{x: '{d.date()}', y: {v:.2f}}}"
                  for d, v in s.items() if not np.isnan(v)]
        return f"const {name} = [{', '.join(points)}];"

    paper_js = _series_to_js(paper_nav, "paperData") if not paper_nav.empty else "const paperData = [];"
    theo_js  = _series_to_js(theo_nav,  "theoData")  if not theo_nav.empty else "const theoData = [];"

    p_ret  = paper_stats.get("total_return", 0.0)
    t_ret  = theo_stats.get("total_return",  0.0)
    p_nav  = paper_stats.get("current_nav",  initial_equity)

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <title>P&amp;L Report — Mega Portfolio</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f8f9fa; }}
    h1   {{ color: #1a1a2e; }}
    .card {{ background: white; border-radius: 12px; padding: 24px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    .stats {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }}
    .stat  {{ text-align: center; padding: 16px; background: #f0f4ff; border-radius: 8px; }}
    .stat .val {{ font-size: 1.6em; font-weight: 700; }}
    .stat .lbl {{ font-size: .85em; color: #666; margin-top: 4px; }}
    .green {{ color: #16a34a; }} .red {{ color: #dc2626; }}
    pre {{ background: #1e1e2e; color: #cdd6f4; padding: 20px; border-radius: 8px; font-size: .85em; overflow-x: auto; }}
    canvas {{ max-height: 380px; }}
    .ts {{ font-size: .8em; color: #999; }}
  </style>
</head>
<body>
  <h1>📊 P&amp;L Report — Mega Portfolio</h1>
  <p class="ts">Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

  <div class="card stats">
    <div class="stat">
      <div class="val">${p_nav:,.0f}</div>
      <div class="lbl">NAV Paper attuale</div>
    </div>
    <div class="stat">
      <div class="val {'green' if p_ret >= 0 else 'red'}">{p_ret:+.2%}</div>
      <div class="lbl">Return Paper totale</div>
    </div>
    <div class="stat">
      <div class="val {'green' if t_ret >= 0 else 'red'}">{t_ret:+.2%}</div>
      <div class="lbl">Return Teorico totale</div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">NAV: Paper vs Teorico</h2>
    <canvas id="navChart"></canvas>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Summary statistiche</h2>
    <pre>{summary_txt}</pre>
  </div>

  <script>
    {paper_js}
    {theo_js}

    new Chart(document.getElementById('navChart'), {{
      type: 'line',
      data: {{
        datasets: [
          {{
            label: 'Paper Account',
            data: paperData,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,.08)',
            borderWidth: 2,
            pointRadius: 2,
            tension: .3,
            fill: true,
          }},
          {{
            label: 'Backtest Teorico',
            data: theoData,
            borderColor: '#f97316',
            backgroundColor: 'rgba(249,115,22,.05)',
            borderWidth: 2,
            borderDash: [6,3],
            pointRadius: 1,
            tension: .3,
            fill: false,
          }},
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{
          legend: {{ position: 'top' }},
          tooltip: {{
            callbacks: {{
              label: ctx => ` ${{ctx.dataset.label}}: $${{ctx.raw.y.toFixed(2)}}`,
            }}
          }}
        }},
        scales: {{
          x: {{ type: 'time', time: {{ unit: 'month' }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ callback: v => '$' + v.toLocaleString() }} }},
        }}
      }}
    }});
  </script>
</body>
</html>"""

    with open(PNL_REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"✅ Report HTML salvato: {PNL_REPORT_HTML}")
    print(f"   Apri nel browser: {PNL_REPORT_HTML}")


# =================================================================================
# CLI
# =================================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Tracker P&L paper account vs backtest teorico"
    )
    parser.add_argument("--log", action="store_true",
                        help="Legge NAV da IB e lo registra in pnl_history.csv")
    parser.add_argument("--report", action="store_true",
                        help="Genera report P&L (grafico + statistiche)")
    parser.add_argument("--equity", type=float, default=10_000,
                        help="Equity iniziale in USD (default: 10000)")
    parser.add_argument("--port", type=int, default=4002,
                        help="Porta IB Gateway (default: 4002)")
    parser.add_argument("--client-id", type=int, default=11,
                        help="Client ID IB (default: 11)")
    parser.add_argument("--mock-nav", type=float, default=None,
                        help="Test: usa NAV simulato invece di connettersi a IB")
    args = parser.parse_args()

    if not args.log and not args.report:
        parser.print_help()
        print("\n💡 Esempi:\n"
              "   python pnl_tracker.py --log            # registra NAV oggi\n"
              "   python pnl_tracker.py --report         # genera report\n"
              "   python pnl_tracker.py --log --report   # entrambi\n")
        sys.exit(0)

    if args.log:
        if args.mock_nav is not None:
            # Modalità test (senza IB)
            nav_data = {
                "nav": args.mock_nav, "cash": args.mock_nav * 0.3,
                "unrealized_pnl": args.mock_nav - args.equity,
                "realized_pnl": 0.0, "account_id": "TEST",
            }
            logger.info(f"[MOCK] NAV simulato: ${args.mock_nav:,.2f}")
        else:
            nav_data = fetch_paper_nav(port=args.port, client_id=args.client_id)

        if nav_data and nav_data.get("nav"):
            log_nav(nav_data, initial_equity=args.equity)
        else:
            logger.error("NAV non disponibile — verificare connessione IB Gateway")
            if not args.report:
                sys.exit(1)

    if args.report:
        generate_report(initial_equity=args.equity)
