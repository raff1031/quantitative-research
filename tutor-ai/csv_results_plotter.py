# =================================================================================
# CSV Results Plotter - Visualizzazione Risultati XGBoost Trading Model
#
# DESCRIZIONE:
# Script standalone per visualizzare i risultati del backtest OOS.
# Genera grafici interattivi per analizzare le performance di trading.
#
# FEATURES:
# - Equity curves per ogni asset e strategia
# - Confronto performance vs Buy & Hold
# - Analisi distribuzione trade (win rate, holding time, MFE/MAE)
# - Heatmap performance multi-asset
# - Report statistici dettagliati
# =================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Configurazione stile grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =================================================================================
# CONFIGURAZIONE
# =================================================================================

# Directory contenente i risultati (MODIFICA QUI SE NECESSARIO)
RESULTS_DIR = r"C:\Users\sas\run_v72.7_final_backtester_20251023_225614"

# Asset da analizzare
ASSETS = ['AAPL', 'GOOG', 'TSLA', '^IXIC', 'MRNA', 'LLY', 'ETH-USD', 'SOL-USD']
ASSET_NAMES = {
    'AAPL': 'Apple',
    'GOOG': 'Google',
    'TSLA': 'Tesla',
    '^IXIC': 'NASDAQ',
    'MRNA': 'Moderna',
    'LLY': 'Eli Lilly',
    'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana'
}

# Strategie
STRATEGIES = ['Overnight', 'Trend-Following']

# Directory output per i grafici
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================================
# FUNZIONI DI CARICAMENTO DATI
# =================================================================================

def load_csv_data(results_dir):
    """
    Carica tutti i file CSV dalla directory dei risultati.

    Returns:
        dict: Dizionario con struttura {asset: {strategy: {top_n: {summary, trades}}}}
    """
    print("="*80)
    print("CARICAMENTO DATI CSV")
    print("="*80)

    results_dir = Path(results_dir)
    all_data = {}

    for asset in ASSETS:
        all_data[asset] = {}

        for strategy in STRATEGIES:
            all_data[asset][strategy] = {}

            for top_n in [1, 2, 3]:
                csv_file = results_dir / f"{asset}_{strategy}_top{top_n}_oos.csv"

                if not csv_file.exists():
                    print(f"[WARN] File non trovato: {csv_file.name}")
                    continue

                try:
                    # Leggi le prime righe per ottenere le metriche summary
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Parse summary metrics (riga 2: valori)
                    summary_line = lines[2].strip().split(',')
                    summary = {
                        'total_return_pct': float(summary_line[0]),
                        'max_drawdown_pct': float(summary_line[1]),
                        'calmar_ratio': float(summary_line[2]),
                        'window_size': int(summary_line[3]),
                        'num_features': summary_line[4]
                    }

                    # Leggi trade details (skip le prime 5 righe: # Summary, header, valori, blank, # Trade Details)
                    trades_df = pd.read_csv(csv_file, skiprows=5)

                    # Converti date
                    if not trades_df.empty:
                        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

                    all_data[asset][strategy][top_n] = {
                        'summary': summary,
                        'trades': trades_df
                    }

                    print(f"[OK] Caricato: {csv_file.name} ({len(trades_df)} trades)")

                except Exception as e:
                    print(f"[ERROR] Errore caricando {csv_file.name}: {e}")

    print(f"\n[OK] Caricamento completato: {sum(len(s) for a in all_data.values() for s in a.values())} file processati\n")
    return all_data

def load_oos_report(results_dir):
    """
    Carica il report OOS finale.

    Returns:
        dict: Dizionario con struttura {asset: {'buy_hold': %, 'single_day': {...}, 'trend_following': {...}}}
    """
    report_file = Path(results_dir) / "oos_final_report.txt"

    if not report_file.exists():
        print(f"[WARN] Report OOS non trovato: {report_file}")
        return {}

    print("Caricamento OOS Final Report...")

    report_data = {}
    current_asset = None
    current_strategy = None

    with open(report_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Parse asset section
            if "Risultati OOS per:" in line:
                current_asset = line.split(":")[-1].strip()
                report_data[current_asset] = {}

            # Parse Buy & Hold
            elif "Buy & Hold Return" in line and current_asset:
                bh_return = float(line.split(":")[-1].strip().replace('%', ''))
                report_data[current_asset]['buy_hold'] = bh_return

            # Parse Strategy section
            elif "Strategia:" in line and current_asset:
                strategy_name = line.split(":")[-1].strip().lower()
                current_strategy = strategy_name
                report_data[current_asset][current_strategy] = {}

            # Parse metrics
            elif current_asset and current_strategy:
                if "Media:" in line and "Calmar" in f.readline():
                    continue
                elif "Media:" in line:
                    value = float(line.split(":")[-1].strip().replace('%', ''))
                    metric = "mean"
                    if "Calmar" in line:
                        metric = "calmar_mean"
                    elif "Ritorno" in line:
                        metric = "return_mean"
                    report_data[current_asset][current_strategy][metric] = value

    print(f"[OK] Report OOS caricato: {len(report_data)} assets\n")
    return report_data

# =================================================================================
# FUNZIONI DI ANALISI
# =================================================================================

def calculate_equity_curve(trades_df, initial_capital=100):
    """
    Calcola la equity curve dai trade con COMPOUNDING CORRETTO.

    Args:
        trades_df: DataFrame con i trade
        initial_capital: Capitale iniziale (default 100 per normalizzazione)

    Returns:
        pd.Series: Equity curve indicizzata per data
    """
    if trades_df.empty:
        return pd.Series(dtype=float)

    # Ordina per exit_date
    trades_sorted = trades_df.sort_values('exit_date').copy()

    # CORREZIONE: Calcola equity con compounding corretto
    # pnl_net è già in percentuale, quindi dividiamo per 100 per ottenere il fattore moltiplicativo
    # Poi usiamo cumprod() per il compounding invece di cumsum()
    trades_sorted['return_factor'] = 1 + (trades_sorted['pnl_net'] / 100)
    trades_sorted['equity'] = initial_capital * trades_sorted['return_factor'].cumprod()

    # Crea serie temporale
    equity_curve = pd.Series(
        trades_sorted['equity'].values,
        index=trades_sorted['exit_date']
    )

    # Aggiungi punto iniziale
    if not equity_curve.empty:
        equity_curve = pd.concat([
            pd.Series([initial_capital], index=[trades_sorted['entry_date'].iloc[0]]),
            equity_curve
        ])

    return equity_curve

def calculate_trade_statistics(trades_df):
    """
    Calcola statistiche dettagliate sui trade.

    Returns:
        dict: Dizionario con statistiche
    """
    if trades_df.empty:
        return {}

    winning_trades = trades_df[trades_df['pnl_net'] > 0]
    losing_trades = trades_df[trades_df['pnl_net'] <= 0]

    stats = {
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'avg_win': winning_trades['pnl_net'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl_net'].mean() if len(losing_trades) > 0 else 0,
        'largest_win': trades_df['pnl_net'].max(),
        'largest_loss': trades_df['pnl_net'].min(),
        'avg_holding_time': trades_df['holding_time'].mean(),
        'avg_mfe': trades_df['mfe'].mean(),
        'avg_mae': trades_df['mae'].mean(),
        'profit_factor': abs(winning_trades['pnl_net'].sum() / losing_trades['pnl_net'].sum()) if len(losing_trades) > 0 and losing_trades['pnl_net'].sum() != 0 else float('inf'),
        'long_trades': len(trades_df[trades_df['position'] == 'LONG']),
        'short_trades': len(trades_df[trades_df['position'] == 'SHORT'])
    }

    return stats

# =================================================================================
# FUNZIONI DI PLOTTING
# =================================================================================

def plot_equity_curves(data, oos_report):
    """
    Crea grafici delle equity curves per tutti gli asset.
    """
    print("="*80)
    print("GENERAZIONE EQUITY CURVES")
    print("="*80)

    for asset in ASSETS:
        if asset not in data:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(f'Equity Curves - {ASSET_NAMES.get(asset, asset)}', fontsize=16, fontweight='bold')

        for idx, strategy in enumerate(STRATEGIES):
            ax = axes[idx]

            if strategy not in data[asset]:
                continue

            # Plot top 3 candidates
            for top_n in [1, 2, 3]:
                if top_n not in data[asset][strategy]:
                    continue

                trades_df = data[asset][strategy][top_n]['trades']
                summary = data[asset][strategy][top_n]['summary']

                if trades_df.empty:
                    continue

                # Calcola equity curve
                equity = calculate_equity_curve(trades_df)

                # Plot
                label = f"Top {top_n} (Return: {summary['total_return_pct']:.1f}%, Calmar: {summary['calmar_ratio']:.1f})"
                ax.plot(equity.index, equity.values, label=label, linewidth=2, alpha=0.7)

            # Linea Buy & Hold di riferimento
            if asset in oos_report and 'buy_hold' in oos_report[asset]:
                bh_return = oos_report[asset]['buy_hold']
                ax.axhline(y=100 + bh_return, color='black', linestyle='--', linewidth=1.5,
                          label=f'Buy & Hold: {bh_return:.1f}%', alpha=0.6)

            ax.set_title(f'{strategy} Strategy', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Equity (Base 100)', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, f'{asset}_equity_curves.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[OK] Salvato: {asset}_equity_curves.png")

    print()

def plot_trade_distributions(data):
    """
    Crea istogrammi delle distribuzioni dei trade.
    """
    print("="*80)
    print("GENERAZIONE TRADE DISTRIBUTIONS")
    print("="*80)

    for asset in ASSETS:
        if asset not in data:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Trade Distributions - {ASSET_NAMES.get(asset, asset)}', fontsize=16, fontweight='bold')

        for strategy_idx, strategy in enumerate(STRATEGIES):
            if strategy not in data[asset] or 1 not in data[asset][strategy]:
                continue

            # Usa solo il top 1 candidate per le distribuzioni
            trades_df = data[asset][strategy][1]['trades']

            if trades_df.empty:
                continue

            row = strategy_idx

            # 1. PnL Distribution
            ax = axes[row, 0]
            ax.hist(trades_df['pnl_net'], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
            ax.set_title(f'{strategy} - PnL Distribution', fontweight='bold')
            ax.set_xlabel('PnL Net (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # 2. Holding Time Distribution
            ax = axes[row, 1]
            ax.hist(trades_df['holding_time'], bins=20, alpha=0.7, edgecolor='black', color='orange')
            ax.set_title(f'{strategy} - Holding Time', fontweight='bold')
            ax.set_xlabel('Days')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # 3. MFE vs MAE
            ax = axes[row, 2]
            ax.scatter(trades_df['mae'], trades_df['mfe'], alpha=0.6, s=50)
            ax.plot([0, trades_df['mae'].max()], [0, trades_df['mae'].max()],
                   'r--', linewidth=1, alpha=0.5, label='MFE = MAE')
            ax.set_title(f'{strategy} - MFE vs MAE', fontweight='bold')
            ax.set_xlabel('MAE (%)')
            ax.set_ylabel('MFE (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, f'{asset}_trade_distributions.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[OK] Salvato: {asset}_trade_distributions.png")

    print()

def plot_performance_heatmap(data):
    """
    Crea heatmap delle performance multi-asset.
    """
    print("="*80)
    print("GENERAZIONE PERFORMANCE HEATMAP")
    print("="*80)

    # Raccogli dati per heatmap
    performance_data = []

    for asset in ASSETS:
        if asset not in data:
            continue

        for strategy in STRATEGIES:
            if strategy not in data[asset] or 1 not in data[asset][strategy]:
                continue

            summary = data[asset][strategy][1]['summary']

            performance_data.append({
                'Asset': ASSET_NAMES.get(asset, asset),
                'Strategy': strategy,
                'Return (%)': summary['total_return_pct'],
                'Calmar Ratio': summary['calmar_ratio'],
                'Max DD (%)': summary['max_drawdown_pct']
            })

    if not performance_data:
        print("[WARN] Nessun dato disponibile per heatmap")
        return

    df = pd.DataFrame(performance_data)

    # Crea 3 heatmap (Return, Calmar, Max DD)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Performance Heatmap - Top 1 Candidates', fontsize=16, fontweight='bold')

    metrics = ['Return (%)', 'Calmar Ratio', 'Max DD (%)']
    cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn_r']  # _r per invertire colori su Max DD

    for idx, (metric, cmap) in enumerate(zip(metrics, cmaps)):
        pivot = df.pivot(index='Asset', columns='Strategy', values=metric)

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap,
                   center=0 if metric == 'Max DD (%)' else None,
                   cbar_kws={'label': metric}, ax=axes[idx], linewidths=0.5)

        axes[idx].set_title(metric, fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'performance_heatmap.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Salvato: performance_heatmap.png\n")

def plot_strategy_comparison(data):
    """
    Confronto diretto tra strategie Overnight vs Trend-Following.
    """
    print("="*80)
    print("GENERAZIONE STRATEGY COMPARISON")
    print("="*80)

    comparison_data = []

    for asset in ASSETS:
        if asset not in data:
            continue

        overnight_return = None
        trend_return = None

        if 'Overnight' in data[asset] and 1 in data[asset]['Overnight']:
            overnight_return = data[asset]['Overnight'][1]['summary']['total_return_pct']

        if 'Trend-Following' in data[asset] and 1 in data[asset]['Trend-Following']:
            trend_return = data[asset]['Trend-Following'][1]['summary']['total_return_pct']

        if overnight_return is not None and trend_return is not None:
            comparison_data.append({
                'Asset': ASSET_NAMES.get(asset, asset),
                'Overnight': overnight_return,
                'Trend-Following': trend_return
            })

    if not comparison_data:
        print("[WARN] Nessun dato disponibile per confronto strategie")
        return

    df = pd.DataFrame(comparison_data)

    # Bar chart comparativo
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['Overnight'], width, label='Overnight', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['Trend-Following'], width, label='Trend-Following', alpha=0.8)

    ax.set_xlabel('Asset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Comparison: Overnight vs Trend-Following (Top 1 Candidates)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Asset'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Aggiungi valori sopra le barre
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'strategy_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Salvato: strategy_comparison.png\n")

def generate_summary_report(data, oos_report):
    """
    Genera report testuale riassuntivo.
    """
    print("="*80)
    print("GENERAZIONE SUMMARY REPORT")
    print("="*80)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SUMMARY REPORT - TRADING MODEL BACKTEST RESULTS")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Results Directory: {RESULTS_DIR}")
    report_lines.append("="*80)
    report_lines.append("")

    for asset in ASSETS:
        if asset not in data:
            continue

        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"ASSET: {ASSET_NAMES.get(asset, asset)} ({asset})")
        report_lines.append(f"{'='*80}")

        # Buy & Hold reference
        if asset in oos_report and 'buy_hold' in oos_report[asset]:
            bh_return = oos_report[asset]['buy_hold']
            report_lines.append(f"\n[REF] Buy & Hold Return (OOS): {bh_return:.2f}%")

        for strategy in STRATEGIES:
            if strategy not in data[asset]:
                continue

            report_lines.append(f"\n--- {strategy} Strategy ---")

            for top_n in [1, 2, 3]:
                if top_n not in data[asset][strategy]:
                    continue

                summary = data[asset][strategy][top_n]['summary']
                trades_df = data[asset][strategy][top_n]['trades']
                stats = calculate_trade_statistics(trades_df)

                report_lines.append(f"\n  Top {top_n} Candidate:")
                report_lines.append(f"    Config: Window={summary['window_size']}, Features={summary['num_features']}")
                report_lines.append(f"    Total Return: {summary['total_return_pct']:.2f}%")
                report_lines.append(f"    Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
                report_lines.append(f"    Calmar Ratio: {summary['calmar_ratio']:.2f}")

                if stats:
                    report_lines.append(f"    Total Trades: {stats['total_trades']}")
                    report_lines.append(f"    Win Rate: {stats['win_rate']:.1f}%")
                    report_lines.append(f"    Profit Factor: {stats['profit_factor']:.2f}")
                    report_lines.append(f"    Avg Win: {stats['avg_win']:.2f}% | Avg Loss: {stats['avg_loss']:.2f}%")
                    report_lines.append(f"    Avg Holding Time: {stats['avg_holding_time']:.1f} days")

    report_lines.append("\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    # Salva report
    report_file = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"[OK] Salvato: summary_report.txt\n")

    # Stampa anche a console
    print("\n" + "\n".join(report_lines[:50]))  # Prime 50 righe
    if len(report_lines) > 50:
        print(f"\n... (vedi {report_file} per report completo)")

# =================================================================================
# MAIN
# =================================================================================

def main():
    """
    Main function - orchestra tutto il processo di visualizzazione.
    """
    print("\n" + "="*80)
    print("CSV RESULTS PLOTTER - XGBoost Trading Model")
    print("="*80)
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*80 + "\n")

    # 1. Carica dati
    data = load_csv_data(RESULTS_DIR)
    oos_report = load_oos_report(RESULTS_DIR)

    if not data:
        print("[ERROR] Nessun dato caricato. Verifica la directory dei risultati.")
        return

    # 2. Genera grafici
    plot_equity_curves(data, oos_report)
    plot_trade_distributions(data)
    plot_performance_heatmap(data)
    plot_strategy_comparison(data)

    # 3. Genera report
    generate_summary_report(data, oos_report)

    print("="*80)
    print("[OK] PROCESSO COMPLETATO")
    print("="*80)
    print(f"Tutti i grafici sono stati salvati in: {OUTPUT_DIR}")
    print(f"Report testuale: {os.path.join(OUTPUT_DIR, 'summary_report.txt')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
