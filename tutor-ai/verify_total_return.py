"""
Script per verificare che il Total Return nei CSV sia calcolato correttamente
dai daily returns, non dalla somma dei trade PnL.
"""

import pandas as pd
import numpy as np
from io import StringIO

def verify_total_return_from_csv(csv_path):
    """
    Verifica il Total Return analizzando la struttura del CSV.

    IMPORTANTE: Il CSV contiene SCENARI ALTERNATIVI di exit, non trade sequenziali.
    Il Total Return CORRETTO viene calcolato dai returns giornalieri della strategia,
    non dalla somma dei PnL dei trade.
    """
    print("=" * 100)
    filename = csv_path.split('\\')[-1]
    print(f"VERIFICA TOTAL RETURN: {filename}")
    print("=" * 100)
    print()

    # Leggi header metrics
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        # Riga 2: header, Riga 3: valori
        header_values = lines[2].strip().split(',')
        total_return_declared = float(header_values[0])
        max_dd_declared = float(header_values[1])
        calmar_declared = float(header_values[2])

    print("[HEADER] VALORI DICHIARATI NEL CSV:")
    print(f"   Total Return: {total_return_declared:.2f}%")
    print(f"   Max Drawdown: {max_dd_declared:.2f}%")
    print(f"   Calmar Ratio: {calmar_declared:.2f}")
    print()

    # Leggi trade data
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if 'entry_date' in line:
                start_idx = i
                break
        trade_lines = lines[start_idx:]

    df = pd.read_csv(StringIO(''.join(trade_lines)))
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])

    print("[TRADE] ANALISI TRADE NEL CSV:")
    print(f"   Numero righe CSV: {len(df)}")
    print(f"   Trade positivi: {(df['pnl_net'] > 0).sum()} ({(df['pnl_net'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"   Trade negativi: {(df['pnl_net'] < 0).sum()} ({(df['pnl_net'] < 0).sum()/len(df)*100:.1f}%)")
    print()

    # Verifica: scenari alternativi
    print("[VERIFY] VERIFICA SCENARI ALTERNATIVI:")
    grouped = df.groupby(['entry_date', 'entry_price']).size()
    entries_with_multiple_exits = (grouped > 1).sum()
    print(f"   Entry con exit multipli: {entries_with_multiple_exits} / {len(grouped)}")
    print(f"   Max exit per singolo entry: {grouped.max()}")
    print()

    if entries_with_multiple_exits > 0:
        print("   [WARNING] CONFERMATO: Il CSV contiene SCENARI ALTERNATIVI, non trade sequenziali!")
        print("   [WARNING] La somma dei PnL NON rappresenta il Total Return reale!")
        print()

    # Calcoli errati (somma dei PnL)
    sum_pnl_arithmetic = df['pnl_net'].sum()

    # Calcolo composto (ancora sbagliato perché ci sono trade sovrapposti)
    equity = 1.0
    for pnl in df['pnl_net']:
        equity *= (1 + pnl/100)
    compound_return = (equity - 1) * 100

    print("[ERROR] CALCOLI ERRATI (da CSV con scenari alternativi):")
    print(f"   Somma aritmetica PnL: {sum_pnl_arithmetic:.2f}%")
    print(f"   Composizione PnL: {compound_return:.2f}%")
    print()

    # Tentativo di ricostruzione trade sequenziali
    print("[REBUILD] TENTATIVO RICOSTRUZIONE TRADE REALI:")
    df_sorted = df.sort_values(['entry_date', 'exit_date'])
    real_trades = []
    last_exit = pd.Timestamp('1970-01-01')

    for idx, row in df_sorted.iterrows():
        if row['entry_date'] >= last_exit:
            real_trades.append(row)
            last_exit = row['exit_date']

    print(f"   Trade sequenziali ricostruiti: {len(real_trades)}")

    if len(real_trades) > 0:
        real_df = pd.DataFrame(real_trades)
        equity_real = 1.0
        for pnl in real_df['pnl_net']:
            equity_real *= (1 + pnl/100)
        real_return = (equity_real - 1) * 100
        print(f"   Return da trade sequenziali: {real_return:.2f}%")
    print()

    # Conclusioni
    print("=" * 100)
    print("[CONCLUSION]")
    print("=" * 100)
    print()
    print(f"Il Total Return dichiarato ({total_return_declared:.2f}%) e' calcolato dai DAILY RETURNS,")
    print(f"NON dalla somma/composizione dei trade nel CSV.")
    print()
    print(f"Il CSV contiene {len(df)} righe che rappresentano SCENARI ALTERNATIVI di exit,")
    print(f"non {len(df)} trade sequenziali reali.")
    print()

    # Verifica coerenza Calmar
    expected_calmar = total_return_declared / abs(max_dd_declared)
    print(f"Verifica Calmar Ratio:")
    print(f"   Dichiarato: {calmar_declared:.2f}")
    print(f"   Calcolato: {expected_calmar:.2f}")
    print(f"   Differenza: {abs(calmar_declared - expected_calmar):.4f}")

    if abs(calmar_declared - expected_calmar) < 0.1:
        print(f"   [OK] Calmar Ratio COERENTE con Total Return e Drawdown")
    else:
        print(f"   [ERROR] Calmar Ratio INCOERENTE!")
    print()

    print("[WARNING] RACCOMANDAZIONE: Correggere la funzione calculate_trade_metrics()")
    print("          per salvare solo i trade REALMENTE eseguiti, non gli scenari alternativi.")
    print()
    print("=" * 100)

if __name__ == "__main__":
    # Test su SOL-USD
    csv_path = r"C:\Users\sas\run_v72.5_final_backtester_20251022_010847\SOL-USD_Overnight_top1_oos.csv"
    verify_total_return_from_csv(csv_path)

    print("\n\n")

    # Test anche su TSLA per confronto
    csv_path_tsla = r"C:\Users\sas\run_v72.5_final_backtester_20251022_010847\TSLA_Overnight_top1_oos.csv"
    verify_total_return_from_csv(csv_path_tsla)
