import pandas as pd
import os

results_dir = 'C:/Users/sas/run_v72.7_final_backtester_20251023_225614'
csv_files = [f for f in os.listdir(results_dir) if f.endswith('_Overnight_top1_oos.csv')]

print('='*80)
print('ANALISI DISCREPANZA: Total Return vs Trade PnL')
print('='*80)

for csv_file in csv_files:
    asset = csv_file.split('_')[0]
    full_path = os.path.join(results_dir, csv_file)

    # Leggi header
    with open(full_path, 'r') as f:
        lines = f.readlines()
        header_values = lines[2].strip().split(',')
        total_return_header = float(header_values[0])

    # Leggi trades
    trades_df = pd.read_csv(full_path, skiprows=5)

    if len(trades_df) > 0:
        trades_df_sorted = trades_df.sort_values('exit_date')
        return_factors = 1 + (trades_df_sorted['pnl_net'] / 100)
        cumulative_pnl = (return_factors.cumprod().iloc[-1] - 1) * 100

        diff = total_return_header - cumulative_pnl
        if cumulative_pnl != 0:
            ratio = total_return_header / cumulative_pnl
        else:
            ratio = 0

        print(f'\n{asset}:')
        print(f'  Header Total Return: {total_return_header:.2f}%')
        print(f'  Trade PnL (compounding): {cumulative_pnl:.2f}%')
        print(f'  Differenza: {diff:.2f}%')
        print(f'  Ratio: {ratio:.1f}x')

        if ratio > 5:
            print(f'  [ALERT] Ratio anomalo > 5x!')
