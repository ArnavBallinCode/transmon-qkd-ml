import os
import pandas as pd
import matplotlib.pyplot as plt

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))

def main():
    root_dir = get_project_root()
    
    # 1. Load ML Predictions
    pred_path = os.path.join(root_dir, 'results', 'hardware', 'predicted_qber_per_qubit.csv')
    df_preds = pd.read_csv(pred_path)
    # df_preds has columns: qubit, predicted_qber_linear, predicted_qber_rf, predicted_qber_nn
    
    # 2. Load Hardware Runs
    hw_dir = os.path.join(root_dir, 'results', 'hardware')
    all_runs = []
    for n_val in [1000, 3000]:
        summary_file = os.path.join(hw_dir, f'N_{n_val}', f'hardware_qber_summary_{n_val}.csv')
        if os.path.exists(summary_file):
            df_hw = pd.read_csv(summary_file)
            df_hw['sample_size'] = n_val
            all_runs.append(df_hw)
            
    if not all_runs:
        print("No valid hardware runs with physical_qubit data found yet. (Waiting for N=1000 or N=3000)")
        return
        
    df_actual = pd.concat(all_runs, ignore_index=True)
    
    # 3. Merge Actual vs Predicted
    # hardware summary has 'physical_qubit', pred has 'qubit'
    df_merged = pd.merge(df_actual, df_preds, left_on='physical_qubit', right_on='qubit')
    
    # 4. Save validation table
    out_table_path = os.path.join(root_dir, 'results', 'tables', '5_ml_validation_results.csv')
    df_merged.to_csv(out_table_path, index=False)
    print(f"Saved validation table to {out_table_path}")
    
    # 5. Plot Actual vs Predicted
    plt.figure(figsize=(10, 8))
    
    plt.scatter(df_merged['qber'], df_merged['predicted_qber_linear'], label='Linear Regression', alpha=0.7, marker='s')
    plt.scatter(df_merged['qber'], df_merged['predicted_qber_rf'], label='Random Forest', alpha=0.7, marker='o')
    plt.scatter(df_merged['qber'], df_merged['predicted_qber_nn'], label='Neural Network', alpha=0.7, marker='^')
    
    # Plot y=x perfect prediction line
    max_val = max(df_merged['qber'].max(), df_merged[['predicted_qber_linear', 'predicted_qber_rf', 'predicted_qber_nn']].max().max())
    plt.plot([0, max_val*1.1], [0, max_val*1.1], 'r--', label='Perfect Prediction (y=x)')
    
    plt.title('ML Predicted QBER vs Actual Hardware QBER')
    plt.xlabel('Actual QBER (Hardware)')
    plt.ylabel('Predicted QBER (ML)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_fig_path = os.path.join(root_dir, 'results', 'figures', 'ml_validation_scatter.png')
    plt.savefig(out_fig_path)
    print(f"Saved validation plot to {out_fig_path}")

if __name__ == "__main__":
    main()
