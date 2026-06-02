import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))

def generate_ml_model_comparison(root_dir):
    # This was previously evaluated on the dataset.
    # We can rebuild the table manually or by predicting.
    df = pd.read_csv(os.path.join(root_dir, 'results', 'hardware', 'predicted_qber_per_qubit.csv'))
    
    # We'll summarize the inference predictions
    summary = []
    for model, col in [('Linear Regression', 'predicted_qber_linear'), 
                       ('Random Forest', 'predicted_qber_rf'), 
                       ('Neural Network', 'predicted_qber_nn')]:
        summary.append({
            'Model': model,
            'Mean Predicted QBER': df[col].mean(),
            'Median Predicted QBER': df[col].median(),
            'Min Predicted QBER': df[col].min(),
            'Max Predicted QBER': df[col].max(),
            'Std Dev': df[col].std()
        })
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(root_dir, 'results', 'tables', '1_ml_model_comparison.csv'), index=False)
    return df_summary

def generate_ibm_calibration_summary(root_dir):
    df = pd.read_csv(os.path.join(root_dir, 'data', 'hardware', 'ibm_marrakesh_qubit_calibration.csv'))
    summary = df.describe()
    summary.to_csv(os.path.join(root_dir, 'results', 'tables', '2_ibm_calibration_summary.csv'))
    
    plt.figure(figsize=(10, 5))
    plt.hist(df['t1_us'].dropna(), bins=20, alpha=0.6, label='T1 (us)')
    plt.hist(df['t2_us'].dropna(), bins=20, alpha=0.6, label='T2 (us)')
    plt.title('IBM Marrakesh Coherence Times Distribution')
    plt.xlabel('Time (us)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(root_dir, 'results', 'figures', 'coherence_times_dist.png'))
    plt.close()

def generate_hardware_bb84_summary(root_dir):
    hw_dir = os.path.join(root_dir, 'results', 'hardware')
    
    # Process all available N summaries
    all_dfs = []
    for n_val in [100, 500, 1000, 3000]:
        file_path = os.path.join(hw_dir, f'N_{n_val}', f'hardware_qber_summary_{n_val}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['sample_size'] = n_val
            all_dfs.append(df)
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        summary = combined_df.describe()
        summary.to_csv(os.path.join(root_dir, 'results', 'tables', '3_hardware_bb84_summary.csv'))
        
        plt.figure(figsize=(10, 6))
        
        # Plot a line for each sample size
        colors = {100: 'blue', 500: 'green', 1000: 'orange', 3000: 'red'}
        for n_val, group in combined_df.groupby('sample_size'):
            plt.plot(group['run_id'], group['qber'], marker='o', linestyle='-', 
                     color=colors.get(n_val, 'purple'), label=f'N={n_val}')
            
        plt.title('Hardware BB84 QBER across Runs')
        plt.xlabel('Run ID')
        plt.ylabel('QBER')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        # Ensure x-ticks only show integer run IDs (1, 2, 3, 4, 5)
        run_ids = combined_df['run_id'].unique()
        plt.xticks(run_ids)
        plt.savefig(os.path.join(root_dir, 'results', 'figures', 'hardware_qber_trend.png'))
        plt.close()
    else:
        print("Hardware summary files not available.")

def generate_predicted_qber_summary(root_dir):
    df = pd.read_csv(os.path.join(root_dir, 'results', 'hardware', 'predicted_qber_per_qubit.csv'))
    
    # Sort and take top 10 best and worst
    top_10_best = df.sort_values(by='predicted_qber_rf').head(10)
    top_10_worst = df.sort_values(by='predicted_qber_rf', ascending=False).head(10)
    
    combined = pd.concat([
        top_10_best[['qubit', 'predicted_qber_rf']].assign(Category='Best'),
        top_10_worst[['qubit', 'predicted_qber_rf']].assign(Category='Worst')
    ])
    combined.to_csv(os.path.join(root_dir, 'results', 'tables', '4_predicted_qber_summary.csv'), index=False)

def main():
    root_dir = get_project_root()
    os.makedirs(os.path.join(root_dir, 'results', 'tables'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'results', 'figures'), exist_ok=True)
    
    print("Generating ML model comparison table...")
    generate_ml_model_comparison(root_dir)
    
    print("Generating IBM calibration summary...")
    generate_ibm_calibration_summary(root_dir)
    
    print("Generating Hardware BB84 summary (if data exists)...")
    generate_hardware_bb84_summary(root_dir)
    
    print("Generating predicted QBER summary...")
    generate_predicted_qber_summary(root_dir)
    
    print("All tables and figures generated in results/ directory!")

if __name__ == "__main__":
    main()
