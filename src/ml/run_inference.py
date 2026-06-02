import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))

def load_models(root_dir):
    """
    Load the trained ML models from the models/ directory.
    """
    models_dir = os.path.join(root_dir, 'models')
    print(f"Loading models from {models_dir}...")
    
    lr_model = joblib.load(os.path.join(models_dir, 'linear_regression.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
    nn_model = joblib.load(os.path.join(models_dir, 'neural_network.pkl'))
    
    return lr_model, rf_model, nn_model

def load_and_merge_hardware_data(root_dir):
    """
    Load the real IBM Marrakesh calibration and readout error data and merge them.
    """
    calib_path = os.path.join(root_dir, 'data', 'hardware', 'ibm_marrakesh_qubit_calibration.csv')
    ro_path = os.path.join(root_dir, 'data', 'hardware', 'ibm_marrakesh_readout_errors.csv')
    
    df_calib = pd.read_csv(calib_path)
    df_ro = pd.read_csv(ro_path)
    
    df_hw = pd.merge(df_calib, df_ro, on='qubit')
    # Drop rows with NaN in essential columns
    df_hw = df_hw.dropna(subset=['t1_us', 't2_us', 'readout_error'])
    return df_hw

def feature_engineering(df_hw):
    """
    Create p_relax, p_dephase, and p_readout features.
    p_relax is proportional to 1/t1, normalized to [0,1].
    p_dephase is proportional to 1/t2, normalized to [0,1].
    p_readout is the readout_error (already a probability, no normalization needed).
    """
    df_hw['inv_t1'] = 1.0 / df_hw['t1_us']
    df_hw['inv_t2'] = 1.0 / df_hw['t2_us']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df_hw[['inv_t1', 'inv_t2']])
    
    df_hw['p_relax'] = scaled[:, 0]
    df_hw['p_dephase'] = scaled[:, 1]
    df_hw['p_readout'] = df_hw['readout_error']
    
    return df_hw

def run_inference(df_hw, lr_model, rf_model, nn_model):
    """
    Run the ML models on the engineered hardware features.
    """
    X_hw = df_hw[['p_relax', 'p_dephase', 'p_readout']]
    
    df_hw['predicted_qber_linear'] = lr_model.predict(X_hw)
    df_hw['predicted_qber_rf'] = rf_model.predict(X_hw)
    df_hw['predicted_qber_nn'] = nn_model.predict(X_hw)
    
    return df_hw

def save_plots(df_hw, root_dir):
    """
    Generate and save evaluation plots for the predicted QBERs.
    We will use the Random Forest predictions as the primary metric for plots, 
    as it is highly accurate on the non-linear boundaries.
    """
    figures_dir = os.path.join(root_dir, 'results', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Predicted QBER vs Qubit Index
    plt.figure(figsize=(12, 6))
    plt.scatter(df_hw['qubit'], df_hw['predicted_qber_rf'], alpha=0.7)
    plt.title('Predicted QBER vs Qubit Index (Random Forest)')
    plt.xlabel('Qubit Index')
    plt.ylabel('Predicted QBER')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(figures_dir, 'predicted_qber_vs_qubit.png'))
    plt.close()
    
    # 2. Histogram of Predicted QBER
    plt.figure(figsize=(10, 6))
    plt.hist(df_hw['predicted_qber_rf'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Predicted QBER (Random Forest)')
    plt.xlabel('Predicted QBER')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(figures_dir, 'predicted_qber_histogram.png'))
    plt.close()
    
    # 3. Top 10 Worst Qubits by Predicted QBER
    worst_10 = df_hw.sort_values(by='predicted_qber_rf', ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(worst_10['qubit'].astype(str), worst_10['predicted_qber_rf'], color='salmon')
    plt.title('Top 10 Worst Qubits by Predicted QBER')
    plt.xlabel('Qubit Index')
    plt.ylabel('Predicted QBER')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(figures_dir, 'worst_10_qubits.png'))
    plt.close()
    
    # 4. Top 10 Best Qubits by Predicted QBER
    best_10 = df_hw.sort_values(by='predicted_qber_rf', ascending=True).head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(best_10['qubit'].astype(str), best_10['predicted_qber_rf'], color='lightgreen')
    plt.title('Top 10 Best Qubits by Predicted QBER')
    plt.xlabel('Qubit Index')
    plt.ylabel('Predicted QBER')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(figures_dir, 'best_10_qubits.png'))
    plt.close()

def main():
    root_dir = get_project_root()
    
    lr_model, rf_model, nn_model = load_models(root_dir)
    
    print("Loading and merging IBM hardware data...")
    df_hw = load_and_merge_hardware_data(root_dir)
    
    print("Performing feature engineering...")
    df_hw = feature_engineering(df_hw)
    
    print("Running inference for all qubits...")
    df_hw = run_inference(df_hw, lr_model, rf_model, nn_model)
    
    # Select requested columns
    final_cols = [
        'qubit', 't1_us', 't2_us', 'readout_error', 
        'p_relax', 'p_dephase', 'p_readout', 
        'predicted_qber_linear', 'predicted_qber_rf', 'predicted_qber_nn'
    ]
    df_final = df_hw[final_cols]
    
    # Save the dataframe
    results_hw_dir = os.path.join(root_dir, 'results', 'hardware')
    os.makedirs(results_hw_dir, exist_ok=True)
    out_csv = os.path.join(results_hw_dir, 'predicted_qber_per_qubit.csv')
    df_final.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    print("Generating and saving plots...")
    save_plots(df_final, root_dir)
    print(f"Plots saved to {os.path.join(root_dir, 'results', 'figures')}")
    
    # Calculate statistics for all models
    models_stats = {}
    for model_name, col in [('Linear Regression', 'predicted_qber_linear'), 
                            ('Random Forest', 'predicted_qber_rf'), 
                            ('Neural Network', 'predicted_qber_nn')]:
        best_qubit = df_final.loc[df_final[col].idxmin(), 'qubit']
        worst_qubit = df_final.loc[df_final[col].idxmax(), 'qubit']
        models_stats[model_name] = {
            'Best Qubit': int(best_qubit),
            'Worst Qubit': int(worst_qubit),
            'Mean QBER': df_final[col].mean(),
            'Median QBER': df_final[col].median(),
            'Std QBER': df_final[col].std()
        }
    
    # Markdown Summary Table
    markdown_table = f"""
### Prediction Summary (All Models)

| Model | Best Qubit | Worst Qubit | Mean QBER | Median QBER | Std Deviation |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for model_name, stats in models_stats.items():
        markdown_table += f"| {model_name} | Qubit {stats['Best Qubit']} | Qubit {stats['Worst Qubit']} | {stats['Mean QBER']:.4f} | {stats['Median QBER']:.4f} | {stats['Std QBER']:.4f} |\n"
    
    print("\\n=============================================")
    print("MARKDOWN SUMMARY TABLE:")
    print("=============================================")
    print(markdown_table)
    print("=============================================")

if __name__ == "__main__":
    main()
