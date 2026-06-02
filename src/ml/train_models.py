import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))

def train_and_save_models():
    root_dir = get_project_root()
    dataset_path = os.path.join(root_dir, 'data', 'generated', 'qber_dataset_full.csv')
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    X = df[['p_relax', 'p_dephase', 'p_readout']]
    y = df['avg_qber']
    
    print("Training Linear Regression...")
    lr_model = LinearRegression().fit(X, y)
    joblib.dump(lr_model, os.path.join(models_dir, 'linear_regression.pkl'))
    
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    
    print("Training Neural Network...")
    nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42).fit(X, y)
    joblib.dump(nn_model, os.path.join(models_dir, 'neural_network.pkl'))
    
    print(f"Models successfully trained and saved to {models_dir}")

if __name__ == "__main__":
    train_and_save_models()
