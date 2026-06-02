import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('../data/qber_dataset_full.csv')
X = df[['p_relax', 'p_dephase', 'p_readout']]
y = df['avg_qber']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(), 
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42), 
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({'Model': name, 'MAE': round(mae, 5), 'RMSE': round(rmse, 5), 'R²': round(r2, 5)})
    
print(pd.DataFrame(results).to_markdown(index=False))
