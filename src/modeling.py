import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import src.config as config

def select_features(X_train, y_train, threshold="median"):
    fs_model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    fs_model.fit(X_train, y_train)
    selector = SelectFromModel(fs_model, prefit=True, threshold=threshold)
    return selector

def evaluate_model(y_true, y_pred, model_name):
    rmsle = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} Evaluation ---")
    print(f"Root Mean Squared Log Error (RMSLE): {rmsle:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}\n")
    return rmsle, r2

def train_linear_regression(X_train, y_train, X_test, y_test):
    print("--- Training Linear Regression ---")
    selector = select_features(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred = lr_model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred, "Linear Regression")
    
    coeff = pd.DataFrame({"Feature": selected_features, "Coefficient": lr_model.coef_})
    print("--- Linear Regression Coefficients ---")
    print(coeff.reindex(coeff.Coefficient.abs().sort_values(ascending=False).index).head(10))
    print("\n")

def train_random_forest(X_train, y_train, X_test, y_test):
    print("--- Training Random Forest ---")
    selector = select_features(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
        param_grid=config.RF_PARAM_GRID,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=1
    )
    grid_search.fit(X_train_selected, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test_selected)
    
    evaluate_model(y_test, y_pred, "Random Forest")
    
    importances = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("--- Top 10 Most Important Features (Random Forest) ---")
    print(importances.head(10))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importances.head(20), palette='viridis')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_PATH}rf_feature_importance.png')
    plt.close()