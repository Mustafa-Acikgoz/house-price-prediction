# src/eda.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import src.config as config
import os

def run_eda(X_train, y_train_log):
    """Generates and saves all EDA plots."""
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    if not os.path.exists(config.PLOTS_PATH):
        os.makedirs(config.PLOTS_PATH)

    # Combine for easier plotting
    train_eda_df = X_train.copy()
    train_eda_df[config.LOG_TARGET_VARIABLE] = y_train_log
    
    # Plot target variable distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(y_train_log, kde=True, bins=50)
    plt.title('Log-Transformed SalePrice Distribution')
    plt.savefig(f'{config.PLOTS_PATH}saleprice_log_distribution.png')
    plt.close()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    corrmat = train_eda_df.corr()
    top_corr_features = corrmat[config.LOG_TARGET_VARIABLE].abs().sort_values(ascending=False).head(15).index
    sns.heatmap(train_eda_df[top_corr_features].corr(), annot=True, cmap="viridis")
    plt.title('Top 15 Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_PATH}correlation_heatmap.png')
    plt.close()

    print("EDA plots saved to 'plots/' directory.")