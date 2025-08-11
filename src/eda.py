import matplotlib.pyplot as plt
import seaborn as sns
import src.config as config
import os

def run_eda(X_train, y_train_log):
    if not os.path.exists(config.PLOTS_PATH):
        os.makedirs(config.PLOTS_PATH)
        
    train_eda_df = X_train.copy()
    train_eda_df[config.LOG_TARGET_VARIABLE] = y_train_log
    
    # Plot distribution of log-transformed SalePrice
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train_log, kde=True, bins=50)
    plt.title('Log-Transformed SalePrice Distribution')
    plt.xlabel('Log(SalePrice)')
    plt.ylabel('Frequency')
    plt.savefig(f"{config.PLOTS_PATH}saleprice_log_distribution.png")
    plt.close()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    # Calculate correlation with the target variable
    corrmat = train_eda_df.corr(numeric_only=True)
    # Get top 15 features with highest absolute correlation
    top_corr_features = corrmat[config.LOG_TARGET_VARIABLE].abs().sort_values(ascending=False).head(15).index
    # Create heatmap of correlations among these top features
    sns.heatmap(train_eda_df[top_corr_features].corr(numeric_only=True), annot=True, cmap="viridis")
    plt.title("Top 15 Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{config.PLOTS_PATH}correlation_heatmap.png")
    plt.close()