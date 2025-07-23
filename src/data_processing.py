# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from scipy.special import boxcox1p
import src.config as config

def load_data(path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}")
        return df
    except FileNotFoundError:
        print(f"Error: '{path}' not found.")
        exit()

def engineer_features(df):
    """Creates new features from existing ones."""
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                            df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    return df

def preprocess_data():
    """Main function to orchestrate the data preprocessing workflow."""
    df = load_data(config.DATA_PATH)

    # 1. Split data
    X = df.drop(['Id', config.TARGET_VARIABLE], axis=1)
    y = df[config.TARGET_VARIABLE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # 2. Log transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # 3. Handle Missing Values
    for col in config.IMPUTE_NONE_COLS:
        X_train[col] = X_train[col].fillna('None')
        X_test[col] = X_test[col].fillna('None')
    for col in config.IMPUTE_ZERO_COLS:
        X_train[col] = X_train[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    # Impute remaining with median/mode
    for col in X_train.columns:
        if X_train[col].isnull().any():
            if X_train[col].dtype == 'object':
                mode_val = X_train[col].mode()[0]
                X_train[col] = X_train[col].fillna(mode_val)
                X_test[col] = X_test[col].fillna(mode_val)
            else:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)

    # 4. Feature Engineering
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # 5. Transform Skewed Features
    numeric_feats = X_train.select_dtypes(include=np.number).columns
    skewed_feats = X_train[numeric_feats].apply(lambda x: skew(x.dropna()))
    highly_skewed_feats = skewed_feats[abs(skewed_feats) > config.SKEW_THRESHOLD].index
    for feat in highly_skewed_feats:
        X_train[feat] = boxcox1p(X_train[feat], 0.15)
        X_test[feat] = boxcox1p(X_test[feat], 0.15)

    # 6. One-Hot Encode and Align
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    
    print("Data preprocessing complete.")
    return X_train, X_test, y_train_log, y_test_log