import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from scipy.special import boxcox1p
import src.config as config

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Data loaded from {path}")
        return df
    except FileNotFoundError:
        print(f"Error: {path} not found")
        exit()

def engineer_features(df):
    df["TotalSF"] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["TotalBathrooms"] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                            df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
    return df

def preprocess_data():
    df = load_data(config.DATA_PATH)
    df = engineer_features(df)
    
    X = df.drop(["Id", config.TARGET_VARIABLE], axis=1)
    y = df[config.TARGET_VARIABLE]
    
    # Handle missing values and skewness before splitting
    # Impute missing values
    for col in config.IMPUTE_NONE_COLS:
        if col in X.columns:
            X[col] = X[col].fillna('None')
            
    for col in config.IMPUTE_ZERO_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    # Impute remaining NaNs with the median of the column
    X = X.fillna(X.median(numeric_only=True))

    # Log transform skewed features
    numeric_feats = X.dtypes[X.dtypes != "object"].index
    skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > config.SKEW_THRESHOLD]
    skewed_feats = skewed_feats.index
    X[skewed_feats] = np.log1p(X[skewed_feats])
    
    # One-hot encode categorical features
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Align columns between train and test sets
    train_cols = X_train.columns
    test_cols = X_test.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
        
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
        
    X_test = X_test[train_cols]
    
    return X_train, X_test, y_train_log, y_test_log