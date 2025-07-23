# src/config.py

import numpy as np

# --- File Paths ---
DATA_PATH = 'data/train.csv'
PLOTS_PATH = 'plots/'

# --- Data Splitting ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Feature Engineering & Preprocessing ---
TARGET_VARIABLE = 'SalePrice'
LOG_TARGET_VARIABLE = 'SalePrice_Log'

# Columns for specific imputation strategies
IMPUTE_NONE_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]

IMPUTE_ZERO_COLS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]

# --- Skewness Transformation ---
SKEW_THRESHOLD = 0.75

# --- Modeling ---
# Random Forest Hyperparameter Grid for GridSearchCV
RF_PARAM_GRID = {
    'n_estimators': [200, 300],
    'max_depth': [15, 25],
    'min_samples_leaf': [1, 3]
}