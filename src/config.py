import numpy as np

DATA_PATH = "data/train.csv"
PLOTS_PATH = "plots/"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_VARIABLE = "SalePrice"
LOG_TARGET_VARIABLE = "SalePrice_Log"

IMPUTE_NONE_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]

IMPUTE_ZERO_COLS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]

SKEW_THRESHOLD = 0.75

RF_PARAM_GRID = {
    "n_estimators": [200, 250, 300],
    "max_depth": [15, 25],
    "min_samples_leaf": [1, 3]
}