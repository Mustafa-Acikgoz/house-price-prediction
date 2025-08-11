import src.data_processing as dp
import src.eda as eda
import src.modeling as model

def main():
    # Preprocessing, feature engineering, and transformations
    X_train, X_test, y_train_log, y_test_log = dp.preprocess_data()
    
    # EDA
    eda.run_eda(X_train, y_train_log)
    
    # Modelling and evaluation
    model.train_linear_regression(X_train, y_train_log, X_test, y_test_log)
    model.train_random_forest(X_train, y_train_log, X_test, y_test_log)
    
    print("Project Execution Completed")

if __name__ == "__main__":
    main()