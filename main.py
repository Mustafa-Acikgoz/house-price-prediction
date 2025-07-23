# main.py

import src.data_processing as dp
import src.eda as eda
import src.modeling as model

def main():
    """Main function to run the entire ML pipeline."""
    
    # Phase 1 & 2: Preprocessing, Feature Engineering, and Transformation
    X_train, X_test, y_train_log, y_test_log = dp.preprocess_data()
    
    # Phase 2 (cont.): Exploratory Data Analysis
    eda.run_eda(X_train, y_train_log)
    
    # Phase 3, 4, 5: Modeling and Evaluation
    model.train_linear_regression(X_train, y_train_log, X_test, y_test_log)
    model.train_random_forest(X_train, y_train_log, X_test, y_test_log)
    
    print("\n--- Project Execution Complete ---")

if __name__ == '__main__':
    main()