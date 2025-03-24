import numpy as np
from mlscore import score, score_c, score_r

def main():
    # Example 1: Using score_c for classification
    print("\nClassification Example using score_c")
    print("-"*30)
    
    y_true_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_binary = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    
    # This will automatically print the results
    metrics_binary = score_c(y_true_binary, y_pred_binary)
    
    # Example 2: Using score_r for regression
    print("\nRegression Example using score_r")
    print("-"*30)
    
    np.random.seed(42)
    y_true_reg = np.random.rand(10) * 10
    y_pred_reg = y_true_reg + np.random.randn(10) * 0.5
    
    # This will automatically print the results
    metrics_reg = score_r(y_true_reg, y_pred_reg)
    
    # Example 3: Using general score function
    print("\nGeneral Example using score")
    print("-"*30)
    
    y_true_multiclass = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred_multiclass = np.array([0, 1, 2, 3, 4, 0, 2, 2, 3, 4])
    
    # This will automatically print the results
    metrics_multiclass = score(y_true_multiclass, y_pred_multiclass)
    
    # Example 4: Getting metrics without printing
    print("\nGetting metrics without printing")
    print("-"*30)
    
    metrics_no_print = score_c(y_true_binary, y_pred_binary, print_results=False)
    print("Metrics dictionary:", metrics_no_print)

if __name__ == "__main__":
    main() 