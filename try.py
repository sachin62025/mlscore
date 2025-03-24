from mlscore import score_c
import numpy as np

# Create sample data
y_true_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
y_pred_binary = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])

# This will automatically print the results
metrics_binary = score_c(y_true_binary, y_pred_binary,  print_results=False)

# If you want to see the metrics dictionary as well
print("\nMetrics Dictionary:")
print(metrics_binary['Accuracy'])


