import numpy as np
import pandas as pd

# -----------------------------------
# Optimal threshold
# -----------------------------------
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# Prepartion for sample data creation
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# Assume that the true and predicted values are train_y and train_pred_prob, respectively
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# When the threshold is 0.5, F1 is 0.722
init_threshold = 0.5
init_score = f1_score(train_y, train_pred_prob >= init_threshold)
print(init_threshold, init_score)


# Objective function for optimization
def f1_opt(x):
    return -f1_score(train_y, train_pred_prob >= x)


# Use scipy.optimize minimize() function to find optimal threshold
# F1 is 0.756 when obtained with the optimal threshold
result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
best_threshold = result['x'].item()
best_score = f1_score(train_y, train_pred_prob >= best_threshold)
print(best_threshold, best_score)
