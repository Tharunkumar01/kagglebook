import numpy as np
import pandas as pd

# -----------------------------------
# Optimization of out-of-fold threshold
# -----------------------------------
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# Prepartion for sample data creation
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# Assume that the true and predicted values are train_y and train_pred_prob, respectively
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# Find thresholds using cross-validation framework
thresholds = []
scores_tr = []
scores_va = []

kf = KFold(n_splits=4, random_state=71, shuffle=True)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_pred_prob)):
    tr_pred_prob, va_pred_prob = train_pred_prob[tr_idx], train_pred_prob[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # Objective function for optimization
    def f1_opt(x):
        return -f1_score(tr_y, tr_pred_prob >= x)

    # Optimize thresholds with training data, and evaluate with validation data
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    threshold = result['x'].item()
    score_tr = f1_score(tr_y, tr_pred_prob >= threshold)
    score_va = f1_score(va_y, va_pred_prob >= threshold)
    print(threshold, score_tr, score_va)

    thresholds.append(threshold)
    scores_tr.append(score_tr)
    scores_va.append(score_va)

# Apply mean of the fold thresholds to the test data 
threshold_test = np.mean(thresholds)
print(threshold_test)
