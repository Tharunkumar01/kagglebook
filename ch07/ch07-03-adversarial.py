# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# Data creation (just random data)
rand = np.random.RandomState(71)
train_x = pd.DataFrame(rand.uniform(0.0, 1.0, (10000, 2)), columns=['model1', 'model2'])
adv_train = pd.Series(rand.uniform(0.0, 1.0, 10000))
w = np.array([0.3, 0.7]).reshape(1, -1)
train_y = pd.Series((train_x.values * w).sum(axis=1) > 0.5)

# ---------------------------------
# adversarial stochastic blending
# ----------------------------------
# Use adversarial validation to calculate weights for averaging predicted values from models
# train_x: Predicted probabilities from each model (actually using results that have been ordered)
# train_y: Target values
# adv_train: Values that represent likelihood that training data was also test data

from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

n_sampling = 50  # Number of times to sample
frac_sampling = 0.5  # Fraction of training data to take when sampling


def score(x, data_x, data_y):
    # Use AUC as evaluation metric
    y_prob = data_x['model1'] * x + data_x['model2'] * (1 - x)
    return -roc_auc_score(data_y, y_prob)


# Repeatedly use sampling to calculate weights for weighted averaging
results = []
for i in range(n_sampling):
    # Perform sampling
    seed = i
    idx = pd.Series(np.arange(len(train_y))).sample(frac=frac_sampling, replace=False,
                                                    random_state=seed, weights=adv_train)
    x_sample = train_x.iloc[idx]
    y_sample = train_y.iloc[idx]

    # Want to use sampling data to find most optimum weights for weighted averaging
    # As there are constraints use the COBYLA algorithm
    init_x = np.array(0.5)
    constraints = (
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: 1.0 - x},
    )
    result = minimize(score, x0=init_x,
                      args=(x_sample, y_sample),
                      constraints=constraints,
                      method='COBYLA')
    results.append((result.x, 1.0 - result.x))

# Weights for model1 and model2 weighted averaging
results = np.array(results)
w_model1, w_model2 = results.mean(axis=0)
