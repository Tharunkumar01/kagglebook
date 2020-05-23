# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y contains the target values, test_x is the test data
# stored in pandas DataFrames and Series (numpy arrays also used)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# Split the training data into training and validation data
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# Examples of custom metrics and objective functions in xgboost
# (Reference) https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# Convert features and target values into xgboost data structure
# Test features and target values are tr_x, tr_x, validation features and target values va_x, va_y
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)


# Custom objective function (logless in this case, which is equivalent to xgboost's 'binary:logistic')
def logregobj(preds, dtrain):
    labels = dtrain.get_label()  # Get labels of true values
    preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid function
    grad = preds - labels  # Gradient
    hess = preds * (1.0 - preds)  # Second derivative
    return grad, hess


# Custom metric (error rate in this case)
def evalerror(preds, dtrain):
    labels = dtrain.get_label()  # Get labels of true values
    return 'custom-error', float(sum(labels != (preds > 0.0))) / len(labels)


# Set hyperparameters
params = {'silent': 1, 'random_state': 71}
num_round = 50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# Train the model
bst = xgb.train(params, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)

# Unlike when binary:logistic is specified as the objective function,
# the values outputted are not probabilities so they need to be converted
pred_val = bst.predict(dvalid)
pred = 1.0 / (1.0 + np.exp(-pred_val))
logloss = log_loss(va_y, pred)
print(logloss)

# For reference results from normal training method
params = {'silent': 1, 'random_state': 71, 'objective': 'binary:logistic'}
bst = xgb.train(params, dtrain, num_round, watchlist)

pred = bst.predict(dvalid)
logloss = log_loss(va_y, pred)
print(logloss)
