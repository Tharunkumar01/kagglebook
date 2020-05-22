# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y is the target values, and test_x is the test data
# stored in pandas DataFrames and Series (numpy arrays also used)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# Split the training data into training and validation data
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# xgboost implementation
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# Change the features and target values into format suitable for xgboost
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

# Set the hyperparameters
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50

# Train the model
# Pass the validation data to the model, and monitor how the score changes during training
# In watchlist put the training and validation data
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)

# Check the score using the validation data
va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# Output prediction (not a binary value but a probability)
pred = model.predict(dtest)

# -----------------------------------
# Monitor the scores for the training and validation data
# -----------------------------------
# Monitor the logless metric, set number of early stopping rounds to 20
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71,
          'eval_metric': 'logloss'}
num_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist,
                  early_stopping_rounds=20)

# Use the optimal decision tree to make predictions
pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
