# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y is the target values, and test_x is the test data
# stored in pandas DataFrames and Series (also possible to use numpy arrays)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

import xgboost as xgb


# The Model class to operate the code
class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        self.model = xgb.train(params, dtrain, num_round)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# Model training and prediction
# -----------------------------------
# Specify the model hyperparameters
params = {'param1': 10, 'param2': 100}

# Define the Model class
# The Model class has functions fit for training and predict for outputting predicted probabilities

# Define the Model class
model = Model(params)

# Use the training data to train the model
model.fit(train_x, train_y)

# Output predictions for the test data
pred = model.predict(test_x)

# -----------------------------------
# Validation
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Create an index in order to split the training and validation data
# Split the training data into 4, and keep aside 1 quarter for validation
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# Split the training data into training and validation data
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# Define the model
model = Model(params)

# Use the training data to train the model
# Depending on the model, validation data can be supplied at the same time in order to monitor the score
model.fit(tr_x, tr_y)

# Make predictions with the validation data, and calculate the score
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# -----------------------------------
# Cross validation
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Split the training data into 4, and keep aside 1 quarter for validation
# Change the quarter used for validation and evaluate the score 4 times
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = Model(params)
    model.fit(tr_x, tr_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# Output the mean cross validation score
print(f'logloss: {np.mean(scores):.4f}')
