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

# Class for training and making predictions using xgboost
import xgboost as xgb


class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# hold-out method
# -----------------------------------
# Partition validation data for hold-out method

from sklearn.model_selection import train_test_split

# Use train_test_split function for partitioning
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,
                                          test_size=0.25, random_state=71, shuffle=True)

# -----------------------------------
# Perform validation with hold-out method

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Assume Model class has been predefined
# Model class performs fitting and returns predicted probabilities for each outcome

# Use train_test_split function for partitioning
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,
                                          test_size=0.25, random_state=71, shuffle=True)

# Train the model, output predictions and calculate score
model = Model()
model.fit(tr_x, tr_y, va_x, va_y)
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(score)

# -----------------------------------
# Use the KFold class to partition validation data for hold-out method

from sklearn.model_selection import KFold

# Use KFold class to parition for hold-out method
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# Cross validation
# -----------------------------------
# Partition data for cross-validation

from sklearn.model_selection import KFold

# Use KFold class for partitioning for cross-valdiation
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# Perform cross-validation

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# It is assumed that the Model class has been predefined
# Model class performs fitting and returns predicted probabilities for each outcome

scores = []

# Use KFold class for partitioning for cross-valdiation
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # Train the model, output predictions and calculate score
    model = Model()
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# Take average of scores for each fold
print(np.mean(scores))

# -----------------------------------
# Stratified K-Fold
# -----------------------------------
from sklearn.model_selection import StratifiedKFold

# Use the StratifiedKFold class to perform partitioning into stratified folds
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x, train_y):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# GroupKFold
# -----------------------------------
# It is assumed that the data has the same users repeated 4 times
train_x['user_id'] = np.arange(0, len(train_x)) // 4
# -----------------------------------

from sklearn.model_selection import KFold, GroupKFold

# Partition taking the user_id column to be the customer ID
user_id = train_x['user_id']
unique_user_ids = user_id.unique()

# Use the KFold class and partition using the customer ID
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_group_idx, va_group_idx in kf.split(unique_user_ids):
    # Parition using the customer ID (into data for training and validation)
    tr_groups, va_groups = unique_user_ids[tr_group_idx], unique_user_ids[va_group_idx]

    # Partition records based on whether the customer ID is in train/valid
    is_tr = user_id.isin(tr_groups)
    is_va = user_id.isin(va_groups)
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]

# (For reference)GroupKFold is difficult to use as you cannot shuffle or specify the random number seed
kf = GroupKFold(n_splits=4)
for tr_idx, va_idx in kf.split(train_x, train_y, user_id):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# leave-one-out
# -----------------------------------
# Assume that there are only 100 data
train_x = train_x.iloc[:100, :].copy()
# -----------------------------------
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for tr_idx, va_idx in loo.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
