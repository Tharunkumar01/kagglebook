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

# As time-series data assume a period variable is set that changes with time
train_x['period'] = np.arange(0, len(train_x)) // (len(train_x) // 4)
train_x['period'] = np.clip(train_x['period'], 0, 3)
test_x['period'] = 4

# -----------------------------------
# Hold-out method for time-series data
# -----------------------------------
# Partition using the period variable as the basis (0 to 3 are the training data, 4 is the test data)
# Here for within the training data period 3 is used for validation and periods 0 to 2 are used for training
is_tr = train_x['period'] < 3
is_va = train_x['period'] == 3
tr_x, va_x = train_x[is_tr], train_x[is_va]
tr_y, va_y = train_y[is_tr], train_y[is_va]

# -----------------------------------
# Cross validation for time-series data (use method that follows time)
# -----------------------------------
# Partition using the period variable as the basis (0 to 3 are the training data, 4 is the test data)
# Periods 1, 2 and 3 are each used for cross-validation, and the preceding periods are used for training

va_period_list = [1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] < va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]

# (For reference) Using TimeSeriesSplit() function is difficult as only the order of the data can be used
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=4)
for tr_idx, va_idx in tss.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# Cross validation for time-series data (method to simply partition by time)
# -----------------------------------
# Partition using the period variable as the basis (0 to 3 are the training data, 4 is the test data)
# Periods 1, 2 and 3 are each used for cross-validation, and the preceding periods are used for training

va_period_list = [0, 1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] != va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]
