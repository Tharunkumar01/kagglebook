# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y is the target values, and test_x is the test data
# stored in pandas DataFrames and Series (also possible to use numpy arrays)）

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
# lightgbm implementation
# -----------------------------------
import lightgbm as lgb
from sklearn.metrics import log_loss

# Change the features and target values into format suitable for lightgbm
lgb_train = lgb.Dataset(tr_x, tr_y)
lgb_eval = lgb.Dataset(va_x, va_y)

# Set the hyperparameters
params = {'objective': 'binary', 'seed': 71, 'verbose': 0, 'metrics': 'binary_logloss'}
num_round = 100

# Perform training
# Specify categorical features as a parameter
# Pass the validation data to the model, and monitor how the score changes during training
categorical_features = ['product', 'medical_info_b2', 'medical_info_b3']
model = lgb.train(params, lgb_train, num_boost_round=num_round,
                  categorical_feature=categorical_features,
                  valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval])

# Check score for validation data
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# Predictions
pred = model.predict(test_x)
