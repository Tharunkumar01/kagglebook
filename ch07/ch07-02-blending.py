# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y is the target values, and test_x is the test data
# stored in pandas DataFrames and Series (numpy arrays also used as well)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# Data for neural network
train_nn = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x_nn = train_nn.drop(['target'], axis=1)
train_y_nn = train_nn['target']
test_x_nn = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# Ensemble using predictions from hold-out data
# ----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_index = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_index]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_index]
tr_x_nn, va_x_nn = train_x_nn.iloc[tr_idx], train_x_nn.iloc[va_index]

# Assume Model1_1, Model1_2 and Model2 are specified in models.py
# For each class train using fit and output prediction probabilities using predict
from models import Model1Xgb, Model1NN, Model2Linear

# First level model
# Train using training data, output predictions for hold-out and test data
model_1a = Model1Xgb()
model_1a.fit(tr_x, tr_y, va_x, va_y)
va_pred_1a = model_1a.predict(va_x)
test_pred_1a = model_1a.predict(test_x)

model_1b = Model1NN()
model_1b.fit(tr_x_nn, tr_y, va_x_nn, va_y)
va_pred_1b = model_1b.predict(va_x_nn)
test_pred_1b = model_1b.predict(test_x_nn)

# Score when using hold-out data
print(f'logloss: {log_loss(va_y, va_pred_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(va_y, va_pred_1b, eps=1e-7):.4f}')

# Make predictions from hold-out and test data a feature and create data frame
va_x_2 = pd.DataFrame({'pred_1a': va_pred_1a, 'pred_1b': va_pred_1b})
test_x_2 = pd.DataFrame({'pred_1a': test_pred_1a, 'pred_1b': test_pred_1b})

# Second level model
# Trained using all hold-out data so cannot evaluate score
# In order to score, a method further cross validating hold-out data can be considered
model2 = Model2Linear()
model2.fit(va_x_2, va_y, None, None)
pred_test_2 = model2.predict(test_x_2)
