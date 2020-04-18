# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y is the target values, and test_x is the test data
# stored in pandas DataFrames and Series (also possible to use numpy arrays)）
# Load one-hot encoded data

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# Split the training data into training and validation data
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# Linear model implementation
# -----------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# Data scaling
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x)
va_x = scaler.transform(va_x)
test_x = scaler.transform(test_x)

# Construction and training of linear model
model = LogisticRegression(C=1.0)
model.fit(tr_x, tr_y)

# Check score for validation data
# Use predict_proba to output probabilities. (predict outputs binary class predicitons)
va_pred = model.predict_proba(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# Predictions
pred = model.predict(test_x)
