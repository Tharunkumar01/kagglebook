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

# Data for neural network
train_nn = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x_nn = train_nn.drop(['target'], axis=1)
train_y_nn = train_nn['target']
test_x_nn = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# Stacking
# ----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Assume Model1Xgb, Model1NN and Model2Linear are specified in models.py
# For each class train using fit and output prediction probabilities using predict

from models import Model1Xgb, Model1NN, Model2Linear


# Function that returns predictions for training data without knowing the target values, and predictions for the test data
def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=71)

    # Train and make predictions using cross validation, save indices of predictions
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # Link using predictions from validation data, then return to original order
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # Take average of predictions from test data
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test


# First level model
# pred_train_1a, pred_train_1b are predictions from training data using cross validation
# pred_test_1a, pred_test_1b are predictions from test data
model_1a = Model1Xgb()
pred_train_1a, pred_test_1a = predict_cv(model_1a, train_x, train_y, test_x)

model_1b = Model1NN()
pred_train_1b, pred_test_1b = predict_cv(model_1b, train_x_nn, train_y, test_x_nn)

# Score for first level model
print(f'logloss: {log_loss(train_y, pred_train_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(train_y, pred_train_1b, eps=1e-7):.4f}')

# Make predictions a feature and create a data frame
train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})

# Second level model
# pred_train_2 are predictions from training data using cross validation via second level model
# pred_test_2 are predictions from test data via second level model
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv(model_2, train_x_2, train_y, test_x_2)
print(f'logloss: {log_loss(train_y, pred_train_2, eps=1e-7):.4f}')
