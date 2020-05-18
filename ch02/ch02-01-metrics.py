import numpy as np
import pandas as pd

# -----------------------------------
# Regression
# -----------------------------------
# rmse

from sklearn.metrics import mean_squared_error

# y_true are the true values、y_pred are the predictions
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5532

# -----------------------------------
# Binary classification
# -----------------------------------
# Confusion matrix

from sklearn.metrics import confusion_matrix

# True values and predicted values are binary classifiers: either 0 or 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                              [fn, tn]])
print(confusion_matrix1)
# array([[3, 1],
#        [2, 2]])

# Can also be created using confusion_matrix() from scikit-learn's metrics, but
# beware that the arrangement of the confusion matrix elements may be different
confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)
# array([[2, 1],
#        [2, 3]])

# -----------------------------------
# accuracy

from sklearn.metrics import accuracy_score

# True values and predicted values are binary classifiers: either 0 or 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625

# -----------------------------------
# logloss

from sklearn.metrics import log_loss

# True values are binary (0 or 1), predicted values are probabilities
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7136

# -----------------------------------
# Multi-class classification
# -----------------------------------
# multi-class logloss

from sklearn.metrics import log_loss

# True values are 3-class classifiers, predicted values are probabilities for each class
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3626

# -----------------------------------
# Multi-label classification
# -----------------------------------
# mean_f1, macro_f1, micro_f1

from sklearn.metrics import f1_score

# For calculating performance metric of multi-label classification, it is easier to handle the true / predicted values as binary matrices of record x class
# True values - [[1,2], [1], [1,2,3], [2,3], [3]]
y_true = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [0, 0, 1]])

# Predicted values - [[1,3], [2], [1,3], [3], [3]]
y_pred = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

# mean_f1 is the mean of the F1-scores for each record
mean_f1 = np.mean([f1_score(y_true[i, :], y_pred[i, :]) for i in range(len(y_true))])

# macro_f1 is the mean of the F1-scores for each class
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])

# micro-f1 is the F1-score calculate using the true/predicted values for each record-class pair 
micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)
# 0.5933, 0.5524, 0.6250

# Can also be calculated using a scikit-learn method
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

# -----------------------------------
# Multi-class classification with ordered classes
# -----------------------------------
# quadratic weighted kappa

from sklearn.metrics import confusion_matrix, cohen_kappa_score


# Function for calculating quadratic weighted kappa
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij

    return 1.0 - numer / denom


# y_true is the true class list, y_pred is the predicted class list
y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

# Calculate the confusion matrix
c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

# Calculate quadratic weighted kappa
kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
# 0.6153

# Can also be calculated using a scikit-learn method
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# -----------------------------------
# Recommendation
# -----------------------------------
# MAP@K

# K=3, with 5 records and 4 class types
K = 3

# True values for each record
y_true = [[1, 2], [1, 2], [4], [1, 2, 3, 4], [3, 4]]

# Predicted values for each record - as K=3, usually predict order of 3 records for each class
y_pred = [[1, 2, 4], [4, 1, 2], [1, 4, 3], [1, 2, 3], [1, 2, 4]]


# Function to calculate the average precision for each record
def apk(y_i_true, y_i_pred):
    # Length of y_pred must be less than or equal to K, and all elements must be unique
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)


# Function for calculating MAP@K
def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])


# Calculate MAP@K
print(mapk(y_true, y_pred))
# 0.65

# Even if the number of true values is the same, if the order is different then the score should be different
print(apk(y_true[0], y_pred[0]))
print(apk(y_true[1], y_pred[1]))
# 1.0, 0.5833
