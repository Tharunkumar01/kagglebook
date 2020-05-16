# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y is the target values, and test_x is the test data
# stored in pandas DataFrames and Series (numpy arrays also used as well)

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# Use argsort to do index sort
# ---------------------------------
# Arrays can be ordered using index sort into ascending and descending order with argsort
ary = np.array([10, 20, 30, 0])
idx = ary.argsort()
print(idx)  # Ascending order - [3 0 1 2]
print(idx[::-1])  # Descending order - [2 1 0 3]

print(ary[idx[::-1][:3]])  # Output best three - [30, 20, 10]

# ---------------------------------
# Correlation coefficient
# ---------------------------------
import scipy.stats as st

# Correlation coefficient
corrs = []
for c in train_x.columns:
    corr = np.corrcoef(train_x[c], train_y)[0, 1]
    corrs.append(corr)
corrs = np.array(corrs)

# Spearman's rank correlation coefficient
corrs_sp = []
for c in train_x.columns:
    corr_sp = st.spearmanr(train_x[c], train_y).correlation
    corrs_sp.append(corr_sp)
corrs_sp = np.array(corrs_sp)

# Output in order to top importance (maximum of top 5)
# Using np.argsort, you can get the indices of the ordered values
idx = np.argsort(np.abs(corrs))[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

idx2 = np.argsort(np.abs(corrs_sp))[::-1]
top_cols2, top_importances2 = train_x.columns.values[idx][:5], corrs_sp[idx][:5]
print(top_cols2, top_importances2)

# ---------------------------------
# Chi-square statistic
# ---------------------------------
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Chi-square statistic
x = MinMaxScaler().fit_transform(train_x)
c2, _ = chi2(x, train_y)

# Output in order to top importance (maximum of top 5)
idx = np.argsort(c2)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

# ---------------------------------
# Mutual information
# ---------------------------------
from sklearn.feature_selection import mutual_info_classif

# Mutual information
mi = mutual_info_classif(train_x, train_y)

# Output in order to top importance (maximum of top 5)
idx = np.argsort(mi)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)
