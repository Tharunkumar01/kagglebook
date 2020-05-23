# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y contains the target values, test_x is the test data
# stored in pandas DataFrames and Series (numpy arrays also used)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# Save training and test datasets in their original form for explanations
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# Function to recover original training and test datasets
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# Store names of numerical variables to be converted in list
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

# -----------------------------------
# Standardization
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# Compute standardization parameters for multiple columns of the training data
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# Replace columns with standardized values
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# Compute standardization parameters for multiple columns from combined training and test data
scaler = StandardScaler()
scaler.fit(pd.concat([train_x[num_cols], test_x[num_cols]]))

# Replace columns with standardized values
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# Standardize training and test data separately (bad example)
scaler_train = StandardScaler()
scaler_train.fit(train_x[num_cols])
train_x[num_cols] = scaler_train.transform(train_x[num_cols])
scaler_test = StandardScaler()
scaler_test.fit(test_x[num_cols])
test_x[num_cols] = scaler_test.transform(test_x[num_cols])

# -----------------------------------
# Min-Max scaling
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import MinMaxScaler

# Compute parameters for min-max scaling for multiple columns of the training data
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# Replace columns with min-max scaled values
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# Logarithmic transformation
# -----------------------------------
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

# Take simple logarithm
x1 = np.log(x)

# Take logarithm of x+1
x2 = np.log1p(x)

# Apply original sign to logarithm taken of absolute value
x3 = np.sign(x) * np.log(np.abs(x))

# -----------------------------------
# Box-Cox transformation
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------

# Store only columns that take positive values in a list for transformation
# Note: when including missing values it is necessary to use (~(train_x[c] <= 0.0)).all() etc.
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

from sklearn.preprocessing import PowerTransformer

# Fit Box-Cox transformation to the columns with positive values in the training data
pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

# Replace columns with transformed data
train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

# -----------------------------------
# Yeo-Johnson transformation
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------

from sklearn.preprocessing import PowerTransformer

# Compute parameters for Yeo-Johnnson transformation for multiple columns of the training data
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# Replace columns with transformed data
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])

# -----------------------------------
# Clipping
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
# Calculate 1% and 99% limits of each column of the training data
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

# Clip out values in the 1st and 99th percentiles
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

# -----------------------------------
# Binning
# -----------------------------------
x = [1, 7, 5, 4, 6, 3]

# Use cut() function in pandas for binning

# Case where you specify the number of bins
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - shows which of the three bins the converted values are in

# Case where you specify the bin ranges (<3.0, 3.0->5.0, >5.0）
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - shows which of the three bins the converted values are in

# -----------------------------------
# Rank transformation
# -----------------------------------
x = [10, 20, 30, 0, 40, 40]

# Use rank() function in pandas for rank transformation
rank = pd.Series(x).rank()
print(rank.values)
# First value is 1, mean rank is given for values in equal position
# [2. 3. 4. 1. 5.5 5.5]

# Also possible to to apply argsort() function in numpy twice to make rank transformation
order = np.argsort(x)
rank = np.argsort(order)
print(rank)
# First value is zero, equal position values are ordered by whichever is first
# [1 2 3 0 4 5]

# -----------------------------------
# RankGauss
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import QuantileTransformer

# Compute parameters for Rank-Gauss transformation for multiple columns of the training data
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# Replace columns with transformed data
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])
