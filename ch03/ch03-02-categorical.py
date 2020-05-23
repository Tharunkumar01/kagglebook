# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y contains the target values, test_x is the test data
# stored in pandas DataFrames and Series (numpy arrays also used)

train = pd.read_csv('../input/sample-data/train.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test.csv')

# Save training and test datasets in their original form for explanations
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# Function to recover original training and test datasets
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# Store names of categorical variables to be converted in list
cat_cols = ['sex', 'product', 'medical_info_b2', 'medical_info_b3']

# -----------------------------------
# One-hot encoding
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------

# Concatenate the training and test datasets, and apply one-hot encoding via get_dummies()
all_x = pd.concat([train_x, test_x])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# Resplit into training and test data
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)

# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# Encoding with the OneHotEncoder() function
ohe = OneHotEncoder(sparse=False, categories='auto')
ohe.fit(train_x[cat_cols])

# Create column names for dummy variables
columns = []
for i, c in enumerate(cat_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# Put created dummy variables into data frames
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[cat_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[cat_cols]), columns=columns)

# Join the remaining variables
train_x = pd.concat([train_x.drop(cat_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(cat_cols, axis=1), dummy_vals_test], axis=1)

# -----------------------------------
# Label encoding
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# Loop over the categorical variables and apply label encoding
for c in cat_cols:
    # Define labels based on the training data
    le = LabelEncoder()
    le.fit(train_x[c])
    train_x[c] = le.transform(train_x[c])
    test_x[c] = le.transform(test_x[c])

# -----------------------------------
# Feature hashing
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.feature_extraction import FeatureHasher

# Loop over the categorical variables and apply feature hashing
for c in cat_cols:
    # Using the FeatureHasher() function is slightly different from other encoders

    fh = FeatureHasher(n_features=5, input_type='string')
    # Convert the variable to a string and apply the FeatureHasher() function
    hash_train = fh.transform(train_x[[c]].astype(str).values)
    hash_test = fh.transform(test_x[[c]].astype(str).values)
    # Add to a data frame
    hash_train = pd.DataFrame(hash_train.todense(), columns=[f'{c}_{i}' for i in range(5)])
    hash_test = pd.DataFrame(hash_test.todense(), columns=[f'{c}_{i}' for i in range(5)])
    # Join with the original data frame
    train_x = pd.concat([train_x, hash_train], axis=1)
    test_x = pd.concat([test_x, hash_test], axis=1)

# Drop the original categorical variable columns
train_x.drop(cat_cols, axis=1, inplace=True)
test_x.drop(cat_cols, axis=1, inplace=True)

# -----------------------------------
# Frequency encoding
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
# Loop over the categorical variables and apply frequency encoding
for c in cat_cols:
    freq = train_x[c].value_counts()
    # Replace each categorical variable with its frequency of occurrence
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)

# -----------------------------------
# Target encoding
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# Loop over the categorical variables and apply target encoding
for c in cat_cols:
    # Calculate the average of the target for each categorical value in the training data
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # Replace the categorical variables in the test data
    test_x[c] = test_x[c].map(target_mean)

    # Prepare an array to store the converted training data
    tmp = np.repeat(np.nan, train_x.shape[0])

    # Split the training data
    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train_x):
        # Calculate the average of the target values for the out-of-fold categorical variables
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        # Store the converted values temporarily in an array
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    # Replace the original data with the converted values
    train_x[c] = tmp

# -----------------------------------
# Target encoding - for each fold of cross validation
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# Apply target encoding for each cross validation fold
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

    # Split the validation data off from the training data
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # Loop over the categorical variables and apply target encoding
    for c in cat_cols:
        # Calculate the average of the target for each categorical value in the training data
        data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # Replace the categorical variables in the validation data
        va_x.loc[:, c] = va_x[c].map(target_mean)

        # Prepare an array to store the converted training data
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            # Calculate the average of the target values for the out-of-fold categorical variables
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # Store the converted values temporarily in an array
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

        tr_x.loc[:, c] = tmp

    # Remember to save the encoded features so you can come back and read the data later if necessary

# -----------------------------------
# Target encoding - when the cross validation and target encoded folds need to be partitioned
# -----------------------------------
# Load the data
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# Define the cross validation folds
kf = KFold(n_splits=4, shuffle=True, random_state=71)

# Loop over the categorical variables and apply target encoding
for c in cat_cols:

    # Add the target values
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    # Store the converted values temporarily in an array
    tmp = np.repeat(np.nan, train_x.shape[0])

    # Split off the cross validation 
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        # Calculate the average of the target values for each category for the training data
        target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()
        # For the validation data, store the converted values temporarily in an array
        tmp[va_idx] = train_x[c].iloc[va_idx].map(target_mean)

    # Replace the original data with the converted values
    train_x[c] = tmp
