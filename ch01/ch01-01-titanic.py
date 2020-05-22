import numpy as np
import pandas as pd

# -----------------------------------
# Load the training and test data
# -----------------------------------
# Load the training and test data
train = pd.read_csv('../input/ch01-titanic/train.csv')
test = pd.read_csv('../input/ch01-titanic/test.csv')

# Split the training data into features and target values
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# The test data only contains features, so is ok to use as it
test_x = test.copy()

# -----------------------------------
# Feature engineering
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# Drop the PassengerID variables
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# Drop the Name, Ticket & Cabin variables
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Apply label encoding to categorical variables
for c in ['Sex', 'Embarked']:
    # Fit the labels using the training data
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    # Return the encoded labels for the training and test data
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# -----------------------------------
# Model creation
# -----------------------------------
from xgboost import XGBClassifier

# Create the model and fit it using the training data
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# Output prediction probabilities for the test data
pred = model.predict_proba(test_x)[:, 1]

# Convert into binary predictions 
pred_label = np.where(pred > 0.5, 1, 0)

# Create a submission file
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# Score: 0.7799 (it is possible for this value to differ from the one in the book)

# -----------------------------------
# Validation
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# Create lists to store the scores for each fold
scores_accuracy = []
scores_logloss = []

# Perform cross validation
# Split the training data into 4, use 1 part for validation, then use the next part for validation, and so on...
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # Split the training data into training and validation sets
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # Train the model
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # Output prediction probabilities for the validation data
    va_pred = model.predict_proba(va_x)[:, 1]

    # Calculate scores for the validation data
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # Store the scores for this fold
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

# Calculate the mean scores using all folds
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148 (it is possible for these value to be different from the book)

# -----------------------------------
# Model tuning
# -----------------------------------
import itertools

# Prepare candidate tuning parameters
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# Hyperparamter combinations to try
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# Create lists to store scores for the different hyperparameter combinations
params = []
scores = []

# Perform cross-validation for each hyperparameter combination
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    # Perform cross-validation
    # Split the training data into 4, use 1 part for validation, then use the next part for validation, and so on...
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # Split the training data into training and validation sets
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # Train the model
        model = XGBClassifier(n_estimators=20, random_state=71,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # Output prediction probabilities for the validation data
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)

    # Calculate the mean score using all folds
    score_mean = np.mean(score_folds)

    # Store the scores for this hyperparameter combination
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# Set the best parameters to those giving the highest score
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# Best score is with max_depth=7, min_child_weight=2.0 


# -----------------------------------
# Create features for logistic regression
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# Copy the original datasets
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# Drop the PassengerID variables
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# Drop the Name, Ticket & Cabin variables
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Perform one-hot encoding
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

# Create column names for dummy one-hot encoding variables
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# Create DataFrames for one-hot encoding
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# Drop the original columns that were used for one-hot encoding
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# Append the one-hot encoded columns 
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# Replace missing values in these columns with the means of the values that exist
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# Make a logarithmic transformation of the Fare variables
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# Ensembling
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboost model
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# Logistic regression model
# As the xgboost model uses differently engineered features, train_x2, test_x2 were created separately
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# Take a weighted average of the predicted values
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)
