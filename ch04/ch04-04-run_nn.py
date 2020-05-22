# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, train_y is the target values, and test_x is the test data
# stored in pandas DataFrames and Series (numpy arrays also used)
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

# Suppress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -----------------------------------
# Neural network implementation
# -----------------------------------
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# Data scaling
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x)
va_x = scaler.transform(va_x)
test_x = scaler.transform(test_x)

# Construct the neural network
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(train_x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Perform training
# Pass the validation data to the model, and monitor how the score changes during training
batch_size = 128
epochs = 10
history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y))

# Check score for validation data
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred, eps=1e-7)
print(f'logloss: {score:.4f}')

# Predictions
pred = model.predict(test_x)

# -----------------------------------
# Early stopping
# -----------------------------------
from keras.callbacks import EarlyStopping

# Set number of early stopping rounds to 20
# By setting restore_best_weights, we use the model from the best epoch
epochs = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y), callbacks=[early_stopping])
pred = model.predict(test_x)
