import numpy as np
import pandas as pd

# -----------------------------------
# Wide format, long format
# -----------------------------------

# Load wide format data
df_wide = pd.read_csv('../input/ch03/time_series_wide.csv', index_col=0)
# Convert the index column to datetime dtype
df_wide.index = pd.to_datetime(df_wide.index)

print(df_wide.iloc[:5, :3])
'''
              A     B     C
date
2016-07-01  532  3314  1136
2016-07-02  798  2461  1188
2016-07-03  823  3522  1711
2016-07-04  937  5451  1977
2016-07-05  881  4729  1975
'''

# Convert to long format
df_long = df_wide.stack().reset_index(1)
df_long.columns = ['id', 'value']

print(df_long.head(10))
'''
           id  value
date
2016-07-01  A    532
2016-07-01  B   3314
2016-07-01  C   1136
2016-07-02  A    798
2016-07-02  B   2461
2016-07-02  C   1188
2016-07-03  A    823
2016-07-03  B   3522
2016-07-03  C   1711
2016-07-04  A    937
...
'''

# Restore wide format
df_wide = df_long.pivot(index=None, columns='id', values='value')

# -----------------------------------
# Lag variables
# -----------------------------------
# Set data to wide format
x = df_wide
# -----------------------------------
# x is the wide format data frame
# The index is the date or timestamp, assume the columns store data of interest such as sales etc. for users or stores

# Create lag data for one period ago
x_lag1 = x.shift(1)

# Create lag data for seven periods ago
x_lag7 = x.shift(7)

# -----------------------------------
# Calculate moving averages for three periods from one period before
x_avg3 = x.shift(1).rolling(window=3).mean()

# -----------------------------------
# Calculate max values over seven periods from one period before
x_max7 = x.shift(1).rolling(window=7).max()

# -----------------------------------
# Calculate average of data from 7, 14, 21 and 28 periods before
x_e7_avg = (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4.0

# -----------------------------------
# Create values for one period ahead
x_lead1 = x.shift(-1)

# -----------------------------------
# Lag variables
# -----------------------------------
# Load the data
train_x = pd.read_csv('../input/ch03/time_series_train.csv')
event_history = pd.read_csv('../input/ch03/time_series_events.csv')
train_x['date'] = pd.to_datetime(train_x['date'])
event_history['date'] = pd.to_datetime(event_history['date'])
# -----------------------------------

# train_x is training data in a data frame with columns for user id and date
# event_history contains data from past events in a data frame with date and event columns

# occurrences is a data frame with columns for date and whether a sale was made or not
dates = np.sort(train_x['date'].unique())
occurrences = pd.DataFrame(dates, columns=['date'])
sale_history = event_history[event_history['event'] == 'sale']
occurrences['sale'] = occurrences['date'].isin(sale_history['date'])

# Take cumulative sums to calculate to number of occurrences on each date
# occurrences is now a data frame with columns for date and cumulative number of sales on that date
occurrences['sale'] = occurrences['sale'].cumsum()

# Using the timestamp as a key, combine with the training dataset
train_x = train_x.merge(occurrences, on='date', how='left')
