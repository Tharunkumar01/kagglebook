import numpy as np
import pandas as pd

# -----------------------------------
# Merging data
# -----------------------------------
# Load the data
train = pd.read_csv('../input/ch03/multi_table_train.csv')
product_master = pd.read_csv('../input/ch03/multi_table_product.csv')
user_log = pd.read_csv('../input/ch03/multi_table_log.csv')

# -----------------------------------
# Suppose we have a data frame in the format shown in the diagram
# train         : Training data (UserID, ProductID, Target value columns etc.)
# product_master : Product data (ProductID, Product information columns etc.)
# user_log      : User actions log data (UserID, Columns recording user action data etc.)

# Combine the product data and training data
train = train.merge(product_master, on='product_id', how='left')

# Aggregate the lines containing data for each user, and append to the training data
user_log_agg = user_log.groupby('user_id').size().reset_index().rename(columns={0: 'user_count'})
train = train.merge(user_log_agg, on='user_id', how='left')
