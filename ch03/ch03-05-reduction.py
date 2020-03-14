# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd

# train_x is the training data, with train_y target values, and test_x is the test data
# Saving in pandas DataFrame and Series objects. (Also possible to use numpy arrays)

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# For explanations save the original forms of the training and test data
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Function to return standardized versions of the original training and test data
def load_standarized_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# Function to return MinMax scaled versions of the original training and test data
def load_minmax_scaled_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    # Apply Min-Max Scaling
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([train_x, test_x], axis=0))
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# -----------------------------------
# PCA
# -----------------------------------
# Use the standardized data
train_x, test_x = load_standarized_data()
# -----------------------------------
# PCA
from sklearn.decomposition import PCA

# Assume that the data has been preprocessed e.g. by standardization to make the scale uniform

# Fit the PCA transformation using the training data
pca = PCA(n_components=5)
pca.fit(train_x)

# Apply the transformation
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

# -----------------------------------
# Use the standardized data
train_x, test_x = load_standarized_data()
# -----------------------------------
# TruncatedSVD
from sklearn.decomposition import TruncatedSVD

# Assume that the data has been preprocessed e.g. by standardization to make the scale uniform

# Fit the SVD transformation using the training data
svd = TruncatedSVD(n_components=5, random_state=71)
svd.fit(train_x)

# Apply the transformation
train_x = svd.transform(train_x)
test_x = svd.transform(test_x)

# -----------------------------------
# NMF
# -----------------------------------
# So that the data are non-negative, use the MinMax scaled data
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import NMF

# Assume the data only contains non-negative values

# Fit the NMF transformation using the training data
model = NMF(n_components=5, init='random', random_state=71)
model.fit(train_x)

# Apply the transformation
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# LatentDirichletAllocation
# -----------------------------------
# Use the MinMax scaled data
# Although this is not a matrix of counts, as the values are all non-negative it is still possible to calculate
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import LatentDirichletAllocation

# Assume the data is a matrix of counts of words in a document

# Fit the Latent Dirichlet Allocation transformation using the training data
model = LatentDirichletAllocation(n_components=5, random_state=71)
model.fit(train_x)

# Apply the transformation
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# LinearDiscriminantAnalysis
# -----------------------------------
# Use the standardized data
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Assume that the data has been preprocessed e.g. by standardization to make the scale uniform

# Fit the Linear Discriminant Analysis transformation using the training data
lda = LDA(n_components=1)
lda.fit(train_x, train_y)

# Apply the transformation
train_x = lda.transform(train_x)
test_x = lda.transform(test_x)

# -----------------------------------
# t-sne
# -----------------------------------
# Use the standardized data
train_x, test_x = load_standarized_data()
# -----------------------------------
import bhtsne

# Assume that the data has been preprocessed e.g. by standardization to make the scale uniform

# Transform using t-sne
data = pd.concat([train_x, test_x])
embedded = bhtsne.tsne(data.astype(np.float64), dimensions=2, rand_seed=71)

# -----------------------------------
# UMAP
# -----------------------------------
# Use the standardized data
train_x, test_x = load_standarized_data()
# -----------------------------------
import umap

# Assume that the data has been preprocessed e.g. by standardization to make the scale uniform

# Fit the UMAP transformation using the training data
um = umap.UMAP()
um.fit(train_x)

# Apply the transformation
train_x = um.transform(train_x)
test_x = um.transform(test_x)

# -----------------------------------
# Clustering
# -----------------------------------
# Use the standardized data
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.cluster import MiniBatchKMeans

# Assume that the data has been preprocessed e.g. by standardization to make the scale uniform

#  Fit the Mini-Batch K-Means using the training data
kmeans = MiniBatchKMeans(n_clusters=10, random_state=71)
kmeans.fit(train_x)

# Output the clusters to which each class belongs
train_clusters = kmeans.predict(train_x)
test_clusters = kmeans.predict(test_x)

# Output the distance to the center for each cluster
train_distances = kmeans.transform(train_x)
test_distances = kmeans.transform(test_x)
