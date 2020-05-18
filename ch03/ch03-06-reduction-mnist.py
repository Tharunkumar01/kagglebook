# ---------------------------------
# Prepare the data etc.
# ----------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Visualization of MNIST data

# Import MNIST data from keras.datasets
from keras.datasets import mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Convert to 2D data
train_x = train_x.reshape(train_x.shape[0], -1)

# Decrease size by taking only first 1000 data
train_x = pd.DataFrame(train_x[:1000, :])
train_y = train_y[:1000]

# -----------------------------------
# PCA
# -----------------------------------
from sklearn.decomposition import PCA

# Fit the PCA transformation by using the training data
pca = PCA()
x_pca = pca.fit_transform(train_x)

# Plot in 2D, differentiating each class by color 
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_pca[mask, 0], x_pca[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

# -----------------------------------
# LDA (Linear Discriminant Analysis)
# -----------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Derive the 2 axes that best split the classes using linear discriminat analysis
lda = LDA(n_components=2)
x_lda = lda.fit_transform(train_x, train_y)

# Plot in 2D, differentiating each class by color
# Note that the division is good, but this method is using the target values which gives it an advantage over other methods
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_lda[mask, 0], x_lda[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

# -----------------------------------
# t-sne
# -----------------------------------
from sklearn.manifold import TSNE

# Transform using t-sne
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(train_x)

# Plot in 2D, differentiating each class by color
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_tsne[mask, 0], x_tsne[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

# -----------------------------------
# UMAP
# -----------------------------------
import umap

# Transform using UMAP
um = umap.UMAP()
x_umap = um.fit_transform(train_x)

# Plot in 2D, differentiating each class by color
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_umap[mask, 0], x_umap[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()
