# -*- coding: utf-8 -*-

# Authors: Vlad Niculae, Alexandre Gramfort, Slim Essid
# License: BSD

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition

# Prepare data and define utility functions
n_row, n_col = 2, 5
n_components = n_row * n_col
image_shape = (64, 64)
rng = np.random.RandomState(0)

# Load faces data
dataset = fetch_olivetti_faces(
    data_home='c:/tmp/', shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# Global centering
faces_centered = faces - faces.mean(axis=0).astype(np.float64)

print(f"Dataset consists of {n_samples} faces")


def plot_gallery(title, images):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        comp = comp.reshape(image_shape)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp, cmap='gray', interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


# Plot a sample of the input data
plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# Decomposition methods
estimators = [
    ('PCA', 'Eigenfaces - PCA',
     decomposition.PCA(n_components=n_components, whiten=True), True),
    ('NMF', 'Non-negative components - NMF', decomposition.NMF(n_components=n_components,
     init='nndsvd', tol=1e-6, max_iter=1000), False)
]

# Transform and classify
labels = dataset.target
X = faces
X_ = faces_centered

for shortname, name, estimator, center in estimators:
    print(f"Extracting the top {n_components} {name}...")
    t0 = time()

    data = X_ if center else X
    data = estimator.fit_transform(data)

    train_time = time() - t0
    print(f"done in {train_time:.3f}s")

    components_ = estimator.components_

    plot_gallery(f'{name} - Train time {train_time:.1f}s',
                 components_[:n_components])