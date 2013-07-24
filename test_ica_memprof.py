import numpy as np

from sklearn.decomposition import FastICA

X = np.random.random([1e3, 20])

FastICA().fit_transform(X.T)
