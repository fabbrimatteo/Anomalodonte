import numpy as np
from sklearn.neighbors import LocalOutlierFactor
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

clf = LocalOutlierFactor(n_neighbors=2)
labels = clf.fit_predict(X)
score = clf.score_samples(X)
print()