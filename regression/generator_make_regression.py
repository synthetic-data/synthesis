"""
Generating data with datasets.make_regression function
"""

import numpy as np
# print('NumPy:{}'.format(np.__version__))
np.random.seed(123)

import matplotlib.pyplot as plt

import sklearn as sk
# print('Scikit Learn:{}'.format(sk.__version__))
from sklearn import model_selection as skms
from sklearn import datasets as skds
from sklearn import preprocessing as skpp

X, y = skds.make_regression(
    n_samples=200, n_features=1, n_informative=1, n_targets=1, noise=20.0)
if (y.ndim == 1):
    y = y.reshape(-1, 1)

plt.figure(figsize=(14,8))
plt.plot(X,y,'b.')
plt.title('Original Dataset')
plt.show()