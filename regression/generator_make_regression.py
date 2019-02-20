"""
Generating data with datasets.make_regression function
"""

import numpy as np
# print('NumPy:{}'.format(np.__version__))
np.random.seed(123)

import matplotlib.pyplot as plt

import sklearn as sk
# print('Scikit Learn:{}'.format(sk.__version__))
from sklearn import datasets as skds

# the function make_regression returns a tuple
X, y = skds.make_regression(
    n_samples=4,        # integer, default = 100
    n_features=1,       # integer, default = 100
    n_informative=1,    # integer, default = 10
                        # the number of features used to build the linear
                        # model used to generate the output.
    n_targets=1,        # integer, default = 1
                        # The number of regression targets, i.e., the
                        # dimension of the y output vector associated
                        # with a sample. By default, a scalar.
    noise=.001)         # what _exactly_ is this?

print(X, "\n")
print(y, "\n")

# reshape?
if (y.ndim == 1):
    y = y.reshape(-1, 1)

# plot the points
plt.figure(figsize=(14,8))
plt.plot(X,y,'b.')
plt.title('The Synthetic Dataset')
plt.show()