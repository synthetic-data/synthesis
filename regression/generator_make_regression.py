"""
Generating data with datasets.make_regression function

It creates a linear model y = w*X + b, then generates
'outputs' y and adds noise to X.

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression

"""

#import numpy as np
# print('NumPy:{}'.format(np.__version__))
# np.random.seed(123)

import matplotlib.pyplot as plt

import sklearn as sk
# print('Scikit Learn:{}'.format(sk.__version__))
from sklearn import datasets as skds

# the function make_regression returns a tuple
# the _generated_ X matrix is called 'inputs'
# the _generated_ y matrix is called 'targets'

X, y, C = skds.make_regression(
    n_samples=10,       # integer, default = 100
    n_features=1,       # integer, default = 100
    n_informative=1,    # integer, default = 10
                        # the number of features used to build the linear
                        # model used to generate the output.
    n_targets=1,        # integer, default = 1
                        # The number of regression targets, i.e., the
                        # dimension of the y output vector associated
                        # with a sample. By default, a scalar.
    bias=0.0,           # float, optional (default=0.0) in units of y
                        # The bias term in the underlying linear model.
    effective_rank=None,# the 'approximate effective' rank of the input
                        # matrix. If None the input set is 'well
                        # conditioned, centered and Gaussian with unit
                        # variance.
    tail_strength=0.5,  # float between 0.0 and 1.0, optional (default=0.5)
                        # The relative importance of the fat noisy tail
                        # of the singular values profile if
                        # effective_rank is not None.
    noise=0.0,          # the standard deviation of Gaussian noise, added
                        # to the output - y.
    coef=True,          # boolean, optional (default=False)
                        #   If True, the coefficients of the underlying
                        #   linear model are returned.
    random_state=789)   # the seed-like integer for reproducibility

print(X, "\n")
print(y, "\n")
print(C, "\n")

# reshape?
if (y.ndim == 1):
    y = y.reshape(-1, 1) # Transposes y which originally is a line (not column).

print(y, "\n")
# plot the points
plt.figure(figsize=(14,8))
plt.plot(X,y,'b.')
plt.title('The Synthetic Dataset')
plt.show()

"""
Now I understand.
the "features" are generated to be Gaussian and "centered" (around zero)
and having a standard deviation of 1, which means that they can be positive, 
negative and are potentially unlimited. This fake 'input' is put through
a linear model generated in a way that a standart deviation of 1 would 
transform into a 'target' of approximately 100. The 'bias' is the "b" term
of the linear model so it shifts the picture by the same number of units

What I don't understand is: why not define the coefficients of the linear
model? Why not let scale the output? 

Caught the constant of the linear model, finally.
"""