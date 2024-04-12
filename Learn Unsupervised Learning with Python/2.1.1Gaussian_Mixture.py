import numpy as np
from scipy.ndimage import convolve

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
    ]

    def shift(x, w):
        return convolve(x, mode="constant", weights=w).ravel()

    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

X, y = datasets.load_digits(return_X_y=True)
print(type(X))
#print(X[0:10,0:10])
#print(X.shape)

X_test = np.array([[1,2,3], [0,1,0], [3,4,5]])
print((X_test))
Y_test = np.array([1,2,3])

X, Y = nudge_dataset(X_test, Y_test)


