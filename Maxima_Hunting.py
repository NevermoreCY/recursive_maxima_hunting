import dcor
import scipy
import numpy as np
import scipy.signal
from dcor import u_distance_correlation_sqr

# help function for computing dependence
def compute_dependence(X,y,):
    # we use u_distance_correlation_sqr for depencency measure
    # This function returns the squared energy distance correlation coefficient,
    # a value between 0 and 1. A value of 0 indicates no dependence between the variables,
    # while a value of 1 indicates complete dependence.
    dependence_measure = dcor.u_distance_correlation_sqr
    # align the input shape
    X_ndarray = X.data_matrix
    input_shape = X_ndarray.shape[1:-1]
    X_ndarray = np.moveaxis(X_ndarray, 0, -2)
    X_ndarray = X_ndarray.reshape(-1, X_ndarray.shape[-2], X_ndarray.shape[-1])
    if y.ndim == 1:
        y = np.atleast_2d(y).T
    Y = np.array([y] * len(X_ndarray))
    # calculate the dependence
    dependence_results = dcor.rowwise(dependence_measure,X_ndarray,Y)
    # get back to the input shape
    result = dependence_results.reshape( 1,input_shape[0],1 )
    X.data_matrix = result
    return X

# helper function to find the relative maxima point
def select_relative_maxima(X, *, order: int = 1):

    X_array = X.data_matrix[0, ..., 0]
    indexes = scipy.signal.argrelextrema(X_array,comparator=np.greater_equal,order=order,)[0]
    maxima = X_array[indexes]
    left_points = np.take(X_array, indexes - 1, mode='clip')
    right_points = np.take(X_array, indexes + 1, mode='clip')
    is_not_flat = (maxima > left_points) | (maxima > right_points)

    return indexes[is_not_flat]

# the maxima hunting class
class Maxima_hunting:

    def __init__(self,dependence_measure = u_distance_correlation_sqr,smooth = 1):
        # define the dependence function
        self.dependence_measure = dependence_measure
        self.smooth = smooth
    # fit the training data
    def fit(self,X,y,):
        X2 = X.copy()
        dependence = compute_dependence(X2, y.astype(np.float_))
        indexes = select_relative_maxima(dependence,order=self.smooth)
        sorting_indexes = np.argsort(
            dependence.data_matrix[0, indexes, 0])[::-1]
        self.sorted_indexes_ = indexes[sorting_indexes]

        return self
    # transform the test data to lower dimension
    def transform(self,X):
        return X.data_matrix[:, self.sorted_indexes_].reshape(X.n_samples, -1)
