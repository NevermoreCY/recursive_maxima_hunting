import dcor
import math
import numpy.ma as ma
import scipy
import numpy as np


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

# help function for chi_bound
def chi_bound(x,y, sig):
    x_dist = dcor.distances.pairwise_distances(x)
    y_dist = dcor.distances.pairwise_distances(y)
    t2 = np.mean(x_dist) * np.mean(y_dist)
    chi_quant = scipy.stats.chi2.ppf(1 - sig, df=1)
    return float(chi_quant * t2 / x_dist.shape[0])

def _transform_to_2d(t):
    t = np.asfarray(t)
    dim = len(t.shape)
    assert dim <= 2
    if dim < 2:
        t = np.atleast_2d(t).T
    return t

#Get the mask of points that have a large dependence with another.
def get_mask_large_dependence(X, t_max_index, redundancy_condition, old_mask):
    sl = slice(None)
    def get_index(index,):
        return (sl,) + tuple(index) + (np.newaxis,)
    def is_redundant(index):
        max_point = np.squeeze(X[get_index(t_max_index)], axis=1)
        test_point = np.squeeze(X[get_index(index)], axis=1)
        return bool(dcor.u_distance_covariance_sqr(max_point, test_point) <redundancy_condition )
    def adjacent_indexes(index):
        for i, coord in enumerate(index):
            # Out of bounds right check
            if coord < (X.shape[i + 1] - 1):
                new_index = list(index)
                new_index[i] += 1
                yield tuple(new_index)
            # Out of bounds left check
            if coord > 0:
                new_index = list(index)
                new_index[i] -= 1
                yield tuple(new_index)

    def update_mask(new_mask,index):
        indexes = [index]
        while indexes:
            index = indexes.pop()
            # Check if it wasn't masked before
            if (
                    not old_mask[index] and not new_mask[index]
                    and is_redundant(index)
            ):
                new_mask[index] = True
                for i in adjacent_indexes(index):
                    indexes.append(i)

    new_mask = np.zeros_like(old_mask)
    update_mask(new_mask, t_max_index)
    # The selected point is masked even if min_redundancy is high
    new_mask[t_max_index] = True
    old_mask |= new_mask
    return old_mask

def conditional_mean(X,selected_index,cov_model):

    T = X.grid_points[0]
    t_0 = T[selected_index]
    x_index = (slice(None),) + tuple(selected_index) + (np.newaxis,)
    x_0 = X.data_matrix[x_index]
    T = _transform_to_2d(T)
    var = cov_model(t_0, t_0)
    expectation = np.ones_like(T, dtype=float) * 0
    t_0_expectation = expectation[selected_index]
    b_T = cov_model(T, t_0)

    cond_expectation = (
            expectation
            + b_T / var
            * (x_0.T - t_0_expectation)
    ) if var else expectation + np.zeros_like(x_0.T)

    return X.copy(data_matrix=cond_expectation.T,sample_names=None)

class Brownian:
    def __init__(self, *, variance: float = 1, origin: float = 0) -> None:
        self.variance = variance
        self.origin = origin
    def __call__(self, x, y):
        x = _transform_to_2d(x) - self.origin
        y = _transform_to_2d(y) - self.origin

        sum_norms = np.add.outer(
            np.linalg.norm(x, axis=-1),
            np.linalg.norm(y, axis=-1),
        )
        norm_sub = np.linalg.norm(
            x[:, np.newaxis, :] - y[np.newaxis, :, :],
            axis=-1,
        )
        return (
            self.variance * (sum_norms - norm_sub) / 2
        )

class Recursive_maxima_hunting():
    #The recursive maximum hunting algorithm.

    def __init__(self,s=0.01, r=0.9):
        self.s_value = s
        self.r_value = r

    def fit(self,X,y):
        self.features_shape_ = X.data_matrix.shape[1:]
        y = np.asfarray(y)  # return array with float type
        max_features = math.inf
        mask = np.zeros([len(t) for t in X.grid_points], dtype=bool)
        indexes = []
        first_pass = True
        c = 0
        while True:  # while loop for the algorithm
            X2 = X.copy()
            dependences = compute_dependence(X2,y)
            # find the maximum for dependences
            masked_function = ma.array( dependences.data_matrix,mask=mask)
            t_max = ma.argmax(masked_function)
            t_max_index = np.unravel_index(t_max, dependences.data_matrix.shape[1:-1])
            repeated_point = mask[t_max_index]
            # check whether we reached stop condition
            sig_value = self.s_value
            selected_variable = X.data_matrix[(slice(None),) + tuple(t_max_index)]
            bound = chi_bound(selected_variable, y, sig_value)
            stopping_condition_reached = bool(dcor.u_distance_covariance_sqr(selected_variable, y) < bound)

            # check if we meet terminate condition
            if (( len(indexes) >= max_features or repeated_point or stopping_condition_reached) and not first_pass):
                # store the parameters then finish fit function
                self.indexes_ = tuple(np.transpose(indexes).tolist())
                return self
            indexes.append(t_max_index)

            mask = get_mask_large_dependence(X=X.data_matrix,t_max_index=t_max_index,
                                                    redundancy_condition=self.r_value,old_mask=mask)
            # Exclude redundant points
            if first_pass:
                x_index = (slice(None),) + t_max_index
                x_0 = X.data_matrix[x_index]
                X = X - x_0

            else:
                X = X - conditional_mean(X,t_max_index, self.cov_model)

            self.cov_model = Brownian(origin=X.grid_points[0][t_max_index])

            first_pass = False
            c += 1


    def transform(self, X):
        # to transform test data into lower dimension
        X_matrix = X.data_matrix
        output = X_matrix[(slice(None),) + self.indexes_]
        return output.reshape(X.n_samples,-1,)
