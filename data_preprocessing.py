import skfda
import numpy as np

import ssl
from simulation import peak1,peak2,sin,square,mix_data,exp,tanh
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda.misc.kernels import normal
ssl._create_default_https_context = ssl._create_unverified_context

n = 200  # trajectories are discretized in 200 points as state in paper
d = 1000  # we will have at most 2000 trajectories

def get_growth():
    X, y = skfda.datasets.fetch_growth(return_X_y=True, as_frame=True)
    X = X.iloc[:, 0].values
    y = y.values
    y = y.codes
    return X, y


def get_phoneme():
    X, y = skfda.datasets.fetch_phoneme(return_X_y=True)

    X = X[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]
    n_points = 50


    new_points = X.grid_points[0][:n_points]
    new_data = X.data_matrix[:, :n_points]

    X = X.copy(
        grid_points=new_points,
        data_matrix=new_data,
        domain_range=(np.min(new_points), np.max(new_points)),
    )

    smoother = KernelSmoother(NadarayaWatsonHatMatrix(bandwidth=0.2))
    X_smooth = smoother.fit_transform(X)
    return X_smooth, y

def get_tecator():
    X, y = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=True)
    X = X.iloc[:, 0].values
    fdd = X.derivative(order=2)
    fat = y['fat'].values
    low_fat = fat < 20
    labels = np.full(fdd.n_samples, 0)  # high fat = 0
    labels[low_fat] = 1   #low fat = 1
    return fdd,labels

def get_medflies():
    X, y = skfda.datasets.fetch_medflies(return_X_y=True, as_frame=True)
    X = X.iloc[:, 0].values
    y = y.values
    y = y.codes
    return X,y

def get_simulation_peak1():
    B, B2, avg_line, grid_points = peak1(n=n, d=d, T=1.)
    X, y = mix_data(B,B2, grid_points)
    return X,y

def get_simulation_peak2():
    B, B2, avg_line, grid_points = peak2(n=n, d=d, T=1.)
    X, y = mix_data(B,B2, grid_points)
    return X,y

def get_simulation_square():
    B, B2, avg_line, grid_points = square(n=n, d=d, T=1.)
    X, y = mix_data(B,B2, grid_points)
    return X,y

def get_simulation_sin():
    B, B2, avg_line, grid_points = sin(n=n, d=d, T=1.)
    X, y = mix_data(B,B2, grid_points)
    return X,y

def get_simulation_tanh():
    B, B2, avg_line, grid_points = tanh(n=n, d=d, T=1.)
    X, y = mix_data(B,B2, grid_points)
    return X,y

def get_simulation_exp():
    B, B2, avg_line, grid_points = exp(n=n, d=d, T=1.)
    X, y = mix_data(B,B2, grid_points)
    return X,y
