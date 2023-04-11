import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from skfda.ml.classification import KNeighborsClassifier
from skfda.representation.grid import FDataGrid
from tqdm import tqdm
import random




def indicator_function(left,right,s):
    if left <= s <= right:
        return 1
    else:
        return 0


def brownian_motion(n=10000, d=200, T = 1.):
    # n = 10000  # number of samples
    # d = 200 # number of brownian motion
    # T = 1.  # time horizon
    times = np.linspace(0., T , n)
    dt = times[1] -  times[0]

    # brownian motion is independent normally distributed increment
    # bt2 - bt1 ~ Normally distributed with mean 0 and variance t2-t1 which is dt.
    dB = np.sqrt(dt) *  np.random.normal(size=(n-1, d))  # variance equal to dt which means std = sqrt(dt)
    # we assume brownian motion starts with value zero
    B0 = np.zeros(shape=(1,d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0) ), axis=0 )
    plt.plot(times, B)
    plt.show()
    return B

    # observation: even though our brownian motion was made up of totally independent increments
    # every single line is generated randomly, but we can see a structure here,
    # lines are tend to stay in a certain parabolic shape region.
    #  * law of the iterated logarithm for brownian motion


def peak1(n=10000,d=200,T=1.):
    k = 3
    m = 3
    times = np.linspace(0., T, n)
    dt = times[1] - times[0]

    coef = np.sqrt(2**(m-1))
    thresh1 = (2*k - 2) / (2**m)
    thresh2 = (2*k - 1) / (2**m)
    thresh3 = (2*k ) / (2**m)

    m_values= []
    for s in times:
        m_values.append(2 * coef * dt * (indicator_function(thresh1, thresh2, s) - indicator_function(thresh2, thresh3, s)))


    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B0 = np.zeros(shape=(1,d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    dB2 = dB + np.array([m_values[1:]]).T
    B2 = np.concatenate((B0, np.cumsum(dB2, axis=0)), axis=0)
    avg_line = np.average(B2, axis=1)

    # plt.plot(times,B)
    plt.plot(times,B2)
    plt.plot(times,avg_line, color='black', linewidth=5)
    plt.ylim((-3, 3))
    plt.title("peak 1 ")
    plt.show()
    plt.close()

    return B,B2,avg_line , times

def peak2(n=10000,d=200,T=1.):

    times = np.linspace(0., T, n)
    dt = times[1] - times[0]

    # 3_2
    m = 3
    k = 2
    coef_32 = np.sqrt(2**(m-1))
    thresh1_32 = (2*k - 2) / (2**m)
    thresh2_32 = (2*k - 1) / (2**m)
    thresh3_32 = (2*k ) / (2**m)

    # 3 3
    m = 3
    k = 3
    coef_33 = np.sqrt(2**(m-1))
    thresh1_33 = (2*k - 2) / (2**m)
    thresh2_33 = (2*k - 1) / (2**m)
    thresh3_33 = (2*k ) / (2**m)

    # 2 2
    m = 2
    k = 2
    coef_22 = np.sqrt(2**(m-1))
    thresh1_22 = (2*k - 2) / (2**m)
    thresh2_22 = (2*k - 1) / (2**m)
    thresh3_22 = (2*k ) / (2**m)

    m_values= []

    for s in times:
        thi_3_2 = coef_32 * dt * (
                indicator_function(thresh1_32, thresh2_32, s) - indicator_function(thresh2_32, thresh3_32, s))
        thi_3_3 = coef_33 * dt * (
                    indicator_function(thresh1_33, thresh2_33, s) - indicator_function(thresh2_33, thresh3_33, s))
        thi_2_2 = coef_22 * dt * (
                    indicator_function(thresh1_22, thresh2_22, s) - indicator_function(thresh2_22, thresh3_22, s))

        m_value = 2 * thi_3_2 + 3 * thi_3_3 - 2 * thi_2_2
        m_values.append(m_value)


    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B0 = np.zeros(shape=(1,d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    dB2 = dB + np.array([m_values[1:]]).T
    B2 = np.concatenate((B0, np.cumsum(dB2, axis=0)), axis=0)
    avg_line = np.average(B2, axis=1)

    # plt.plot(times,B)
    plt.plot(times,B2)
    plt.plot(times,avg_line, color='black', linewidth=5)
    plt.ylim((-3, 3))
    plt.title("peak 2 ")
    plt.show()
    plt.close()

    return B, B2,avg_line , times


def square(n=10000,d=200,T=1.):

    times = np.linspace(0., T, n)
    dt = times[1] - times[0]

    m_values= []
    # print(times[-10:])
    for s in times:
        m_values.append( 2*s**2)

    # print(m_values[-10:])
    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B0 = np.zeros(shape=(1,d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    B2 = B + np.array([m_values]).T
    avg_line = np.average(B2, axis=1)

    # plt.plot(times,B)
    plt.plot(times,B2)
    plt.plot(times,avg_line, color='black', linewidth=5)
    plt.ylim((-3, 3))
    plt.title("square")
    plt.show()
    plt.close()

    plt.plot(times,B)
    plt.plot(times,avg_line, color='black', linewidth=5)
    plt.ylim((-3, 3))
    plt.title("B")
    plt.show()
    plt.close()

    return B, B2,avg_line , times


def sin(n=10000,d=200,T=1.):

    times = np.linspace(0., T, n)
    dt = times[1] - times[0]

    m_values= []
    # print(times[-10:])
    for s in times:
        m_values.append( 0.5 * np.sin(2*np.pi * s))

    # print(m_values[-10:])
    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B0 = np.zeros(shape=(1,d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    B2 = B + np.array([m_values]).T
    avg_line = np.average(B2, axis=1)

    # plt.plot(times,B)
    plt.plot(times,B2)
    plt.plot(times,avg_line, color='black', linewidth=5)
    plt.ylim((-3, 3))
    plt.title("sin")
    plt.show()
    plt.close()

    return B, B2,avg_line, times


def tanh(n=10000,d=200,T=1.):

    times = np.linspace(0., T, n)
    dt = times[1] - times[0]

    m_values= []
    # print(times[-10:])
    for s in times:
        m_values.append(np.tanh(4 * s))

    # print(m_values[-10:])
    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B0 = np.zeros(shape=(1,d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    B2 = B + np.array([m_values]).T
    avg_line = np.average(B2, axis=1)

    # plt.plot(times,B)
    plt.plot(times,B2)
    plt.plot(times,avg_line, color='black', linewidth=5)
    plt.ylim((-3, 3))
    plt.title("sin")
    plt.show()
    plt.close()

    return B, B2,avg_line, times


def exp(n=10000,d=200,T=1.):

    times = np.linspace(0., T, n)
    dt = times[1] - times[0]

    m_values= []

    for s in times:
        m_values.append(np.exp(s) -1 )

    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B0 = np.zeros(shape=(1,d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    B2 = B + np.array([m_values]).T
    avg_line = np.average(B2, axis=1)

    # plt.plot(times,B)
    plt.plot(times,B2)
    plt.plot(times,avg_line, color='black', linewidth=5)
    plt.ylim((-3, 3))
    plt.title("sin")
    plt.show()
    plt.close()

    return B, B2,avg_line, times

def mix_data(B, B2 , grid_points):
    # B shape: (n,d)
    # B2 shape: (n,d)
    n,d = B.shape

    X = np.concatenate( (B.T,B2.T),axis= 0)
    y = np.array([0] * d + [1] * d)
    X = FDataGrid(X, grid_points)

    return X , y

def mix_data_arr(B, B2 , grid_points):
    # B shape: (n,d)
    # B2 shape: (n,d)
    n,d = B.shape

    X = np.concatenate( (B.T,B2.T),axis= 0)
    y = np.array([0] * d + [1] * d)

    return X , y
