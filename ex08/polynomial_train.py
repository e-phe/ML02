#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from multivariate_linear_model import MyMultiLinearRegression as MyLR
import matplotlib.pyplot as plt
import numpy as np
import os


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of shape m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(x, np.ndarray)
        and x.size != 0
        and x.shape[1] == 1
        and isinstance(power, int)
    ):
        return np.vander(x.reshape(1, -1)[0], power + 1, increasing=True)[:, 1:]
    return


if __name__ == "__main__":
    try:
        if os.stat("../resources/are_blue_pills_magics.csv").st_size > 0:
            data = np.loadtxt(
                "../resources/are_blue_pills_magics.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    x = data[:, [1]]
    y = data[:, [2]]

    figure, axis = plt.subplots(1, 6)
    even = 1
    mse = np.zeros(6)
    for i in range(1, 7):
        if i == 4:
            theta = np.array([[-20], [160], [-80], [10], [-1]])
        elif i == 5:
            theta = np.array([[1140], [-1850], [1110], [-305], [40], [-2]])
        elif i == 6:
            theta = np.array(
                [[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]]
            )
        else:
            theta = np.ones(i + 1)

        if i == 1:
            alpha = 1e-2
            odd = alpha
        else:
            if i % 2 == 0:
                alpha = even * 1e-3
                even = alpha
            else:
                alpha = odd * 1e-3
                odd = alpha

        my_lr = MyLR(theta.reshape(-1, 1), alpha, 100000)
        x_ = add_polynomial_features(x, i)
        my_lr.fit_(x_, y)

        mse[i - 1] = my_lr.mse_(x_, y)
        print("MSE", mse[i - 1])

        continuous_x = np.arange(0, 8, 0.01).reshape(-1, 1)
        x_ = add_polynomial_features(continuous_x, i)
        y_hat = my_lr.predict_(x_)

        axis[i - 1].scatter(x, y)
        axis[i - 1].plot(continuous_x, y_hat, color="orange")
        axis[i - 1].set_ylim([20, 90])
    plt.show()

    plt.xlabel("polynomial degree")
    plt.ylabel("mse")
    plt.bar(np.arange(1, 7), mse)
    plt.show()
