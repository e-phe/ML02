#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from multivariate_linear_model import MyMultiLinearRegression as MyLR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def vander_matrix(x, y):
    x_ = np.zeros(y.shape)
    for x_train in x.T:
        x_ = np.concatenate(
            (x_, add_polynomial_features(x_train.reshape(-1, 1), i)), axis=1
        )
    return x_[:, 1:]


if __name__ == "__main__":
    try:
        if (
            os.stat("../resources/space_avocado.csv").st_size > 0
            and os.stat("models.csv").st_size > 0
        ):
            data = np.loadtxt(
                "../resources/space_avocado.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
            models = np.genfromtxt("models.csv", dtype=str, delimiter="\t")
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    (x, x_test, y, y_test) = data_spliter(data[:, 1:4], data[:, [4]], 0.7)

    theta = []
    for el in models:
        theta.append(np.fromstring(el, sep=","))
    models = theta

    figure, axis = plt.subplots(3, 4)
    mse_train = np.zeros(4)
    mse_test = np.zeros(4)
    for i in range(1, 5):
        my_lr = MyLR(theta[i].reshape(-1, 1), 1e-7, 10000)
        x_ = vander_matrix(x, y)
        if i == 1:
            my_lr = MyLR(np.ones(x_.shape[1] + 1).reshape(-1, 1), 1e-7, 10000)

        if i == 1:
            my_lr.fit_(x_, y)
        y_hat = my_lr.predict_(x_)

        x_test_ = vander_matrix(x_test, y_test)
        mse_train[i - 1] = my_lr.mse_(x_, y)
        mse_test[i - 1] = my_lr.mse_(x_test_, y_test)

        x_name = ["weight(in ton)", "prod_distance (in Mkm)", "time_delivery (in days)"]
        for j in range(x.shape[1]):
            axis[j, i - 1].set_xlabel(x_name[j])
            axis[j, i - 1].set_ylabel("target (in trantorian unit)")

            axis[j, i - 1].scatter(x[:, j], y, label="dataset_train")
            axis[j, i - 1].scatter(x[:, j], y_hat, marker=".", label="prediction")
            axis[j, i - 1].legend()
    plt.show()

    plt.xlabel("polynomial degree")
    plt.ylabel("mse")

    plt.plot(np.arange(1, 5), mse_train, label="mse_train")
    plt.scatter(np.arange(1, 5), mse_train, label="mse_train")
    plt.plot(np.arange(1, 5), mse_test, label="mse_test")
    plt.scatter(np.arange(1, 5), mse_test, marker=".", label="mse_test")

    plt.legend()
    plt.show()
