#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os


class MyMultiLinearRegression:
    """
    Description:
        My personal linear regression class to fit like a boss.
    """

    def __init__(self, theta=[[0.0], [0.0], [0.0], [0.0]], alpha=5e-5, max_iter=100000):
        try:
            theta = np.array(theta)
        except:
            return
        if (
            isinstance(theta, np.ndarray)
            and theta.size != 0
            and isinstance(alpha, float)
            and isinstance(max_iter, int)
        ):
            self.theta = theta
            self.alpha = alpha
            self.max_iter = max_iter
        else:
            return

    def fit_(self, x, y):
        if (
            isinstance(x, np.ndarray)
            and x.size != 0
            and isinstance(y, np.ndarray)
            and y.size != 0
            and y.shape[1] == 1
            and x.shape[0] == y.shape[0]
            and x.shape[1] + 1 == self.theta.shape[0]
            and self.theta.shape[1] == 1
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            for _ in range(self.max_iter):
                gradient = np.dot(x.transpose(), x @ self.theta - y) / x.shape[0]
                self.theta = self.theta - self.alpha * gradient
            return self.theta
        return

    def predict_(self, x):
        if (
            isinstance(x, np.ndarray)
            and x.size != 0
            and x.shape[1] + 1 == self.theta.shape[0]
            and self.theta.shape[1] == 1
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return x @ self.theta
        return

    def plot_(self, x, y, predict):
        if (
            isinstance(x, np.ndarray)
            and x.size != 0
            and isinstance(y, np.ndarray)
            and y.size != 0
            and isinstance(predict, np.ndarray)
            and predict.size != 0
            and x.shape[0] == y.shape[0]
            and y.shape == predict.shape
        ):
            for i in range(x.shape[1]):
                plt.grid()
                plt.scatter(x[:, i], y, label="Sell price")
                plt.scatter(x[:, i], predict, marker=".", label="Predicted sell price")
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y: sell price (in keuros)")
                plt.show()

    def mse_(self, y, y_hat):
        y = self.predict_(y)
        if (
            isinstance(y, np.ndarray)
            and y.size != 0
            and isinstance(y_hat, np.ndarray)
            and y_hat.size != 0
            and y.shape == y_hat.shape
            and y.shape[1] == 1
        ):
            return np.square(y_hat - y).sum() / y.shape[0]
        return


if __name__ == "__main__":
    try:
        if os.stat("../resources/spacecraft_data.csv").st_size > 0:
            data = np.loadtxt(
                "../resources/spacecraft_data.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    x = data[:, :-1]
    y = data[:, [data.shape[1] - 1]]

    mlr = MyMultiLinearRegression()
    print(mlr.fit_(x, y))
    predict = mlr.predict_(x)
    mlr.plot_(x, y, predict)
    print(mlr.mse_(x, y))
