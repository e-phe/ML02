#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os


class MyLinearRegression:
    def __init__(self, theta=[[0.0], [0.0]], alpha=1.5e-4, max_iter=200000):
        try:
            theta = np.array(theta)
        except:
            return
        if (
            isinstance(theta, np.ndarray)
            and theta.size != 0
            and theta.shape == (2, 1)
            and isinstance(alpha, float)
            and isinstance(max_iter, int)
        ):
            self.theta = theta
            self.alpha = alpha
            self.max_iter = max_iter
        else:
            return

    def fit_(self, x, y):
        if check_matrix(x) and check_matrix(y) and x.shape == y.shape:
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            for _ in range(self.max_iter):
                gradient = np.dot(np.transpose(x), x @ self.theta - y) / x.shape[0]
                self.theta = self.theta - self.alpha * gradient
            return self.theta

    def predict_(self, x):
        if check_matrix(x):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return x @ self.theta

    def plot_(self, x, y, predict):
        if (
            check_matrix(x)
            and check_matrix(y)
            and check_matrix(predict)
            and x.shape == y.shape
            and x.shape == predict.shape
        ):
            plt.grid()
            plt.scatter(x, predict, marker=".", label="Predicted sell price")
            plt.scatter(x, y, label="Sell price")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y: sell price (in keuros)")
            plt.show()

    def mse_(self, y, predict):
        if check_matrix(y) and check_matrix(predict) and y.shape == predict.shape:
            return np.square(predict - y).sum() / y.shape[0]


def check_matrix(matrix):
    if (
        isinstance(matrix, np.ndarray)
        and matrix.size != 0
        and len(matrix.shape) == 2
        and matrix.shape[1] == 1
    ):
        return True
    exit("Error matrix")


class MyMultiLinearRegression:
    """
    Description:
        My personal linear regression class to fit like a boss.
    """

    def __init__(
        self, theta=[[0.0], [0.0], [0.0], [0.0]], alpha=8e-5, max_iter=1000000
    ):
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
            and len(x.shape) == 2
            and check_matrix(y)
            and x.shape[0] == y.shape[0]
            and check_matrix(self.theta)
            and x.shape[1] + 1 == self.theta.shape[0]
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
            and len(x.shape) == 2
            and check_matrix(self.theta)
            and x.shape[1] + 1 == self.theta.shape[0]
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return x @ self.theta
        return

    def plot_(self, x, y, predict):
        if (
            isinstance(x, np.ndarray)
            and x.size != 0
            and len(x.shape) == 2
            and check_matrix(y)
            and x.shape[0] == y.shape[0]
            and check_matrix(predict)
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
        if check_matrix(y) and check_matrix(y_hat) and y.shape == y_hat.shape:
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

    alpha = [1e-3, 1.5e-4, 2.2e-4]
    for i in range(data.shape[1] - 1):
        x = data[:, [i]]
        y = data[:, [data.shape[1] - 1]]

        mlr = MyLinearRegression(alpha=alpha[i])
        print(mlr.fit_(x, y))
        predict = mlr.predict_(x)
        print("MSE", mlr.mse_(y, predict))
        mlr.plot_(x, y, predict)

    x = data[:, :-1]
    y = data[:, [data.shape[1] - 1]]

    mmlr = MyMultiLinearRegression()
    print(mmlr.fit_(x, y))
    predict = mmlr.predict_(x)
    print("MSE", mmlr.mse_(x, y))
    mmlr.plot_(x, y, predict)
