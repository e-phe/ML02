#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class MyLinearRegression:
    """
    Description:
        My personal linear regression class to fit like a boss.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if (
            isinstance(theta, list)
            and len(theta) != 0
            and isinstance(alpha, float)
            and isinstance(max_iter, int)
        ):
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = np.array(theta)
        else:
            return

    def fit_(self, x, y):
        if (
            isinstance(x, np.ndarray)
            and x.size != 0
            and len(x.shape) == 2
            and isinstance(y, np.ndarray)
            and y.size != 0
            and len(y.shape) == 2
            and y.shape[1] == 1
            and x.shape[0] == y.shape[0]
            and x.shape[1] + 1 == self.theta.shape[0]
            and isinstance(self.theta, np.ndarray)
            and self.theta.size != 0
            and len(self.theta.shape) == 2
            and self.theta.shape[1] == 1
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            for _ in range(self.max_iter):
                gradient = x.T @ (x @ self.theta - y) / x.shape[0]
                self.theta = self.theta - self.alpha * gradient
            return self.theta
        return

    def predict_(self, x):
        if (
            isinstance(x, np.ndarray)
            and x.size != 0
            and len(x.shape) == 2
            and isinstance(self.theta, np.ndarray)
            and self.theta.size != 0
            and len(self.theta.shape) == 2
            and self.theta.shape[1] == 1
            and x.shape[1] + 1 == self.theta.shape[0]
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return x @ self.theta
        return

    def loss_elem_(self, y, y_hat):
        y = self.predict_(y)
        if (
            isinstance(y, np.ndarray)
            and y.size != 0
            and len(y.shape) == 2
            and isinstance(y_hat, np.ndarray)
            and y_hat.size != 0
            and y.shape == y_hat.shape
            and y.shape[1] == 1
        ):
            return np.array(
                [(y_hat[i] - y[i]) * (y_hat[i] - y[i]) for i in range(y.shape[0])]
            )
        return

    def loss_(self, y, y_hat):
        y = self.predict_(y)
        if (
            isinstance(y, np.ndarray)
            and y.size != 0
            and len(y.shape) == 2
            and isinstance(y_hat, np.ndarray)
            and y_hat.size != 0
            and y.shape == y_hat.shape
            and y.shape[1] == 1
        ):
            return np.square(y_hat - y).sum() / (2 * y.shape[0])
        return


if __name__ == "__main__":
    X = np.array(
        [[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]]
    )
    Y = np.array([[23.0], [48.0], [218.0]])
    mylr = MyLinearRegression([[1.0], [1.0], [1.0], [1.0], [1]])

    print(mylr.predict_(X))
    # Output: array([[8.], [48.], [323.]])

    print(mylr.loss_elem_(X, Y))
    # Output: array([[225.], [0.], [11025.]])

    print(mylr.loss_(X, Y))
    # Output: 1875.0

    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000

    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output: array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

    print(mylr.predict_(X))
    # Output: array([[23.417..], [47.489..], [218.065...]])

    print(mylr.loss_elem_(X, Y))
    # Output: array([[0.174..], [0.260..], [0.004..]])

    print(mylr.loss_(X, Y))
    # Output: 0.0732..
