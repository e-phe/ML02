#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of shape m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of shape m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of shape (number of features + 1, 1).
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(x, np.ndarray)
        and x.size != 0
        and isinstance(y, np.ndarray)
        and y.size != 0
        and y.shape[1] == 1
        and x.shape[0] == y.shape[0]
        and isinstance(theta, np.ndarray)
        and theta.size != 0
        and theta.shape[1] == 1
        and x.shape[1] + 1 == theta.shape[0]
        and isinstance(alpha, float)
        and isinstance(max_iter, int)
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        for _ in range(max_iter):
            gradient = np.dot(x.transpose(), x @ theta - y) / x.shape[0]
            theta = theta - alpha * gradient
        return theta
    return


from prediction import predict_

if __name__ == "__main__":
    x = np.array(
        [[0.2, 2.0, 20.0], [0.4, 4.0, 40.0], [0.6, 6.0, 60.0], [0.8, 8.0, 80.0]]
    )
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.0], [1.0], [1.0], [1.0]])

    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    print(theta2)
    # Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])

    print(predict_(x, theta2))
    # Output: array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
