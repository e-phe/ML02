#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(x, np.ndarray)
        and x.size != 0
        and len(x.shape) == 2
        and isinstance(theta, np.ndarray)
        and theta.size != 0
        and len(theta.shape) == 2
        and x.shape[1] + 1 == theta.shape[0]
        and theta.shape[1] == 1
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        return x @ theta
    return


if __name__ == "__main__":
    x = np.arange(1, 13).reshape((4, 3))

    theta1 = np.array([[5], [0], [0], [0]])
    print(simple_predict(x, theta1))
    # Output: array([[5.],[5.],[5.],[5.]])

    theta2 = np.array([[0], [1], [0], [0]])
    print(simple_predict(x, theta2))
    # Output: array([[1.],[4.],[7.],[10.]])

    theta3 = np.array([[-1.5], [0.6], [2.3], [1.98]])
    print(simple_predict(x, theta3))
    # Output: array([[9.64],[24.28],[38.92],[53.56]])

    theta4 = np.array([[-3], [1], [2], [3.5]])
    print(simple_predict(x, theta4))
    # Output: array([[12.5],[32.],[51.5],[71.]])
