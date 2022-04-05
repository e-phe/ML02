#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible shapes.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of shapes n * 1,
    containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(x, np.ndarray)
        and x.size != 0
        and len(x.shape) == 2
        and isinstance(y, np.ndarray)
        and y.size != 0
        and len(y.shape) == 2
        and y.shape[1] == 1
        and x.shape[0] == y.shape[0]
        and isinstance(theta, np.ndarray)
        and theta.size != 0
        and len(theta.shape) == 2
        and theta.shape[1] == 1
        and x.shape[1] + 1 == theta.shape[0]
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        return x.T @ (x @ theta - y) / x.shape[0]
    return


if __name__ == "__main__":
    x = np.array(
        [
            [-6, -7, -9],
            [13, -2, 14],
            [-7, 14, -1],
            [-8, -4, 6],
            [-5, -9, 6],
            [1, -5, 11],
            [9, -11, 8],
        ]
    )
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    theta1 = np.array([[0], [3], [0.5], [-6]])
    print(gradient(x, y, theta1))
    # Output: array([[ -33.71428571],[ -37.35714286],[ 183.14285714],[ -393.]])

    theta2 = np.array([[0], [0], [0], [0]])
    print(gradient(x, y, theta2))
    # Output: array([[ -0.71428571],[ 0.85714286],[ 23.28571429],[ -26.42857143]])
