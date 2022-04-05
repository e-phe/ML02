#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Return:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same shapes.
    None if y or y_hat is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(y, np.ndarray)
        and y.size != 0
        and isinstance(y_hat, np.ndarray)
        and y_hat.size != 0
        and y.shape == y_hat.shape
        and y.shape[1] == 1
    ):
        return np.square(y_hat - y).sum() / (2 * y.shape[0])
    return


if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    print(loss_(X, Y))
    # Output: 2.1428571428571436

    print(loss_(X, X))
    # Output: 0.0
