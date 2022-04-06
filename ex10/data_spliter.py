#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
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
        and x.shape[0] == y.shape[0]
        and y.shape[1] == 1
        and isinstance(proportion, float)
        and proportion <= 1
    ):
        np.random.shuffle(x)
        np.random.shuffle(y)
        x_train, x_test = np.split(x, [int(proportion * x.shape[0])])
        y_train, y_test = np.split(y, [int(proportion * y.shape[0])])
        return (x_train, x_test, y_train, y_test)
    return


if __name__ == "__main__":
    x1 = np.array([[1], [42], [300], [10], [59]])
    y = np.array([[0], [1], [0], [1], [0]])

    print(data_spliter(x1, y, 0.8))

    print(data_spliter(x1, y, 0.5))

    x2 = np.array([[1, 42], [300, 10], [59, 1], [300, 59], [10, 42]])
    y = np.array([[0], [1], [0], [1], [0]])

    print(data_spliter(x2, y, 0.8))

    print(data_spliter(x2, y, 0.5))
