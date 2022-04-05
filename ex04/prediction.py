#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shapes m * n.
    theta: has to be an numpy.array, a vector of shapes (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of shapes m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(x, np.ndarray)
        and x.size != 0
        and isinstance(theta, np.ndarray)
        and theta.size != 0
        and x.shape[1] + 1 == theta.shape[0]
        and theta.shape[1] == 1
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        return x @ theta
    return
