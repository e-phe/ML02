#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from multivariate_linear_model import MyLinearRegression as MyLR
from multivariate_linear_model import MyMultiLinearRegression as MyMLR
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

if __name__ == "__main__":
    print("Linear Regresion\n")
    data = pd.read_csv("../resources/spacecraft_data.csv")
    X = np.array(data[["Age"]])
    Y = np.array(data[["Sell_price"]])
    myLR_age = MyLR(theta=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)

    print(myLR_age.fit_(X[:, 0].reshape(-1, 1), Y))

    print(myLR_age.mse_(X[:, 0].reshape(-1, 1), Y))
    # Output: 57636.77729...
    print(
        myLR_age.mse_(X[:, 0].reshape(-1, 1), Y)
        == mean_squared_error(X[:, 0].reshape(-1, 1), Y)
    )

    # data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    # Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    # Yscore = np.array(data["Score"]).reshape(-1, 1)

    # linear_model1 = MyLR(np.array([[89.0], [-8]]))
    # Y_model1 = linear_model1.predict_(Xpill)
    # print(linear_model1.mse_(Yscore, Y_model1))
    # # Output: 57.60304285714282
    # print(linear_model1.mse_(Yscore, Y_model1) == mean_squared_error(Yscore, Y_model1))

    # linear_model2 = MyLR(np.array([[89.0], [-6]]))
    # Y_model2 = linear_model2.predict_(Xpill)
    # print(linear_model2.mse_(Yscore, Y_model2))
    # # Output: 232.16344285714285
    # print(linear_model2.mse_(Yscore, Y_model2) == mean_squared_error(Yscore, Y_model2))

    print("\nMultivariate Linear Regresion\n")
    data = pd.read_csv("../resources/spacecraft_data.csv")
    X = np.array(data[["Age", "Thrust_power", "Terameters"]])
    Y = np.array(data[["Sell_price"]])
    my_lreg = MyMLR(theta=[[1.0], [1.0], [1.0], [1.0]], alpha=5e-5, max_iter=600000)

    print(my_lreg.mse_(X, Y))
    # Output: 144044.877...
    print(my_lreg.mse_(X, Y) == mean_squared_error(my_lreg.predict_(X), Y))

    print(my_lreg.fit_(X, Y))
    # Output: array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

    print(my_lreg.mse_(X, Y))
    # Output: 586.896999...
    print(my_lreg.mse_(X, Y) == mean_squared_error(my_lreg.predict_(X), Y))
