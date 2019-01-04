import numpy as np


def l2_cost(x_arr, y_arr, predict_f):
    y_predict = predict_f(x_arr)
    return np.sum(0.5 * (y_predict - y_arr) ** 2)


def ce_cost(x_arr, y_arr, predict_f):
    y_eq_0 = (y_arr == 0).nonzero()[1]
    y_eq_1 = (y_arr == 1).nonzero()[1]
    a = predict_f(x_arr)
    cost = np.sum(-np.log2(a[0][y_eq_1])) + np.sum(-np.log2(1-a[0][y_eq_0]))
    return cost
