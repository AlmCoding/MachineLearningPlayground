import numpy as np


def l2_cost(x_D, y_D, predict_f):
    y_predict = predict_f(x_D)
    return np.sum(0.5*(y_predict-y_D)**2)
