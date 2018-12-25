"""
    Single Neuron Linear Regression with data normalization
    N = #Samples
    i = [0, ... ,N]
    M = Dimension of samples +1(offset b)
    Output a_i
    a_i = w.transpose()*x_i
    Cost C
    C = sum(0.5*(a_i-y_i)**2)
"""

import numpy as np
from dataset1_linreg import DataSet


normalize_data = False
y_D, x_D = DataSet.get_data()
DataSet.plot_data()
# Initialize weights
w = np.array([[0.1], [0.1]])


def get_norm_params(x_arr):
    mean_x = np.sum(x_arr)/x_arr.size
    xmm = x_arr - mean_x
    stdd_x = np.sqrt(np.sum(xmm**2)/x_arr.size)
    return mean_x, stdd_x


mean, stdd = get_norm_params(x_D)


def predict_y1(x_arr):
    if normalize_data:
        # Normalize Input Samples
        x_arr = (x_arr - mean) / stdd

    a = w[1][0] * x_arr + w[0][0]
    return a


DataSet.plot_model(predict_y1)


def l2_cost(x_arr, y_arr, predict_f):
    cost = np.sum((y_arr - predict_f(x_arr)) ** 2) * 0.5
    return cost


print('Initial cost: %.3f' % l2_cost(x_D, y_D, predict_y1))


# Compute gradient of cost function
def gradient_w(x_arr, y_arr, predict_f):
    amy = (predict_f(x_arr) - y_arr)
    ones = np.ones(x_arr.shape)
    X = np.vstack((ones, x_arr))
    grad_w = amy.dot(X.transpose())
    return grad_w.transpose()


grad = gradient_w(x_D, y_D, predict_y1)
print("Initial gradient:{}".format(grad))

# Weight tuning
i = 0
while l2_cost(x_D, y_D, predict_y1) > 0.9:
    w = w - 0.01 * gradient_w(x_D, y_D, predict_y1)
    i += 1

DataSet.plot_model(predict_y1)
print('Final Cost: %f' % l2_cost(x_D, y_D, predict_y1))
print("Iteration needed: {}".format(i))

