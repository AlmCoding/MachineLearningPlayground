import numpy as np
import matplotlib.pyplot as plt


def identity(x):
    return x


def d_identity(x):
    return np.ones(shape=x.shape)


def relu(x):
    y = np.copy(x)
    threshold = x < np.zeros(shape=x.shape)
    y[threshold] = 0
    return y


def d_relu(x):
    y = np.ones(shape=x.shape)
    threshold = x < np.zeros(shape=x.shape)
    y[threshold] = 0
    return y


def sigmoid(x):
    return 1/(1+np.exp(-x))


def d_sigmoid(x):
    return 1/(1+np.exp(-x))**2


if __name__ == '__main__':
    x = np.arange(-6, 6, 0.1)
    y = identity(x)
    plt.plot(x, y)
    y = d_identity(x)
    plt.plot(x, y)
    plt.show()

    y = relu(x)
    plt.plot(x, y)
    y = d_relu(x)
    plt.plot(x, y)
    plt.show()

    y = sigmoid(x)
    plt.plot(x, y)
    y = d_sigmoid(x)
    plt.plot(x, y)
    plt.show()




