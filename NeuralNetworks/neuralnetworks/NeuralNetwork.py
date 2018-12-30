import numpy as np


class NeuralNetwork:
    """
    Class implementing a feed forawrd neural network.
    Object fields:
        layers = a tuple containing numbers of neurons in each layer, starting from the input layer

        L = depth of the NN, eg, with depth L there are L matrices W: W[1], ...,W[L]

        act_hid   = activation function for neurons in the hidden layer
        d_act_hid = derivative of the activation function for neurons in the hidden layer
        act_out   = activation function for neuron(s) in the output layer
        d_act_out = derivative of the activation function for neuron(s) in the output layer

        W = dictionary containing the W matrices for each layer. The keys are arranged such that the matrices
            stored in the dictionary corresponds to the notation form the lecture. Ie, W[1] is the matrix which
            describes the connections between the layer 0 and layer 1. The matrix stored at W[1] is a numpy array
            with dimensions (number of neurons in the layer 1) x (number of neurons in the layer 0)

        b = dictionary containing the b vectors for each layer. The indexing corresponds to the indexing from
            the lecture. See above. Eg, dimensions of b[1] (number of neurons in the layer 1) x  1
    """

    def __init__(self, layers, act_hid, d_act_hid, act_out, d_act_out):
        self.layers = layers
        self.L = len(layers) - 1
        self.act_hid = act_hid
        self.d_act_hid = d_act_hid
        self.act_out = act_out
        self.d_act_out = d_act_out
        self.W, self.b = self.init_Wb()

    def init_Wb(self):
        """
        Initialize the matrices W[1],...,W[L] and the vectors b[1],...,b[L] with random numbers from gaussian
        distribution with 0-mean, and 0.25 variance. Note that W, b are dictionaries with integer keys.
        """
        W, b = {}, {}
        mu = 0
        sigma = np.sqrt(0.25)
        for l in range(1, self.L + 1):
            W[l] = np.random.normal(mu, sigma, (self.layers[l], self.layers[l-1]))
            b[l] = np.random.normal(mu, sigma, (self.layers[l], 1))
        return W, b

    def fp(self, x):
        """
        Forward propagation. Uses the current parameters W, b
        Inputs:
            x = np.array of size self.layers[0] x N. This means that this function
                performs the forward propagation for N input vectors (columns).
        Outputs:
            a = dictionary containing output of each layer of NN. Each dictionary stores N outputs
                for each of the inputs. Eg., a[1] should be np.array of size self.layers[1] x N
                The indexing corresponds to the indexing from the lecture. E.g. a[0]=x because a[0]
                contains the N outputs of the input layer, which is the input x.
            z = dictionary containing input to each layer of NN. The indexing corresponds to the indexing
                from the lecture. E.g. z[1]=W[1].dot(a[0])+b[1].
        """
        a, z = {}, {}
        a[0] = x
        for l in range(1, self.L + 1):
            B = np.tile(self.b[l], (1, x.shape[1]))
            z[l] = self.W[l].dot(a[l-1]) + B
            if l == self.L:
                a[l] = self.act_out(z[l])
            else:
                a[l] = self.act_hid(z[l])
        return a, z

    def output(self, x):
        """
        Provides the output from the last layer of NN.
        """
        a, _ = self.fp(x)
        a_out = a[self.L]
        return a_out

    def bp(self, x, y):
        """
        Backpropagation. Uses the current parameters W, b
        Args:
            x = np.array of size self.layers[0] x N (contains N input vectors from the training set)
            y = np.array of size self.layers[L] x N (contains N output vectors from the training set)
        Returns:
            dW = dictionary corresponding to W, where each corresponding key contains a matrix of the
                 same size, eg, W[i].shape = dW[i].shape for all i. It contains the partial derivatives
                 of the cost function with respect to each entry entry of W.
            db = dictionary corresponding to b, where each corresponding key contains a matrix of the
                 same size, eg, b[i].shape = bW[i].shape for all i. It contains the partial derivatives
                 of the cost function with respect to each entry entry of b.
        """

        a, z = self.fp(x)
        L = self.L

        dCdz = {L: a[L] - y}
        for l in range(L - 1, 0, -1):
            dCdz[l] = self.W[l + 1].T.dot(dCdz[l + 1]) * self.d_act_hid(z[l])

        db = {}
        for l in range(1, L + 1):
            db[l] = np.sum(dCdz[l], axis=1).reshape((-1, 1))

        dW = {}
        for l in range(1, L + 1):
            dW[l] = dCdz[l].dot(a[l - 1].T)
        return dW, db

    def gd_learn(self, iter_num, l_rate, x, y):
        """
        Performs gradient descent learning.
        iter_num = number of iterations of GD
        l_rate = learning rate
        x = nparray with the training inputs
        y = nparray with the training outputs
        """
        for i in range(iter_num):
            dW, db = self.bp(x, y)
            for l in range(1, self.L + 1):
                self.W[l] = self.W[l] - l_rate*dW[l]
                self.b[l] = self.b[l] - l_rate*db[l]







