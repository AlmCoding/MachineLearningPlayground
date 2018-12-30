import numpy as np
import matplotlib.pyplot as plt
from dataset2_linreg import DataSet
from neuralnetworks.NeuralNetwork import NeuralNetwork
from neuralnetworks import activation_functions as af
from neuralnetworks import cost_functions as cf


y_D, x_D = DataSet.get_data()
DataSet.plot_data()
plt.show()


NN = NeuralNetwork((1, 4, 4, 1), af.sigmoid, af.d_sigmoid, af.identity, af.d_identity)
DataSet.plot_model(NN.output)
plt.show()
print('Cost: %f' % cf.l2_cost(x_D, y_D, NN.output))


NN.gd_learn(100000, 0.02, x_D, y_D)

# plot and compute cost
DataSet.plot_model(NN.output)
plt.show()
print('Cost: %f' % cf.l2_cost(x_D, y_D, NN.output))
