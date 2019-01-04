import matplotlib.pyplot as plt
from dataset2_logreg import DataSet
from neuralnetworks.NeuralNetwork import NeuralNetwork
from neuralnetworks import activation_functions as af
from neuralnetworks import cost_functions as cf


# get and plot the data
y_D, x_D = DataSet.get_data()
DataSet.plot_data()
plt.show()

lbd_regularization = 0.07
NN = NeuralNetwork((2, 6, 6, 6, 1), af.sigmoid, af.d_sigmoid, af.sigmoid, af.d_sigmoid)
NN.gd_learn(70000, 0.01, x_D, y_D, NN.bp_ce, lbd_regularization)

# plot and compute cost
DataSet.plot_decision_boundary(NN.output)
plt.show()
print('Cost:%f' % cf.ce_cost(x_D, y_D, NN.output))
