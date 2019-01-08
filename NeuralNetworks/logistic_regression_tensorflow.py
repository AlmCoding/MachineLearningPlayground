import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Download mnist data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Show 25 training images as an example
plt.figure(1)
for i in range(0, 25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i][:][:], cmap='gray')
plt.show()


def convert_labels(x):
    y = np.zeros(shape=(10,), dtype=np.uint8)
    y[x] = 1
    return y


# Bring data into the right form for our model
y_train = np.array([convert_labels(y) for y in y_train]).T
x_train = np.array([np.reshape(x/255.0, (784,)) for x in x_train]).T

batch_size = 100
num_batches_per_epoch = int(y_train.shape[1] / batch_size)
num_epochs = 100

X = tf.placeholder(tf.float64, shape=(784, batch_size), name='X')
Y = tf.placeholder(tf.float64, shape=(10, batch_size), name='Y')

W = tf.get_variable('W', shape=(10, 784), dtype=tf.float64, initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable('b', shape=(10, 1), dtype=tf.float64, initializer=tf.zeros_initializer())

Y_model = tf.nn.softmax(tf.matmul(W, X) + b, axis=0)

# loss_fun = -tf.reduce_sum(tf.log(tf.nn.softmax(Y_model, axis=0)) * Y, axis=0)
loss_fun = -tf.reduce_sum(Y*tf.log(Y_model), axis=0)
loss_fun = tf.reduce_mean(loss_fun)

#
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
opt = optimizer.minimize(loss_fun)

with tf.Session() as sess:
    # tf.summary.FileWriter('tmp/optim', tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    total_loss = 0.0
    for i in range(num_epochs):
        for j in range(num_batches_per_epoch):
            batch_x = x_train[:, j*batch_size:(j+1)*batch_size]
            batch_y = y_train[:, j*batch_size:(j+1)*batch_size]
            l, _ = sess.run([loss_fun, opt], feed_dict={X: batch_x, Y: batch_y})
            total_loss += l
        print('{}: {}'.format(i, total_loss/(i+1)))

    W_final, b_final = sess.run([W, b])

# Performance evaluation
y_test = np.array([convert_labels(y) for y in y_test]).T
x_test = np.array([np.reshape(x/255.0, (784,)) for x in x_test]).T

# def np_softmax(x):
#    return np.exp(x) / np.sum(np.exp(x))

tmp = tf.nn.softmax(np.matmul(W_final, x_test) + b_final, axis=0) # b_final is broadcasted here on the resulting matrix
with tf.Session() as sess:
    tmp = sess.run(tmp)
Yfinal = np.zeros((y_test.shape[1], 10))
correct = np.zeros(y_test.shape[1])
for i in range(Yfinal.shape[0]):
    # correct[i] = np.argmax(y_test[:, i]) == np.argmax(np_softmax(tmp[:, i]))
    correct[i] = np.argmax(y_test[:, i]) == np.argmax(tmp[:, i])
print('Score on the test data set: {}'.format(np.mean(correct)))


"""
# Evaluation part 2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    W.assign(W_final)
    b.assign(b_final)
    correct_assignments = 0
    for j in range(int(y_test.shape[1] / batch_size)):
        batch_x = x_test[:, j*batch_size:(j+1)*batch_size]
        batch_y = y_test[:, j*batch_size:(j+1)*batch_size]
        y_results = sess.run(Y_model, feed_dict={X: batch_x})

        def discretize(x):
            y = np.zeros(shape=(10,), dtype=np.uint8)
            y[np.unravel_index(np.argmax(x), x.shape)] = 1
            return y

        y_results = np.array([discretize(element) for element in y_results.T]).T

        matches = batch_y == y_results
        a = np.sum(matches, axis=0)
        b = a/10
        c = b == 1
        correct_assignments += np.sum(c, axis=0)

    a = correct_assignments/y_test.shape[0]
    print("Correct classifications: {}%".format(correct_assignments/y_test.shape[0]))
"""

"""
tf.cast(x, dtype=tf.float64)
            def reshape_samples(x):
                y = np.reshape(x, (in_rows,))
                return y
"""
