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
y_train = np.array([convert_labels(y) for y in y_train])
x_train = np.array([np.reshape(x/255.0, (784,)) for x in x_train])
y_test = np.array([convert_labels(y) for y in y_test])
x_test = np.array([np.reshape(x/255.0, (784,)) for x in x_test])


tf.reset_default_graph()
data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
data.shuffle(10000)
batch_size = 128
num_epochs = 100
data = data.batch(batch_size)
it = data.make_initializable_iterator()

X, Y = it.get_next()
print(X)

N1 = 16     # number of neurons of layer 1
N2 = 16     # number of neurons of layer 2

W1 = tf.get_variable('W1', shape=(N1, 784), dtype=tf.float64, initializer=tf.random_normal_initializer(0, 0.1))
W2 = tf.get_variable('W2', shape=(N2, N1), dtype=tf.float64, initializer=tf.random_normal_initializer(0, 0.1))
W3 = tf.get_variable('W3', shape=(10, N2), dtype=tf.float64, initializer=tf.random_normal_initializer(0, 0.1))

b1 = tf.get_variable('b1', shape=(N1, 1), dtype=tf.float64, initializer=tf.zeros_initializer())
b2 = tf.get_variable('b2', shape=(N2, 1), dtype=tf.float64, initializer=tf.zeros_initializer())
b3 = tf.get_variable('b3', shape=(10, 1), dtype=tf.float64, initializer=tf.zeros_initializer())

# z1 = tf.nn.relu(np.array([tf.matmul(W1, X)]).T + b1)
z1 = tf.nn.relu(np.transpose(tf.matmul(X, W1.T)) + b1)
z2 = tf.nn.relu(tf.matmul(W2, z1) + b2)
Y_model = tf.matmul(W3, z2) + b3
print(Y_model)

loss_fun = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_model, labels=Y)
loss_fun = tf.reduce_mean(loss_fun)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
opt = optimizer.minimize(loss_fun)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tot_loss = 0
    for i in range(num_epochs):
        sess.run(it.initializer)  # important: The iterator has to be initialized in each epoch!
        try:
            while 1:
                l, _ = sess.run([loss_fun, opt])
                tot_loss = tot_loss + l
        except  tf.errors.OutOfRangeError:
            pass
        print('Loss {}: {}'.format(i, tot_loss / (i + 1)))

    pred = tf.nn.softmax(Y_model)
    res_pred, = sess.run([pred], feed_dict={X: x_test})
    correct = np.zeros((res_pred.shape[0], 1))
    for j in range(len(res_pred)):
        correct[j] = (np.argmax(res_pred[j, :]) == np.argmax(y_test[j, :]))
    print(np.mean(correct))