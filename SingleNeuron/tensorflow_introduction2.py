import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


a_real = 4.5
b_real = -2.0
x_real = np.arange(-3.0, 5.0, 0.1)
y_real = a_real * x_real + b_real

sigma2 = 4.0
y_noisy = y_real + np.sqrt(sigma2) * np.random.randn(100, y_real.shape[0])
print(y_noisy.shape)

plt.plot(x_real, y_real)
plt.plot(x_real, y_noisy[1, :])
plt.plot(x_real, y_noisy[2, :])
plt.show()


tf.reset_default_graph()
X = tf.placeholder(tf.float64, name='X')
Y = tf.placeholder(tf.float64, name='Y')
a_model = tf.get_variable('a_model', initializer=tf.constant(1, dtype=tf.float64))
b_model = tf.get_variable('b_model', initializer=tf.constant(1, dtype=tf.float64))

Y_model = a_model * X + b_model
loss_fun = (Y_model - Y)**2

opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = opt.minimize(loss_fun)

with tf.Session() as sess:
    tf.summary.FileWriter('tmp/optim', tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    # perform training for 100 epochs
    for i in range(100):
        l, _ = sess.run([loss_fun, optimizer], feed_dict={X: x_real, Y: y_noisy[i, :]})
    a_opt, b_opt = sess.run([a_model, b_model])

    print(a_opt)
    print(b_opt)
    print(l.shape)

    plt.plot(x_real, y_real)
    plt.plot(x_real, a_opt*x_real+b_opt)
    plt.show()
    print(a_opt, b_opt)
