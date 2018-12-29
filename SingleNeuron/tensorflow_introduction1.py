import tensorflow as tf

EXAMPLE = 5
print('Example: ' + str(EXAMPLE))

a = tf.add(3, 5)
b = tf.multiply(a, 4)
c = tf.subtract(b, 1)

if EXAMPLE == 1:
    print(a)
    print(b)
    print(c)

    sess = tf.Session()
    res = sess.run([c])
    print(res)
    sess.close()

    with tf.Session() as sess:
        res = sess.run([a, b, c])
        print(res)

    tf.reset_default_graph()


if EXAMPLE == 2:

    sess = tf.Session()
    writer = tf.summary.FileWriter('tmp/tf_intro', tf.get_default_graph())
    res = sess.run([c])
    print(res)
    sess.close()

    tf.reset_default_graph()
    a = tf.add(3, 5, name='add_op')
    b = tf.multiply(a, 4, name='mul_op')
    c = tf.subtract(b, 1, name='sub_op')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tmp/tf_intro', tf.get_default_graph())
        res = sess.run([c])
        print(res)


if EXAMPLE == 3:
    tf.reset_default_graph()
    a = tf.constant(1, name='avar')
    b = tf.constant(2, name='bvar')
    print(a)
    print(b)
    c = a + b
    d = tf.add(a, b)
    print(c)
    print(d)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tmp/tf_intro', tf.get_default_graph())
        res = sess.run([c])
        print(res)


if EXAMPLE == 4:
    tf.reset_default_graph()
    a = tf.get_variable('a', initializer=tf.constant([5, 6], shape=(2, 1)))
    b = tf.get_variable('b', initializer=tf.constant([3, 4], shape=(1, 2)))
    c = a + b
    print(a)
    print(b)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tmp/tf_intro', tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        res = sess.run([a, b, c])
        print(res)


if EXAMPLE == 5:
    tf.reset_default_graph()
    a = tf.placeholder(tf.float32, shape=(1, 2))
    b = tf.placeholder(tf.float32, shape=(1, 2))
    c = a + b

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tmp/tf_intro1', tf.get_default_graph())
        res = sess.run(c, feed_dict={a: [[2, 3]], b: [[0, 1]]})
        print(res)




