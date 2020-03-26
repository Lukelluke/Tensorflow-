# 更简洁的一种写法
# 直接用上自带的dense全连接来实现：
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# Hyperparameters
Batch_size = 64
learning_rate = 0.001
N_test_IMG = 5

tf_x = tf.placeholder(tf.float32, [None, 28 * 28])

# encoder
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
encoded = tf.layers.dense(en2, 3, tf.nn.tanh)

# decoder
de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
decoded = tf.layers.dense(de2, 28 * 28, tf.nn.tanh)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        train_images, train_labels = mnist.train.next_batch(Batch_size)
        _, loss_ = sess.run([train, loss], feed_dict={tf_x: train_images})

        if step % 100 == 0:
            print('step:{}，train loss = {:.4f}'.format(step, loss_))
