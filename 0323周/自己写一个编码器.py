# 模仿着，写出自己的三层自编码机，是将中间过程看作 矩阵乘法+b 的形式
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

learning_rate = 0.001
training_epochs = 1
batch_size = 256
display_step = 1

n_input = 784
X = tf.placeholder("float", [None, n_input])

n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 2

# 产生指定尺寸的W，可以用：
# tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))
# 返回6*6的矩阵，产生于low和high之间，产生的值是均匀分布的。
# 也可以用 tf.truncated_normal(shape, mean, stddev)
# 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成
# n_input = [None,784]
weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),  # [784,512]——>[None,512]
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),  # [512,256]——>[None,256]
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),  # [256,2]——>[None,2]

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),  # [2,256]——>[None,256]
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_1],)),  # [256,512]——>[None,512]
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),  # [512,784]——>[None,784]
}

biases = {
    'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],)),  # [512,]
    'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],)),  # [256,]
    'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],)),  # [2,]

    'decoder_b1': tf.Variable(tf.truncated_normal([n_hidden_2],)),  # [256,]
    'decoder_b2': tf.Variable(tf.truncated_normal([n_hidden_1],)),  # [512,]
    'decoder_b3': tf.Variable(tf.truncated_normal([n_input],)),     # [784,]
}


def encoder(x):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))  #  [None,784]*[784,512]+[512,] = [None,512]+[512,]
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))  # [None,512]*[512,256] + [256,] = [None,256]
    layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))  # [None,256]*[256,2]+[2,] = [None,2]
    return layer_3

def decoder(x):
    # 小心，下面第一层的输入，是这个函数的输入：x，所以应该是x，而不是上面的 decoder（x）!!!
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))  # [None,2]*[2,256] + [256,] = [None,256]
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))  # [None,256]*[256,512] + [512,] = [None,512]
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))  # [None,512]*[512.]
    return layer_3


encode_op = encoder(X)
decode_op = decoder(encode_op)

y_pred = decode_op  # 这个地方别搞乱，这是预测的，所以应该是解码出来的
y_true = X  # 这事真实值，所以用该用输入X

# cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        if epoch % display_step == 0:
            print("Epoch:", '%04d'%(epoch+1),
                  "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    # encoder_result = sess.run(encode_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()

    encoder_result = sess.run(encode_op, feed_dict={X: mnist.test.images})

    # print(mnist.test.labels.shape)
    # c=mnist.test.labels
    # print(type(c))
    # 这里，如果在最开始导入mnist的时候，one_hot选择true，那么c的尺寸就是（10000，10）
    # 而one_hot=false的时候，c = (10000,)
    # 所以，暂时理解（要画图的话，就不用one_hot独热编码了）
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # 说明，参数c就是color，赋值为可迭代参数对象，这里表示以labels的种类数目伟分类的颜色数目
    plt.colorbar()
    plt.show()




