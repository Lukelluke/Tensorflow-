
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)  # 训练集
print(mnist.test.images.shape, mnist.test.labels.shape)    # 测试集
print(mnist.validation.images.shape, mnist.validation.labels.shape)  # 验证集
# (55000, 784) (55000, 10)
# (10000, 784) (10000, 10)
# (5000, 784) (5000, 10)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])  # x 也是真实值，是图像数据，这里只需要输入的维度是784就可以，不限制是多少张图片；
W = tf.Variable(tf.zeros([784, 10]))  # 这行出问题，原本写错写成了placeholder，这个是要预测的值，所以不应该是placeholder
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # x * W + b
# [None,784]x[784,10] = [None,10]  + (10,)
# (10,)说明是一维的向量，总共有十个数字，[x1,x2,x3, ... ,x10]

# define loss 损失函数:
# y：预测值
# y_：样本真实值
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 定义训练步骤：
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局初始化，并直接执行全局参数初始化器的 run 方法：

init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, {x: batch_xs, y_: batch_ys})
    acc = sess.run(accuracy, {x:mnist.test.images, y_: mnist.test.labels})
    print("Iter " + str(i) + ",Testing Accuracy " + str(acc))

# 上下这两种，是tensorflow的sess调用的两种方法：
# tf.global_variables_initializer().run()
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)  # 查查看其他数据类型情况下，怎么获得下一个batch数据
#     # print(batch_xs.shape,batch_ys.shape)  # (100, 784) (100, 10)
#     train_step.run({x: batch_xs, y_: batch_ys})
#
# # 下面开始进行对上面训练结果的准确率做计算，并输出准确率：
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval({x:mnist.test.images, y_: mnist.test.labels}))
# # tf.argmax(input, axis=None, name=None, dimension=None)
# # axis：0表示按列，1表示按行






# 下面是网上找到的另一份很漂亮的代码
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# # 载入数据集
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# # 每个批次100张照片
# batch_size = 100
#
# # 计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# # 定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
#
# # 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(x, W) + b)
#
# # 二次代价函数
# # loss = tf.reduce_mean(tf.square(y-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#
# # 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 结果存放在一个布尔型列表中
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# # 求准确率
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(11):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
#         acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#         print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
#









# 这是网上一份，显示mnist数据格式的漂亮代码；验证了自己是labels 写错，写成了labels；正确写法是：labels
# from tensorflow.examples.tutorials.mnist import input_data
#
# print('packs loaded')
# print('download and extract mnist set')
# mnist = input_data.read_data_sets('MNIST_data/',
#                                   one_hot=True)  # one_hot 是0,1的编码格式,read_data_sets 用于下载数据集(文件目录里有就不用下载,改成直接读取,没有就需要下载;
#
# print('the type of mnist:%s' % type(mnist))
# print('the number of train data:%d' % (mnist.train.num_examples))
# print('the number of test data:%d' % (mnist.test.num_examples))
#
# print('把数据集分开,分别有四个:')
# print('1:train images,2:train labels,3:test images,4:test labels')
# print('what does the mnist look like?')
#
# #image图片，训练数据
# train_img = mnist.train.images
#
# #训练标签，
# train_label = mnist.train.labels
#
# #测试数据
# test_img = mnist.test.images
#
# #测试标签
# test_label = mnist.test.labels
# print('type-------------->>>>>')
# print('train images type%s'%type(train_img))
# print('train labels type%s'%type(train_label))
# print('test images type%s'%type(test_img))
# print('test labels type%s'%type(test_label))
# print('shape------------------->>>')
# print('train images shape%s'%(train_img.shape,)) #特别要注意这个逗号,不然会报TypeError: not all arguments converted during string formatting
# print('train labels shape%s'%(train_label.shape,))
# print('test images shape%s'%(test_img.shape,))
# print('test labels shape%s'%(test_label.shape,))