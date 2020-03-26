# 写成函数 & 函数实例的形式
# 写的很美
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 构建一个初始化器：
# tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))
# 返回6*6的矩阵，产生于low和high之间，产生的值是均匀分布的。
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in / fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low,
                             maxval=high, dtype=tf.float32)

"""
n_input ：输入变量数目
n_hidden ： 隐含层节点数
transfer_function ；隐含层激活函数，这里采用softplus
optimizer ：优化器，这里用Adam
scale ：高斯噪声系数，用来制造干扰项的一个系数
"""


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])

        # self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
        #                                              self.weights["w1"]), self.weights["b1"]))
        # self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights["w2"]),
        #                              self.weights["b2"])
        # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # self.optimizer = optimizer.minimize(self.cost)
        #
        # init = tf.global_variables_initializer()
        # self.sess = tf.Session()
        # self.sess.run(init)
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        #init = tf.global_variables_initializer()
        init = tf.global_variables_initializer()  # 别漏掉，更别漏掉括号（我的心好痛！）
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()  # 创建一个空字典的一种方式
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 执行一步训练的函数；
    # 功能就是：用一个batch的数据进行训练，并返回当前的损失cost：
    # 执行两个计算图节点：损失cost & 训练过程（优化器）optimizer
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 用在最后的测试集上的计算cost的函数；与上面的不一样。上面是训练中使用的；
    # 执行一个计算图节点：self.cost
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 用来返回隐含层的输出：
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 将隐含层的输出作为输入，将隐含层的高阶特征复原为原始数据：
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'].shape)
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

    # 包含了上面的两个环节：高阶特征提取 & 复原数据
    # 输入：源数据；输出：复原后的数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    # 获取隐含层的权重 w1：
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐含层的偏置系数 b1：
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 这事一个对训练、测试数据 进行 标准化处理的函数。变成 ：均值：0 ， 标准差：1
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)  # 先在训练集上 fit 出一个Scaler（噪声？）
    X_train = preprocessor.transform(X_train)  # 可以直接使用上面定义过得transform 方法函数吗？
    X_test = preprocessor.transform(X_test)
    return X_train,X_test


# 说这事不放回抽样？取走就没放回了吗？没明白；
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# if __name__ == '__main__':
# 加载数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 对数据进行标准化变换
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
# 定义几个常用参数：总训练样本数，最大训练轮数（epochs），batch_size，设置每隔一轮就显示一次损失cost
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

# 创建一个自编码器的 实例 ，定义模型的输入节点n_input为784，自编码器的隐含层节点数n_hidden为200，
# 隐含层激活函数为softplus，优化器为Adam且学习率为0.001，噪音系数scale设为0.01
autoencoder = AdditiveGaussianNoiseAutoencoder(
    n_input=784,
    n_hidden=200,
    transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    scale=0.01
)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size


    if epoch % display_step == 0:
        print("Epoch: %04d" % (epoch + 1), "cost={:.9f}".format(avg_cost))

print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))










