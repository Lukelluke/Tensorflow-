# https://blog.csdn.net/khy19940520/article/details/89934335

# coding: utf-8
import tensorflow as tf
 
num_filters = 2  # num_filters = 2定义有两个过滤器
kernel_size = 2  
# kernel_size = 2定义每个卷积核的宽度为2
# 长度即为字向量长度5。
batch_size = 1  # 一篇文章 
seq_length = 4  # 文本中有 4 个字（假设以字为单位）
embedding_dim = 5  
# 定义一个字的向量长度为5
# 这里初始化每个字的向量都为[1, 1, 1, 1, 1]

embedding_inputs = tf.constant(-1.0, shape=[batch_size, seq_length, embedding_dim], dtype=tf.float32)
# print('************************')
# print(embedding_inputs[0][0][1])
# print('************************')
with tf.name_scope("cnn"):
    conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size, name='conv')

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print('Before: embedding_inputs.shape = '+str(embedding_inputs.shape))
	sess.run(conv)
	print('hello,now conv.shape = '+str(conv.shape))

# session = tf.Session()
# session.run(tf.global_variables_initializer())
 
# print (session.run(conv).shape)  # (1,3,2)


"""
inputs： 输入tensor， 
维度(batch_size, seq_length, embedding_dim) 是一个三维的tensor；
其中，batch_size指每次输入的文本数量；
seq_length指每个文本的词语数或者单字数；
embedding_dim指每个词语或者每个字的向量长度；
例如每次训练输入2篇文本，每篇文本有100个词，每个词的向量长度为20，
那input维度即为(2, 100, 20)。

filters：过滤器（卷积核）的数目
kernel_size：卷积核的大小，卷积核本身应该是二维的，这里只需要指定一维，因为第二个维度即长度与词向量的长度一致，卷积核只能从上往下走，不能从左往右走，即只能按照文本中词的顺序，也是列的顺序。

***************************************
首先，batch_size = 1即为一篇文本，seq_length = 4定义文本中有4个字
（假设以字为单位），embedding_dim = 5定义一个字的向量长度为5，
这里初始化每个字的向量都为[1, 1, 1, 1, 1]，
num_filters = 2定义有两个过滤器，
kernel_size = 2定义每个卷积核的宽度为2，长度即为字向量长度5。
一个卷积核通过卷积操作之后得到
(4-2+1)*1（seq_length - kernel_size + 1）即3*1的向量，
一共有两个卷积核，所以卷积出来的数据维度(1, 3, 2)其中1指一篇文本。

***************************************

经过卷积之后得到2个feature maps，分别经过pooling层之后，
两个3*1的向量就变成两个1*1的常数，
在把这两个1*1的常数拼接在一起变成2*1向量，
之后就可以进行下一步比如全连接或者softmax操作了。
"""

"""
tf.layers.conv1d(
inputs,
filters,
kernel_size,
strides=1,
padding='valid',
data_format='channels_last',
dilation_rate=1,
activation=None,
use_bias=True,
kernel_initializer=None,
bias_initializer=tf.zeros_initializer(),
kernel_regularizer=None,
bias_regularizer=None,
activity_regularizer=None,
kernel_constraint=None,
bias_constraint=None,
trainable=True,
name=None,
reuse=None
)
"""