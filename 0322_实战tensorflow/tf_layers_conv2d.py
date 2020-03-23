import tensorflow as tf
#输入为100个数据集，28*28像素，3个通道
input = tf.Variable(tf.random_normal([100,28,28,3]))
#卷积核的尺寸为3*3，步幅为2，卷积核个数为32
# filters是一个表示卷积核数量的整数,也就是最终输出的最后一个数字：输出通道数
# 而tf.nn.conv2d的filter是我们已知的卷积核

# kernel_size：一个整数，或者包含了两个整数的元组/队列，表示卷积窗的高和宽。
# 如果是一个整数，则宽高相等。 
output = tf.layers.conv2d(input, filters=32, kernel_size=3,
                         strides=2, padding="SAME",
                         activation=tf.nn.relu, name="")
# padding = "SAME"，则计数从 0 开始，最后多出来的补 0 ，凑成一组；
print(output)  # SAME的时候：Tensor("conv2d/Relu:0", shape=(100, 14, 14, 32), dtype=float32)
			   # VALID的时候：Tensor("conv2d/Relu:0", shape=(100, 13, 13, 32), dtype=float32)


# ********************************************* #
#以下是tf.nn.conv2d()的一个例子：
#输入为100个数据集，28*28像素，3个通道
# input = tf.Variable(tf.random_normal([100,28,28,3]))
# #卷积核过滤器尺寸为3*3，输入3个通道，输出10个通道，也就是10个卷积核
# filter = tf.Variable(tf.random_normal([3,3,3,10]))
 
# output = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='VALID')
# print(output)  # Tensor("Conv2D:0", shape=(100, 13, 13, 10), dtype=float32)

"""
tf.layers.conv2d(
    inputs,
    filters,  # filters是一个表示卷积核数量的整数
    		  # 而tf.nn.conv2d的filter是我们已知的卷积核
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
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