import tensorflow as tf
import numpy as np

# a = tf.Variable(tf.random_normal([16,100,80,1]))
# a = tf.Variable(np.random.rand(16,100,80,1))
# a = tf.convert_to_tensor(np.random.rand(16,100,80,1))  # 将ndarray 转化为 tensor，因为要送进卷积的，应该是tensor数据类型才可以
# a = tf.random_normal((16,100,80,1))
a = tf.random_uniform((16,100,80,1))

print(a.shape)
print(type(a))

a1 = tf.layers.conv2d(a, filters=24, kernel_size=(3,1), strides=1, padding="VALID", activation=tf.nn.relu)

print(a1.shape)
# a1 = tf.layers.conv2d(a,)
