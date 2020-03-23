"""
tf.split(
    value,
    num_or_size_splits,
    axis=0,
    num=None,
    name='split'
)

value:　　待切分的张量
num_or_size_splits:　　切分的个数
axis： 沿着哪个维度切分

value： 输入的tensor
num_or_size_splits: 如果是个整数n，就将输入的tensor分为n个子tensor。
如果是个tensor T，就将输入的tensor分为len(T)个子tensor。

axis： 默认为0，计算value.shape[axis], 一定要能被num_or_size_splits整除。
"""
import tensorflow as tf
import numpy as np
# 例子：
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
value = np.reshape(range(150),(5,30))

split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
print(split0.shape, split1.shape, split2.shape) 
# (5, 4) (5, 15) (5, 11)
# [5, 4]
# [5, 15]
# [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0)  # [5, 10]
print(split0.shape, split1.shape, split2.shape)
# (5, 10) (5, 10) (5, 10)


