import tensorflow as tf
import numpy as np
"""
tf.concat是连接两个矩阵的操作
# def concat(values, axis, name="concat"):

如果concat_dim是0，那么在某一个shape的第一个维度上连，对应到实际，就是叠放到列上
如果concat_dim是1，那么在某一个shape的第二个维度上连
如果有更高维，最后连接的依然是指定那个维：

例子：https://blog.csdn.net/leviopku/article/details/82380118
# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
"""

data = np.array([
    [1,2,3,4,5,6,7,8,9,0],
    [11,12,13,14,15,16,17,18,19,1],
    [21,22,23,24,25,26,27,28,29,30],
    [31,32,33,34,35,36,37,38,39,40]
])
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 按行切分（按第一维度）
data1,data2 = tf.split(data,[2,2],0)

row1 =  data1.eval()
print(row1)
#[[ 1  2  3  4  5  6  7  8  9  0]
# [11 12 13 14 15 16 17 18 19  1]]

row2 = data2.eval()
print(row2)
#[[21 22 23 24 25 26 27 28 29 30]
# [31 32 33 34 35 36 37 38 39 40]]

# 按列切分（第二维度）
col1,col2 = tf.split(data,[5,5],1)

col1_eval = col1.eval()
print(col1_eval)
"""
[[ 1  2  3  4  5]
 [11 12 13 14 15]
 [21 22 23 24 25]
 [31 32 33 34 35]]
"""

col2_eval = col2.eval()
col2.eval()
print(col2_eval)
"""
[[ 6  7  8  9  0]
 [16 17 18 19  1]
 [26 27 28 29 30]
 [36 37 38 39 40]]
"""
# 在列方向上拼接
data1 = tf.concat([row1,row2],axis=1)
data1.eval()
print(data1.eval())
"""
[[ 1  2  3  4  5  6  7  8  9  0 21 22 23 24 25 26 27 28 29 30]
 [11 12 13 14 15 16 17 18 19  1 31 32 33 34 35 36 37 38 39 40]]
"""

# 在行方向上拼接
data2 = tf.concat([col1_eval,col2_eval],0)
print(data2.eval())
"""
[[ 1  2  3  4  5]
 [11 12 13 14 15]
 [21 22 23 24 25]
 [31 32 33 34 35]
 [ 6  7  8  9  0]
 [16 17 18 19  1]
 [26 27 28 29 30]
 [36 37 38 39 40]]

"""
# numpy 版本  列方向上（第二维度）均分
data_np_sp = np.split(data,2,axis=1)
print(data_np_sp)
"""
[array([[ 1,  2,  3,  4,  5],
       [11, 12, 13, 14, 15],
       [21, 22, 23, 24, 25],
       [31, 32, 33, 34, 35]]), array([[ 6,  7,  8,  9,  0],
       [16, 17, 18, 19,  1],
       [26, 27, 28, 29, 30],
       [36, 37, 38, 39, 40]])]
"""

#列方向（第二维度）可以不均分，列表的第一个数量根据情况变化
data_np_asp = np.array_split(data,3,axis=1)
print(data_np_asp)

"""
[array([[ 1,  2,  3,  4],
       [11, 12, 13, 14],
       [21, 22, 23, 24],
       [31, 32, 33, 34]]), array([[ 5,  6,  7],
       [15, 16, 17],
       [25, 26, 27],
       [35, 36, 37]]), array([[ 8,  9,  0],
       [18, 19,  1],
       [28, 29, 30],
       [38, 39, 40]])]
"""
# 在行方向上拼接（第一维）
data_np_con_0 = np.concatenate(np.array(data_np_sp),axis = 0)
print(data_np_con_0)

"""
[[ 1  2  3  4  5]
 [11 12 13 14 15]
 [21 22 23 24 25]
 [31 32 33 34 35]
 [ 6  7  8  9  0]
 [16 17 18 19  1]
 [26 27 28 29 30]
 [36 37 38 39 40]]
"""

# 在列方向上拼接
data_np_con_1 = np.concatenate((data_np_asp[0],data_np_asp[1],data_np_asp[2]),axis = 1)
print(data_np_con_1)
"""
[[ 1  2  3  4  5  6  7  8  9  0]
 [11 12 13 14 15 16 17 18 19  1]
 [21 22 23 24 25 26 27 28 29 30]
 [31 32 33 34 35 36 37 38 39 40]]
"""