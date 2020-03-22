import numpy as np
import tensorflow as tf

## 1 测试 np.ndarray（）和np.asarray()常规创建方法
#https://blog.csdn.net/Rex_WUST/article/details/85205179

a = np.array([2,3,4])
b = np.array([2.0,3.0,4.0])
c = np.array([[1.0,2.0],[3.0,4.0]])
d = np.array([[1,2],[3,4]],dtype=complex) # 指定数据类型
print (a, a.dtype)
print (b, b.dtype)
print (c, c.dtype)
print (d, d.dtype)

data1 = np.array(a)
data2 = np.asarray(a)
print(data1,data2)

print('\n****************\n')
##	2	测试 tf.cast() 数据类型转换

t1 = tf.Variable([1,2,3,4,5])
t2 = tf.cast(t1,dtype=tf.float32)
 
print('t1: {}'.format(t1))
print('t2: {}'.format(t2))
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(t2)
    print(t2.eval())
    # print(sess.run(t2))