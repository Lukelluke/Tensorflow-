import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

# 7 是int类型，7. 或者 7.0 是float类型（ python 类型 ）
# 后面的 dict 是python里面的字典，feed_dict = { 键:[值]， 键:[值]，... }
# 在求解 output 的时候，再给他相应的数值
# 用了 placeholder 就和 feed_dict 是绑定的了

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.5], input2:[2.5]}))

# 小结：因为这里没用 tf.Variable 所以不需要 tf.global_variables_initializer()
# 用 placeholder 占位符的好处是，可以在不知道数据的情况下，提前编写运算逻辑；
