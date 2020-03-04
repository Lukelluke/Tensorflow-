import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

state = tf.Variable(0,name='counter')
# print(state.name)

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()   # 旧版本是：tf.initialize_all_variables()
                                           # 如果有定义tf.Variable(),那么一定要有这句变量初始化！
with tf.Session() as sess:
    sess.run(init)                         # 定义好了 init 之后一定要在 sess 里面run一次 init
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
