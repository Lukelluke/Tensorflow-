import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

state = tf.Variable(0,name='counter')

print(state.name)
print('hello')
print(f'hello2')