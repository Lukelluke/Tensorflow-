from __ future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation=None):
	Weight = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zero([1,out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs,Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus,b)
	return outputs

def compute_accuracy(v_xs,v_ys):
	global prediction # 这句是什么意思？
	y_pre = sess.run(prediction,feed_dict={xs:v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
	return result

xs = tf.placeholder(tf.float32,[None,784]) # 28 x 28
ys = tf.placeholder(tf.float32,[None,10])


# add output layer
prediction = add_layer(xs,784,10,activation_function = )