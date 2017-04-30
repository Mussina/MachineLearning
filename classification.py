import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# y = wx + b
def add_layer(input, in_size, out_size, activate_fnction = None):
	with tf.name_scope('layer'):
		with tf.name_scope('weight'):
			Weight = tf.Variable(tf.random_normal([in_size, out_size]))
		with tf.name_scope('bias'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
		with tf.name_scope('Wx_Plus_b'):
			Wx_Plus_b = tf.matmul(input,Weight) + biases
		if activate_fnction is None:
			output = Wx_Plus_b
		else :
			output = activate_fnction(Wx_Plus_b)	
		return output
	

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
	return result
	


# input data
with tf.name_scope('input'):
	xs = tf.placeholder(tf.float32,[None,784], name='x_input') # 28x28
	ys = tf.placeholder(tf.float32,[None,10], name='y_input')

# output layer
prediction = add_layer(xs , 784 , 10, tf.nn.softmax)

# loss function
with tf.name_scope('loss'):
	#cross_entropy = tf.reduce_mean(
      #tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
	cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1] ))

	
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# train
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))











