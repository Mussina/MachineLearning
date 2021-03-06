import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs,keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys,keep_prob:1})
	return result
	
def weight_variable(shape):
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init)

def bias_variable(shape):
	init = tf.constant(0.1, shape = shape)
	return tf.Variable(init)
	
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
	
def max_pool_2X2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	
# input data
xs = tf.placeholder(tf.float32,[None,784], name='x_input') # 28x28
ys = tf.placeholder(tf.float32,[None,10], name='y_input')
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])
print(x_image.shape) #(n_sample, 28,28,1)

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) # patch 5x5, in size 1 , out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2X2(h_conv1)	# output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5, in size 32 , out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2X2(h_conv2)	# output size 7x7x64

## function1 layer ##
W_fcl = weight_variable([7*7*64, 1024])
b_fcl = bias_variable([1024])

# [n_sample, 7,7,64] --> [n_sample, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)
h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob) 

## function2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)



# loss function
with tf.name_scope('loss'):
	#cross_entropy = tf.reduce_mean(
      #tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
	cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1] ))

	
with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# train
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys,keep_prob:0.5})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))











