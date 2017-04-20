import tensorflow as tf
import numpy as np

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
			return Wx_Plus_b
		else :
			return activate_fnction(Wx_Plus_b)
		
		
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('input'):
	xs = tf.placeholder(tf.float32,[None,1], name='x_input')
	ys = tf.placeholder(tf.float32,[None,1], name='y_input')

l1 = add_layer(xs , 1 , 10, activate_fnction =tf.nn.relu)
prediction = add_layer(l1 , 10 , 1)

with tf.name_scope('loss'):
	loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys - prediction), reduction_indices=[1] ))

with tf.name_scope('train'):
	train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter("logs/", sess.graph)













