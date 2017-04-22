import tensorflow as tf
import numpy as np

# y = wx + b
def add_layer(input, in_size, out_size, n_layers, activate_fnction = None):
	layer_name = 'layer%s' % n_layers
	with tf.name_scope(layer_name):
		with tf.name_scope('weight'):
			Weight = tf.Variable(tf.random_normal([in_size, out_size]))
			tf.summary.histogram(layer_name, Weight)
		with tf.name_scope('bias'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
			tf.summary.histogram(layer_name , biases)
		with tf.name_scope('Wx_Plus_b'):
			Wx_Plus_b = tf.matmul(input,Weight) + biases
		if activate_fnction is None:
			output = Wx_Plus_b
		else :
			output = activate_fnction(Wx_Plus_b)
		tf.summary.histogram(layer_name , output)	
		return output
		
# make up some data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# input data
with tf.name_scope('input'):
	xs = tf.placeholder(tf.float32,[None,1], name='x_input')
	ys = tf.placeholder(tf.float32,[None,1], name='y_input')

l1 = add_layer(xs , 1 , 10, n_layers=1, activate_fnction =tf.nn.relu)
prediction = add_layer(l1 , 10 , 1, n_layers=2)

with tf.name_scope('loss'):
	loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys - prediction), reduction_indices=[1] ))
	tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
sess.run(init)

writer = tf.summary.FileWriter("logs/", sess.graph)

# train
for i in range(1000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i % 50 == 0:
		result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
		writer.add_summary(result, i)












