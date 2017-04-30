import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# y = wx + b
def add_layer(input, in_size, out_size, activate_fnction = None):
	Weight = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_Plus_b = tf.matmul(input,Weight) + biases
	if activate_fnction is None:
		return Wx_Plus_b
	else :
		return activate_fnction(Wx_Plus_b)
		
		
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs , 1 , 10, activate_fnction =tf.nn.relu)
prediction = add_layer(l1 , 10 , 1)

loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys - prediction), reduction_indices=[1] ))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# draw pic
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion() # block = false
plt.show()

for i in range(1000):
	sess.run(train, feed_dict={xs:x_data, ys:y_data})
	if i % 50 == 0:
		predict_value = sess.run(prediction, feed_dict={xs:x_data})
		lines = ax.plot(x_data,predict_value, 'r-', lw = 5)
		plt.pause(0.3)
		ax.lines.remove(lines[0])
	
	
	









