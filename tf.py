import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from matplotlib import pyplot

data = pd.read_csv("YES.csv")


#prepearin the data for standirisation
ts = data['Close Price']
print("When taking without array value\n")
print("ts")
print(ts)
TS = np.array(ts)
print("TS")
print(TS)
len(TS)
num_periods = 20
f_horizon = 1 #forecast horizon one priod into the future

x_data = TS[:(len(TS)-(len(TS)%num_periods))]#making a bunch of 140 sets
print("x_data")
print(x_data)
x_batches = x_data.reshape(-1, 20 ,1)#reshaping the dimwnsion of the datasets with 20 bunch of 7 sets
print("x_ batches")
print(x_batches)

y_data = TS[1:(len(TS)-(len(TS)%num_periods))+f_horizon]
print("y_data")
print(y_data)
y_batches = y_data.reshape(-1,20,1)
#print(len(x_batches))
#print(x_batches.shape)
#print(x_batches[0:2])

#print(y_batches)

def test_data(series, forecast, num_periods):
	test_x_setup = TS[-(num_periods + forecast):]
	testX = test_x_setup[:num_periods].reshape(-1,20,1)
	testY =TS[-(num_periods):].reshape(-1,20,1)
	return testX, testY

X_test , Y_test = test_data(TS, f_horizon, num_periods)

print(X_test.shape)
#print(x_test)

#this would reset the graph
tf.reset_default_graph()

num_periods = 20
inputs = 1
hidden =100
output = 1

#creating the variable object
X = tf.placeholder(tf.float32 ,[None ,num_periods ,inputs])
y = tf.placeholder(tf.float32 ,[None ,num_periods ,output])

#create our RNN object
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden , activation=tf.nn.relu)

#create dynamic over static
rnn_output ,states = tf.nn.dynamic_rnn(basic_cell, X ,dtype=tf.float32)

#small leanring rate so that we don't overshoot the minimum
learning_rate = 0.001

#change the form into tensor
stacked_rnn_output = tf.reshape(rnn_output ,[-1, hidden])
#specify the type of layers(dense)
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
#shape of the result
outputs = tf.reshape(stacked_outputs ,[-1, num_periods, output])

#define the cost function which evaluates the quality of our model
#since it is the regression problem we are using MSE in other case we will use cross entropy
#if it is the case for classification problem
loss = tf.reduce_sum(tf.square(outputs-y))

#gradient descent method
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

#train the result of the application of the cost_function
training_op = optimizer.minimize(loss)


#initialize all variables 
init = tf.global_variables_initializer()



#implementing model on our training data
epochs = 5600 #forward + backward = 1 epochs

#tf.Session() it is the graph object
with tf.Session() as sess:
	init.run()
	for ep in range(epochs):
		sess.run(training_op, feed_dict={X:x_batches, y:y_batches})
		if ep%100 == 0:
			mse = loss.eval(feed_dict={X:x_batches ,y:y_batches})
			print(ep ,"\tMSE", mse)

	y_pred = sess.run(outputs, feed_dict={X:X_test})

	print(y_pred)


#plotting the data
# Plot
actual_line = pyplot.plot(np.ravel(Y_test), marker='s', label='Actual Price')
predicted_line = pyplot.plot(np.ravel(y_pred), color='red', marker='o', label='Predicted Price')
pyplot.legend(loc='upper left')
pyplot.ylabel('Price')
pyplot.xlabel('Date')
pyplot.title('Forecast Results')
pyplot.grid()
pyplot.show()
