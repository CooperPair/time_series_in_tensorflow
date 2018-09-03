import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import mean
import os
import matplotlib.pyplot as plt
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
import common
import crayons
from pandas import Series

#input data from the command shell
data = pd.read_csv(sys.argv[1])

#prepearing data for standirisation
ts = data['Adj Close']
ts_mean = ts-ts.mean()
ts_var = ts/ts.var()

TS = np.array(ts)#using data directly without normalising


num_periods = 20
f_horizon = 1 #forecast horizon one priod into the future

x_data = TS[:(len(TS)-(len(TS)%num_periods))]#making a bunch of sets
x_batches = x_data.reshape(-1, 20 ,1)#reshaping the dimwnsion of the datasets


y_data = TS[1:(len(TS)-(len(TS)%num_periods))+f_horizon]
y_batches = y_data.reshape(-1,20,1)

def test_data(series, forecast, num_periods):
	test_x_setup = TS[-(num_periods + forecast):]
	testX = test_x_setup[:num_periods].reshape(-1,20,1)
	testY =TS[-(num_periods):].reshape(-1,20,1)
	return testX, testY

X_test , Y_test = test_data(TS, f_horizon, num_periods)

print(Y_test.shape)
print(Y_test)
y1 = np.reshape(Y_test, 20)#reshape the data into 1d
y = pd.Series(y1)

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
epochs = 1000 #forward + backward = 1 epochs
scores = list()
#tf.Session() it is the graph object
with tf.Session() as sess:
	init.run()
	for ep in range(epochs):
		sess.run(training_op, feed_dict={X:x_batches, y:y_batches})
		if ep%100 == 0:
			mse = loss.eval(feed_dict={X:x_batches ,y:y_batches})
			print(ep ,"\tMSE", mse)#this item we have to run for optimizing the error.
	y_pred = sess.run(outputs, feed_dict={X:X_test})
	ypred = np.reshape(y_pred, 20)
	#print(ypred)


#to do !-- the tak of generating the result accuracy on the basis of up and down
	print(crayons.blue("[*] Finding the actual trend."))
	ACTUAL_TRENDS = common.find_trend(y1, y1)
	print(crayons.blue("[*] Finding predicted trend."))
	PREDICTED_TRENDS = common.find_trend(ypred, y1)
	print(crayons.blue("[*] Calculating accuracy of the prediction."))
	correct_pred, incorrect_pred = common.accuracy(PREDICTED_TRENDS, ACTUAL_TRENDS)
	pred_accuracy = (correct_pred/len(PREDICTED_TRENDS))*100
	print(crayons.yellow(f'\t[*] No. of correct prediction : {correct_pred}', bold=True))
	print(crayons.yellow(f'\t[*] No. of incorrect prediction : {incorrect_pred}', bold=True))
	print(crayons.yellow(f'\t[*] Prediction Accuracy : {pred_accuracy} %', bold=True))

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