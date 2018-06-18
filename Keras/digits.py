from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import scipy.io as sio
import keras
from keras.layers import Dense, Input
from keras.models import Sequential
from keras import backend as K
import matplotlib.pyplot as plt

# Simple Sequential Dense Neural Network

model = Sequential()
#https://github.com/keras-team/keras/issues/4813
reg = keras.regularizers.l2(0.01)
model.add(Dense(25, activation=keras.activations.sigmoid, 
                input_dim=400,
                kernel_regularizer=reg))
model.add(Dense(10, 
                activation=keras.activations.sigmoid,
                kernel_regularizer=reg))
                

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


## input dataset
filename = ("../coursera/machine-learning-ex4/ex4/ex4data1.mat")

data = sio.loadmat(filename)


# the dataset is sorted, so limiting to 4000 won't train on certain numbers
# let's shuffle it
order = numpy.random.permutation(range(len(data['X'])))
X = data['X'][order]
Y = data['y'][order]

ts = 4000

train_datasetX = X[:ts]
train_datasetY = numpy.zeros(( ts , 10)) 
for i in range(ts):
    j = Y[i]
    if Y[i] == 10:
        j = 0
    train_datasetY[i][ j ] = 1

test_datasetX = X[ts:]
test_datasetY = numpy.zeros((len(X)-ts, 10)) 
for i in range(len(X)-ts):
    j = Y[i+ts]
    if Y[i+ts] == 10:
        j = 0
    test_datasetY[i][ j ] = 1


hist = model.fit(train_datasetX, train_datasetY, epochs=50)
print (hist.history)
score = model.evaluate(test_datasetX, test_datasetY)

vis = 3
p = model.predict(train_datasetX[0:vis])
py = numpy.zeros(numpy.shape(p)) 
for i in range(vis):
    py[i][numpy.argmax(p[i])] = 1
