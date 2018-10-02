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


class L1L2_m(keras.regularizers.Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.01):
        with K.name_scope(self.__class__.__name__):
            self.l1 = K.variable(l1,name='l1')
            self.l2 = K.variable(l2,name='l2')
            self.val_l1 = l1
            self.val_l2 = l2
            
    def set_l1_l2(self,l1,l2):
        K.set_value(self.l1,l1)
        K.set_value(self.l2,l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def __call__(self, x):
        regularization = 0.
        if self.val_l1 > 0.:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.val_l2 > 0.:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        config = {'l1': float(K.get_value(self.l1)),
                  'l2': float(K.get_value(self.l2))}
        return config

# Simple Sequential Dense Neural Network

model = Sequential()
#https://github.com/keras-team/keras/issues/4813
reg = L1L2_m(0.0,0.01)
model.add(Dense(25, activation=keras.activations.sigmoid, 
                input_dim=400,
                kernel_regularizer=reg))
model.add(Dense(10, 
                activation=keras.activations.sigmoid,
                kernel_regularizer=reg))
                
#https://en.wikipedia.org/wiki/Cross_entropy
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

hist = []
lambdas = [ 0.00001, 1e-6 , 0 ]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
axes[0].set_title('loss')
axes[1].set_title('accuracy')

nepochs = 50
epochs = [i for i in range(nepochs)]
for l in lambdas:
    # update the regularisation parameter
    K.set_value(reg.l2, K.cast_to_floatx(l))
    hist.append( 
        model.fit(train_datasetX, train_datasetY, epochs=nepochs)
    )
    #print (hist.history)

    axes[0].plot(epochs,hist[-1].history['loss'],'-', label='lambda {0}'.format(l))
    axes[1].plot(epochs,hist[-1].history['acc'],'-', label='lambda {0}'.format(l))

legend = axes[1].legend(loc='lower center', shadow=True, fontsize='x-large')

plt.show()
score = model.evaluate(test_datasetX, test_datasetY)

vis = 3
p = model.predict(train_datasetX[0:vis])
py = numpy.zeros(numpy.shape(p)) 
for i in range(vis):
    py[i][numpy.argmax(p[i])] = 1
