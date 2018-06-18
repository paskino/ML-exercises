from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
import scipy.io as sio

tf.logging.set_verbosity(tf.logging.INFO)

filename = ("../coursera/machine-learning-ex4/ex4/ex4data1.mat")

data = sio.loadmat(filename)

X = data['X']
Y = data['y']

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(400, activation="relu", input_shape=(400,)) ,
  tf.keras.layers.Dense(25, activation="relu") ,
  tf.keras.layers.Dense(10, activation="relu") ,

])

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train_datasetX = X[:4000]
train_datasetY = Y[:4000]

