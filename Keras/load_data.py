import scipy.io as sio
import numpy
## input dataset
filename = ("../coursera/machine-learning-ex4/ex4/ex4data1.mat")

data = sio.loadmat(filename)


# the dataset is sorted, let's shuffle it
order = numpy.random.permutation(range(len(data['X'])))
X = data['X'][order]
Y = data['y'][order]

ts = len(Y)


train_datasetY = numpy.zeros(( ts , 10)) 
for i in range(ts):
    j = Y[i]
    if Y[i] == 10:
        j = 0
    train_datasetY[i][ j ] = 1

numpy.save('../data/X.npy' , X)
numpy.save('../data/y.npy', Y)
numpy.save('../data/order.npy' , order)