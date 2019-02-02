import os
import numpy
from functools import reduce
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import sys
import scipy.io as sio 


__version__ = '0.1.0'
class MNISTDataset(object):
    '''Create an iterator class as generator for Tensorflow's DataSet

        https://www.tensorflow.org/guide/datasets
        https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, X, y, order, _from, _upto,  batch_size=1):
        super(MNISTDataset, self).__init__()
        self.index = 0
        self.__length = len(X[_from:_upto]) 
        self.batch_size = batch_size
        self.order = order
        self._from = _from
        self._upto = _upto
        self.X = X
        self.y = y
        self.data = X[order][_from:_upto]
        self.label = y[order][_from:_upto]
        
    def __len__(self):
        return self.__length

    def __call__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):

        if self.index == self.__length:
            # raise StopIteration
            self.on_epoch_end()
            
        y = self.data[self.index]
        num = self.label[self.index]
        lab = numpy.zeros((10,), dtype=numpy.int32)
        if num == 10:
             lab[0] = 1
        else:
            lab[num] = 1
        # update the index
        # print (lab)
        self.index_update()
        return numpy.asarray([y]), lab

    def index_update(self):
        self.index += 1

    def __getitem__(self):

        labels = []
        features = []
        for i in range(self.batch_size):
            el = self.__next__()
            labels.append(el[1])
            features.append(el[0])

        labels = numpy.asarray(labels)
        features = numpy.asarray(features)
        return (features, labels)

    def on_epoch_end(self):
        self.__init__(self.X, self.y, self.order, self._from, self._upto,
                self.batch_size)



mnist_data = sio.loadmat(
    os.path.abspath("../coursera/machine-learning-ex4/ex4/ex4data1.mat"))
X = mnist_data['X']
y = mnist_data['y']

if True:
    order = numpy.asarray([i for i in range(len(X))], dtype=int)
    numpy.random.shuffle(order)
    numpy.save('order.npy', order)
else:
    order = numpy.load('order.npy')

upto = int(0.7 * len(order)) 


batchsize = 100
dataset = MNISTDataset(X, y, order, 0, upto, batch_size = batchsize)
dataset2 = MNISTDataset(X,y,order, upto, -1, batch_size = batchsize)

#%%

tf_train = tf.data.Dataset.from_generator(
    dataset, output_types=(tf.float32, tf.int32), 
    #output_shapes=(neig*2,)
    #output_shapes=(tf.TensorShape([neig*2,1]))
    )
tf_cv = tf.data.Dataset.from_generator(
    dataset2, output_types=(tf.float32, tf.int32),
#    output_shapes=(tf.TensorShape([neig*2]), tf.TensorShape([None]))
    )

itera= tf_train.make_initializable_iterator()
el = itera.get_next()
with tf.Session() as sess:
    sess.run(itera.initializer)
    epoch = 2
    i = 0
    while i < epoch * len(dataset):
        sess.run(el)
        print(i)
        i+=1
    
tf_train.repeat()
tf_cv.repeat()
#%%
## Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(25, 
      input_shape=(400,), activation=tf.nn.sigmoid),
  # tf.keras.layers.Dense(25, activation=tf.nn.sigmoid),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  # tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
#%%
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])


# Train model on dataset
#%%
# train
epochs = 500

history = model.fit(tf_train, epochs=epochs,initial_epoch=0 , 
#                    batch_size=350,
#        sample_weight = sample_weights ,
        validation_data = tf_cv,
#        validation_split=0.3,
        shuffle = False,
        verbose = 2, 
        steps_per_epoch=len(dataset)//batchsize,
        validation_steps=len(dataset2)//batchsize
        )
#%%
#model.evaluate(cv_features, cv_labels)
ax = plt.subplot(1,2,1)
ax.set_title("loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
ax = plt.subplot(1,2,2)
ax.set_title("accuracy")
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.legend(['train', 'test'], loc='lower right')
plt.show()

#%%
d = dataset2.__next__()
fig = plt.figure()
ax = plt.imshow(numpy.reshape(d[0], (20,20)).T, cmap='gray_r')
plt.title("Predicted {}".format(model.predict(d[0]).argmax()))
plt.show()
