import numpy
from functools import reduce
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import sys

__version__ = '0.1.0'

# PCA matrix
u = numpy.load("lfwfp1140eigim.npy")

# coordinates
v = numpy.load("lfwfp1140coord.npy")

# neig total number of eigenvectors, ni total number of images
neig, ni = v.shape


#nc, ny, nx = u.shape

#index = index_repeat[i]
#PCA_image = numpy.dot(u.T, v.T[index])
#PCA_image = numpy.reshape(PCA_image, (ny, nx))
# plt.figure()
#plt.title('PCA approximation of the image %d' % i)
#plt.imshow(PCA_image.T, cmap = 'gray')
# plt.show()

# The problem of classification of a picture of a face to the name of the person is not
# trivially treated with a NN. With this dataset the output layer would have more than 5000 output
# neurons, given 1140 inputs.
# A different approach would be to create a NN to compare 2 images, or an image and a tag identifying the
# the person.
# I'll start out with a NN with 1140 + 1 input, where 1140 are the PCA coordinates, and the +1 is a
# unique number for each person in the dataset. Output of this NN will be a single neuron giving the possibility that
# the tag and the coordinates belong to the same person.

# the image set contains 5953 labelled images. The training set will be much bigger as the plan is to
# input in the NN 1140 coordinates + a tag. This means that out of each coordinate we can create N_unique_people input set.
# The input set will be then N_unique_people * len(training_set_indices)


training_set_indices = pickle.load(open("training_set_indices.pkl", "rb"))
cv_set_indices = pickle.load(open("cv_set_indices.pkl", "rb"))
# PCA matrix
u = numpy.load("lfwfp1140eigim.npy")

# coordinates
v = numpy.load("lfwfp1140coord.npy")

nc, ny, nx = u.shape

N_unique_people = training_set_indices[-1][2] + 1

select = 'George_W_Bush'
select = 'Serena_Williams'
select = 'Laura_Bush'
index = 0 

while (not select == cv_set_indices[index][0]):
    index += 1
    print (cv_set_indices[index][0])
    
PCA_image = numpy.dot(u.T, v.T[cv_set_indices[index][1]])
PCA_image = numpy.reshape(PCA_image, (ny, nx))
plt.figure()
plt.title('PCA approximation of the image {}'.format(cv_set_indices[index][0]))
plt.imshow(PCA_image.T, cmap = 'gray')
plt.show()
#sys.exit(0)

class FaceDataset(object):
    '''Create an iterator class as generator for Tensorflow's DataSet

        https://www.tensorflow.org/guide/datasets
        https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    '''
    def __init__(self, indices, eigcoord, batch_size=50):
        self.N_unique = indices[-1][2] + 1
        self.v = eigcoord
        self.indices = indices
        self.index = 0
        self.__length = len(indices) * self.N_unique
        self.dimPic = len(indices)
        self.dimName = self.N_unique
        self.indexPic = 0
        self.indexName = 0
        self.batch_size = batch_size

    def __len__(self):
        return self.__length

    def __call__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):

        if self.index == self.__length:
            raise StopIteration

        face = self.indices[self.indexPic]
        y = (numpy.hstack((self.v.T[self.indexPic], self.indexName)),
             True if face[2] == self.indexName else False)
        # print(face)
        # print(self.indexName, "is the same",
        #      1 if face[2] == self.indexName else 0)

        # update the index
        if self.indexName == self.dimName:
            self.indexName = 0
            self.indexPic += 1
        else:
            self.indexName += 1

        self.index = self.indexPic * self.dimName + self.indexName

        return y

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
        self.__init__(self.indices, self.v,
                self.batch_size)



class LabelDataset(FaceDataset):
    def __init__(self, indices, eigcoord):
        super(LabelDataset, self).__init__(indices, eigcoord)
    def __next__(self):
        return super(LabelDataset, self).__next__()[1]

class XDataset(FaceDataset):
    def __init__(self, indices, eigcoord):
        super(XDataset, self).__init__(indices, eigcoord)
    def __next__(self):
        return super(XDataset, self).__next__()[0]

# next(dataset)
dataset = FaceDataset(training_set_indices , v)
#print ("dataset: " , dataset.__next__())

labels = []
features = []
for i in dataset:
    labels.append(i[1])
    features.append(i[0])

labels = numpy.asarray(labels)
features = numpy.asarray(features)
sample_weights = numpy.ones(labels.shape)
min_label_distribution = 0.5
sample_weights[labels] = min_label_distribution  * \
        (len(labels) - labels.sum()) / labels.sum()

# cross validation
cv_set = FaceDataset(cv_set_indices, v)

cv_labels = []
cv_features = []
for i in cv_set:
    cv_labels.append(i[1])
    cv_features.append(i[0])

cv_labels = numpy.asarray(cv_labels)
cv_features = numpy.asarray(cv_features)

## Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(250, 
      input_shape=(neig + 1,), activation=tf.nn.sigmoid),
  tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# train
history = model.fit(features, labels, epochs=2, batch_size=350,
        sample_weight = sample_weights ,
        validation_data = (cv_features, cv_labels),
        shuffle = True,
        verbose = 2)
#model.evaluate(cv_features, cv_labels)
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1,2,2)
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.legend(['train', 'test'], loc='lower right')
plt.show()




# find best match
match = []
for i in range(cv_set_indices[-1][2]):
    x = numpy.hstack((v.T[cv_set_indices[index][1]],i))
    match.append(
        (model.predict(x, batch_size=None, verbose=0),\
                i,cv_set_indices[index][0])
    )
    
#PCA_image = numpy.dot(u.T, v.T[cv_set_indices[index][1]])
#PCA_image = numpy.reshape(PCA_image, (ny, nx))
#plt.figure()
#plt.title('PCA approximation of the image {}'.format(cv_set_indices[index][0]))
#plt.imshow(PCA_image.T, cmap = 'gray')
#plt.show()
