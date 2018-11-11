import numpy
from functools import reduce
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

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


N_unique_people = training_set_indices[-1][2]


class FaceDataset(object):
    '''Create an iterator class as generator for Tensorflow's DataSet

        https://www.tensorflow.org/guide/datasets
        https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    '''
    def __init__(self, indices, eigcoord):
        self.N_unique = indices[-1][2]
        self.v = eigcoord
        self.indices = indices
        self.index = 0
        self.__length = len(indices) * self.N_unique
        self.dimPic = len(indices)
        self.dimName = self.N_unique
        self.indexPic = 0
        self.indexName = 0

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

min_label_distribution = 1
sample_weights[labels] = min_label_distribution  * \
        (len(labels) - labels.sum()) / labels.sum()

cv_set = FaceDataset(cv_set_indices, v)

cv_labels = []
cv_features = []
for i in cv_set:
    cv_labels.append(i[1])
    cv_features.append(i[0])

cv_labels = numpy.asarray(cv_labels)
cv_features = numpy.asarray(cv_features)
'''
x_train = XDataset(training_set_indices , v)
y_train = LabelDataset(training_set_indices , v)
x_cv = XDataset(cv_set_indices , v)
y_cv = LabelDataset(cv_set_indices , v)
print ("x_train: " , x_train.__next__())
print ("y_train: " , y_train.__next__())

tfx_train = tf.data.Dataset.from_generator(x_train,
                                        (tf.float32, ), 
                                        (tf.TensorShape(neig+1), 
                                            ))
tfy_train = tf.data.Dataset.from_generator(y_train,
                                        tf.bool) 
tfx_test = tf.data.Dataset.from_generator(x_cv,
                                        (tf.float32, ), 
                                        (tf.TensorShape(neig+1), 
                                            ))
tfy_test = tf.data.Dataset.from_generator(y_cv,
                                        (tf.bool, ), 
                                        (tf.TensorShape(None), 
                                            ))

iterator = tf.data.Iterator.from_structure(tfx_train.output_types,
                                           tfx_train.output_shapes)

tfdata = tf.data.Dataset.from_generator(dataset, (tf.float32, tf.uint8), (tf.TensorShape(neig+1), tf.TensorShape(None)))
value = tfdata.make_one_shot_iterator().get_next()
value2 = tfx_train.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    sess.run(value)

'''

#value = tfx_train.make_one_shot_iterator().get_next()
#with tf.Session() as sess:
#    sess.run(value)

# for i, face in enumerate(training_set_indices):
#    faceindex = face[1]
#    print (face[0] , "img index" , face[1])
#    for unique in range(N_unique_people):
#        training_set[i*N_unique_people+unique][:] = numpy.hstack((v.T[faceindex] , unique))
#        training_label[i*N_unique_people+unique] = 1 if face[2] == unique else 0


# for i,face in enumerate(cv_set_indices):
#    faceindex = face[1]
#    for unique in range(N_unique_people):
#        cv_set[i+unique][:] = numpy.hstack((v.T[faceindex] , unique))
#        cv_label[i+unique] = 1 if face[2] == unique else 0
#numpy.save("training_set.npy", training_set)
#numpy.save("training_label.npy", training_label)

# check we are doing something sensible.

#select = 'Vladimir_Putin'
##select = 'Aaron_Peirsol'
#idx = 0
# while (not training_set_indices[idx][0] == select):
#    idx += 1
#set_index = training_set_indices[idx][1]
#unique = training_set_indices[idx][2]

#select_coord = training_set[unique * N_unique_people + set_index]

#nc, ny, nx = u.shape

##index = index_repeat[i]

#PCA_image = numpy.dot(u.T, select_coord[:-1])
#PCA_image = numpy.reshape(PCA_image, (ny, nx))
# plt.figure()
#plt.title('PCA approximation of the image %s' % training_set_indices[idx][0])
#plt.imshow(PCA_image.T, cmap = 'gray')
# plt.show()


## Create the NN
# Logits Layer
# logits = tf.layers.dense(inputs=dropout, units=10)
# predictions = {
#    # Generate predictions (for PREDICT and EVAL mode)
#    "classes": tf.argmax(input=logits, axis=1),
#    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#    # `logging_hook`.
#    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#    }

# layer1 = tf.layers.dense(inputs=1141, units=50, activation=tf.nn.relu)
# layer2 = tf.layers.dense(inputs=layer1 , units = 1, activation=tf.nn.relu)
# outlayer = tf.layers.dense( inputs=layer2, activation=tf.nn.relu)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(250, 
      input_shape=(neig + 1,), activation=tf.nn.relu),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
history = model.fit(features, labels, epochs=50, batch_size=350,
        validation_split=0.2,
        sample_weight = sample_weights ,
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
