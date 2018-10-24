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
#plt.figure()
#plt.title('PCA approximation of the image %d' % i)
#plt.imshow(PCA_image.T, cmap = 'gray')
#plt.show()

## The problem of classification of a picture of a face to the name of the person is not
## trivially treated with a NN. With this dataset the output layer would have more than 5000 output
## neurons, given 1140 inputs.
## A different approach would be to create a NN to compare 2 images, or an image and a tag identifying the
## the person. 
## I'll start out with a NN with 1140 + 1 input, where 1140 are the PCA coordinates, and the +1 is a 
## unique number for each person in the dataset. Output of this NN will be a single neuron giving the possibility that 
## the tag and the coordinates belong to the same person. 

## the image set contains 5953 labelled images. The training set will be much bigger as the plan is to 
## input in the NN 1140 coordinates + a tag. This means that out of each coordinate we can create N_unique_people input set.
## The input set will be then N_unique_people * len(training_set_indices)



training_set_indices = pickle.load(open("training_set_indices.pkl", "rb"))
cv_set_indices = pickle.load(open("cv_set_indices.pkl", "rb"))



N_unique_people = training_set_indices[-1][2]


#training_set = numpy.zeros((len(training_set_indices) * N_unique_people, neig + 1), dtype=v.dtype)
#training_label = numpy.zeros((len(training_set_indices) * N_unique_people, 1), dtype=numpy.int8)
#cv_set = numpy.zeros((len(cv_set_indices) * N_unique_people, neig + 1), dtype=v.dtype)
#cv_label = numpy.zeros((len(cv_set_indices) * N_unique_people, 1), dtype=numpy.int8)

class FaceDataset(object):
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
        #y = numpy.zeros((self.v.shape[0] + 1), dtype=v.dtype)

        face = self.indices[self.indexPic]
        y = ( numpy.hstack((self.v.T[self.indexPic] , self.indexName)) ,
                 1 if face[2] == self.indexName else 0 )
        print (face)
        print (self.indexName, "is the same" , 1 if face[2] == self.indexName else 0)
        
        # update the index
        if self.indexName == self.dimName:
            self.indexName = 0
            self.indexPic += 1
        else:
            self.indexName += 1

        self.index = self.indexPic * self.dimName + self.indexName
        
        return y
         

dataset = FaceDataset(training_set_indices, v)
# next(dataset)
tfdata = tf.data.Dataset.from_generator(dataset, (tf.float32, tf.uint8), (tf.TensorShape(neig+1), tf.TensorShape(None)))
value = tfdata.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    sess.run(value)

#for i, face in enumerate(training_set_indices):
#    faceindex = face[1]
#    print (face[0] , "img index" , face[1])
#    for unique in range(N_unique_people):
#        training_set[i*N_unique_people+unique][:] = numpy.hstack((v.T[faceindex] , unique))
#        training_label[i*N_unique_people+unique] = 1 if face[2] == unique else 0
    
    
#for i,face in enumerate(cv_set_indices):
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
#while (not training_set_indices[idx][0] == select):
#    idx += 1
#set_index = training_set_indices[idx][1]
#unique = training_set_indices[idx][2]

#select_coord = training_set[unique * N_unique_people + set_index]

#nc, ny, nx = u.shape

##index = index_repeat[i]

#PCA_image = numpy.dot(u.T, select_coord[:-1])
#PCA_image = numpy.reshape(PCA_image, (ny, nx))
#plt.figure()
#plt.title('PCA approximation of the image %s' % training_set_indices[idx][0])
#plt.imshow(PCA_image.T, cmap = 'gray')
#plt.show()