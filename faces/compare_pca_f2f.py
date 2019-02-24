#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:22:34 2019

@author: edo
"""

import numpy
from functools import reduce
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import sys
import PIL

__version__ = '0.1.0'



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
neig = 128

# PCA matrix
U = numpy.load("lfwfp1140eigim.npy")
U = numpy.load("lfw128eigim.npy")

u = U[:neig]
nc, ny, nx = u.shape

# coordinates
V = numpy.load("lfwfp1140coord.npy")
V = numpy.load("lfw128coord.npy")
v = V[:neig]

# neig total
# number of eigenvectors, ni total number of images
neig, ni = v.shape


N_unique_people = training_set_indices[-1][2] + 1

# Count the number of pictures per person
for i, el in enumerate(training_set_indices):
    if i == 0:
        name = el[0] 
        count = 1
        positives = 0
    else:
        if el[0] == name:
            count += 1
        else:
            print ("Found {} of {}".format(count, name))
            positives += count * (count -1)
            name = el[0]
            count = 1
            

select = 'George_W_Bush'
#select = 'Serena_Williams'
#select = 'Laura_Bush'
index = 0 

while (not select == cv_set_indices[index][0]):
    index += 1
    print (cv_set_indices[index][0])
#%%    
    
def plot_image(u,v,index, n):
    PCA_image = numpy.dot(u[:n].T, v[:n].T[cv_set_indices[index][1]])
    PCA_image = numpy.reshape(PCA_image, (ny, nx))
    plt.figure()
    plt.title('PCA {} approximation of the image {} '.format(n, cv_set_indices[index][0]))
    plt.imshow(PCA_image.T, cmap = 'gray')
    plt.show()
#sys.exit(0)
index = 345
#plot_image(U,V,index,U.shape[0])
n = 500
#plot_image(U,V,index,n)
n = 700
#plot_image(U,V,index,n)
n = 800
#plot_image(U,V,index,n)

def plot_outer(u,v,n,index1, index2):
    PCA_image = numpy.outer(v[:n].T[cv_set_indices[index][1]], v[:n].T[cv_set_indices[index2][1]])
    PCA_image = numpy.reshape(PCA_image, (n, n))
    plt.figure()
    plt.title('PCA {} approximation of the image {} '.format(n, cv_set_indices[index][0]))
    plt.imshow(PCA_image.T)
    plt.show()


class FaceFaceDataset(object):
    '''Create an iterator class as generator for Tensorflow's DataSet

        https://www.tensorflow.org/guide/datasets
        https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, indices, eigcoord, batch_size=50, weight=1):
        super(FaceFaceDataset, self).__init__()
        self.N_unique = indices[-1][2] + 1
        self.v = eigcoord
        # contains ('name', index in v, name index)
        self.indices = indices
        self.index = 0
        self.__length = len(indices) * (len(indices) - 1)
        self.dimPic = len(indices)
        self.dimName = self.N_unique
        self.indexPic = 0
        self.secondIndexPic = 0
        self.indexName = 0
        self.batch_size = batch_size
        self.weight = weight

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
        if self.indexPic == self.secondIndexPic:
            self.index_update()
        face1 = self.indices[self.indexPic]
        face2 = self.indices[self.secondIndexPic]
        y = numpy.hstack((self.v.T[self.indexPic], 
                           self.v.T[self.secondIndexPic])),
        lab = True if face1[2] == face2[2] else False
        w = self.weight if lab else 1
        # print(face)
        # print(self.indexName, "is the same",
        #      1 if face[2] == self.indexName else 0)

        # update the index
        self.index_update()
        #if self.secondIndexPic == self.dimPic:
        #    self.secondIndexPic = 0
        #    self.indexPic += 1


        return y, lab

    def index_update(self):
        self.secondIndexPic += 1
        if self.secondIndexPic == self.dimPic:
            self.secondIndexPic = 0
            self.indexPic += 1

        self.index = self.indexPic * self.dimPic + self.secondIndexPic
#         print ("pic index 1 {} 2 {} {}".format(self.indexPic,
#             self.secondIndexPic, self.dimPic))
        if self.index >= self.__length:
            raise StopIteration('stop it!')

    def __getitem__(self):

        #labels = []
        #features = []
        n,l = self.v.shape
        labels = numpy.zeros((self.batch_size, ), dtype=numpy.bool)
        features = numpy.zeros((self.batch_size, n))
        for i in range(self.batch_size):
            el = self.__next__()
            #labels.append(el[1])
            #features.append(el[0])
            labels[i] = el[1]
            features[i] = el[0][:]

        #labels = numpy.asarray(labels)
        #features = numpy.asarray(features)
        return (features, labels)

    def on_epoch_end(self):
        self.__init__(self.indices, self.v,
                self.batch_size)


batchsize = 1
dataset = FaceFaceDataset(training_set_indices , v, batch_size=batchsize)
dataset.weight = len(dataset)/positives
dataset2 = FaceFaceDataset(cv_set_indices , v, batch_size=batchsize)


threshold = numpy.infty
for i,el in enumerate(dataset):
    label = el[1]
    if label:
        v1 = el[0][0][:128]
        v2 = el[0][0][128:]
        #print ( "processing ", i)
        distance = numpy.linalg.norm(v1-v2)/numpy.linalg.norm(v1)    
        if distance < threshold:
            threshold = distance   
#%%            
results_ok = 0
results_nok = 0 
print ("Found Threshold {}".format(threshold))

# test if threshold found is ok for the CV set
for i,el in enumerate(dataset2):
#for i in range(100000):
#    el = dataset.__next__()
    v1 = el[0][0][:128]
    v2 = el[0][0][128:]
    distance = numpy.linalg.norm(v1-v2)/numpy.linalg.norm(v1)    
            
    label = el[1]
    if label == (distance <= threshold):
        #print ( "processing ", i)
        results_ok += 1
    else:
        results_nok += 1

#%%
# check the cat how it scores
nc, ny, nx = U.shape
n = nx*ny
u = numpy.reshape(U, (nc, n))
cat = 1-numpy.asarray(PIL.Image.open('cat.jpg'))
image = numpy.reshape(cat, (n,))
w = numpy.dot(u, image)
mdistance = numpy.infty
for i,el in enumerate(V.T):
#for i in range(100000):
#    el = dataset.__next__()
    distance = numpy.linalg.norm(w-el)/numpy.linalg.norm(w)  
    if distance < mdistance:
        mdistance = distance


print ("Cat distance with a face {}".format(mdistance))


