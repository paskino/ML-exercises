import numpy
from functools import reduce
import matplotlib.pyplot as plt
import pickle

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

training_set = numpy.zeros((len(training_set_indices) * N_unique_people, neig + 1), dtype=v.dtype)
training_label = numpy.zeros((len(training_set_indices) * N_unique_people, 1), dtype=numpy.int8)
#cv_set = numpy.zeros((len(cv_set_indices) * N_unique_people, neig + 1), dtype=v.dtype)
#cv_label = numpy.zeros((len(cv_set_indices) * N_unique_people, 1), dtype=numpy.int8)

for i,face in enumerate(training_set_indices):
    faceindex = face[1]
    for unique in range(N_unique_people):
        training_set[i+unique][:] = numpy.hstack((v.T[faceindex] , unique))
        training_label[i+unique] = 1 if face[2] == unique else 0
    
#for i,face in enumerate(cv_set_indices):
#    faceindex = face[1]
#    for unique in range(N_unique_people):
#        cv_set[i+unique][:] = numpy.hstack((v.T[faceindex] , unique))
#        cv_label[i+unique] = 1 if face[2] == unique else 0


# check we are doing something sensible. 
select = 'Vladimir_Putin'
idx = 0
while (not training_set_indices[idx][0] == select):
    idx += 1
set_index = training_set_indices[idx][1]
unique = training_set_indices[idx][2]

select_coord = training_set[unique * N_unique_people + set_index]

nc, ny, nx = u.shape

#index = index_repeat[i]

PCA_image = numpy.dot(u.T, select_coord[:-1])
PCA_image = numpy.reshape(PCA_image, (ny, nx))
plt.figure()
plt.title('PCA approximation of the image %s' % training_set_indices[idx][0])
plt.imshow(PCA_image.T, cmap = 'gray')
plt.show()