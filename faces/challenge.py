'''
Facial recognition data challenge

Data from Evgueni Ovtchinnikov: https://www.dropbox.com/sh/a62wxyw9fpzwt95/AABJE0CEAtqOuLXKo_sOTFMVa?dl=0
https://github.com/evgueni-ovtchinnikov

1. clean the dataset: select only images with more than one face
2. select 70% train 30% cross validation

'''
import numpy
from functools import reduce
import matplotlib.pyplot as plt
import pickle

__version__ = '0.1.0'
#from docopt import docopt
''' 
args = docopt(__doc__, version=__version__)
file = args['<images>']
pref = args['<prefix>']
path = args['--path']
 
print('loading images from %s...' % (path + '/' + file))
images = numpy.load(file)
ni, ny, nx = images.shape
''' 
# link images numbers to names
names = []
num = []
#index = numpy.ndarray((ni,), dtype = numpy.int16)
# the number following the name indicates at which index the images of 
# the person start. 

off = []
count = 0
with open('lfw_names.txt') as fp:
    line = fp.readline()
    while line:
        theline = line.split(' ')
        names.append(theline[0])
        num.append(int(theline[1]))
        line = fp.readline()

def count_img(num):
    return [num[i+1] - num[i] for i in range(len(num)) if i < len(num)-1] 

# PCA matrix
u = numpy.load("lfwfp1140eigim.npy")

# coordinates
v = numpy.load("lfwfp1140coord.npy")

# total number of images
ni = v.shape[1]

count = count_img(num)
# correct the last count
if ni - num[-1] > 1:
    count.append(ni-num[-1])

names_repeat = []
index_repeat = []
name_index = {}
min_num_pics = 2
for i in range (len(count)):
    if count[i] >= min_num_pics:
        for j in range(count[i]):
            names_repeat.append(names[i])
            index_repeat.append(num[i] + j)


select = 'Bill_Clinton'
select = 'Vladimir_Putin'

i = 0 
while (names_repeat[i] != select): 
    i+=1                            

nselect = reduce(lambda x,y: x + 1 if y == select else x, names_repeat,0)
# the selected person will be in the range [i, i-nselect-1]
nc, ny, nx = u.shape
index = index_repeat[i]
PCA_image = numpy.dot(u.T, v.T[index])
PCA_image = numpy.reshape(PCA_image, (ny, nx))
plt.figure()
plt.title('PCA approximation of the image %d' % i)
plt.imshow(PCA_image.T, cmap = 'gray')
plt.show()

#n = nx*ny
#u = numpy.reshape(u, (nc, n))

# create the test set and cross validation set
# if a person has:
# n pics, train/cross validation split
# 2     , 1-1
# 3     , 2-1
# 4     , 3-1
# 5     , 3-2
# 6     , 4-2
# 7     , 5-2
# 8     , 5-3
# 9     , 6-3
# 10    , 70%-30%
# 11    , idem
# 12    , 

training_set_indices = []
cv_set_indices = []

face_index = 0
for select in names:
    if select in names_repeat:
        i = 0 
        while (i < len(names_repeat) and names_repeat[i] != select): 
            i+=1                            
        #print (select, i)

        nselect = reduce(lambda x,y: x + 1 if y == select else x, names_repeat,0)
        #print ("{0}, found {1} images".format(select, nselect))
        if nselect == 2:
            nts = 1
            ncv = 1
        elif nselect == 3:
            nts = 2
            ncv = 1
        elif nselect == 4:
            nts = 3
            ncv = 1
        elif nselect == 5:
            nts = 3
            ncv = 2
        elif nselect == 6:
            nts = 4
            ncv = 2
        elif nselect == 7:
            nts = 5
            ncv = 2
        elif nselect == 8:
            nts = 5
            ncv = 3
        elif nselect == 9:
            nts = 6    
            ncv = 3
        else:
            nts = int(nselect * 0.7)
            ncv = nselect - nts
    
            #print ("   Number of images in training set {0}".format(nts))
            #print ("   Number of images in cross validation set {0}".format(ncv))
        for n in range(nts):
            training_set_indices.append((select, index_repeat[i+n], face_index))
        for n in range(ncv):
            cv_set_indices.append((select, index_repeat[i+nts+n], face_index))
            
        face_index += 1



        
neig = v.shape[0]        
training_set = numpy.zeros((len(training_set_indices), neig), dtype=v.dtype)
cv_set = numpy.zeros((len(cv_set_indices), neig), dtype=v.dtype)

for i,face in enumerate(training_set_indices):
    faceindex = face[1]
    training_set[i][:] = v.T[faceindex]
    
for i,face in enumerate(cv_set_indices):
    faceindex = face[1]
    cv_set[i][:] = v.T[faceindex]

    
# show that we are doing well
select = 'Vladimir_Putin'
select = 'Colin_Powell'
index = 0 
while (not select == training_set_indices[index][0]):
    index += 1
    
PCA_image = numpy.dot(u.T, training_set[index])
PCA_image = numpy.reshape(PCA_image, (ny, nx))
plt.figure()
plt.title('PCA approximation of the image {}'.format(training_set_indices[index][0]))
plt.imshow(PCA_image.T, cmap = 'gray')
plt.show()

# save description of dataset
pickle.dump(training_set_indices, open("training_set_indices.pkl", "wb"))
pickle.dump(cv_set_indices, open("cv_set_indices.pkl", "wb"))
