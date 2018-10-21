'''
Facial recognition data challenge

Initial release from https://www.dropbox.com/sh/a62wxyw9fpzwt95/AABJE0CEAtqOuLXKo_sOTFMVa?dl=0&preview=assemble.py

1. clean the dataset: select only images with more than one face
2. select 70% train 30% cross validation

'''
import numpy
from functools import reduce

__version__ = '0.1.0'
from docopt import docopt
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
        #k = line.find(' ')
        #names += [line[:k].replace('_', ' ')]
        theline = line.split(' ')
        names.append(theline[0])
        num.append(int(theline[1]))
	#new_off = int(line[k:])
        #off += [new_off]
        #count += 1
        line = fp.readline()
#off += [ni]
#i = 0
#for c in range(count):
#    for j in range(off[c], off[c + 1]):
#        index[j] = i
#    i += 1

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
for i in range (len(count)):
    if count[i] > 1:
        for j in range(count[i]):
            names_repeat.append(names[i])
            index_repeat.append(num[i] + j)

'''
filename = path + '/%seigim.npy' % pref
print('loading eigenimages from %s...' % filename)
u = numpy.load(filename)
 
filename = path + '/%scoord.npy' % pref
print('loading images coordinates in eigenimages basis from %s...' % filename)
v = numpy.load(filename)
 
nc, ny, nx = u.shape
ni = v.shape[1]
'''

select = 'Bill_Clinton'
select = 'Vladimir_Putin'

i = 0 
while (names_repeat[i] != select): 
    i+=1                            

nselect = reduce(lambda x,y: x + 1 if y == select else x, names_repeat,0)
# the selected person will be in the range [i, i-nselect-1]
nc, ny, nx = u.shape
index = index_repeat[i]
PCA_image = numpy.dot(u.T, v[:,index])
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
# 10    , int (10*0.3)
# 11    , idem
# 12    , 

training_set = []
cv_set = []

for select in names:
    #select = 'Bill_Clinton'
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
    
            print ("   Number of images in training set {0}".format(nts))
            print ("   Number of images in cross validation set {0}".format(ncv))
        for n in range(nts):
            training_set.append((select, index_repeat[i+n]))
        for n in range(ncv):
            cv_set.append((select, index_repeat[i+nts+n]))


