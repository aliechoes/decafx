import os
import decafx as dx
import numpy as np
#os.chdir('jan/')
#(X,y,fnames)=dx.import_ideas

folder = '../jan/extracted_tifs'
imsize = (32,32)
channels = (('Ch1','Ch2'),('Ch1','Ch2'))
#folder = 'F:\\Imaging Flow Cytometry Tutorial\\brocker data\data_Aug_2016\\train_set\\extracted_tifs'
(X,y,fnames) = dx.import_ideas(folder = folder,
                 classes = ('apoptotic_cells','attached_cells'),
                 channels = channels,
                 imsize = imsize,
                 imtype='uint16')

(X2,y2,fnames2) = dx.import_ideas(folder = folder,
                 classes = ('apoptotic_cells','attached_cells'),
                 channels = channels,
                 imsize = imsize,
                 imtype='uint16',
                 image_dim_ordering='tf')

X.shape

X2.shape

import matplotlib.pyplot as plt

#32x32 image of ch0 of first cell in theano ordering
plt.imshow(X[0,0,:,:],cmap='gray')

plt.imshow(X2[0,0,:,:],cmap='gray')#this is just a 32x2 slice

#32x32 image of ch0 of first cell in tensorflow ordering
plt.imshow(X2[0,:,:,0],cmap='gray')

X[0,0,1:5,0]==X2[0,1:5,0,0]#All true

np.array_equal(X[0,0,:,:],X2[0,:,:,0])#True