# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:31:21 2017

@author: N.Chlis
"""

import numpy as np
#import skimage.io
import skimage.util
#import skimage.transform
import glob
import cv2

def import_ideas(folder = 'extracted_tifs',
                 classes = ('apoptotic_cells','attached_cells'),
                 channels = (('Ch1','Ch2'),('Ch1','Ch2')),
                 imsize = (32,32),
                 imtype='uint16',
                 image_dim_ordering='tf'):
    """
    searches in the 'folder' folder and
    imports the images in subfolders given by 'classes',
    while the channel information is given by 'channels'.
    Crops or pads the images to fit in the dimensions of
    'imsize'. If imsize is None, the images are not processed.
    imtype sets the dtype of the ndarray used to store the images.
    image_dim_ordering is either 'th' (theano: nchannels x rows x cols)
    or 'tf' (tensorflow: rows x cols x nchannels)
    """
    #print('folder', folder)
    #print('classes', classes)
    #print('channels', channels)
    #initialize the array to save the images
    #X=np.zeros(shape=(len(channels),imsize[0],imsize[1]),dtype=imtype)
    
    if(type(folder)==list):
        folder=tuple(folder)
    if(type(folder)!=tuple):
        print('Warning: folder not supplied as a tuple, trying to convert...')
        folder=(folder,)#assuming a string was provided
    
    all_folders=folder
    
    listX=list()
    listy=list()
    listfnames=list()
    for folder in all_folders:
        #scan for all filenames
        fnames_class=[None]*len(classes)#one element per class
        for cl in range(len(classes)):
            print('Scanning in: '+folder+'/'+classes[cl])
            #number of images per channel
            image_count_ch = np.zeros(len(channels[0]));
            fnames_channel=[None]*len(channels[0])#one element per channel
            for ch in range(len(channels[0])):
                #read all the filenames of channel 'channels[ch]'
                fname_list = []
                pattern = folder+'/'+classes[cl]+'/*'+channels[cl][ch]+'.ome.tif'
                #find all filenames for given class and channel
                for filename in glob.glob(pattern):
                    fname_list.append(filename)
                image_count_ch[ch]=len(fname_list)#count the number of images
                fnames_channel[ch]=fname_list#save filenames
                print('...channel: '+channels[cl][ch],'detected',len(fname_list),'images')
            #check that all channels have the same number of images
            if(len(np.unique(image_count_ch))!=1):
                print('Unequal number of images per channel for',classes[cl])
            fnames_class[cl]=fnames_channel
        
        #at this point we assume that all channels have the same number of images
        #inside each class. However, different classes can have different number
        #of images each.
        image_count_class = np.zeros(len(classes));
        for i in range(len(classes)):
            #All channels have the same number of images, so just use the 1st ch
            image_count_class[i]=len(fnames_class[i][0])
        
        #print('image count',image_count_class)
        #read all images in a numpy array, X(#images,#channels,#row,#col) 
        #also save filenames in a list. Use only filename for the first channel
        X = np.zeros(shape=(int(image_count_class.sum()),
                            len(channels[0]),imsize[0],imsize[1]),dtype=imtype)
        y = np.zeros(shape=(int(image_count_class.sum()),),dtype='uint8')
        im_filenames = [None]*int(image_count_class.sum())
        offset = 0
        for cl in range(len(classes)):
            print('Loading',classes[cl],'(class'+str(cl)+')')
            for ch in range(len(channels[0])):
                print('...channel', channels[cl][ch])
                ic=int(np.sum(image_count_class[0:cl]))
                #if (cl==0):
                #    ic=0#image count for this channel
                #else:
                #    ic=
                for im in range(int(image_count_class[cl])):#assume all channels same #images
                    #print('im is ',im)
                    fname=fnames_class[cl][ch][im]
                    #temp_image = skimage.io.imread(fname,plugin='freeimage')
                    #temp_image = skimage.io.imread(fname,plugin='tifffile')
                    temp_image = cv2.imread(fname,-1) #-1 flag to read image as-is (in 16bit)
                    X[ic,ch,:,:] = im_adjust(temp_image,imsize)#save the image
                    y[ic] = cl #save the class label
                    #now save the filename, use only the first channel
                    if (ch == 0):
                        im_filenames[ic] = fname
                    ic = ic+1#move to next image
            offset = offset+1
        #return #to return nothing
        
        #optionally convert to tensorflow format by swapping axes
        if(image_dim_ordering=='tf'):
            #originally axis 1 is the channels the last axis len(X.shape)
    		#theano: samples x channels x rows x cols
    		#tensorflow: samples x rows x cols x channels
    		#so rolling the axes is necessary to go from theano to tensorflow ordering
            #equivalent to np.rollaxis(X,axis=1,start=4) for 2D
            #returns a view and not a copy of the X matrix, so it is not slow.
            #a view is just a pointer to the original object.
            X=np.rollaxis(X,axis=1,start=len(X.shape))
        im_filenames=np.array(im_filenames).astype('S')
        listX.append(X)
        listy.append(y)
        listfnames.append(im_filenames)
    
    #concatenate images+metadata of all folders
    X=np.concatenate(listX)
    y=np.concatenate(listy)
    im_filenames=np.concatenate(listfnames)
    return((X,y,im_filenames))

def import_folder(folder = 'extracted_tifs',
                 classes = ('apoptotic_cells','attached_cells'),
                 channels = None,
                 imsize = (32,32),
                 imtype='uint16',
                 image_dim_ordering='tf'):
    """
    searches in the 'folder' folder and
    imports the images in subfolders given by 'classes',
    while the channel information is given by 'channels'.
    Crops or pads the images to fit in the dimensions of
    'imsize'. If imsize is None, the images are not processed.
    imtype sets the dtype of the ndarray used to store the images.
    image_dim_ordering is either 'th' (theano: nchannels x rows x cols)
    or 'tf' (tensorflow: rows x cols x nchannels)
    """
    #print('folder', folder)
    #print('classes', classes)
    #print('channels', channels)
    #initialize the array to save the images
    #X=np.zeros(shape=(len(channels),imsize[0],imsize[1]),dtype=imtype)
    if(channels==None):
        channels=(('Ch1',),)*len(classes)
    
    if(type(folder)!=tuple):
        print('Warning: folder not supplied as a tuple, trying to convert...')
        folder=(folder,)#assuming a string was provided
    
    all_folders=folder
    
    listX=list()
    listy=list()
    listfnames=list()
    for folder in all_folders:
        #scan for all filenames
        fnames_class=[None]*len(classes)#one element per class
        for cl in range(len(classes)):
            print('Scanning in: '+folder+'/'+classes[cl])
            #number of images per channel
            image_count_ch = np.zeros(len(channels[0]));
            fnames_channel=[None]*len(channels[0])#one element per channel
            for ch in range(len(channels[0])):
                #read all the filenames of channel 'channels[ch]'
                fname_list = []
#                pattern = folder+'/'+classes[cl]+'/*'+channels[cl][ch]+'.ome.tif'
                pattern = folder+'/'+classes[cl]+'/*'+'.tif'
                #find all filenames for given class and channel
                for filename in glob.glob(pattern):
                    fname_list.append(filename)
                image_count_ch[ch]=len(fname_list)#count the number of images
                fnames_channel[ch]=fname_list#save filenames
                print('...channel: '+channels[cl][ch],'detected',len(fname_list),'images')
            #check that all channels have the same number of images
            if(len(np.unique(image_count_ch))!=1):
                print('Unequal number of images per channel for',classes[cl])
            fnames_class[cl]=fnames_channel
        
        #at this point we assume that all channels have the same number of images
        #inside each class. However, different classes can have different number
        #of images each.
        image_count_class = np.zeros(len(classes));
        for i in range(len(classes)):
            #All channels have the same number of images, so just use the 1st ch
            image_count_class[i]=len(fnames_class[i][0])
        
        #print('image count',image_count_class)
        #read all images in a numpy array, X(#images,#channels,#row,#col) 
        #also save filenames in a list. Use only filename for the first channel
        X = np.zeros(shape=(int(image_count_class.sum()),
                            len(channels[0]),imsize[0],imsize[1]),dtype=imtype)
        y = np.zeros(shape=(int(image_count_class.sum()),),dtype='uint8')
        im_filenames = [None]*int(image_count_class.sum())
        offset = 0
        for cl in range(len(classes)):
            print('Loading',classes[cl],'(class'+str(cl)+')')
            for ch in range(len(channels[0])):
                print('...channel', channels[cl][ch])
                ic=int(np.sum(image_count_class[0:cl]))
                #if (cl==0):
                #    ic=0#image count for this channel
                #else:
                #    ic=
                for im in range(int(image_count_class[cl])):#assume all channels same #images
                    #print('im is ',im)
                    fname=fnames_class[cl][ch][im]
                    #temp_image = skimage.io.imread(fname,plugin='freeimage')
                    #temp_image = skimage.io.imread(fname,plugin='tifffile')
                    temp_image = cv2.imread(fname,-1) #-1 flag to read image as-is (in 16bit)
                    X[ic,ch,:,:] = im_adjust(temp_image,imsize)#save the image
                    y[ic] = cl #save the class label
                    #now save the filename, use only the first channel
                    if (ch == 0):
                        im_filenames[ic] = fname
                    ic = ic+1#move to next image
            offset = offset+1
        #return #to return nothing
        
        #optionally convert to tensorflow format by swapping axes
        if(image_dim_ordering=='tf'):
            #originally axis 1 is the channels the last axis len(X.shape)
    		#theano: samples x channels x rows x cols
    		#tensorflow: samples x rows x cols x channels
    		#so rolling the axes is necessary to go from theano to tensorflow ordering
            #equivalent to np.rollaxis(X,axis=1,start=4) for 2D
            #returns a view and not a copy of the X matrix, so it is not slow.
            #a view is just a pointer to the original object.
            X=np.rollaxis(X,axis=1,start=len(X.shape))
        im_filenames=np.array(im_filenames).astype('S')
        listX.append(X)
        listy.append(y)
        listfnames.append(im_filenames)
    
    #concatenate images+metadata of all folders
    X=np.concatenate(listX)
    y=np.concatenate(listy)
    im_filenames=np.concatenate(listfnames)
    return((X,y,im_filenames))

def im_adjust(im_raw, imsize = (32,32)):
    """
    adjust a raw image 'im_raw' to a given image size of 'imsize'.
    if a dimension (row,col) is less than specified by imsize
    then it is padded by copying the border values. If it is
    larger than imsize, it is cropped.
    """
    #check row size (image height)
    diff=im_raw.shape[0]-imsize[0]
    if (diff>0):#image is larger, crop
        if (diff % 2 == 0): #even
            before=int(diff/2)
            after=int(diff/2)
        else: #odd
            before=int(diff/2)
            after=int(diff/2)+1
        im_raw = skimage.util.crop(im_raw,((before,after),(0,0)),copy=False)
    elif (diff<0):#image is smaller, pad
        diff=np.abs(diff)#use the absolute value
        if (diff % 2 == 0): #even
            before=int(diff/2)
            after=int(diff/2)
        else: #odd
            before=int(diff/2)
            after=int(diff/2)+1
        im_raw = skimage.util.pad(im_raw,((before,after),(0,0)),'edge')
    else:#do nothing
        pass
    
    #check col size (image width)
    diff=im_raw.shape[1]-imsize[1]
    if (diff>0):#image is larger, crop
        if (diff % 2 == 0): #even
            before=int(diff/2)
            after=int(diff/2)
        else: #odd
            before=int(diff/2)
            after=int(diff/2)+1
        im_raw = skimage.util.crop(im_raw,((0,0),(before,after)),copy=False)
    elif (diff<0):#image is smaller, pad
        diff=np.abs(diff)#use the absolute value
        if (diff % 2 == 0): #even
            before=int(diff/2)
            after=int(diff/2)
        else: #odd
            before=int(diff/2)
            after=int(diff/2)+1
        im_raw = skimage.util.pad(im_raw,((0,0),(before,after)),'edge')
    else:#do nothing
        pass
    #return the adjusted image
    return(im_raw)

#this works fine, without re-exporting numpy to decafx
#just put from import_ideas import test_print in decafx/__init__.py
#def test_print():
#    print(np.arange(3))