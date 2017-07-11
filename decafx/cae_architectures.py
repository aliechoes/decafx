# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:39:00 2017

@author: N.Chlis
"""

#%% import modules
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import merge
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D as Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.merge import concatenate #Concatenate (capital C) not working 
import numpy as np

#%%
def get_encoder(autoencoder):
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='encoder').output)
        return(encoder)

#def cae_encode(data, encoder):
#    imsize= (data.shape[1],data.shape[2])
#    X_enc=encoder.predict([data[:,:,:,0].reshape(data.shape[0], imsize[0], imsize[0], 1),
#                      data[:,:,:,1].reshape(data.shape[0], imsize[0], imsize[0], 1)])
#    num_pixels = X_enc.shape[1] * X_enc.shape[2] * X_enc.shape[3] #encoded pixels
#    return(X_enc.reshape(X_enc.shape[0], num_pixels))
def cae_encode(data, encoder):
    if(len(data.shape)<3):
        print('array of too few dimensions, shape:',data.shape)
        print('expected at least 3 dimensions, returning None')
        return(None)
    data_list=[]
    nchannels = data.shape[-1]
    if(len(data.shape)>3):#batch of images
        imsize = (data.shape[1],data.shape[2])
        for i in np.arange(nchannels):
            data_list.append(data[:,:,:,i].reshape(data.shape[0], imsize[0], imsize[0], 1))
    else:#single image
        imsize = (data.shape[0],data.shape[1])
        for i in np.arange(nchannels):
            data_list.append(data[:,:,i].reshape(1, imsize[0], imsize[0], 1))
    X_enc=encoder.predict(data_list)
    nchannels=X_enc.shape[-1]#number of channels in the encoded dimension
    #print('Xenc shape',X_enc.shape)
    enc_list=[]
    for i in np.arange(nchannels):
        X_cur=X_enc[:,:,:,i]
        #print('i',i,'X_cur shape',X_cur.shape)
        num_pixels = X_cur.shape[1] * X_cur.shape[2]#encoded pixels
        enc_list.append(X_cur.reshape(X_cur.shape[0], num_pixels))
    return(np.concatenate(enc_list,axis=-1))

def cae_autoencode(data, autoencoder):
    if(len(data.shape)<3):
        print('array of too few dimensions, shape:',data.shape)
        print('expected at least 3 dimensions, returning None')
        return(None)
    data_list=[]
    nchannels = data.shape[-1]
    if(len(data.shape)>3):#batch of images
        imsize = (data.shape[1],data.shape[2])
        for i in np.arange(nchannels):
            data_list.append(data[:,:,:,i].reshape(data.shape[0], imsize[0], imsize[0], 1))
    else:#single image
        imsize = (data.shape[0],data.shape[1])
        for i in np.arange(nchannels):
            data_list.append(data[:,:,i].reshape(1, imsize[0], imsize[0], 1))
    X_enc=autoencoder.predict(data_list)
    return(X_enc)
    
#%%
def cae_indepIn(nchannels=2, imsize=(32,32), encoding_dim_multiplier=8,
                  inner_activ='relu', output_activ='linear', mode='tf'):
    """
    Convolutional Auto Encoder for feature image extraction.
    All image channels are encoded indepedently, so there is
    1-to-1 correspondence from channels to extracted features.
    Utilizes 3 convolutional layers in the encoder.
    
    Inputs
    -------
    nchannels: the number of image channels, the default is 3
    imsize: the image size, the default is (32,32)
    encoding_dim_multiplier: controls number of features to extract per channel
        for example, a 32x32 image is reduced to 4x4 after 3 max-pool layers.
        if encoding_dim_multiplier=8, then 4x4x8=128 features will be extracted
        per channel. The default value is 8.
    activ_inner: inner activation function (default 'tanh').
    output_activ: output activation function, 'linear' by default
        
    Returns
    -------
    A keras Model implementing the convolutional autoencoder. It needs to be
    compiled before training.
    
    Examples
    -------
    >> from keras.models import Model
    >> autoencoder = cae3_indepIn()
    >> autoencoder.compile(loss='mse', optimizer='adam')
    >> encoder = Model(input=autoencoder.input, output=autoencoder.get_layer(name='encoder').output)

    @author: N.Chlis
    """
        
    ch_in = [None]*nchannels#input
    conv0 = [None]*nchannels#conv layers
    conv1 = [None]*nchannels
    conv2 = [None]*nchannels
    pool0 = [None]*nchannels#pooling layers
    pool1 = [None]*nchannels
    pool2 = [None]*nchannels
    bnrm0 = [None]*nchannels#batchnorm layers
    bnrm1 = [None]*nchannels
    bnrm2 = [None]*nchannels
    actv0 = [None]*nchannels#activation layers
    actv1 = [None]*nchannels
    actv2 = [None]*nchannels
            
    conv0b = [None]*nchannels
    bnrm0b = [None]*nchannels
    actv0b = [None]*nchannels
    conv1b = [None]*nchannels
    bnrm1b = [None]*nchannels
    actv1b = [None]*nchannels
#    conv2b = [None]*nchannels
#    bnrm2b = [None]*nchannels
#    actv2b = [None]*nchannels
    if mode=='tf':#Tensorflow format
            shape_in=(imsize[0], imsize[1], 1)
            bnorm_axis=-1
    else:#Theano format
            shape_in=(1, imsize[0], imsize[1])
            bnorm_axis=1
            
    # encoder: channel independent convolutions 
    for i in range(nchannels):
        ch_in[i]=Input(shape=shape_in, name='channel'+str(i)+'_input')#32x32#Tensorflow
        conv0[i]=Convolution2D(64, (3, 3), padding='same', name='conv0_ch'+str(i))(ch_in[i])
        bnrm0[i]=BatchNormalization(axis=bnorm_axis)(conv0[i])
        actv0[i]=Activation(inner_activ)(bnrm0[i])
        conv0b[i]=Convolution2D(64, (3, 3), padding='same', name='conv0b_ch'+str(i))(actv0[i])
        bnrm0b[i]=BatchNormalization(axis=bnorm_axis)(conv0b[i])
        actv0b[i]=Activation(inner_activ)(bnrm0b[i])
        pool0[i]=MaxPooling2D((2, 2),)(actv0b[i])#16x16

        conv1[i]=Convolution2D(32, (3, 3), padding='same', name='conv1_ch'+str(i))(pool0[i])
        bnrm1[i]=BatchNormalization(axis=bnorm_axis)(conv1[i])
        actv1[i]=Activation(inner_activ)(bnrm1[i])
        conv1b[i]=Convolution2D(32, (3, 3), padding='same', name='conv1b_ch'+str(i))(actv1[i])
        bnrm1b[i]=BatchNormalization(axis=bnorm_axis)(conv1b[i])
        actv1b[i]=Activation(inner_activ)(bnrm1b[i])
        pool1[i]=MaxPooling2D((2, 2),)(actv1b[i])#8x8

        conv2[i]=Convolution2D(encoding_dim_multiplier, (3, 3), padding='same', name='conv2_ch'+str(i))(pool1[i])
        bnrm2[i]=BatchNormalization(axis=bnorm_axis)(conv2[i])
        actv2[i]=Activation(inner_activ)(bnrm2[i])
        pool2[i]=MaxPooling2D((2, 2),)(actv2[i])#4x4
    
    # merge the features from all channels
#    conv_merge=merge(pool2,mode='concat',concat_axis=1, name='encoder')
    if(nchannels>1):    
        conv_merge=concatenate(pool2, axis=-1, name='encoder')#merge channels
    else:
        conv_merge=Lambda(lambda x: x, name='encoder')(pool2[0])#only 1 channel
    
    # decoder: introduce channel interactions
    x=Convolution2D(encoding_dim_multiplier+1, (3, 3), padding='same')(conv_merge)
    x=BatchNormalization(axis=bnorm_axis)(x)
    x=Activation(inner_activ)(x)
    x=UpSampling2D((2, 2),)(x)#8x8
    
    x=Convolution2D(32, (3, 3), padding='same')(x)
    x=BatchNormalization(axis=bnorm_axis)(x)
    x=Activation(inner_activ)(x)
    x=Convolution2D(32, (3, 3), padding='same')(x)
    x=BatchNormalization(axis=bnorm_axis)(x)
    x=Activation(inner_activ)(x)
    x=UpSampling2D((2, 2),)(x)#16x16
    
    x=Convolution2D(64, (3, 3), padding='same')(x)
    x=BatchNormalization(axis=bnorm_axis)(x)
    x=Activation(inner_activ)(x)
    x=Convolution2D(64, (3, 3), padding='same')(x)
    x=BatchNormalization(axis=bnorm_axis)(x)
    x=Activation(inner_activ)(x)
    x=UpSampling2D((2, 2),)(x)#32x32
    
    out=Convolution2D(nchannels, (3, 3), padding='same',activation=output_activ,name='output')(x)
    model=Model(inputs=ch_in,outputs=out)
    model.compile(loss='mse', optimizer='adam')
    return model
