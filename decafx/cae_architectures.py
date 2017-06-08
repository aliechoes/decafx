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
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D

#%%
def get_encoder(autoencoder):
        encoder = Model(input=autoencoder.input, output=autoencoder.get_layer(name='encoder').output)
        return(encoder)
    
#%%
def cae_indepIn(nchannels=2, imsize=(32,32), encoding_dim_multiplier=8,
                  inner_activ='tanh', output_activ='linear'):
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
    
    # encoder: channel independent convolutions 
    for i in range(nchannels):
        ch_in[i]=Input(shape=(1, imsize[0], imsize[1]),name='channel'+str(i)+'_input')#32x32
        conv0[i]=Convolution2D(64, 3, 3, border_mode='same', name='conv0_ch'+str(i))(ch_in[i])
        bnrm0[i]=BatchNormalization(mode=0, axis=1)(conv0[i])
        actv0[i]=Activation(inner_activ)(bnrm0[i])
        conv0b[i]=Convolution2D(64, 3, 3, border_mode='same', name='conv0b_ch'+str(i))(actv0[i])
        bnrm0b[i]=BatchNormalization(mode=0, axis=1)(conv0b[i])
        actv0b[i]=Activation(inner_activ)(bnrm0b[i])
        pool0[i]=MaxPooling2D((2, 2))(actv0b[i])#16x16

        conv1[i]=Convolution2D(32, 3, 3, border_mode='same', name='conv1_ch'+str(i))(pool0[i])
        bnrm1[i]=BatchNormalization(mode=0, axis=1)(conv1[i])
        actv1[i]=Activation(inner_activ)(bnrm1[i])
        conv1b[i]=Convolution2D(32, 3, 3, border_mode='same', name='conv1b_ch'+str(i))(actv1[i])
        bnrm1b[i]=BatchNormalization(mode=0, axis=1)(conv1b[i])
        actv1b[i]=Activation(inner_activ)(bnrm1b[i])
        pool1[i]=MaxPooling2D((2, 2))(actv1b[i])#8x8

        conv2[i]=Convolution2D(encoding_dim_multiplier, 3, 3, border_mode='same', name='conv2_ch'+str(i))(pool1[i])
        bnrm2[i]=BatchNormalization(mode=0, axis=1)(conv2[i])
        actv2[i]=Activation(inner_activ)(bnrm2[i])
        pool2[i]=MaxPooling2D((2, 2))(actv2[i])#4x4
    
    # merge the features from all channels
    conv_merge=merge(pool2,mode='concat',concat_axis=1, name='encoder')
    
    # decoder: introduce channel interactions
    x=Convolution2D(encoding_dim_multiplier+1, 3, 3, border_mode='same')(conv_merge)
    x=BatchNormalization(mode=0, axis=1)(x)
    x=Activation(inner_activ)(x)
    x=UpSampling2D((2, 2))(x)#8x8
    
    x=Convolution2D(32, 3, 3, border_mode='same')(x)
    x=BatchNormalization(mode=0, axis=1)(x)
    x=Activation(inner_activ)(x)
    x=Convolution2D(32, 3, 3, border_mode='same')(x)
    x=BatchNormalization(mode=0, axis=1)(x)
    x=Activation(inner_activ)(x)
    x=UpSampling2D((2, 2))(x)#16x16
    
    x=Convolution2D(64, 3, 3, border_mode='same')(x)
    x=BatchNormalization(mode=0, axis=1)(x)
    x=Activation(inner_activ)(x)
    x=Convolution2D(64, 3, 3, border_mode='same')(x)
    x=BatchNormalization(mode=0, axis=1)(x)
    x=Activation(inner_activ)(x)
    x=UpSampling2D((2, 2))(x)#32x32
    
    out=Convolution2D(2, 3, 3, border_mode='same',activation=output_activ,name='output')(x)
    model=Model(input=ch_in,output=out)
    return model
