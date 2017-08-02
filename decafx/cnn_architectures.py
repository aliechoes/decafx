# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:35:17 2017

@author: N.Chlis
# deepflow originally introduced in: P. Eulenberg, N. Köhler et al., 2016
# This is a modified version of the keras version originally implemented
# by M. Berthold at https://github.com/moritzbe/CellDeepLearning
"""
#%% import modules
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D as Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.merge import concatenate #Concatenate (capital C) not working 

#%% first define the basic building blocks

# Basic Conv + BN + ReLU
def convFactory(data, num_filter, kernel, stride=(1,1), pad="valid", act_type="relu"): # valid
    conv = Convolution2D(filters=num_filter, kernel_size=kernel, strides=stride, padding=pad)(data)
    bn = BatchNormalization(axis=-1)(conv)
    act = Activation(act_type)(bn)
    return act

# A Simple Downsampling Factory
# pixel dimensions = ((a+2)/2,(a+2)/2)
def downsampleFactory(data, ch_3x3):
    conv = convFactory(data=data, num_filter = ch_3x3, kernel=(3, 3), stride=(2, 2), pad="same", act_type="relu") # same
    pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None)(data) # its actually valid!
    # got shapes [(None, 33, 33, 80), (None, 32, 32, 80)]
    concat = concatenate([conv, pool], axis=-1)
    return concat

# A Simple module
def simpleFactory(data, ch_1x1, ch_3x3):
    conv1x1 = convFactory(data=data, num_filter = ch_1x1, kernel=(1, 1), pad="valid", act_type="relu") # valid
    conv3x3 = convFactory(data=data, num_filter = ch_3x3, kernel=(3, 3), pad="same", act_type="relu") # same
    concat = concatenate([conv1x1, conv3x3], axis=-1)
    return concat

#%% now write down the full model using the building blocks
#def deepflow(n_channels, n_classes):
#	# model = deepflow([1,2,3,4], 4, .01, .09, .0005)
#    #n_channels = len(channels)
#    inputs = Input(shape=(66, 66, n_channels)) # 66x66
#    conv1 = convFactory(data=inputs, num_filter=96 , kernel=(3,3), pad="same", act_type="relu") # same
#    in3a = simpleFactory(conv1, 32, 32)
#    in3b = simpleFactory(in3a, 32, 48)
#    in3c = downsampleFactory(in3b, 80) # 33x33
#    in4a = simpleFactory(in3c, 112, 48)
#    in4b = simpleFactory(in4a, 96, 64)
#    in4c = simpleFactory(in4b, 80, 80)
#    in4d = simpleFactory(in4c, 48, 96)
#    in4e = downsampleFactory(in4d, 96) # 17x17
#    in5a = simpleFactory(in4e, 176, 160)
#    in5b = simpleFactory(in5a, 176, 160)
#    in6a = downsampleFactory(in5b, 96) # 8x8
#    in6b = simpleFactory(in6a, 176, 160)
#    in6c = simpleFactory(in6b, 176, 160)
#    pool = AveragePooling2D(pool_size=(9, 9), strides=None, padding='valid', data_format=None)(in6c) # valid
#    flatten = Flatten(name='flatten-activations')(pool)
#    fc = Dense(n_classes, activation=None, name='fc-activations')(flatten)
#    softmax = Activation(activation="softmax")(fc)
#    model = Model(inputs=inputs, outputs=softmax)
#    model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#    return model

def deepflow(n_channels, n_classes,resize_factor=1):
	# model = deepflow([1,2,3,4], 4, .01, .09, .0005)
    #n_channels = len(channels)
    if(resize_factor<=0):
        print('Warning: a non-positive resize factor was provided, resetting to 1...')
    
    inputs = Input(shape=(66, 66, n_channels)) # 66x66
    conv1 = convFactory(data=inputs, num_filter=int(96*resize_factor) , kernel=(3,3), pad="same", act_type="relu") # same
    in3a = simpleFactory(conv1, int(32*resize_factor), int(32*resize_factor))
    in3b = simpleFactory(in3a, int(32*resize_factor), int(48*resize_factor))
    in3c = downsampleFactory(in3b, int(80*resize_factor)) # 33x33
    in4a = simpleFactory(in3c, int(112*resize_factor), int(48*resize_factor))
    in4b = simpleFactory(in4a, int(96*resize_factor), int(64*resize_factor))
    in4c = simpleFactory(in4b, int(80*resize_factor), int(80*resize_factor))
    in4d = simpleFactory(in4c, int(48*resize_factor), int(96*resize_factor))
    in4e = downsampleFactory(in4d, int(96*resize_factor)) # 17x17
    in5a = simpleFactory(in4e, int(176*resize_factor), int(160*resize_factor))
    in5b = simpleFactory(in5a, int(176*resize_factor), int(160*resize_factor))
    in6a = downsampleFactory(in5b, int(96*resize_factor)) # 8x8
    in6b = simpleFactory(in6a, int(176*resize_factor), int(160*resize_factor))
    in6c = simpleFactory(in6b, int(176*resize_factor), int(160*resize_factor))
    pool = AveragePooling2D(pool_size=(9, 9), strides=None, padding='valid', data_format=None)(in6c) # valid
    flatten = Flatten(name='flatten-activations')(pool)
    fc = Dense(n_classes, activation=None, name='fc-activations')(flatten)
    softmax = Activation(activation="softmax")(fc)
    model = Model(inputs=inputs, outputs=softmax)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
#%%
#%%
def get_last_layer(deepflow):
        last = Model(inputs=deepflow.input, outputs=deepflow.get_layer(name='flatten-activations').output)
        return(last)