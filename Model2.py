#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:04:54 2018

@author: nikita
"""
import os
import numpy as np
from PIL import Image

import cv2
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Input, Dense, merge, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

Nimgs = 20
channels = 1
height = 584
width = 565
dataset_path = "./Medical Project/Datasets/DRIVE/test/"
smooth = 1

dataset = np.zeros((Nimgs,height,width))

def imread(path):
    img = Image.open(path)
    img_conv = img.convert(mode = "L")
    #img_conv.show()
    img_arr = np.asarray(np.uint8(img_conv))
    return img_arr

img1 = []

def load_dataset(dataset_path, label):
    ind = 0
    dataset = np.zeros((Nimgs,channels,height,width))
    labels = np.zeros((Nimgs))
    for roots,dirs,files in os.walk(dataset_path):
        for name in files:
            full_path = os.path.join(roots,name)
            img1 = imread(full_path)
            dataset[ind,0,:,:] = img1[:,:]
            labels[ind] = label
            #print(full_path)
            ind += 1
    return (dataset,labels)
        

ds1,ls1 = load_dataset(dataset_path+"/1st_manual/",0)    
ds2,ls2 = load_dataset(dataset_path+"/2nd_manual/",1)
ds = np.concatenate([ds1,ds2],axis=0)
ls = np.concatenate([ls1,ls2])

X_train, X_test, y_train, y_test = train_test_split(ds, ls, test_size=0.33, random_state=42)


def get_unet(patch_height,patch_width, n_ch):
    
    inputs = Input((patch_height, patch_width, n_ch,))
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print("conv1.shape:", conv1.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print("conv1.shape:", conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1.shape:", pool1.shape)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print("conv2.shape:", conv2.shape)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print("conv2.shape:", conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2.shape:", pool2.shape)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print("conv3.shape:", conv3.shape)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print("conv1.shape:", conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool3.shape:", pool3.shape)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    print("conv4.shape:", conv4.shape)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    print("conv1.shape:", conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    print("drop4.shape:", drop4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print("pool4.shape:", pool4.shape)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    print("conv5.shape:", conv5.shape)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    print("drop5.shape:", drop5.shape)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

model = get_unet(height,width, channels)
get_unet(ds,ls, channels)
model = createModel()
model.summary()
model.fit(X_train, y_train, batch_size = None, epochs = 50, verbose = 1)