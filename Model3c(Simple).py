#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 19:19:05 2018

@author: nikita
"""

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

def createModel():
    
    inputs = Input((height, width, channels,))
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape = inputs))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = createModel()

model.summary()
model.fit(X_train, y_train, batch_size = None, epochs = 50, verbose = 1)