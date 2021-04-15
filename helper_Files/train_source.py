#!/usr/bin/env python
# coding: utf-8

# In[4]:


""" This code contains relevant functions used in training code
"""
# import necessary functions

import tensorflow.keras
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras import activations 
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Input, BatchNormalization, Activation
from tensorflow.keras.layers import concatenate

import numpy as np

import matplotlib.pyplot as plt

from random import randint

from random import random


# In[5]:


# Custom Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_Frame, x_Col, y_Col, directory,tile_Namesake, mask_Namesake, subset = None,
                 horizontal_Flips = False, vertical_Flips = False, rotations = False, batch_Size = 32,
                 split = False, training_Ratio = 1, shuffle = False, dim = (128,128), number_Of_Channels = 3,
                 number_Of_Classes = 2, sample_Mean_Zero_Center_Standarardization = True):
        self.batch_Size = batch_Size
        self.df = data_Frame
        self.x_Col = x_Col
        self.y_Col = y_Col
        self.dim = dim
        self.directory = directory
        self.subset = subset
        self.sample_Mean_Zero_Center_Standarardization = sample_Mean_Zero_Center_Standarardization
        self.number_Of_Classes = number_Of_Classes
        self.number_Of_Channels = number_Of_Channels
        self.shuffle = shuffle
        self.tile_Names = self.df[self.x_Col]
        self.truth_Names = self.df[self.y_Col]
        self.tile_Namesake = tile_Namesake
        self.mask_Namesake = mask_Namesake
        self.horizontal_Flips = horizontal_Flips
        self.vertical_Flips = vertical_Flips
        self.index_List = np.arange(len(self.tile_Names)) + 1
        self.rotations = rotations
        self.training_Samples = int(training_Ratio*len(self.index_List))
        if split == True:
            self.train_Index_List = self.index_List[:self.training_Samples]
            self.validate_Index_List = self.index_List[self.training_Samples:]
        else:
            self.train_Index_List = self.index_List[:]
            self.validate_Index_List = []
        if self.shuffle == True:
            self.on_Epoch_End()
    def __len__(self):
        return int(len(self.train_Index_List)/self.batch_Size)
    def __getitem__(self, index):
        if self.subset == "Training":
            indexes = self.train_Index_List[index*self.batch_Size:(index*self.batch_Size) + self.batch_Size]
            X, y_True = self.generate_Batch(indexes)
        elif self.subset == "Validation":
            indexes = self.validate_Index_List[index*self.batch_Size:(index*self.batch_Size) + self.batch_Size]
            X, y_True = self.generate_Batch(indexes)
        else:
            indexes = self.train_Index_List[index*self.batch_Size:(index*self.batch_Size) + self.batch_Size]
            X, y_True = self.generate_Batch(indexes)
        return X, y_True
    def on_Epoch_End(self):
        if self.shuffle == True:
            np.random.shuffle(self.train_Index_List)
    def generate_Batch(self,indexes):
        X = np.zeros((self.batch_Size, *self.dim, self.number_Of_Channels))
        y_True = np.zeros((self.batch_Size, *self.dim, self.number_Of_Classes))
        for count in range(len(indexes)):
            if self.sample_Mean_Zero_Center_Standarardization == True:
                img = plt.imread(self.directory + self.tile_Namesake + " " + str(indexes[count]) + ".png")[:,:,0:3]
                mask = np.load(self.directory + self.mask_Namesake + " " + str(indexes[count]) + ".npy")
                img, mask = self.augment_Image(img,mask)
                X[count,:,:,:] = self.standard_norm(plt.imread(self.directory + self.tile_Namesake + " " + str(indexes[count]) + ".png")[:,:,0:3])
                y_True[count,:,:,:] = np.load(self.directory + self.mask_Namesake + " " + str(indexes[count]) + ".npy")
            else:
                img = plt.imread(self.directory + self.tile_Namesake + " " + str(indexes[index]) + ".png")[:,:,0:3]
                mask = np.load(self.directory + self.mask_Namesake + " " + str(indexes[index]) + ".npy")
                img, mask = self.augment_Image(img,mask)
                X[index,:,:,:] = img
                y_True[index,:,:,:] = mask
        return X, y_True
    def standard_norm(self,img):
        height, width, channels = img.shape
        for channel in range(channels):
            img[:,:,channel] = (img[:,:,channel] - np.mean(img[:,:,channel]))/np.std(img[:,:,channel])
        return img
    def augment_Image(self, image, mask):
        if self.rotations == True:
            random_Integer = randint(1,5)
            image = np.rot90(image.copy(),random_Integer)
            mask = np.rot90(mask.copy(),random_Integer)
        random_Float = random()
        if self.horizontal_Flips == True and random_Float < 0.5:
            image = np.flip(image.copy(),0)
            mask = np.flip(mask.copy(),0)
        random_Float = random()
        if self.vertical_Flips == True and random_Float < 0.5:
            image = np.flip(image.copy(),1)
            mask = np.flip(mask.copy(),1)
        return image, mask
    
# Model
def Phase1_Net(img_size, num_classes):
    inputs = Input(shape=img_size + (3,))

    x = Conv2D(64,kernel_size = 3, strides = (1,1),
                            padding = "same")(inputs)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    previous_block_concatenate1 = x
    x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)

    x = Conv2D(128,kernel_size = 3, strides = (1,1),
                            padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)

    previous_block_concatenate2 = x

    concate_block_num = 3
    for filters in [256, 512, 512]:
        x = Conv2D(filters,3, strides = (1,1),
                            padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = Conv2D(filters,3, strides = 1,
                         padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)
        globals()['previous_block_concatenate%s' % concate_block_num] = x
        concate_block_num = concate_block_num + 1
        print(("No errors for filter size:" + str(filters)))



    x = Conv2D(512,3, strides = 1,
                            padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)

    x = Conv2D(512,3, strides = 1,
                            padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2DTranspose(256,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate5], axis =-1)

    x = Conv2DTranspose(256,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate4],axis=-1)

    x = Conv2DTranspose(128,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate3],axis=-1)
    
    x = Conv2DTranspose(64,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate2],axis=-1)


    x = Conv2DTranspose(32,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2DTranspose(64,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)


    x = concatenate([x, previous_block_concatenate1],axis=-1)

    x = Conv2D(32,3, strides = (1,1),
                            padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = Conv2D(num_classes,3, strides = (1,1),
                            padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    outputs = Conv2D(num_classes,3, strides = (1,1),
                            activation = 'softmax',
                            padding = 'same',
                            name = 'sRBC_classes')(x)
    model = Model(inputs,outputs)

    return model


# In[ ]:




