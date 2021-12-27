import tensorflow as tf
import os
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import math

class DataGenerator(tf.compat.v1.keras.utils.Sequence):
 
    def __init__(self, X_data , y_data, batch_size, dim, n_classes,
                 to_fit, shuffle = True):
        self.batch_size = batch_size
        self.X_data = X_data
        self.labels = y_data
        self.y_data = y_data
        self.to_fit = to_fit
        self.n_classes = n_classes
        self.dim = dim
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange(len(self.X_data))
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1
        
        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0
        
        return data
    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes)/self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
            (index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X = self._generate_x(list_IDs_temp)
        
        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.X_data))
        
        if self.shuffle: 
            np.random.shuffle(self.indexes)

    def _generate_x(self, list_IDs_temp):
              
        X = np.empty((self.batch_size, *self.dim,5))
        
        for i, ID in enumerate(list_IDs_temp):
            
            x_temp = []

            mask = np.ones_like(self.X_data[ID])
    
            e_1 = np.array(self.X_data[ID]>=mask*25).astype(float)
            # e_2 = np.array(train_data[i]>=mask*50).astype(float)
            e_2 = np.array(self.X_data[ID]>=mask*75).astype(float)
            # e_4 = np.array(train_data[i]>=mask*100).astype(float)
            e_3 = np.array(self.X_data[ID]>=mask*125).astype(float)
            # e_6 = np.array(train_data[i]>=mask*150).astype(float)
            e_4 = np.array(self.X_data[ID]>=mask*175).astype(float)
            # e_8 = np.array(train_data[i]>=mask*200).astype(float)
            e_5 = np.array(self.X_data[ID]>=mask*225).astype(float)

            x_temp.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                            e_4[:,:,np.newaxis],e_5[:,:,np.newaxis]),axis=2))
            X[i,] = np.array(x_temp)
            
            # Normalize data
            # X = (X/255).astype('float32')
            
        return X

    def _generate_y(self, list_IDs_temp):
        
        y = np.empty(self.batch_size)
        
        for i, ID in enumerate(list_IDs_temp):
            
            y[i] = self.y_data[ID]
            
        return keras.utils.to_categorical(
                y,num_classes=self.n_classes)