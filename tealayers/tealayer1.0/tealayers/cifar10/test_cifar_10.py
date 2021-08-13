from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math

import tensorflow as tf
from tensorflow import squeeze
import numpy as np
from keras import backend as K
from keras import Model
from keras.engine.topology import Layer
from keras import initializers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, concatenate,Average, Concatenate, MaxPooling1D, Reshape
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import to_categorical

from tea import Tea
from additivepooling import AdditivePooling

import sys
sys.path.append("../../../rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

# Load MNIST data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# save old labels for later
y_test_not = y_test

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

inputs_1 = Input(shape=(32, 32,3,))

### GET ALL CHANNEL ###  

red = Lambda(lambda x : x[:,:,:,0])(inputs_1)
green = Lambda(lambda x : x[:,:,:,1])(inputs_1)
blue = Lambda(lambda x : x[:,:,:,2])(inputs_1)
inputs = Average()([red,green,blue])
flattened_inputs = Flatten()(inputs)
# flattened_inputs = Activation('sigmoid')(flattened_inputs)

### USING KERNEL  ###

### layer 1 ### 

x1 = Lambda(lambda x : x[:,:256])(flattened_inputs)
x2 = Lambda(lambda x : x[:,96:352])(flattened_inputs)
x3 = Lambda(lambda x : x[:,192:448])(flattened_inputs)
x4 = Lambda(lambda x : x[:,288:544])(flattened_inputs)
x5 = Lambda(lambda x : x[:,384:640])(flattened_inputs)
x6 = Lambda(lambda x : x[:,480:736])(flattened_inputs)
x7 = Lambda(lambda x : x[:,576:832])(flattened_inputs)
x8 = Lambda(lambda x : x[:,672:928])(flattened_inputs)
x9 = Lambda(lambda x : x[:,768:])(flattened_inputs)

x1 = Tea(128)(x1)
x2 = Tea(128)(x2)
x3 = Tea(128)(x3)

x4 = Tea(128)(x4)
x5 = Tea(128)(x5)
x6 = Tea(128)(x6)

x7 = Tea(128)(x7)
x8 = Tea(128)(x8)
x9 = Tea(128)(x9)

x_1 = Concatenate(axis=1)([x1,x2,x3,x4,x5,x6,x7,x8,x9])
x_1 = Reshape((1152,1)) (x_1)
x_1  = MaxPooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x_1)
x_1  = Lambda(lambda x : squeeze(x,2))(x_1)

# 576 feature number # 

x1 = Lambda(lambda x : x[:,:256])(x_1)
x2 = Lambda(lambda x : x[:,40:296])(x_1)
x3 = Lambda(lambda x : x[:,80:336])(x_1)
x4 = Lambda(lambda x : x[:,120:376])(x_1)
x5 = Lambda(lambda x : x[:,160:416])(x_1)
x6 = Lambda(lambda x : x[:,200:456])(x_1)
x7 = Lambda(lambda x : x[:,240:496])(x_1)
x8 = Lambda(lambda x : x[:,280:536])(x_1)
x9 = Lambda(lambda x : x[:,320:])(x_1)

x1 = Tea(85)(x1)
x2 = Tea(85)(x2)
x3 = Tea(85)(x3)

x4 = Tea(85)(x4)
x5 = Tea(85)(x5)
x6 = Tea(85)(x6)

x7 = Tea(85)(x7)
x8 = Tea(85)(x8)
x9 = Tea(85)(x9)

x_2 = Concatenate(axis=1)([x1,x2,x3])
x_3 = Concatenate(axis=1)([x4,x5,x6])
x_4 = Concatenate(axis=1)([x7,x8,x9])

x_2 = Tea(85)(x_2)

x_3 = Tea(85)(x_3)

x_4 = Tea(85)(x_4)

x_out = Concatenate(axis=1)([x_2,x_3,x_4])

x = Tea(250)(x_out)

# Pool spikes and output neurons into 10 classes.

x = AdditivePooling(10)(x)

predictions = Activation('softmax')(x)

model = Model(inputs=inputs_1, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_split=0.2)