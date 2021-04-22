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
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, AveragePooling1D, Reshape, Concatenate, MaxPooling1D,Average 
from keras.datasets import fashion_mnist
from keras.optimizers import Adam,SGD
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tea import Tea
from additivepooling import AdditivePooling

import sys
sys.path.append("/home/phuongdh/Documents/SNN_TeaLearning_Training/rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train =  x_train / 255

x_test  = x_test / 255

# Embedding data via tealayer 

inputs = Input(shape=(28, 28,))

flattened_inputs = Flatten()(inputs)

# 12 feature map #

x0 = Lambda(lambda x : x[:,:256])(flattened_inputs)
x1 = Lambda(lambda x : x[:,48:304])(flattened_inputs)
x2 = Lambda(lambda x : x[:,96:352])(flattened_inputs)
x3 = Lambda(lambda x : x[:,144:400])(flattened_inputs)
x4 = Lambda(lambda x : x[:,192:448])(flattened_inputs)
x5 = Lambda(lambda x : x[:,240:496])(flattened_inputs)
x6 = Lambda(lambda x : x[:,288:544])(flattened_inputs)
x7 = Lambda(lambda x : x[:,336:592])(flattened_inputs)
x8 = Lambda(lambda x : x[:,384:640])(flattened_inputs)
x9 = Lambda(lambda x : x[:,432:688])(flattened_inputs)
x10 = Lambda(lambda x : x[:,480:736])(flattened_inputs)
x11 = Lambda(lambda x : x[:,528:])(flattened_inputs)

# emb via tealayer # output image shape (24 , 24) = 576 pixel

x0 = Tea(48)(x0)
x1 = Tea(48)(x1)
x2 = Tea(48)(x2)
x3 = Tea(48)(x3)
x4 = Tea(48)(x4)
x5 = Tea(48)(x5)
x6 = Tea(48)(x6)
x7 = Tea(48)(x7)
x8 = Tea(48)(x8)
x9 = Tea(48)(x9)
x10 = Tea(48)(x10)
x11 = Tea(48)(x11)

# concanate output # 

x12 = Concatenate(axis=1)([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
# print(x12)
x12_flatten = x12
# x12_flatten = Flatten()(x12)

# 9 feature map #
x13 = Lambda(lambda x : x[:,:256])(x12_flatten)
x14 = Lambda(lambda x : x[:,40:296])(x12_flatten)
x15 = Lambda(lambda x : x[:,80:336])(x12_flatten)
x16 = Lambda(lambda x : x[:,120:376])(x12_flatten)
x17 = Lambda(lambda x : x[:,160:416])(x12_flatten)
x18 = Lambda(lambda x : x[:,200:456])(x12_flatten)
x19 = Lambda(lambda x : x[:,240:496])(x12_flatten)
x20 = Lambda(lambda x : x[:,280:536])(x12_flatten)
x21 = Lambda(lambda x : x[:,320:])(x12_flatten)

# emb via tealayer # output image shape (18, 18) = 324 pixel 

x13 = Tea(36)(x13)
x14 = Tea(36)(x14)
x15 = Tea(36)(x15)
x16 = Tea(36)(x16)
x17 = Tea(36)(x17)
x18 = Tea(36)(x18)
x19 = Tea(36)(x19)
x20 = Tea(36)(x20)
x21 = Tea(36)(x21)

# concanate output # 

x22 = Concatenate(axis=1)([x13,x14,x15,x16,x17,x18,x19,x20,x21])

x22_flatten = x22
# x19_flatten = Flatten()(x19)

# 6 feature map # 

x23 = Lambda(lambda x : x[:,:256])(x22_flatten)
x24 = Lambda(lambda x : x[:,13:269])(x22_flatten)
x25 = Lambda(lambda x : x[:,27:283])(x22_flatten)
x26 = Lambda(lambda x : x[:,40:296])(x22_flatten)
x27 = Lambda(lambda x : x[:,55:311])(x22_flatten)
x28 = Lambda(lambda x : x[:,68:])(x22_flatten)

# # inception ideal model # 
# x_out_1 = Tea(250)(x23)
# x_out_2 = Tea(250)(x24)
# x_out_3 = Tea(250)(x25)
# x_out_4 = Tea(250)(x26)
# x_out_5 = Tea(250)(x27)
# x_out_6 = Tea(250)(x28)

# emb via tealayer # output image shape (16, 16) = 256  

x23 = Tea(42)(x23)
x24 = Tea(42)(x24)
x25 = Tea(42)(x25)
x26 = Tea(42)(x26)
x27 = Tea(42)(x27)
x28 = Tea(42)(x28)

# final out 

x_out_7 = Concatenate(axis=1)([x23,x24,x25,x26,x27,x28])

x_out_7 = Tea(250)(x_out_7) 

# x_out = Concatenate(axis=1)([x_out_1,x_out_2,x_out_3,x_out_4,x_out_5,x_out_6,x_out_7])

x_out = AdditivePooling(10)(x_out_7)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr= 0.001,beta_1 = 0.99,beta_2 = 0.999,epsilon = 1e-05,amsgrad = True),
              metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=128,epochs=100,verbose=1,validation_split=0.2)
# model.save("mnist_12_3_1.h5")
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])