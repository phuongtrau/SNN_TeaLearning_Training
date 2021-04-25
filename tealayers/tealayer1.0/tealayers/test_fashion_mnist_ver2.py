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
sys.path.append("/home/hoangphuong/Documents/FPGA/SNN_TeaLearning_Training/rancutils/rancutils")

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

x1_0 = Lambda(lambda x : x[:,:256])(flattened_inputs)
x1_1 = Lambda(lambda x : x[:,48:304])(flattened_inputs)
x1_2 = Lambda(lambda x : x[:,96:352])(flattened_inputs)
x1_3 = Lambda(lambda x : x[:,144:400])(flattened_inputs)
x1_4 = Lambda(lambda x : x[:,192:448])(flattened_inputs)
x1_5 = Lambda(lambda x : x[:,240:496])(flattened_inputs)
x1_6 = Lambda(lambda x : x[:,288:544])(flattened_inputs)
x1_7 = Lambda(lambda x : x[:,336:592])(flattened_inputs)
x1_8 = Lambda(lambda x : x[:,384:640])(flattened_inputs)
x1_9 = Lambda(lambda x : x[:,432:688])(flattened_inputs)
x1_10 = Lambda(lambda x : x[:,480:736])(flattened_inputs)
x1_11 = Lambda(lambda x : x[:,528:])(flattened_inputs)

# emb via tealayer # output image shape (36 , 36) = 1296 pixel

x1_0 = Tea(108)(x1_0) ### one Tealayer equa a kernel with numer of "output" feature map ### 
x1_1 = Tea(108)(x1_1) ### 108 feature map ### 
x1_2 = Tea(108)(x1_2) 
x1_3 = Tea(108)(x1_3)
x1_4 = Tea(108)(x1_4)
x1_5 = Tea(108)(x1_5)
x1_6 = Tea(108)(x1_6)
x1_7 = Tea(108)(x1_7)
x1_8 = Tea(108)(x1_8)
x1_9 = Tea(108)(x1_9)
x1_10 = Tea(108)(x1_10)
x1_11 = Tea(108)(x1_11)

### 12 kernel => 1296 feature map for 108 feature map for each kernel ### 

### concanate output ### 

x1 = Concatenate(axis=1)([x1_0,x1_1,x1_2,x1_3,x1_4,x1_5,x1_6,x1_7,x1_8,x1_9,x1_10,x1_11])
x1 = Reshape((1296,1)) (x1)
x1 = AveragePooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x1) ### 648 eles left 
x1 = Lambda(lambda x : squeeze(x,2))(x1)

# print(x12)
# x1_flatten = x1
# x1 = Flatten()(x12)

# 9 feature map #
x2_1 = Lambda(lambda x : x[:,:256])(x1)
x2_2 = Lambda(lambda x : x[:,49:305])(x1)
x2_3 = Lambda(lambda x : x[:,98:354])(x1)
x2_4 = Lambda(lambda x : x[:,147:403])(x1)
x2_5= Lambda(lambda x : x[:,196:452])(x1)
x2_6 = Lambda(lambda x : x[:,245:501])(x1)
x2_7= Lambda(lambda x : x[:,294:550])(x1)
x2_8= Lambda(lambda x : x[:,343:599])(x1)
x2_9= Lambda(lambda x : x[:,392:])(x1)

# emb via tealayer # output image shape (27, 27) = 729 pixel 

x2_1 = Tea(81)(x2_1)
x2_2 = Tea(81)(x2_2)
x2_3 = Tea(81)(x2_3)
x2_4 = Tea(81)(x2_4)
x2_5 = Tea(81)(x2_5)
x2_6 = Tea(81)(x2_6)
x2_7 = Tea(81)(x2_7)
x2_8 = Tea(81)(x2_8)
x2_9 = Tea(81)(x2_9)

# concanate output # 

x2 = Concatenate(axis=1)([x2_1,x2_2,x2_3,x2_4,x2_5,x2_6,x2_7,x2_8,x2_9])
x2 = Reshape((729,1)) (x2)
x2  = AveragePooling1D(pool_size=3, strides=2, padding="valid", data_format="channels_last")(x2) ### 364 eles left
x2  = Lambda(lambda x : squeeze(x,2))(x2)


# 4 feature map # 

x3_1 = Lambda(lambda x : x[:,:256])(x2)
x3_2 = Lambda(lambda x : x[:,36:292])(x2)
x3_3 = Lambda(lambda x : x[:,72:328])(x2)
x3_4 = Lambda(lambda x : x[:,108:])(x2)


# x27 = Lambda(lambda x : x[:,55:311])(x22_flatten)
# x28 = Lambda(lambda x : x[:,68:])(x22_flatten)

# # inception ideal model # 
# x_out_1 = Tea(250)(x23)
# x_out_2 = Tea(250)(x24)
# x_out_3 = Tea(250)(x25)
# x_out_4 = Tea(250)(x26)
# x_out_5 = Tea(250)(x27)
# x_out_6 = Tea(250)(x28)

# emb via tealayer # output image shape (16, 16) = 256  

x3_1 = Tea(64)(x3_1)
x3_2 = Tea(64)(x3_2)
x3_3 = Tea(64)(x3_3)
x3_4 = Tea(64)(x3_4)

# x27 = Tea(42)(x27)
# x28 = Tea(42)(x28)

# final out 

x_out_7 = Concatenate(axis=1)([x3_1,x3_2,x3_3,x3_4])

x_out_7 = Tea(250)(x_out_7) 

# x_out = Concatenate(axis=1)([x_out_1,x_out_2,x_out_3,x_out_4,x_out_5,x_out_6,x_out_7])

x_out_7 = AdditivePooling(10)(x_out_7)

predictions = Activation('softmax')(x_out_7)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=128,epochs=30,verbose=1,validation_split=0.2)
# model.save("mnist_12_3_1.h5")
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])