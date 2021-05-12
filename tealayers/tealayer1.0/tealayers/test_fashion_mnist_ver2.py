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
sys.path.append("/../../../rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

# Load FASHION_MNIST data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train =  x_train / 255
x_test  = x_test / 255

# try:
#     x_train = np.load("x_train_2bit.npy")
#     x_test = np.load("x_test_2bit.npy")
#     y_train = np.load("y_train.npy")
#     y_test = np.load("y_test.npy")
# except:
#     x_train_1 =  x_train / 255

#     x_test_1  = x_test / 255

#     x_train_2 = x_train
#     x_test_2 = x_test
#     for ele in x_train_2:
#             for i in range(28):
#                     for j in range(28):
#                             if ele[i][j] <= 128:
#                                     ele[i][j]= ele[i][j] / 128
#                             else:
#                                     ele[i][j]=(ele[i][j]-128)/128 

#     for ele in x_test_2:
#             for i in range(28):
#                     for j in range(28):
#                             if ele[i][j] <= 128:
#                                     ele[i][j]= ele[i][j] / 128
#                             else:
#                                     ele[i][j]=(ele[i][j]-128)/128

#     x_train = np.concatenate([x_train_1[:,:,:,np.newaxis],x_train_2[:,:,:,np.newaxis]],axis=3)
#     np.save("x_train_2bit",x_train)
#     x_test = np.concatenate([x_test_1[:,:,:,np.newaxis],x_test_2[:,:,:,np.newaxis]],axis=3)
#     np.save("x_test_2bit",x_test)
#     np.save("y_train",y_train)
#     np.save("y_test",y_test)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# np.save("y_train",y_train)

# Embedding data via tealayer 

inputs = Input(shape=(28, 28,))

flattened = Flatten()(inputs)
flattened_inputs = Lambda(lambda x : x[:,:784])(flattened)
# flattened_inputs_2 = Lambda(lambda x : x[:,784:])(flattened)
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

# x1_12 = Lambda(lambda x : x[:,:256])(flattened_inputs_2)
# x1_13 = Lambda(lambda x : x[:,48:304])(flattened_inputs_2)
# x1_14 = Lambda(lambda x : x[:,96:352])(flattened_inputs_2)
# x1_15 = Lambda(lambda x : x[:,144:400])(flattened_inputs_2)
# x1_16 = Lambda(lambda x : x[:,192:448])(flattened_inputs_2)
# x1_17 = Lambda(lambda x : x[:,240:496])(flattened_inputs_2)
# x1_18 = Lambda(lambda x : x[:,288:544])(flattened_inputs_2)
# x1_19 = Lambda(lambda x : x[:,336:592])(flattened_inputs_2)
# x1_20 = Lambda(lambda x : x[:,384:640])(flattened_inputs_2)
# x1_21 = Lambda(lambda x : x[:,432:688])(flattened_inputs_2)
# x1_22 = Lambda(lambda x : x[:,480:736])(flattened_inputs_2)
# x1_23 = Lambda(lambda x : x[:,528:])(flattened_inputs_2)

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

# x1_12 = Tea(108)(x1_12) ### one Tealayer equa a kernel with numer of "output" feature map ### 
# x1_13 = Tea(108)(x1_13) ### 108 feature map ### 
# x1_14 = Tea(108)(x1_14) 
# x1_15 = Tea(108)(x1_15)
# x1_16 = Tea(108)(x1_16)
# x1_17 = Tea(108)(x1_17)
# x1_18 = Tea(108)(x1_18)
# x1_19 = Tea(108)(x1_19)
# x1_20 = Tea(108)(x1_20)
# x1_21 = Tea(108)(x1_21)
# x1_22 = Tea(108)(x1_22)
# x1_23 = Tea(108)(x1_23)

# x1_0 = Average()([x1_0,x1_12])
# x1_1 = Average()([x1_1,x1_13])
# x1_2 = Average()([x1_2,x1_14])
# x1_3 = Average()([x1_3,x1_15])
# x1_4 = Average()([x1_4,x1_16])
# x1_5 = Average()([x1_5,x1_17])
# x1_6 = Average()([x1_6,x1_18])
# x1_7 = Average()([x1_7,x1_19])
# x1_8 = Average()([x1_8,x1_20])
# x1_9 = Average()([x1_9,x1_21])
# x1_10 = Average()([x1_10,x1_22])
# x1_11 = Average()([x1_11,x1_23])
### 12 kernel => 1296 feature map for 108 feature map for each kernel ### 

### concanate output ### 

x1 = Concatenate(axis=1)([x1_0,x1_1,x1_2,x1_3,x1_4,x1_5,x1_6,x1_7,x1_8,x1_9,x1_10,x1_11])
x1 = Reshape((1296,1)) (x1)
x1 = MaxPooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x1) ### 648 eles left 
x1 = Lambda(lambda x : squeeze(x,2))(x1)

# x1a = Concatenate(axis=1)([x1_12,x1_13,x1_14,x1_15,x1_16,x1_17,x1_18,x1_19,x1_20,x1_21,x1_22,x1_23])
# x1a = Reshape((1296,1)) (x1a)
# x1a = MaxPooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x1a) ### 648 eles left 
# x1a = Lambda(lambda x : squeeze(x,2))(x1a)

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

# x2_10 = Lambda(lambda x : x[:,:256])(x1a)
# x2_11 = Lambda(lambda x : x[:,49:305])(x1a)
# x2_12 = Lambda(lambda x : x[:,98:354])(x1a)
# x2_13 = Lambda(lambda x : x[:,147:403])(x1a)
# x2_14 = Lambda(lambda x : x[:,196:452])(x1a)
# x2_15 = Lambda(lambda x : x[:,245:501])(x1a)
# x2_16 = Lambda(lambda x : x[:,294:550])(x1a)
# x2_17 = Lambda(lambda x : x[:,343:599])(x1a)
# x2_18 = Lambda(lambda x : x[:,392:])(x1a)


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

# x2_10 = Tea(81)(x2_10)
# x2_11 = Tea(81)(x2_11)
# x2_12 = Tea(81)(x2_12)
# x2_13 = Tea(81)(x2_13)
# x2_14 = Tea(81)(x2_14)
# x2_15 = Tea(81)(x2_15)
# x2_16 = Tea(81)(x2_16)
# x2_17 = Tea(81)(x2_17)
# x2_18 = Tea(81)(x2_18)

# x2_1 = Average()([x2_1,x2_10])
# x2_2 = Average()([x2_2,x2_11])
# x2_3 = Average()([x2_3,x2_12])
# x2_4 = Average()([x2_4,x2_13])
# x2_5 = Average()([x2_5,x2_14])
# x2_6 = Average()([x2_6,x2_15])
# x2_7 = Average()([x2_7,x2_16])
# x2_8 = Average()([x2_8,x2_17])
# x2_9 = Average()([x2_9,x2_18])

# concanate output # 

x2 = Concatenate(axis=1)([x2_1,x2_2,x2_3,x2_4,x2_5,x2_6,x2_7,x2_8,x2_9])
x2 = Reshape((729,1)) (x2)
x2  = MaxPooling1D(pool_size=3, strides=2, padding="valid", data_format="channels_last")(x2) ### 364 eles left
x2  = Lambda(lambda x : squeeze(x,2))(x2)

# x2a = Concatenate(axis=1)([x2_10,x2_11,x2_12,x2_13,x2_14,x2_15,x2_16,x2_17,x2_18])
# x2a = Reshape((729,1)) (x2a)
# x2a  = MaxPooling1D(pool_size=3, strides=2, padding="valid", data_format="channels_last")(x2a) ### 364 eles left
# x2a  = Lambda(lambda x : squeeze(x,2))(x2a)

# 4 feature map # 

x3_1 = Lambda(lambda x : x[:,:256])(x2)
x3_2 = Lambda(lambda x : x[:,36:292])(x2)
x3_3 = Lambda(lambda x : x[:,72:328])(x2)
x3_4 = Lambda(lambda x : x[:,108:])(x2)

# x3_5 = Lambda(lambda x : x[:,:256])(x2a)
# x3_6 = Lambda(lambda x : x[:,36:292])(x2a)
# x3_7 = Lambda(lambda x : x[:,72:328])(x2a)
# x3_8 = Lambda(lambda x : x[:,108:])(x2a)

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

# x3_5 = Tea(64)(x3_5)
# x3_6 = Tea(64)(x3_6)
# x3_7 = Tea(64)(x3_7)
# x3_8 = Tea(64)(x3_8)

# x3_1 = Average()([x3_1,x3_5])
# x3_2 = Average()([x3_2,x3_6])
# x3_3 = Average()([x3_3,x3_7])
# x3_4 = Average()([x3_4,x3_8])

# x27 = Tea(42)(x27)
# x28 = Tea(42)(x28)

# final out 

x_out = Concatenate(axis=1)([x3_1,x3_2,x3_3,x3_4])

x_out = Tea(250)(x_out) 

# x_out_b = Concatenate(axis=1)([x3_5,x3_6,x3_7,x3_8])

# x_out_b = Tea(250)(x_out_b) 

# x_out = Concatenate(axis=1)([x_out_a,x_out_b])

x_out = AdditivePooling(10)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=128,epochs=50,verbose=1,validation_split=0.2)
# model.save("mnist_12_3_1.h5")
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])