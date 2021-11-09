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
import sys
sys.path.append("../../../../rancutils/rancutils")
sys.path.append("../")
from tea import Tea
from deep_tea import DeepTea 
from additivepooling import AdditivePooling
import cv2

# import sys
# sys.path.append("../../../rancutils/rancutils")

# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

## change the input threshold spike ## 
x_tr=[]
x_ts=[]
for i in range(len(x_train)):
    mask = np.ones_like(x_train[i])
    
    x_train_temp = cv2.equalizeHist(x_train[i])

    e_1 = np.array(x_train_temp>=mask*31).astype(float)
    e_2 = np.array(x_train_temp>=mask*63).astype(float)
    e_3 = np.array(x_train_temp>=mask*95).astype(float)
    e_4 = np.array(x_train_temp>=mask*127).astype(float)
    e_5 = np.array(x_train_temp>=mask*159).astype(float)
    e_6 = np.array(x_train_temp>=mask*191).astype(float)
    e_7 = np.array(x_train_temp>=mask*223).astype(float)
    # print(e_1[:,:,np.newaxis].shape)
    x_tr.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                   e_4[:,:,np.newaxis],e_5[:,:,np.newaxis],e_6[:,:,np.newaxis],\
                                   e_7[:,:,np.newaxis]),axis=2))
# print(x_tr.shape)
for i in range(len(x_test)):
    mask = np.ones_like(x_test[i])
    
    x_test_temp = cv2.equalizeHist(x_test[i])

    e_1 = np.array(x_test_temp>=mask*31).astype(float)
    e_2 = np.array(x_test_temp>=mask*63).astype(float)
    e_3 = np.array(x_test_temp>=mask*95).astype(float)
    e_4 = np.array(x_test_temp>=mask*127).astype(float)
    e_5 = np.array(x_test_temp>=mask*159).astype(float)
    e_6 = np.array(x_test_temp>=mask*191).astype(float)
    e_7 = np.array(x_test_temp>=mask*223).astype(float)
    # print(e_1[:,:,np.newaxis].shape)
    x_ts.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                   e_4[:,:,np.newaxis],e_5[:,:,np.newaxis],e_6[:,:,np.newaxis],\
                                   e_7[:,:,np.newaxis]),axis=2))
x_tr = np.array(x_tr)
x_ts = np.array(x_ts)

inputs = Input(shape=(28, 28,7))

flattened_inputs = Flatten()(inputs) 
# print(flattened_inputs.shape)

flattened_inputs_1 = Lambda(lambda x : x[:,      :1*784])(flattened_inputs)
flattened_inputs_2 = Lambda(lambda x : x[:,1*784:2*784])(flattened_inputs)
flattened_inputs_3 = Lambda(lambda x : x[:,2*784:3*784])(flattened_inputs)
flattened_inputs_4 = Lambda(lambda x : x[:,3*784:4*784])(flattened_inputs)
flattened_inputs_5 = Lambda(lambda x : x[:,4*784:5*784])(flattened_inputs)
flattened_inputs_6 = Lambda(lambda x : x[:,5*784:6*784])(flattened_inputs)
flattened_inputs_7 = Lambda(lambda x : x[:,6*784:      ])(flattened_inputs)

# flattened_inputs_1 = Lambda(lambda x : x[:,256:])(flattened_inputs_1)
# flattened_inputs_2 = Lambda(lambda x : x[:,256:])(flattened_inputs_2)
# flattened_inputs_3 = Lambda(lambda x : x[:,256:])(flattened_inputs_3)
# flattened_inputs_4 = Lambda(lambda x : x[:,256:])(flattened_inputs_4)
# flattened_inputs_5 = Lambda(lambda x : x[:,256:])(flattened_inputs_5)
# flattened_inputs_6 = Lambda(lambda x : x[:,256:])(flattened_inputs_6)
# flattened_inputs_7 = Lambda(lambda x : x[:,256:])(flattened_inputs_7)

x1_1  = Lambda(lambda x : x[:,     : 256 ])(flattened_inputs_1)
x2_1  = Lambda(lambda x : x[:, 35  : 291 ])(flattened_inputs_1)
x3_1  = Lambda(lambda x : x[:, 70  : 326 ])(flattened_inputs_1)
x4_1  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_1)
x5_1  = Lambda(lambda x : x[:, 140 : 396 ])(flattened_inputs_1)
x6_1  = Lambda(lambda x : x[:, 175 : 431 ])(flattened_inputs_1)
x7_1  = Lambda(lambda x : x[:, 210 : 466 ])(flattened_inputs_1)
x8_1  = Lambda(lambda x : x[:, 245 : 501 ])(flattened_inputs_1)
x9_1  = Lambda(lambda x : x[:, 280 : 536 ])(flattened_inputs_1)
x10_1  = Lambda(lambda x : x[:,315 : 571 ])(flattened_inputs_1)
x11_1  = Lambda(lambda x : x[:,350 : 606 ])(flattened_inputs_1)
x12_1  = Lambda(lambda x : x[:,385 : 641 ])(flattened_inputs_1)
x13_1  = Lambda(lambda x : x[:,420 : 676 ])(flattened_inputs_1)
x14_1  = Lambda(lambda x : x[:,455 : 711 ])(flattened_inputs_1)
x15_1  = Lambda(lambda x : x[:,490 : 746 ])(flattened_inputs_1)
x16_1  = Lambda(lambda x : x[:,525 : 781 ])(flattened_inputs_1)

x1_1_1  = Tea(64)(x1_1)
x2_1_1  = Tea(64)(x2_1)
x3_1_1  = Tea(64)(x3_1)
x4_1_1  = Tea(64)(x4_1)
x5_1_1  = Tea(64)(x5_1)
x6_1_1  = Tea(64)(x6_1)
x7_1_1  = Tea(64)(x7_1)
x8_1_1  = Tea(64)(x8_1)
x9_1_1  = Tea(64)(x9_1)
x10_1_1  = Tea(64)(x10_1)
x11_1_1  = Tea(64)(x11_1)
x12_1_1  = Tea(64)(x12_1)
x13_1_1  = Tea(64)(x13_1)
x14_1_1  = Tea(64)(x14_1)
x15_1_1  = Tea(64)(x15_1)
x16_1_1  = Tea(64)(x16_1)

x1_2  = Lambda(lambda x : x[:,   :256 ])(flattened_inputs_2)
x2_2  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_2)
x3_2  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_2)
x4_2  = Lambda(lambda x : x[:, 105:361 ])(flattened_inputs_2)
x5_2  = Lambda(lambda x : x[:, 140:396 ])(flattened_inputs_2)
x6_2  = Lambda(lambda x : x[:, 175:431 ])(flattened_inputs_2)
x7_2  = Lambda(lambda x : x[:, 210:466 ])(flattened_inputs_2)
x8_2  = Lambda(lambda x : x[:, 245:501 ])(flattened_inputs_2)
x9_2  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_2)
x10_2  = Lambda(lambda x : x[:,315:571])(flattened_inputs_2)
x11_2  = Lambda(lambda x : x[:,350:606])(flattened_inputs_2)
x12_2  = Lambda(lambda x : x[:,385:641])(flattened_inputs_2)
x13_2  = Lambda(lambda x : x[:,420:676])(flattened_inputs_2)
x14_2  = Lambda(lambda x : x[:,455:711])(flattened_inputs_2)
x15_2  = Lambda(lambda x : x[:,490:746])(flattened_inputs_2)
x16_2  = Lambda(lambda x : x[:,525:781])(flattened_inputs_2)

x1_2_2  = Tea(64)(x1_2)
x2_2_2  = Tea(64)(x2_2)
x3_2_2  = Tea(64)(x3_2)
x4_2_2  = Tea(64)(x4_2)
x5_2_2  = Tea(64)(x5_2)
x6_2_2  = Tea(64)(x6_2)
x7_2_2  = Tea(64)(x7_2)
x8_2_2  = Tea(64)(x8_2)
x9_2_2  = Tea(64)(x9_2)
x10_2_2  = Tea(64)(x10_2)
x11_2_2  = Tea(64)(x11_2)
x12_2_2  = Tea(64)(x12_2)
x13_2_2  = Tea(64)(x13_2)
x14_2_2  = Tea(64)(x14_2)
x15_2_2  = Tea(64)(x15_2)
x16_2_2  = Tea(64)(x16_2)

x1_3  = Lambda(lambda x : x[:,   :256 ])(flattened_inputs_3)
x2_3  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_3)
x3_3  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_3)
x4_3  = Lambda(lambda x : x[:, 105:361 ])(flattened_inputs_3)
x5_3  = Lambda(lambda x : x[:, 140:396 ])(flattened_inputs_3)
x6_3  = Lambda(lambda x : x[:, 175:431 ])(flattened_inputs_3)
x7_3  = Lambda(lambda x : x[:, 210:466 ])(flattened_inputs_3)
x8_3  = Lambda(lambda x : x[:, 245:501 ])(flattened_inputs_3)
x9_3  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_3)
x10_3  = Lambda(lambda x : x[:,315:571])(flattened_inputs_3)
x11_3  = Lambda(lambda x : x[:,350:606])(flattened_inputs_3)
x12_3  = Lambda(lambda x : x[:,385:641])(flattened_inputs_3)
x13_3  = Lambda(lambda x : x[:,420:676])(flattened_inputs_3)
x14_3  = Lambda(lambda x : x[:,455:711])(flattened_inputs_3)
x15_3  = Lambda(lambda x : x[:,490:746])(flattened_inputs_3)
x16_3  = Lambda(lambda x : x[:,525:781])(flattened_inputs_3)

x1_3_3  = Tea(64)(x1_3)
x2_3_3  = Tea(64)(x2_3)
x3_3_3  = Tea(64)(x3_3)
x4_3_3  = Tea(64)(x4_3)
x5_3_3  = Tea(64)(x5_3)
x6_3_3  = Tea(64)(x6_3)
x7_3_3  = Tea(64)(x7_3)
x8_3_3  = Tea(64)(x8_3)
x9_3_3  = Tea(64)(x9_3)
x10_3_3  = Tea(64)(x10_3)
x11_3_3  = Tea(64)(x11_3)
x12_3_3  = Tea(64)(x12_3)
x13_3_3  = Tea(64)(x13_3)
x14_3_3  = Tea(64)(x14_3)
x15_3_3  = Tea(64)(x15_3)
x16_3_3  = Tea(64)(x16_3)

x1_4  = Lambda(lambda x : x[:,   :256 ])(flattened_inputs_4)
x2_4  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_4)
x3_4  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_4)
x4_4  = Lambda(lambda x : x[:, 105:361 ])(flattened_inputs_4)
x5_4  = Lambda(lambda x : x[:, 140:396 ])(flattened_inputs_4)
x6_4  = Lambda(lambda x : x[:, 175:431 ])(flattened_inputs_4)
x7_4  = Lambda(lambda x : x[:, 210:466 ])(flattened_inputs_4)
x8_4  = Lambda(lambda x : x[:, 245:501 ])(flattened_inputs_4)
x9_4  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_4)
x10_4  = Lambda(lambda x : x[:,315:571])(flattened_inputs_4)
x11_4  = Lambda(lambda x : x[:,350:606])(flattened_inputs_4)
x12_4  = Lambda(lambda x : x[:,385:641])(flattened_inputs_4)
x13_4  = Lambda(lambda x : x[:,420:676])(flattened_inputs_4)
x14_4  = Lambda(lambda x : x[:,455:711])(flattened_inputs_4)
x15_4  = Lambda(lambda x : x[:,490:746])(flattened_inputs_4)
x16_4  = Lambda(lambda x : x[:,525:781])(flattened_inputs_4)

x1_4_4  = Tea(64)(x1_4)
x2_4_4  = Tea(64)(x2_4)
x3_4_4  = Tea(64)(x3_4)
x4_4_4  = Tea(64)(x4_4)
x5_4_4  = Tea(64)(x5_4)
x6_4_4  = Tea(64)(x6_4)
x7_4_4  = Tea(64)(x7_4)
x8_4_4  = Tea(64)(x8_4)
x9_4_4  = Tea(64)(x9_4)
x10_4_4  = Tea(64)(x10_4)
x11_4_4  = Tea(64)(x11_4)
x12_4_4  = Tea(64)(x12_4)
x13_4_4  = Tea(64)(x13_4)
x14_4_4  = Tea(64)(x14_4)
x15_4_4  = Tea(64)(x15_4)
x16_4_4  = Tea(64)(x16_4)

x1_5  = Lambda(lambda x : x[:,   :256 ])(flattened_inputs_5)
x2_5  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_5)
x3_5  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_5)
x4_5  = Lambda(lambda x : x[:, 105:361 ])(flattened_inputs_5)
x5_5  = Lambda(lambda x : x[:, 140:396 ])(flattened_inputs_5)
x6_5  = Lambda(lambda x : x[:, 175:431 ])(flattened_inputs_5)
x7_5  = Lambda(lambda x : x[:, 210:466 ])(flattened_inputs_5)
x8_5  = Lambda(lambda x : x[:, 245:501 ])(flattened_inputs_5)
x9_5  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_5)
x10_5  = Lambda(lambda x : x[:,315:571])(flattened_inputs_5)
x11_5  = Lambda(lambda x : x[:,350:606])(flattened_inputs_5)
x12_5  = Lambda(lambda x : x[:,385:641])(flattened_inputs_5)
x13_5  = Lambda(lambda x : x[:,420:676])(flattened_inputs_5)
x14_5  = Lambda(lambda x : x[:,455:711])(flattened_inputs_5)
x15_5  = Lambda(lambda x : x[:,490:746])(flattened_inputs_5)
x16_5  = Lambda(lambda x : x[:,525:781])(flattened_inputs_5)

x1_5_5  = Tea(64)(x1_5)
x2_5_5  = Tea(64)(x2_5)
x3_5_5  = Tea(64)(x3_5)
x4_5_5  = Tea(64)(x4_5)
x5_5_5  = Tea(64)(x5_5)
x6_5_5  = Tea(64)(x6_5)
x7_5_5  = Tea(64)(x7_5)
x8_5_5  = Tea(64)(x8_5)
x9_5_5  = Tea(64)(x9_5)
x10_5_5  = Tea(64)(x10_5)
x11_5_5  = Tea(64)(x11_5)
x12_5_5  = Tea(64)(x12_5)
x13_5_5  = Tea(64)(x13_5)
x14_5_5  = Tea(64)(x14_5)
x15_5_5  = Tea(64)(x15_5)
x16_5_5  = Tea(64)(x16_5)

x1_6  = Lambda(lambda x : x[:,   :256 ])(flattened_inputs_6)
x2_6  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_6)
x3_6  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_6)
x4_6  = Lambda(lambda x : x[:, 105:361 ])(flattened_inputs_6)
x5_6  = Lambda(lambda x : x[:, 140:396 ])(flattened_inputs_6)
x6_6  = Lambda(lambda x : x[:, 175:431 ])(flattened_inputs_6)
x7_6  = Lambda(lambda x : x[:, 210:466 ])(flattened_inputs_6)
x8_6  = Lambda(lambda x : x[:, 245:501 ])(flattened_inputs_6)
x9_6  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_6)
x10_6  = Lambda(lambda x : x[:,315:571])(flattened_inputs_6)
x11_6  = Lambda(lambda x : x[:,350:606])(flattened_inputs_6)
x12_6  = Lambda(lambda x : x[:,385:641])(flattened_inputs_6)
x13_6  = Lambda(lambda x : x[:,420:676])(flattened_inputs_6)
x14_6  = Lambda(lambda x : x[:,455:711])(flattened_inputs_6)
x15_6  = Lambda(lambda x : x[:,490:746])(flattened_inputs_6)
x16_6  = Lambda(lambda x : x[:,525:781])(flattened_inputs_6)

x1_6_6  = Tea(64)(x1_6)
x2_6_6  = Tea(64)(x2_6)
x3_6_6  = Tea(64)(x3_6)
x4_6_6  = Tea(64)(x4_6)
x5_6_6  = Tea(64)(x5_6)
x6_6_6  = Tea(64)(x6_6)
x7_6_6  = Tea(64)(x7_6)
x8_6_6  = Tea(64)(x8_6)
x9_6_6  = Tea(64)(x9_6)
x10_6_6  = Tea(64)(x10_6)
x11_6_6  = Tea(64)(x11_6)
x12_6_6  = Tea(64)(x12_6)
x13_6_6  = Tea(64)(x13_6)
x14_6_6  = Tea(64)(x14_6)
x15_6_6  = Tea(64)(x15_6)
x16_6_6  = Tea(64)(x16_6)

x1_7  = Lambda(lambda x : x[:,   :256 ])(flattened_inputs_7)
x2_7  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_7)
x3_7  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_7)
x4_7  = Lambda(lambda x : x[:, 105:361 ])(flattened_inputs_7)
x5_7  = Lambda(lambda x : x[:, 140:396 ])(flattened_inputs_7)
x6_7  = Lambda(lambda x : x[:, 175:431 ])(flattened_inputs_7)
x7_7  = Lambda(lambda x : x[:, 210:466 ])(flattened_inputs_7)
x8_7  = Lambda(lambda x : x[:, 245:501 ])(flattened_inputs_7)
x9_7  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_7)
x10_7  = Lambda(lambda x : x[:,315:571])(flattened_inputs_7)
x11_7  = Lambda(lambda x : x[:,350:606])(flattened_inputs_7)
x12_7  = Lambda(lambda x : x[:,385:641])(flattened_inputs_7)
x13_7  = Lambda(lambda x : x[:,420:676])(flattened_inputs_7)
x14_7  = Lambda(lambda x : x[:,455:711])(flattened_inputs_7)
x15_7  = Lambda(lambda x : x[:,490:746])(flattened_inputs_7)
x16_7  = Lambda(lambda x : x[:,525:781])(flattened_inputs_7)

x1_7_7  = Tea(64)(x1_7)
x2_7_7  = Tea(64)(x2_7)
x3_7_7  = Tea(64)(x3_7)
x4_7_7  = Tea(64)(x4_7)
x5_7_7  = Tea(64)(x5_7)
x6_7_7  = Tea(64)(x6_7)
x7_7_7  = Tea(64)(x7_7)
x8_7_7  = Tea(64)(x8_7)
x9_7_7  = Tea(64)(x9_7)
x10_7_7  = Tea(64)(x10_7)
x11_7_7  = Tea(64)(x11_7)
x12_7_7  = Tea(64)(x12_7)
x13_7_7  = Tea(64)(x13_7)
x14_7_7  = Tea(64)(x14_7)
x15_7_7  = Tea(64)(x15_7)
x16_7_7  = Tea(64)(x16_7)

x1_1_1 = Average()([x1_1_1,x1_2_2,x1_3_3,x1_4_4,x1_5_5,x1_6_6,x1_7_7])
x2_1_1 = Average()([x2_1_1,x2_2_2,x2_3_3,x2_4_4,x2_5_5,x2_6_6,x2_7_7])
x3_1_1 = Average()([x3_1_1,x3_2_2,x3_3_3,x3_4_4,x3_5_5,x3_6_6,x3_7_7])
x4_1_1 = Average()([x4_1_1,x4_2_2,x4_3_3,x4_4_4,x4_5_5,x4_6_6,x4_7_7])
x5_1_1 = Average()([x5_1_1,x5_2_2,x5_3_3,x5_4_4,x5_5_5,x5_6_6,x5_7_7])
x6_1_1 = Average()([x6_1_1,x6_2_2,x6_3_3,x6_4_4,x6_5_5,x6_6_6,x6_7_7])
x7_1_1 = Average()([x7_1_1,x7_2_2,x7_3_3,x7_4_4,x7_5_5,x7_6_6,x7_7_7])
x8_1_1 = Average()([x8_1_1,x8_2_2,x8_3_3,x8_4_4,x8_5_5,x8_6_6,x8_7_7])
x9_1_1 = Average()([x9_1_1,x9_2_2,x9_3_3,x9_4_4,x9_5_5,x9_6_6,x9_7_7])
x10_1_1 = Average()([x10_1_1,x10_2_2,x10_3_3,x10_4_4,x10_5_5,x10_6_6,x10_7_7])
x11_1_1 = Average()([x11_1_1,x11_2_2,x11_3_3,x11_4_4,x11_5_5,x11_6_6,x11_7_7])
x12_1_1 = Average()([x12_1_1,x12_2_2,x12_3_3,x12_4_4,x12_5_5,x12_6_6,x12_7_7])
x13_1_1 = Average()([x13_1_1,x13_2_2,x13_3_3,x13_4_4,x13_5_5,x13_6_6,x13_7_7])
x14_1_1 = Average()([x14_1_1,x14_2_2,x14_3_3,x14_4_4,x14_5_5,x14_6_6,x14_7_7])
x15_1_1 = Average()([x15_1_1,x15_2_2,x15_3_3,x15_4_4,x15_5_5,x15_6_6,x15_7_7])
x16_1_1 = Average()([x16_1_1,x16_2_2,x16_3_3,x16_4_4,x16_5_5,x16_6_6,x16_7_7])

x1_1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
x2_1 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
x3_1 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
x4_1 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

x1_1 = Tea(128)(x1_1)
x2_1 = Tea(128)(x2_1)
x3_1 = Tea(128)(x3_1)
x4_1 = Tea(128)(x4_1)

x_out_1 = Concatenate(axis=1)([x1_1,x2_1])
x_out_2 = Concatenate(axis=1)([x3_1,x4_1])

x_out_1 = Tea(250)(x_out_1)
x_out_2 = Tea(250)(x_out_2)
x_out = Concatenate(axis=1)([x_out_1,x_out_2])

x_out = AdditivePooling(10)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.002),
              metrics=['accuracy'])

model.fit(x_tr, y_train,batch_size=128,epochs=10,verbose=1,validation_split=0.2)
# model.save("mnist_12_3_1.h5")
score = model.evaluate(x_ts, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



