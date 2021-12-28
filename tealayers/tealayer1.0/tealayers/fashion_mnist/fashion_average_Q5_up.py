from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math

from numpy.core.fromnumeric import mean

import tensorflow as tf
# from tensorflow import squeeze
import numpy as np
# from keras import backend as K
from keras import Model
from keras.layers import Flatten, Activation, Input, Lambda,Average, Add,BatchNormalization ,Concatenate
from keras.datasets import fashion_mnist,mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append("../../../../rancutils/rancutils")
sys.path.append("../")

from tea import Tea
from additivepooling import AdditivePooling
import random
from sklearn.utils import shuffle

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet
from fashion import Fashion
from emulation import write_cores
import cv2

# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train_copy = np.empty_like(x_train)
x_train_copy[:,:,:]=x_train

x_test_copy = np.empty_like(x_test)
x_test_copy[:,:,:]=x_test

# y_train_copy = np.empty_like(y_train)
y_train_copy=y_train

# y_test_copy = np.empty_like(y_test)
y_test_copy=y_test

print(y_test.shape)

del x_train
del x_test
# del y_test
# del y_train

train_data = x_train_copy
test_data = x_test_copy
train_label = y_train_copy
test_label = y_test_copy

y_train_new = []
for i in range(len(y_train)):
    if train_label[i] == 0 :
        y_train_new.append(0)
    elif train_label[i] == 2:
        y_train_new.append(1)
    elif train_label[i] == 4 :
        y_train_new.append(2)
    elif train_label[i] == 6:
        y_train_new.append(3)

y_test_new = []

for i in range(len(y_test)):
    if test_label[i] == 0 :
        y_test_new.append(0)
    elif test_label[i] == 2:
        y_test_new.append(1)
    elif test_label[i] == 4:
        y_test_new.append(2)
    elif test_label[i] == 6:
        y_test_new.append(3)
# print(train_data[0].shape)

x_train = []
for i in range(len(train_data)):
    # edge =cv2.Canny(train_data[i],100,200)
    if train_label[i] == 0 or train_label[i] == 2 or train_label[i] == 4 or train_label[i] == 6:
        mask = np.ones_like(train_data[i])
        # train_data[i] = fft_trans(train_data[i])
        e_1 = np.array(train_data[i]>=mask*25).astype(float)
        # e_2 = np.array(train_data[i]>=mask*50).astype(float)
        e_2 = np.array(train_data[i]>=mask*75).astype(float)
        # e_4 = np.array(train_data[i]>=mask*100).astype(float)
        e_3 = np.array(train_data[i]>=mask*125).astype(float)
        # e_6 = np.array(train_data[i]>=mask*150).astype(float)
        e_4 = np.array(train_data[i]>=mask*175).astype(float)
        # e_8 = np.array(train_data[i]>=mask*200).astype(float)
        e_5 = np.array(train_data[i]>=mask*225).astype(float)

        x_train.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                        e_4[:,:,np.newaxis],e_5[:,:,np.newaxis]),axis=2))


x_test = []
for i in range(len(test_data)):
    # edge =cv2.Canny(test_data[i],100,200)
    if test_label[i] == 0 or test_label[i] == 2 or test_label[i] == 4 or test_label[i] == 6:
        mask = np.ones_like(test_data[i])
        # test_data[i]=fft_trans(test_data[i])

        e_1 = np.array(test_data[i]>=mask*25).astype(float)
        # e_2 = np.array(test_data[i]>=mask*50).astype(float))
        e_2 = np.array(test_data[i]>=mask*75).astype(float)
        # e_4 = np.array(test_data[i]>=mask*100).astype(float)
        e_3 = np.array(test_data[i]>=mask*125).astype(float)
        # e_6 = np.array(test_data[i]>=mask*150).astype(float)
        e_4 = np.array(test_data[i]>=mask*175).astype(float)
        # e_8 = np.array(test_data[i]>=mask*200).astype(float)
        e_5 = np.array(test_data[i]>=mask*225).astype(float)

        # cv2.imwrite("/home/hoangphuong/Documents/image_fashion/{}_edge.jpg".format(i),edge)
        x_test.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                        e_4[:,:,np.newaxis],e_5[:,:,np.newaxis],),axis=2))
x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = to_categorical(y_train_new, 4)
y_test = to_categorical(y_test_new, 4)

random.seed(1)
(x_train,y_train) = shuffle(x_train,y_train)
random.seed(1)
(x_test,y_test) = shuffle(x_test,y_test)


inputs = Input(shape=(28, 28, 5,))
flattened_inputs = Flatten()(inputs)

flattened_inputs_1 = Lambda(lambda x : x[:,     :1*784])(flattened_inputs)
flattened_inputs_2 = Lambda(lambda x : x[:,1*784:2*784])(flattened_inputs)
flattened_inputs_3 = Lambda(lambda x : x[:,2*784:3*784])(flattened_inputs)
flattened_inputs_4 = Lambda(lambda x : x[:,3*784:4*784])(flattened_inputs)
flattened_inputs_5 = Lambda(lambda x : x[:,4*784:5*784])(flattened_inputs)
# flattened_inputs_6 = Lambda(lambda x : x[:,5*784:6*784])(flattened_inputs)
# flattened_inputs_7 = Lambda(lambda x : x[:,6*784:7*784])(flattened_inputs)
# flattened_inputs_8 = Lambda(lambda x : x[:,7*784:8*784])(flattened_inputs)
# flattened_inputs_9 = Lambda(lambda x : x[:,8*784:      ])(flattened_inputs)

x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_1)
x2_1  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_1)
x3_1  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_1)
x4_1  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_1)
x5_1  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_1)
x6_1  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_1)
x7_1  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_1)
x8_1  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_1)
x9_1  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_1)
x10_1  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_1)
x11_1  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_1)
x12_1  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_1)
x13_1  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_1)
x14_1  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_1)
x15_1  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_1)
x16_1  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_1)

x1_1  = Tea(64)(x1_1)
x2_1  = Tea(64)(x2_1)
x3_1  = Tea(64)(x3_1)
x4_1  = Tea(64)(x4_1)
x5_1  = Tea(64)(x5_1)
x6_1  = Tea(64)(x6_1)
x7_1  = Tea(64)(x7_1)
x8_1  = Tea(64)(x8_1)
x9_1  = Tea(64)(x9_1)
x10_1  = Tea(64)(x10_1)
x11_1  = Tea(64)(x11_1)
x12_1  = Tea(64)(x12_1)
x13_1  = Tea(64)(x13_1)
x14_1  = Tea(64)(x14_1)
x15_1  = Tea(64)(x15_1)
x16_1  = Tea(64)(x16_1)

x1_2  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_2)
x2_2  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_2)
x3_2  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_2)
x4_2  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_2)
x5_2  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_2)
x6_2  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_2)
x7_2  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_2)
x8_2  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_2)
x9_2  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_2)
x10_2  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_2)
x11_2  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_2)
x12_2  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_2)
x13_2  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_2)
x14_2  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_2)
x15_2  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_2)
x16_2  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_2)

x1_2  = Tea(64)(x1_2)
x2_2  = Tea(64)(x2_2)
x3_2  = Tea(64)(x3_2)
x4_2  = Tea(64)(x4_2)
x5_2  = Tea(64)(x5_2)
x6_2  = Tea(64)(x6_2)
x7_2  = Tea(64)(x7_2)
x8_2  = Tea(64)(x8_2)
x9_2  = Tea(64)(x9_2)
x10_2  = Tea(64)(x10_2)
x11_2  = Tea(64)(x11_2)
x12_2  = Tea(64)(x12_2)
x13_2  = Tea(64)(x13_2)
x14_2  = Tea(64)(x14_2)
x15_2  = Tea(64)(x15_2)
x16_2  = Tea(64)(x16_2)

x1_3  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_3)
x2_3  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_3)
x3_3  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_3)
x4_3  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_3)
x5_3  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_3)
x6_3  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_3)
x7_3  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_3)
x8_3  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_3)
x9_3  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_3)
x10_3  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_3)
x11_3  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_3)
x12_3  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_3)
x13_3  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_3)
x14_3  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_3)
x15_3  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_3)
x16_3  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_3)

x1_3  = Tea(64)(x1_3)
x2_3  = Tea(64)(x2_3)
x3_3  = Tea(64)(x3_3)
x4_3  = Tea(64)(x4_3)
x5_3  = Tea(64)(x5_3)
x6_3  = Tea(64)(x6_3)
x7_3  = Tea(64)(x7_3)
x8_3  = Tea(64)(x8_3)
x9_3  = Tea(64)(x9_3)
x10_3  = Tea(64)(x10_3)
x11_3  = Tea(64)(x11_3)
x12_3  = Tea(64)(x12_3)
x13_3  = Tea(64)(x13_3)
x14_3  = Tea(64)(x14_3)
x15_3  = Tea(64)(x15_3)
x16_3  = Tea(64)(x16_3)

x1_4  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_4)
x2_4  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_4)
x3_4  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_4)
x4_4  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_4)
x5_4  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_4)
x6_4  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_4)
x7_4  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_4)
x8_4  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_4)
x9_4  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_4)
x10_4  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_4)
x11_4  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_4)
x12_4  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_4)
x13_4  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_4)
x14_4  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_4)
x15_4  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_4)
x16_4  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_4)

x1_4  = Tea(64)(x1_4)
x2_4  = Tea(64)(x2_4)
x3_4  = Tea(64)(x3_4)
x4_4  = Tea(64)(x4_4)
x5_4  = Tea(64)(x5_4)
x6_4  = Tea(64)(x6_4)
x7_4  = Tea(64)(x7_4)
x8_4  = Tea(64)(x8_4)
x9_4  = Tea(64)(x9_4)
x10_4  = Tea(64)(x10_4)
x11_4  = Tea(64)(x11_4)
x12_4  = Tea(64)(x12_4)
x13_4  = Tea(64)(x13_4)
x14_4  = Tea(64)(x14_4)
x15_4  = Tea(64)(x15_4)
x16_4  = Tea(64)(x16_4)

x1_5  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_5)
x2_5  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_5)
x3_5  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_5)
x4_5  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_5)
x5_5  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_5)
x6_5  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_5)
x7_5  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_5)
x8_5  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_5)
x9_5  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_5)
x10_5  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_5)
x11_5  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_5)
x12_5  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_5)
x13_5  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_5)
x14_5  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_5)
x15_5  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_5)
x16_5  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_5)

x1_5  = Tea(64)(x1_5)
x2_5  = Tea(64)(x2_5)
x3_5  = Tea(64)(x3_5)
x4_5  = Tea(64)(x4_5)
x5_5  = Tea(64)(x5_5)
x6_5  = Tea(64)(x6_5)
x7_5  = Tea(64)(x7_5)
x8_5  = Tea(64)(x8_5)
x9_5  = Tea(64)(x9_5)
x10_5  = Tea(64)(x10_5)
x11_5  = Tea(64)(x11_5)
x12_5  = Tea(64)(x12_5)
x13_5  = Tea(64)(x13_5)
x14_5  = Tea(64)(x14_5)
x15_5  = Tea(64)(x15_5)
x16_5  = Tea(64)(x16_5)

# x1_6  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_6)
# x2_6  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_6)
# x3_6  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_6)
# x4_6  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_6)
# x5_6  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_6)
# x6_6  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_6)
# x7_6  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_6)
# x8_6  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_6)
# x9_6  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_6)
# x10_6  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_6)
# x11_6  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_6)
# x12_6  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_6)
# x13_6  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_6)
# x14_6  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_6)
# x15_6  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_6)
# x16_6  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_6)

# x1_6  = Tea(64)(x1_6)
# x2_6  = Tea(64)(x2_6)
# x3_6  = Tea(64)(x3_6)
# x4_6  = Tea(64)(x4_6)
# x5_6  = Tea(64)(x5_6)
# x6_6  = Tea(64)(x6_6)
# x7_6  = Tea(64)(x7_6)
# x8_6  = Tea(64)(x8_6)
# x9_6  = Tea(64)(x9_6)
# x10_6  = Tea(64)(x10_6)
# x11_6  = Tea(64)(x11_6)
# x12_6  = Tea(64)(x12_6)
# x13_6  = Tea(64)(x13_6)
# x14_6  = Tea(64)(x14_6)
# x15_6  = Tea(64)(x15_6)
# x16_6  = Tea(64)(x16_6)

# ### 3 ###

# x1_7  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_7)
# x2_7  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_7)
# x3_7  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_7)
# x4_7  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_7)
# x5_7  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_7)
# x6_7  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_7)
# x7_7  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_7)
# x8_7  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_7)
# x9_7  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_7)
# x10_7  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_7)
# x11_7  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_7)
# x12_7  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_7)
# x13_7  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_7)
# x14_7  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_7)
# x15_7  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_7)
# x16_7  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_7)

# x1_7  = Tea(64)(x1_7)
# x2_7  = Tea(64)(x2_7)
# x3_7  = Tea(64)(x3_7)
# x4_7  = Tea(64)(x4_7)
# x5_7  = Tea(64)(x5_7)
# x6_7  = Tea(64)(x6_7)
# x7_7  = Tea(64)(x7_7)
# x8_7  = Tea(64)(x8_7)
# x9_7  = Tea(64)(x9_7)
# x10_7  = Tea(64)(x10_7)
# x11_7  = Tea(64)(x11_7)
# x12_7  = Tea(64)(x12_7)
# x13_7  = Tea(64)(x13_7)
# x14_7  = Tea(64)(x14_7)
# x15_7  = Tea(64)(x15_7)
# x16_7  = Tea(64)(x16_7)

# x1_8  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_8)
# x2_8  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_8)
# x3_8  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_8)
# x4_8  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_8)
# x5_8  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_8)
# x6_8  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_8)
# x7_8  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_8)
# x8_8  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_8)
# x9_8  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_8)
# x10_8  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_8)
# x11_8  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_8)
# x12_8  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_8)
# x13_8  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_8)
# x14_8  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_8)
# x15_8  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_8)
# x16_8  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_8)

# x1_8  = Tea(64)(x1_8)
# x2_8  = Tea(64)(x2_8)
# x3_8  = Tea(64)(x3_8)
# x4_8  = Tea(64)(x4_8)
# x5_8  = Tea(64)(x5_8)
# x6_8  = Tea(64)(x6_8)
# x7_8  = Tea(64)(x7_8)
# x8_8  = Tea(64)(x8_8)
# x9_8  = Tea(64)(x9_8)
# x10_8  = Tea(64)(x10_8)
# x11_8  = Tea(64)(x11_8)
# x12_8  = Tea(64)(x12_8)
# x13_8  = Tea(64)(x13_8)
# x14_8  = Tea(64)(x14_8)
# x15_8  = Tea(64)(x15_8)
# x16_8  = Tea(64)(x16_8)

# x1_9  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_9)
# x2_9  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_9)
# x3_9  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_9)
# x4_9  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_9)
# x5_9  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_9)
# x6_9  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_9)
# x7_9  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_9)
# x8_9  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_9)
# x9_9  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_9)
# x10_9  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_9)
# x11_9  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_9)
# x12_9  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_9)
# x13_9  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_9)
# x14_9  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_9)
# x15_9  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_9)
# x16_9  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_9)

# x1_9  = Tea(64)(x1_9)
# x2_9  = Tea(64)(x2_9)
# x3_9  = Tea(64)(x3_9)
# x4_9  = Tea(64)(x4_9)
# x5_9  = Tea(64)(x5_9)
# x6_9  = Tea(64)(x6_9)
# x7_9  = Tea(64)(x7_9)
# x8_9  = Tea(64)(x8_9)
# x9_9  = Tea(64)(x9_9)
# x10_9  = Tea(64)(x10_9)
# x11_9  = Tea(64)(x11_9)
# x12_9  = Tea(64)(x12_9)
# x13_9  = Tea(64)(x13_9)
# x14_9  = Tea(64)(x14_9)
# x15_9  = Tea(64)(x15_9)
# x16_9  = Tea(64)(x16_9)

x1_1_1 = Average()([x1_1,x1_2,x1_3,x1_4,x1_5])
x2_1_1 = Average()([x2_1,x2_2,x2_3,x2_4,x2_5])
x3_1_1 = Average()([x3_1,x3_2,x3_3,x3_4,x3_5])
x4_1_1 = Average()([x4_1,x4_2,x4_3,x4_4,x4_5])
x5_1_1 = Average()([x5_1,x5_2,x5_3,x5_4,x5_5])
x6_1_1 = Average()([x6_1,x6_2,x6_3,x6_4,x6_5])
x7_1_1 = Average()([x7_1,x7_2,x7_3,x7_4,x7_5])
x8_1_1 = Average()([x8_1,x8_2,x8_3,x8_4,x8_5])
x9_1_1 = Average()([x9_1,x9_2,x9_3,x9_4,x9_5])
x10_1_1 = Average()([x10_1,x10_2,x10_3,x10_4,x10_5])
x11_1_1 = Average()([x11_1,x11_2,x11_3,x11_4,x11_5])
x12_1_1 = Average()([x12_1,x12_2,x12_3,x12_4,x12_5])
x13_1_1 = Average()([x13_1,x13_2,x13_3,x13_4,x13_5])
x14_1_1 = Average()([x14_1,x14_2,x14_3,x14_4,x14_5])
x15_1_1 = Average()([x15_1,x15_2,x15_3,x15_4,x15_5])
x16_1_1 = Average()([x16_1,x16_2,x16_3,x16_4,x16_5])

x1_1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
x2_1 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
x3_1 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
x4_1 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

x1_1 = Tea(64)(x1_1)
x2_1 = Tea(64)(x2_1)
x3_1 = Tea(64)(x3_1)
x4_1 = Tea(64)(x4_1)

x_out_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])

x_out_1 = Tea(256)(x_out_1)

x_out = AdditivePooling(4)(x_out_1)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

checkpoint_filepath = './ckpt_fashion/4_class_fashion'

import keras 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath + '-epoch-{epoch}',
    save_weights_only=True,)
# model.load_weights('./ckpt_fashion/10_class_fashion-epoch-7')
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          callbacks=[model_checkpoint_callback],
          validation_split=0.2)

import os
scores = []
soure = "./ckpt_fashion"
ckpts = [os.path.join(soure,e) for e in os.listdir(soure)]
for ckpt in ckpts:
    print("======================================")
    print(ckpt)
    print("======================================")
    model.load_weights(ckpt)      
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    scores.append(score[1])

print("Max accuracy:",max(scores))
print("Best epoch:",ckpts[scores.index(max(scores))])

# score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss new:', score[0])
# print('Test accuracy new:', score[1])

# cores_sim = create_cores(model, 20,neuron_reset_type=0 ) 
# write_cores(cores_sim,output_path="/home/phuongdh/Documents/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/out_new")