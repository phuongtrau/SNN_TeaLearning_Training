from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras import Model
from keras.engine.topology import Layer
from keras import initializers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, Concatenate, Average, Permute, Multiply
from keras.datasets import mnist,fashion_mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
import sys
sys.path.append("../../../rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet
# sys.path.append("../")
# from tea import Tea
from additivepooling import AdditivePooling
import helper
from tea import Tea
import random
from sklearn.utils import shuffle
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
import cv2

exp_i_data = helper.load_exp_i_new("../dataset/experiment-i")
exp_i_data_3_class = helper.load_exp_i_new("../dataset/experiment-i",preprocess=True)

exp_i_data_supine = helper.load_exp_i_supine("../dataset/experiment-i")

datasets_supine = {"Base":exp_i_data_supine}
datasets = {"Base":exp_i_data}
datasets_3_class = {"Base":exp_i_data_3_class}

subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]
sub="S6"

subjects.remove(sub)
random.seed(1)
random.shuffle(subjects)

train_data         = helper.Mat_Dataset(datasets,["Base"],subjects)
train_data_3_class = helper.Mat_Dataset(datasets_3_class,["Base"],subjects)

test_data           = helper.Mat_Dataset(datasets,["Base"],[sub])
test_data_3_class   = helper.Mat_Dataset(datasets_3_class,["Base"],[sub])
test_data_supine    = helper.Mat_Dataset(datasets_supine,["Base"],[sub])

x_test_only_supine = []

for i in range(len(test_data_supine.samples)):

    test_data_supine.samples[i] = cv2.equalizeHist(test_data_supine.samples[i])
    
    heat = cv2.applyColorMap(test_data_supine.samples[i], cv2.COLORMAP_JET)
    mask = np.ones_like(heat)
    bin1 = np.array(heat>=mask*63).astype(np.uint8)
    bin2 = np.array(heat>=mask*127).astype(np.uint8)
    bin3 = np.array(heat>=mask*190).astype(np.uint8)
    bin_out = np.concatenate((bin1,bin2,bin3),axis=2)

    x_test_only_supine.append(bin_out)

x_test_only_supine = np.array(x_test_only_supine).astype(np.uint8)

y_test_only_supine = to_categorical(test_data_supine.labels, 9)

#### Supine ####

x_test = []

for i in range(len(test_data.samples)):

    temp_i = cv2.equalizeHist(test_data.samples[i])
    
    heat = cv2.applyColorMap(temp_i, cv2.COLORMAP_JET)
    mask = np.ones_like(heat)
    bin1 = np.array(heat>=mask*63).astype(np.uint8)
    bin2 = np.array(heat>=mask*127).astype(np.uint8)
    bin3 = np.array(heat>=mask*190).astype(np.uint8)
    bin_out = np.concatenate((bin1,bin2,bin3),axis=2)

    x_test.append(bin_out)

x_test_supine = np.array(x_test).astype(np.uint8)

x_train = []

for i in range(len(train_data.samples)):

    temp_i = cv2.equalizeHist(train_data.samples[i])
    
    heat = cv2.applyColorMap(temp_i, cv2.COLORMAP_JET)
    mask = np.ones_like(heat)
    bin1 = np.array(heat>=mask*63).astype(np.uint8)
    bin2 = np.array(heat>=mask*127).astype(np.uint8)
    bin3 = np.array(heat>=mask*190).astype(np.uint8)
    bin_out = np.concatenate((bin1,bin2,bin3),axis=2)

    x_train.append(bin_out)

x_train_supine = np.array(x_train).astype(np.uint8)

#### Left ####

x_test = []

for i in range(len(test_data.samples)):
    mask = np.ones_like(test_data.samples[i])
    
    test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])

    e_1 = np.array(test_data.samples[i]>=mask*25).astype(float)
    e_2 = np.array(test_data.samples[i]>=mask*50).astype(float)
    e_3 = np.array(test_data.samples[i]>=mask*75).astype(float)
    e_4 = np.array(test_data.samples[i]>=mask*100).astype(float)
    e_5 = np.array(test_data.samples[i]>=mask*125).astype(float)
    e_6 = np.array(test_data.samples[i]>=mask*150).astype(float)
    e_7 = np.array(test_data.samples[i]>=mask*175).astype(float)
    e_8 = np.array(test_data.samples[i]>=mask*200).astype(float)
    e_9 = np.array(test_data.samples[i]>=mask*225).astype(float)


    # print(e_1[:,:,np.newaxis].shape)
    x_test.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                   e_4[:,:,np.newaxis],e_5[:,:,np.newaxis],e_6[:,:,np.newaxis],\
                                   e_7[:,:,np.newaxis],e_8[:,:,np.newaxis],e_9[:,:,np.newaxis]),axis=2))
x_test_left = np.array(x_test)

x_train = []

for i in range(len(train_data.samples)):
    mask = np.ones_like(train_data.samples[i])
    
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])

    e_1 = np.array(train_data.samples[i]>=mask*25).astype(float)
    e_2 = np.array(train_data.samples[i]>=mask*50).astype(float)
    e_3 = np.array(train_data.samples[i]>=mask*75).astype(float)
    e_4 = np.array(train_data.samples[i]>=mask*100).astype(float)
    e_5 = np.array(train_data.samples[i]>=mask*125).astype(float)
    e_6 = np.array(train_data.samples[i]>=mask*150).astype(float)
    e_7 = np.array(train_data.samples[i]>=mask*175).astype(float)
    e_8 = np.array(train_data.samples[i]>=mask*200).astype(float)
    e_9 = np.array(train_data.samples[i]>=mask*225).astype(float)
    
    # print(e_1[:,:,np.newaxis].shape)
    x_train.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                   e_4[:,:,np.newaxis],e_5[:,:,np.newaxis],e_6[:,:,np.newaxis],\
                                   e_7[:,:,np.newaxis],e_8[:,:,np.newaxis],e_9[:,:,np.newaxis]),axis=2))
x_train_left = np.array(x_train)


#### Right ####
x_test = []

for i in range(len(test_data.samples)):

    mask = np.ones_like(test_data.samples[i])

    e_1 = np.array(test_data.samples[i]>=mask*25).astype(float)
    e_2 = np.array(test_data.samples[i]>=mask*50).astype(float)
    e_3 = np.array(test_data.samples[i]>=mask*75).astype(float)
    e_4 = np.array(test_data.samples[i]>=mask*100).astype(float)
    e_5 = np.array(test_data.samples[i]>=mask*125).astype(float)
    e_6 = np.array(test_data.samples[i]>=mask*150).astype(float)
    e_7 = np.array(test_data.samples[i]>=mask*175).astype(float)
    e_8 = np.array(test_data.samples[i]>=mask*200).astype(float)
    e_9 = np.array(test_data.samples[i]>=mask*225).astype(float)

    x_test.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                   e_4[:,:,np.newaxis],e_5[:,:,np.newaxis],e_6[:,:,np.newaxis],\
                                   e_7[:,:,np.newaxis],e_8[:,:,np.newaxis],e_9[:,:,np.newaxis]),axis=2))
x_test_right = np.array(x_test)

x_train = []

for i in range(len(train_data.samples)):

    mask = np.ones_like(train_data.samples[i])
    
    e_1 = np.array(train_data.samples[i]>=mask*25).astype(float)
    e_2 = np.array(train_data.samples[i]>=mask*50).astype(float)
    e_3 = np.array(train_data.samples[i]>=mask*75).astype(float)
    e_4 = np.array(train_data.samples[i]>=mask*100).astype(float)
    e_5 = np.array(train_data.samples[i]>=mask*125).astype(float)
    e_6 = np.array(train_data.samples[i]>=mask*150).astype(float)
    e_7 = np.array(train_data.samples[i]>=mask*175).astype(float)
    e_8 = np.array(train_data.samples[i]>=mask*200).astype(float)
    e_9 = np.array(train_data.samples[i]>=mask*225).astype(float)
    
    # print(e_1[:,:,np.newaxis].shape)
    x_train.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                   e_4[:,:,np.newaxis],e_5[:,:,np.newaxis],e_6[:,:,np.newaxis],\
                                   e_7[:,:,np.newaxis],e_8[:,:,np.newaxis],e_9[:,:,np.newaxis]),axis=2))

x_train_right = np.array(x_train)

#### 3_Classes ####

x_test_3_class=[]
for i in range(len(test_data_3_class.samples)):
    # test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
    x_test_3_class.append(cv2.equalizeHist(test_data_3_class.samples[i]))

x_test_3_class = np.array(x_test_3_class).astype('float32')
# x_test_3_class = test_data.samples.astype('float32')

x_test_3_class /= 255

x_train_3_class=[]
for i in range(len(train_data_3_class.samples)):
    x_train_3_class.append(cv2.equalizeHist(train_data_3_class.samples[i]))

x_train_3_class = np.array(x_train_3_class).astype('float32')

x_train_3_class /= 255

#### Label ####

y_test = to_categorical(test_data.labels, 17)
y_train = to_categorical(train_data.labels, 17)

#### MODEL ####
#### Model 3 Classes #### 

inputs_3_classes = Input(shape=(64, 32,))

flattened_inputs = Flatten()(inputs_3_classes)

x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs)
x2_1  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs)
x3_1  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs)
x4_1  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs)
x5_1  = Lambda(lambda x : x[:, 476:732])(flattened_inputs)
x6_1  = Lambda(lambda x : x[:, 595:851])(flattened_inputs)
x7_1  = Lambda(lambda x : x[:, 714:970])(flattened_inputs)
x8_1  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs)
x9_1  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs)
x10_1  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs)
x11_1  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs)
x12_1  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs)
x13_1  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs)
x14_1  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs)
x15_1  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs)
x16_1  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs)

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

x1_1_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
x2_1_1 = Concatenate(axis=1)([x5_1,x6_1,x7_1,x8_1])
x3_1_1 = Concatenate(axis=1)([x9_1,x10_1,x11_1,x12_1])
x4_1_1 = Concatenate(axis=1)([x13_1,x14_1,x15_1,x16_1])

x1_1 = Tea(64)(x1_1_1)
x2_1 = Tea(64)(x2_1_1)
x3_1 = Tea(64)(x3_1_1)
x4_1 = Tea(64)(x4_1_1)

x_out = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
x_out = Tea(255)(x_out)
x_out = AdditivePooling(3)(x_out)

predictions_3_classes = Activation('softmax')(x_out)
model_3_classes = Model(inputs=inputs_3_classes, outputs=predictions_3_classes)

#### Model Supine ####

inputs_supine = Input(shape=(64, 32,9))

flattened = Flatten()(inputs_supine)

flattened_inputs_1 = Lambda(lambda x : x[:,      :2048*3 ])(flattened)
flattened_inputs_2 = Lambda(lambda x : x[:,2048*3:2048*6])(flattened)
flattened_inputs_3 = Lambda(lambda x : x[:,2048*6: ])(flattened)

R_1 = Lambda(lambda x : x[:,     :2048 ])(flattened_inputs_1)
G_1 = Lambda(lambda x : x[:, 2048:4096 ])(flattened_inputs_1)
B_1 = Lambda(lambda x : x[:, 4096:     ])(flattened_inputs_1)

R_2 = Lambda(lambda x : x[:,     :2048 ])(flattened_inputs_2)
G_2 = Lambda(lambda x : x[:, 2048:4096 ])(flattened_inputs_2)
B_2 = Lambda(lambda x : x[:, 4096:     ])(flattened_inputs_2)

R_3 = Lambda(lambda x : x[:,     :2048 ])(flattened_inputs_3)
G_3 = Lambda(lambda x : x[:, 2048:4096 ])(flattened_inputs_3)
B_3 = Lambda(lambda x : x[:, 4096:     ])(flattened_inputs_3)

x1_1  = Lambda(lambda x : x[:,     :256 ])(R_1)
x2_1  = Lambda(lambda x : x[:, 119 : 375 ])(R_1)
x3_1  = Lambda(lambda x : x[:, 238 :494 ])(R_1)
x4_1  = Lambda(lambda x : x[:, 357 : 613])(R_1)
x5_1  = Lambda(lambda x : x[:, 476:732])(R_1)
x6_1  = Lambda(lambda x : x[:, 595:851])(R_1)
x7_1  = Lambda(lambda x : x[:, 714:970])(R_1)
x8_1  = Lambda(lambda x : x[:, 833:1089])(R_1)
x9_1  = Lambda(lambda x : x[:, 952:1208])(R_1)
x10_1  = Lambda(lambda x : x[:, 1071:1327])(R_1)
x11_1  = Lambda(lambda x : x[:, 1190:1446])(R_1)
x12_1  = Lambda(lambda x : x[:, 1309:1565])(R_1)
x13_1  = Lambda(lambda x : x[:, 1428:1684])(R_1)
x14_1  = Lambda(lambda x : x[:, 1547:1803])(R_1)
x15_1  = Lambda(lambda x : x[:, 1666:1922])(R_1)
x16_1  = Lambda(lambda x : x[:, 1785:2041])(R_1)

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

x1_2  = Lambda(lambda x : x[:,     :256 ])(G_1)
x2_2  = Lambda(lambda x : x[:, 119 : 375 ])(G_1)
x3_2  = Lambda(lambda x : x[:, 238 :494 ])(G_1)
x4_2  = Lambda(lambda x : x[:, 357 : 613])(G_1)
x5_2  = Lambda(lambda x : x[:, 476:732])(G_1)
x6_2  = Lambda(lambda x : x[:, 595:851])(G_1)
x7_2  = Lambda(lambda x : x[:, 714:970])(G_1)
x8_2  = Lambda(lambda x : x[:, 833:1089])(G_1)
x9_2  = Lambda(lambda x : x[:, 952:1208])(G_1)
x10_2  = Lambda(lambda x : x[:, 1071:1327])(G_1)
x11_2  = Lambda(lambda x : x[:, 1190:1446])(G_1)
x12_2  = Lambda(lambda x : x[:, 1309:1565])(G_1)
x13_2  = Lambda(lambda x : x[:, 1428:1684])(G_1)
x14_2  = Lambda(lambda x : x[:, 1547:1803])(G_1)
x15_2  = Lambda(lambda x : x[:, 1666:1922])(G_1)
x16_2  = Lambda(lambda x : x[:, 1785:2041])(G_1)

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

x1_3  = Lambda(lambda x : x[:,     :256 ])(B_1)
x2_3  = Lambda(lambda x : x[:, 119 : 375 ])(B_1)
x3_3  = Lambda(lambda x : x[:, 238 :494 ])(B_1)
x4_3  = Lambda(lambda x : x[:, 357 : 613])(B_1)
x5_3  = Lambda(lambda x : x[:, 476:732])(B_1)
x6_3  = Lambda(lambda x : x[:, 595:851])(B_1)
x7_3  = Lambda(lambda x : x[:, 714:970])(B_1)
x8_3  = Lambda(lambda x : x[:, 833:1089])(B_1)
x9_3  = Lambda(lambda x : x[:, 952:1208])(B_1)
x10_3  = Lambda(lambda x : x[:, 1071:1327])(B_1)
x11_3  = Lambda(lambda x : x[:, 1190:1446])(B_1)
x12_3  = Lambda(lambda x : x[:, 1309:1565])(B_1)
x13_3  = Lambda(lambda x : x[:, 1428:1684])(B_1)
x14_3  = Lambda(lambda x : x[:, 1547:1803])(B_1)
x15_3  = Lambda(lambda x : x[:, 1666:1922])(B_1)
x16_3  = Lambda(lambda x : x[:, 1785:2041])(B_1)

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

### 2 ###

x1_4  = Lambda(lambda x : x[:,     :256 ])(R_2)
x2_4  = Lambda(lambda x : x[:, 119 : 375 ])(R_2)
x3_4  = Lambda(lambda x : x[:, 238 :494 ])(R_2)
x4_4  = Lambda(lambda x : x[:, 357 : 613])(R_2)
x5_4  = Lambda(lambda x : x[:, 476:732])(R_2)
x6_4  = Lambda(lambda x : x[:, 595:851])(R_2)
x7_4  = Lambda(lambda x : x[:, 714:970])(R_2)
x8_4  = Lambda(lambda x : x[:, 833:1089])(R_2)
x9_4  = Lambda(lambda x : x[:, 952:1208])(R_2)
x10_4  = Lambda(lambda x : x[:, 1071:1327])(R_2)
x11_4  = Lambda(lambda x : x[:, 1190:1446])(R_2)
x12_4  = Lambda(lambda x : x[:, 1309:1565])(R_2)
x13_4  = Lambda(lambda x : x[:, 1428:1684])(R_2)
x14_4  = Lambda(lambda x : x[:, 1547:1803])(R_2)
x15_4  = Lambda(lambda x : x[:, 1666:1922])(R_2)
x16_4  = Lambda(lambda x : x[:, 1785:2041])(R_2)

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

x1_5  = Lambda(lambda x : x[:,     :256 ])(G_2)
x2_5  = Lambda(lambda x : x[:, 119 : 375 ])(G_2)
x3_5  = Lambda(lambda x : x[:, 238 :494 ])(G_2)
x4_5  = Lambda(lambda x : x[:, 357 : 613])(G_2)
x5_5  = Lambda(lambda x : x[:, 476:732])(G_2)
x6_5  = Lambda(lambda x : x[:, 595:851])(G_2)
x7_5  = Lambda(lambda x : x[:, 714:970])(G_2)
x8_5  = Lambda(lambda x : x[:, 833:1089])(G_2)
x9_5  = Lambda(lambda x : x[:, 952:1208])(G_2)
x10_5  = Lambda(lambda x : x[:, 1071:1327])(G_2)
x11_5  = Lambda(lambda x : x[:, 1190:1446])(G_2)
x12_5  = Lambda(lambda x : x[:, 1309:1565])(G_2)
x13_5  = Lambda(lambda x : x[:, 1428:1684])(G_2)
x14_5  = Lambda(lambda x : x[:, 1547:1803])(G_2)
x15_5  = Lambda(lambda x : x[:, 1666:1922])(G_2)
x16_5  = Lambda(lambda x : x[:, 1785:2041])(G_2)

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

x1_6  = Lambda(lambda x : x[:,     :256 ])(B_2)
x2_6  = Lambda(lambda x : x[:, 119 : 375 ])(B_2)
x3_6  = Lambda(lambda x : x[:, 238 :494 ])(B_2)
x4_6  = Lambda(lambda x : x[:, 357 : 613])(B_2)
x5_6  = Lambda(lambda x : x[:, 476:732])(B_2)
x6_6  = Lambda(lambda x : x[:, 595:851])(B_2)
x7_6  = Lambda(lambda x : x[:, 714:970])(B_2)
x8_6  = Lambda(lambda x : x[:, 833:1089])(B_2)
x9_6  = Lambda(lambda x : x[:, 952:1208])(B_2)
x10_6  = Lambda(lambda x : x[:, 1071:1327])(B_2)
x11_6  = Lambda(lambda x : x[:, 1190:1446])(B_2)
x12_6  = Lambda(lambda x : x[:, 1309:1565])(B_2)
x13_6  = Lambda(lambda x : x[:, 1428:1684])(B_2)
x14_6  = Lambda(lambda x : x[:, 1547:1803])(B_2)
x15_6  = Lambda(lambda x : x[:, 1666:1922])(B_2)
x16_6  = Lambda(lambda x : x[:, 1785:2041])(B_2)

x1_6  = Tea(64)(x1_6)
x2_6  = Tea(64)(x2_6)
x3_6  = Tea(64)(x3_6)
x4_6  = Tea(64)(x4_6)
x5_6  = Tea(64)(x5_6)
x6_6  = Tea(64)(x6_6)
x7_6  = Tea(64)(x7_6)
x8_6  = Tea(64)(x8_6)
x9_6  = Tea(64)(x9_6)
x10_6  = Tea(64)(x10_6)
x11_6  = Tea(64)(x11_6)
x12_6  = Tea(64)(x12_6)
x13_6  = Tea(64)(x13_6)
x14_6  = Tea(64)(x14_6)
x15_6  = Tea(64)(x15_6)
x16_6  = Tea(64)(x16_6)

### 3 ###

x1_7  = Lambda(lambda x : x[:,     :256 ])(R_3)
x2_7  = Lambda(lambda x : x[:, 119 : 375 ])(R_3)
x3_7  = Lambda(lambda x : x[:, 238 :494 ])(R_3)
x4_7  = Lambda(lambda x : x[:, 357 : 613])(R_3)
x5_7  = Lambda(lambda x : x[:, 476:732])(R_3)
x6_7  = Lambda(lambda x : x[:, 595:851])(R_3)
x7_7  = Lambda(lambda x : x[:, 714:970])(R_3)
x8_7  = Lambda(lambda x : x[:, 833:1089])(R_3)
x9_7  = Lambda(lambda x : x[:, 952:1208])(R_3)
x10_7  = Lambda(lambda x : x[:, 1071:1327])(R_3)
x11_7  = Lambda(lambda x : x[:, 1190:1446])(R_3)
x12_7  = Lambda(lambda x : x[:, 1309:1565])(R_3)
x13_7  = Lambda(lambda x : x[:, 1428:1684])(R_3)
x14_7  = Lambda(lambda x : x[:, 1547:1803])(R_3)
x15_7  = Lambda(lambda x : x[:, 1666:1922])(R_3)
x16_7  = Lambda(lambda x : x[:, 1785:2041])(R_3)

x1_7  = Tea(64)(x1_7)
x2_7  = Tea(64)(x2_7)
x3_7  = Tea(64)(x3_7)
x4_7  = Tea(64)(x4_7)
x5_7  = Tea(64)(x5_7)
x6_7  = Tea(64)(x6_7)
x7_7  = Tea(64)(x7_7)
x8_7  = Tea(64)(x8_7)
x9_7  = Tea(64)(x9_7)
x10_7  = Tea(64)(x10_7)
x11_7  = Tea(64)(x11_7)
x12_7  = Tea(64)(x12_7)
x13_7  = Tea(64)(x13_7)
x14_7  = Tea(64)(x14_7)
x15_7  = Tea(64)(x15_7)
x16_7  = Tea(64)(x16_7)

x1_8  = Lambda(lambda x : x[:,     :256 ])(G_3)
x2_8  = Lambda(lambda x : x[:, 119 : 375 ])(G_3)
x3_8  = Lambda(lambda x : x[:, 238 :494 ])(G_3)
x4_8  = Lambda(lambda x : x[:, 357 : 613])(G_3)
x5_8  = Lambda(lambda x : x[:, 476:732])(G_3)
x6_8  = Lambda(lambda x : x[:, 595:851])(G_3)
x7_8  = Lambda(lambda x : x[:, 714:970])(G_3)
x8_8  = Lambda(lambda x : x[:, 833:1089])(G_3)
x9_8  = Lambda(lambda x : x[:, 952:1208])(G_3)
x10_8  = Lambda(lambda x : x[:, 1071:1327])(G_3)
x11_8  = Lambda(lambda x : x[:, 1190:1446])(G_3)
x12_8  = Lambda(lambda x : x[:, 1309:1565])(G_3)
x13_8  = Lambda(lambda x : x[:, 1428:1684])(G_3)
x14_8  = Lambda(lambda x : x[:, 1547:1803])(G_3)
x15_8  = Lambda(lambda x : x[:, 1666:1922])(G_3)
x16_8  = Lambda(lambda x : x[:, 1785:2041])(G_3)

x1_8  = Tea(64)(x1_8)
x2_8  = Tea(64)(x2_8)
x3_8  = Tea(64)(x3_8)
x4_8  = Tea(64)(x4_8)
x5_8  = Tea(64)(x5_8)
x6_8  = Tea(64)(x6_8)
x7_8  = Tea(64)(x7_8)
x8_8  = Tea(64)(x8_8)
x9_8  = Tea(64)(x9_8)
x10_8  = Tea(64)(x10_8)
x11_8  = Tea(64)(x11_8)
x12_8  = Tea(64)(x12_8)
x13_8  = Tea(64)(x13_8)
x14_8  = Tea(64)(x14_8)
x15_8  = Tea(64)(x15_8)
x16_8  = Tea(64)(x16_8)

x1_9  = Lambda(lambda x : x[:,     :256 ])(B_3)
x2_9  = Lambda(lambda x : x[:, 119 : 375 ])(B_3)
x3_9  = Lambda(lambda x : x[:, 238 :494 ])(B_3)
x4_9  = Lambda(lambda x : x[:, 357 : 613])(B_3)
x5_9  = Lambda(lambda x : x[:, 476:732])(B_3)
x6_9  = Lambda(lambda x : x[:, 595:851])(B_3)
x7_9  = Lambda(lambda x : x[:, 714:970])(B_3)
x8_9  = Lambda(lambda x : x[:, 833:1089])(B_3)
x9_9  = Lambda(lambda x : x[:, 952:1208])(B_3)
x10_9  = Lambda(lambda x : x[:, 1071:1327])(B_3)
x11_9  = Lambda(lambda x : x[:, 1190:1446])(B_3)
x12_9  = Lambda(lambda x : x[:, 1309:1565])(B_3)
x13_9  = Lambda(lambda x : x[:, 1428:1684])(B_3)
x14_9  = Lambda(lambda x : x[:, 1547:1803])(B_3)
x15_9  = Lambda(lambda x : x[:, 1666:1922])(B_3)
x16_9  = Lambda(lambda x : x[:, 1785:2041])(B_3)

x1_9  = Tea(64)(x1_9)
x2_9  = Tea(64)(x2_9)
x3_9  = Tea(64)(x3_9)
x4_9  = Tea(64)(x4_9)
x5_9  = Tea(64)(x5_9)
x6_9  = Tea(64)(x6_9)
x7_9  = Tea(64)(x7_9)
x8_9  = Tea(64)(x8_9)
x9_9  = Tea(64)(x9_9)
x10_9  = Tea(64)(x10_9)
x11_9  = Tea(64)(x11_9)
x12_9  = Tea(64)(x12_9)
x13_9  = Tea(64)(x13_9)
x14_9  = Tea(64)(x14_9)
x15_9  = Tea(64)(x15_9)
x16_9  = Tea(64)(x16_9)

x1_1_1 = Average()([x1_1,x1_2,x1_3,x1_4,x1_5,x1_6,x1_7,x1_8,x1_9])
x2_1_1 = Average()([x2_1,x2_2,x2_3,x2_4,x2_5,x2_6,x2_7,x2_8,x2_9])
x3_1_1 = Average()([x3_1,x3_2,x3_3,x3_4,x3_5,x3_6,x3_7,x3_8,x3_9])
x4_1_1 = Average()([x4_1,x4_2,x4_3,x4_4,x4_5,x4_6,x4_7,x4_8,x4_9])
x5_1_1 = Average()([x5_1,x5_2,x5_3,x5_4,x5_5,x5_6,x5_7,x5_8,x5_9])
x6_1_1 = Average()([x6_1,x6_2,x6_3,x6_4,x6_5,x6_6,x6_7,x6_8,x6_9])
x7_1_1 = Average()([x7_1,x7_2,x7_3,x7_4,x7_5,x7_6,x7_7,x7_8,x7_9])
x8_1_1 = Average()([x8_1,x8_2,x8_3,x8_4,x8_5,x8_6,x8_7,x8_8,x8_9])
x9_1_1 = Average()([x9_1,x9_2,x9_3,x9_4,x9_5,x9_6,x9_7,x9_8,x9_9])
x10_1_1 = Average()([x10_1,x10_2,x10_3,x10_4,x10_5,x10_6,x10_7,x10_8,x10_9])
x11_1_1 = Average()([x11_1,x11_2,x11_3,x11_4,x11_5,x11_6,x11_7,x11_8,x11_9])
x12_1_1 = Average()([x12_1,x12_2,x12_3,x12_4,x12_5,x12_6,x12_7,x12_8,x12_9])
x13_1_1 = Average()([x13_1,x13_2,x13_3,x13_4,x13_5,x13_6,x13_7,x13_8,x13_9])
x14_1_1 = Average()([x14_1,x14_2,x14_3,x14_4,x14_5,x14_6,x14_7,x14_8,x14_9])
x15_1_1 = Average()([x15_1,x15_2,x15_3,x15_4,x15_5,x15_6,x15_7,x15_8,x15_9])
x16_1_1 = Average()([x16_1,x16_2,x16_3,x16_4,x16_5,x16_6,x16_7,x16_8,x16_9])

x1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
x2 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
x3 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
x4 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

x1_1 = Tea(64)(x1)
x2_1 = Tea(64)(x2)
x3_1 = Tea(64)(x3)
x4_1 = Tea(64)(x4)

x_out = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
x_out = Tea(252)(x_out)

x_out = AdditivePooling(9)(x_out)

predictions_supine = Activation('softmax')(x_out)

model_supine = Model(inputs=inputs_supine, outputs=predictions_supine)

#### Model left ####

inputs_left = Input(shape=(64, 32,9,))

permute = Permute((2,1,3))(inputs_left)

flattened_inputs = Flatten()(permute)

flattened_inputs_1 = Lambda(lambda x : x[:,      :1*2048])(flattened_inputs)
flattened_inputs_2 = Lambda(lambda x : x[:,1*2048:2*2048])(flattened_inputs)
flattened_inputs_3 = Lambda(lambda x : x[:,2*2048:3*2048])(flattened_inputs)
flattened_inputs_4 = Lambda(lambda x : x[:,3*2048:4*2048])(flattened_inputs)
flattened_inputs_5 = Lambda(lambda x : x[:,4*2048:5*2048])(flattened_inputs)
flattened_inputs_6 = Lambda(lambda x : x[:,5*2048:6*2048])(flattened_inputs)
flattened_inputs_7 = Lambda(lambda x : x[:,6*2048:7*2048])(flattened_inputs)
flattened_inputs_8 = Lambda(lambda x : x[:,7*2048:8*2048])(flattened_inputs)
flattened_inputs_9 = Lambda(lambda x : x[:,8*2048:      ])(flattened_inputs)

# flattened_inputs_1 = Lambda(lambda x : x[:,256:])(flattened_inputs_1)
# flattened_inputs_2 = Lambda(lambda x : x[:,256:])(flattened_inputs_2)
# flattened_inputs_3 = Lambda(lambda x : x[:,256:])(flattened_inputs_3)
# flattened_inputs_4 = Lambda(lambda x : x[:,256:])(flattened_inputs_4)
# flattened_inputs_5 = Lambda(lambda x : x[:,256:])(flattened_inputs_5)
# flattened_inputs_6 = Lambda(lambda x : x[:,256:])(flattened_inputs_6)
# flattened_inputs_7 = Lambda(lambda x : x[:,256:])(flattened_inputs_7)
# flattened_inputs_8 = Lambda(lambda x : x[:,256:])(flattened_inputs_8)
# flattened_inputs_9 = Lambda(lambda x : x[:,256:])(flattened_inputs_9)

x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_1)
x2_1  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_1)
x3_1  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_1)
x4_1  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_1)
x5_1  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_1)
x6_1  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_1)
x7_1  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_1)
x8_1  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_1)
x9_1  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_1)
x10_1  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_1)
x11_1  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_1)
x12_1  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_1)
x13_1  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_1)
x14_1  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_1)
x15_1  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_1)
x16_1  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_1)

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
x2_2  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_2)
x3_2  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_2)
x4_2  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_2)
x5_2  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_2)
x6_2  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_2)
x7_2  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_2)
x8_2  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_2)
x9_2  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_2)
x10_2  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_2)
x11_2  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_2)
x12_2  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_2)
x13_2  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_2)
x14_2  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_2)
x15_2  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_2)
x16_2  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_2)

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
x2_3  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_3)
x3_3  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_3)
x4_3  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_3)
x5_3  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_3)
x6_3  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_3)
x7_3  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_3)
x8_3  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_3)
x9_3  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_3)
x10_3  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_3)
x11_3  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_3)
x12_3  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_3)
x13_3  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_3)
x14_3  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_3)
x15_3  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_3)
x16_3  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_3)

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

### 2 ###

x1_4  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_4)
x2_4  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_4)
x3_4  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_4)
x4_4  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_4)
x5_4  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_4)
x6_4  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_4)
x7_4  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_4)
x8_4  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_4)
x9_4  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_4)
x10_4  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_4)
x11_4  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_4)
x12_4  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_4)
x13_4  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_4)
x14_4  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_4)
x15_4  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_4)
x16_4  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_4)

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
x2_5  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_5)
x3_5  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_5)
x4_5  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_5)
x5_5  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_5)
x6_5  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_5)
x7_5  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_5)
x8_5  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_5)
x9_5  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_5)
x10_5  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_5)
x11_5  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_5)
x12_5  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_5)
x13_5  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_5)
x14_5  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_5)
x15_5  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_5)
x16_5  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_5)

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

x1_6  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_6)
x2_6  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_6)
x3_6  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_6)
x4_6  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_6)
x5_6  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_6)
x6_6  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_6)
x7_6  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_6)
x8_6  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_6)
x9_6  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_6)
x10_6  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_6)
x11_6  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_6)
x12_6  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_6)
x13_6  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_6)
x14_6  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_6)
x15_6  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_6)
x16_6  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_6)

x1_6  = Tea(64)(x1_6)
x2_6  = Tea(64)(x2_6)
x3_6  = Tea(64)(x3_6)
x4_6  = Tea(64)(x4_6)
x5_6  = Tea(64)(x5_6)
x6_6  = Tea(64)(x6_6)
x7_6  = Tea(64)(x7_6)
x8_6  = Tea(64)(x8_6)
x9_6  = Tea(64)(x9_6)
x10_6  = Tea(64)(x10_6)
x11_6  = Tea(64)(x11_6)
x12_6  = Tea(64)(x12_6)
x13_6  = Tea(64)(x13_6)
x14_6  = Tea(64)(x14_6)
x15_6  = Tea(64)(x15_6)
x16_6  = Tea(64)(x16_6)

### 3 ###

x1_7  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_7)
x2_7  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_7)
x3_7  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_7)
x4_7  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_7)
x5_7  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_7)
x6_7  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_7)
x7_7  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_7)
x8_7  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_7)
x9_7  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_7)
x10_7  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_7)
x11_7  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_7)
x12_7  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_7)
x13_7  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_7)
x14_7  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_7)
x15_7  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_7)
x16_7  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_7)

x1_7  = Tea(64)(x1_7)
x2_7  = Tea(64)(x2_7)
x3_7  = Tea(64)(x3_7)
x4_7  = Tea(64)(x4_7)
x5_7  = Tea(64)(x5_7)
x6_7  = Tea(64)(x6_7)
x7_7  = Tea(64)(x7_7)
x8_7  = Tea(64)(x8_7)
x9_7  = Tea(64)(x9_7)
x10_7  = Tea(64)(x10_7)
x11_7  = Tea(64)(x11_7)
x12_7  = Tea(64)(x12_7)
x13_7  = Tea(64)(x13_7)
x14_7  = Tea(64)(x14_7)
x15_7  = Tea(64)(x15_7)
x16_7  = Tea(64)(x16_7)

x1_8  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_8)
x2_8  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_8)
x3_8  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_8)
x4_8  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_8)
x5_8  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_8)
x6_8  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_8)
x7_8  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_8)
x8_8  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_8)
x9_8  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_8)
x10_8  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_8)
x11_8  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_8)
x12_8  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_8)
x13_8  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_8)
x14_8  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_8)
x15_8  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_8)
x16_8  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_8)

x1_8  = Tea(64)(x1_8)
x2_8  = Tea(64)(x2_8)
x3_8  = Tea(64)(x3_8)
x4_8  = Tea(64)(x4_8)
x5_8  = Tea(64)(x5_8)
x6_8  = Tea(64)(x6_8)
x7_8  = Tea(64)(x7_8)
x8_8  = Tea(64)(x8_8)
x9_8  = Tea(64)(x9_8)
x10_8  = Tea(64)(x10_8)
x11_8  = Tea(64)(x11_8)
x12_8  = Tea(64)(x12_8)
x13_8  = Tea(64)(x13_8)
x14_8  = Tea(64)(x14_8)
x15_8  = Tea(64)(x15_8)
x16_8  = Tea(64)(x16_8)

x1_9  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_9)
x2_9  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_9)
x3_9  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_9)
x4_9  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_9)
x5_9  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_9)
x6_9  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_9)
x7_9  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_9)
x8_9  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_9)
x9_9  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_9)
x10_9  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_9)
x11_9  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_9)
x12_9  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_9)
x13_9  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_9)
x14_9  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_9)
x15_9  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_9)
x16_9  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_9)

x1_9  = Tea(64)(x1_9)
x2_9  = Tea(64)(x2_9)
x3_9  = Tea(64)(x3_9)
x4_9  = Tea(64)(x4_9)
x5_9  = Tea(64)(x5_9)
x6_9  = Tea(64)(x6_9)
x7_9  = Tea(64)(x7_9)
x8_9  = Tea(64)(x8_9)
x9_9  = Tea(64)(x9_9)
x10_9  = Tea(64)(x10_9)
x11_9  = Tea(64)(x11_9)
x12_9  = Tea(64)(x12_9)
x13_9  = Tea(64)(x13_9)
x14_9  = Tea(64)(x14_9)
x15_9  = Tea(64)(x15_9)
x16_9  = Tea(64)(x16_9)

x1_1_1 = Average()([x1_1,x1_2,x1_3,x1_4,x1_5,x1_6,x1_7,x1_8,x1_9])
x2_1_1 = Average()([x2_1,x2_2,x2_3,x2_4,x2_5,x2_6,x2_7,x2_8,x2_9])
x3_1_1 = Average()([x3_1,x3_2,x3_3,x3_4,x3_5,x3_6,x3_7,x3_8,x3_9])
x4_1_1 = Average()([x4_1,x4_2,x4_3,x4_4,x4_5,x4_6,x4_7,x4_8,x4_9])
x5_1_1 = Average()([x5_1,x5_2,x5_3,x5_4,x5_5,x5_6,x5_7,x5_8,x5_9])
x6_1_1 = Average()([x6_1,x6_2,x6_3,x6_4,x6_5,x6_6,x6_7,x6_8,x6_9])
x7_1_1 = Average()([x7_1,x7_2,x7_3,x7_4,x7_5,x7_6,x7_7,x7_8,x7_9])
x8_1_1 = Average()([x8_1,x8_2,x8_3,x8_4,x8_5,x8_6,x8_7,x8_8,x8_9])
x9_1_1 = Average()([x9_1,x9_2,x9_3,x9_4,x9_5,x9_6,x9_7,x9_8,x9_9])
x10_1_1 = Average()([x10_1,x10_2,x10_3,x10_4,x10_5,x10_6,x10_7,x10_8,x10_9])
x11_1_1 = Average()([x11_1,x11_2,x11_3,x11_4,x11_5,x11_6,x11_7,x11_8,x11_9])
x12_1_1 = Average()([x12_1,x12_2,x12_3,x12_4,x12_5,x12_6,x12_7,x12_8,x12_9])
x13_1_1 = Average()([x13_1,x13_2,x13_3,x13_4,x13_5,x13_6,x13_7,x13_8,x13_9])
x14_1_1 = Average()([x14_1,x14_2,x14_3,x14_4,x14_5,x14_6,x14_7,x14_8,x14_9])
x15_1_1 = Average()([x15_1,x15_2,x15_3,x15_4,x15_5,x15_6,x15_7,x15_8,x15_9])
x16_1_1 = Average()([x16_1,x16_2,x16_3,x16_4,x16_5,x16_6,x16_7,x16_8,x16_9])

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

predictions_left = Activation('softmax')(x_out)
model_left = Model(inputs=inputs_left, outputs=predictions_left)

#### Model Right ####

inputs_right = Input(shape=(64, 32,9,))

permute = Permute((2,1,3))(inputs_right)

flattened_inputs = Flatten()(permute)

flattened_inputs_1 = Lambda(lambda x : x[:,      :1*2048])(flattened_inputs)
flattened_inputs_2 = Lambda(lambda x : x[:,1*2048:2*2048])(flattened_inputs)
flattened_inputs_3 = Lambda(lambda x : x[:,2*2048:3*2048])(flattened_inputs)
flattened_inputs_4 = Lambda(lambda x : x[:,3*2048:4*2048])(flattened_inputs)
flattened_inputs_5 = Lambda(lambda x : x[:,4*2048:5*2048])(flattened_inputs)
flattened_inputs_6 = Lambda(lambda x : x[:,5*2048:6*2048])(flattened_inputs)
flattened_inputs_7 = Lambda(lambda x : x[:,6*2048:7*2048])(flattened_inputs)
flattened_inputs_8 = Lambda(lambda x : x[:,7*2048:8*2048])(flattened_inputs)
flattened_inputs_9 = Lambda(lambda x : x[:,8*2048:      ])(flattened_inputs)

# flattened_inputs_1 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_1)
# flattened_inputs_2 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_2)
# flattened_inputs_3 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_3)
# flattened_inputs_4 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_4)
# flattened_inputs_5 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_5)
# flattened_inputs_6 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_6)
# flattened_inputs_7 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_7)
# flattened_inputs_8 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_8)
# flattened_inputs_9 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_9)

x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_1)
x2_1  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_1)
x3_1  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_1)
x4_1  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_1)
x5_1  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_1)
x6_1  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_1)
x7_1  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_1)
x8_1  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_1)
x9_1  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_1)
x10_1  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_1)
x11_1  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_1)
x12_1  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_1)
x13_1  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_1)
x14_1  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_1)
x15_1  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_1)
x16_1  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_1)

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
x2_2  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_2)
x3_2  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_2)
x4_2  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_2)
x5_2  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_2)
x6_2  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_2)
x7_2  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_2)
x8_2  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_2)
x9_2  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_2)
x10_2  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_2)
x11_2  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_2)
x12_2  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_2)
x13_2  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_2)
x14_2  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_2)
x15_2  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_2)
x16_2  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_2)

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
x2_3  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_3)
x3_3  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_3)
x4_3  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_3)
x5_3  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_3)
x6_3  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_3)
x7_3  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_3)
x8_3  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_3)
x9_3  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_3)
x10_3  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_3)
x11_3  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_3)
x12_3  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_3)
x13_3  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_3)
x14_3  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_3)
x15_3  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_3)
x16_3  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_3)

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

### 2 ###

x1_4  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_4)
x2_4  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_4)
x3_4  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_4)
x4_4  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_4)
x5_4  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_4)
x6_4  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_4)
x7_4  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_4)
x8_4  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_4)
x9_4  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_4)
x10_4  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_4)
x11_4  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_4)
x12_4  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_4)
x13_4  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_4)
x14_4  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_4)
x15_4  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_4)
x16_4  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_4)

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
x2_5  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_5)
x3_5  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_5)
x4_5  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_5)
x5_5  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_5)
x6_5  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_5)
x7_5  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_5)
x8_5  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_5)
x9_5  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_5)
x10_5  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_5)
x11_5  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_5)
x12_5  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_5)
x13_5  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_5)
x14_5  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_5)
x15_5  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_5)
x16_5  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_5)

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

x1_6  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_6)
x2_6  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_6)
x3_6  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_6)
x4_6  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_6)
x5_6  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_6)
x6_6  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_6)
x7_6  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_6)
x8_6  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_6)
x9_6  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_6)
x10_6  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_6)
x11_6  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_6)
x12_6  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_6)
x13_6  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_6)
x14_6  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_6)
x15_6  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_6)
x16_6  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_6)

x1_6  = Tea(64)(x1_6)
x2_6  = Tea(64)(x2_6)
x3_6  = Tea(64)(x3_6)
x4_6  = Tea(64)(x4_6)
x5_6  = Tea(64)(x5_6)
x6_6  = Tea(64)(x6_6)
x7_6  = Tea(64)(x7_6)
x8_6  = Tea(64)(x8_6)
x9_6  = Tea(64)(x9_6)
x10_6  = Tea(64)(x10_6)
x11_6  = Tea(64)(x11_6)
x12_6  = Tea(64)(x12_6)
x13_6  = Tea(64)(x13_6)
x14_6  = Tea(64)(x14_6)
x15_6  = Tea(64)(x15_6)
x16_6  = Tea(64)(x16_6)

### 3 ###

x1_7  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_7)
x2_7  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_7)
x3_7  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_7)
x4_7  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_7)
x5_7  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_7)
x6_7  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_7)
x7_7  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_7)
x8_7  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_7)
x9_7  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_7)
x10_7  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_7)
x11_7  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_7)
x12_7  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_7)
x13_7  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_7)
x14_7  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_7)
x15_7  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_7)
x16_7  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_7)

x1_7  = Tea(64)(x1_7)
x2_7  = Tea(64)(x2_7)
x3_7  = Tea(64)(x3_7)
x4_7  = Tea(64)(x4_7)
x5_7  = Tea(64)(x5_7)
x6_7  = Tea(64)(x6_7)
x7_7  = Tea(64)(x7_7)
x8_7  = Tea(64)(x8_7)
x9_7  = Tea(64)(x9_7)
x10_7  = Tea(64)(x10_7)
x11_7  = Tea(64)(x11_7)
x12_7  = Tea(64)(x12_7)
x13_7  = Tea(64)(x13_7)
x14_7  = Tea(64)(x14_7)
x15_7  = Tea(64)(x15_7)
x16_7  = Tea(64)(x16_7)

x1_8  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_8)
x2_8  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_8)
x3_8  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_8)
x4_8  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_8)
x5_8  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_8)
x6_8  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_8)
x7_8  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_8)
x8_8  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_8)
x9_8  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_8)
x10_8  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_8)
x11_8  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_8)
x12_8  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_8)
x13_8  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_8)
x14_8  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_8)
x15_8  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_8)
x16_8  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_8)

x1_8  = Tea(64)(x1_8)
x2_8  = Tea(64)(x2_8)
x3_8  = Tea(64)(x3_8)
x4_8  = Tea(64)(x4_8)
x5_8  = Tea(64)(x5_8)
x6_8  = Tea(64)(x6_8)
x7_8  = Tea(64)(x7_8)
x8_8  = Tea(64)(x8_8)
x9_8  = Tea(64)(x9_8)
x10_8  = Tea(64)(x10_8)
x11_8  = Tea(64)(x11_8)
x12_8  = Tea(64)(x12_8)
x13_8  = Tea(64)(x13_8)
x14_8  = Tea(64)(x14_8)
x15_8  = Tea(64)(x15_8)
x16_8  = Tea(64)(x16_8)

x1_9  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_9)
x2_9  = Lambda(lambda x : x[:, 119 : 375 ])(flattened_inputs_9)
x3_9  = Lambda(lambda x : x[:, 238 :494 ])(flattened_inputs_9)
x4_9  = Lambda(lambda x : x[:, 357 : 613])(flattened_inputs_9)
x5_9  = Lambda(lambda x : x[:, 476:732])(flattened_inputs_9)
x6_9  = Lambda(lambda x : x[:, 595:851])(flattened_inputs_9)
x7_9  = Lambda(lambda x : x[:, 714:970])(flattened_inputs_9)
x8_9  = Lambda(lambda x : x[:, 833:1089])(flattened_inputs_9)
x9_9  = Lambda(lambda x : x[:, 952:1208])(flattened_inputs_9)
x10_9  = Lambda(lambda x : x[:, 1071:1327])(flattened_inputs_9)
x11_9  = Lambda(lambda x : x[:, 1190:1446])(flattened_inputs_9)
x12_9  = Lambda(lambda x : x[:, 1309:1565])(flattened_inputs_9)
x13_9  = Lambda(lambda x : x[:, 1428:1684])(flattened_inputs_9)
x14_9  = Lambda(lambda x : x[:, 1547:1803])(flattened_inputs_9)
x15_9  = Lambda(lambda x : x[:, 1666:1922])(flattened_inputs_9)
x16_9  = Lambda(lambda x : x[:, 1785:2041])(flattened_inputs_9)

x1_9  = Tea(64)(x1_9)
x2_9  = Tea(64)(x2_9)
x3_9  = Tea(64)(x3_9)
x4_9  = Tea(64)(x4_9)
x5_9  = Tea(64)(x5_9)
x6_9  = Tea(64)(x6_9)
x7_9  = Tea(64)(x7_9)
x8_9  = Tea(64)(x8_9)
x9_9  = Tea(64)(x9_9)
x10_9  = Tea(64)(x10_9)
x11_9  = Tea(64)(x11_9)
x12_9  = Tea(64)(x12_9)
x13_9  = Tea(64)(x13_9)
x14_9  = Tea(64)(x14_9)
x15_9  = Tea(64)(x15_9)
x16_9  = Tea(64)(x16_9)

x1_1_1 = Average()([x1_1,x1_2,x1_3,x1_4,x1_5,x1_6,x1_7,x1_8,x1_9])
x2_1_1 = Average()([x2_1,x2_2,x2_3,x2_4,x2_5,x2_6,x2_7,x2_8,x2_9])
x3_1_1 = Average()([x3_1,x3_2,x3_3,x3_4,x3_5,x3_6,x3_7,x3_8,x3_9])
x4_1_1 = Average()([x4_1,x4_2,x4_3,x4_4,x4_5,x4_6,x4_7,x4_8,x4_9])
x5_1_1 = Average()([x5_1,x5_2,x5_3,x5_4,x5_5,x5_6,x5_7,x5_8,x5_9])
x6_1_1 = Average()([x6_1,x6_2,x6_3,x6_4,x6_5,x6_6,x6_7,x6_8,x6_9])
x7_1_1 = Average()([x7_1,x7_2,x7_3,x7_4,x7_5,x7_6,x7_7,x7_8,x7_9])
x8_1_1 = Average()([x8_1,x8_2,x8_3,x8_4,x8_5,x8_6,x8_7,x8_8,x8_9])
x9_1_1 = Average()([x9_1,x9_2,x9_3,x9_4,x9_5,x9_6,x9_7,x9_8,x9_9])
x10_1_1 = Average()([x10_1,x10_2,x10_3,x10_4,x10_5,x10_6,x10_7,x10_8,x10_9])
x11_1_1 = Average()([x11_1,x11_2,x11_3,x11_4,x11_5,x11_6,x11_7,x11_8,x11_9])
x12_1_1 = Average()([x12_1,x12_2,x12_3,x12_4,x12_5,x12_6,x12_7,x12_8,x12_9])
x13_1_1 = Average()([x13_1,x13_2,x13_3,x13_4,x13_5,x13_6,x13_7,x13_8,x13_9])
x14_1_1 = Average()([x14_1,x14_2,x14_3,x14_4,x14_5,x14_6,x14_7,x14_8,x14_9])
x15_1_1 = Average()([x15_1,x15_2,x15_3,x15_4,x15_5,x15_6,x15_7,x15_8,x15_9])
x16_1_1 = Average()([x16_1,x16_2,x16_3,x16_4,x16_5,x16_6,x16_7,x16_8,x16_9])

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

predictions_right = Activation('softmax')(x_out)

model_right = Model(inputs=inputs_right, outputs=predictions_right)

#### Load_weight #### 

model_3_classes.load_weights("bed_posture/ckpt_3_classes_median/3_class-{}".format(sub))
model_supine.load_weights("bed_posture/ckpt_supine/9_class_deep-{}".format(sub))
model_left.load_weights("bed_posture/ckpt_left_9_cores_nocrop/4_class_deep-{}".format(sub))
model_right.load_weights("bed_posture/ckpt_right_9_cores_nocrop/4_class_deep-{}".format(sub))

#### Combine model ####
model_3_classes.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.003),
            metrics=['accuracy'])
model_supine.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.003),
            metrics=['accuracy'])
model_left.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.003),
            metrics=['accuracy'])
model_right.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.003),
            metrics=['accuracy'])
#### Merge Output ####

lambda_supine = Lambda(lambda x : x[:,0])(model_3_classes.output)
lambda_left = Lambda(lambda x : x[:,1])(model_3_classes.output)
lambda_right = Lambda(lambda x : x[:,2])(model_3_classes.output)

out_supine = Multiply()([model_supine.output,lambda_supine])
out_left = Multiply()([model_left.output,lambda_left])
out_right = Multiply()([model_right.output,lambda_right])

# model_3_classes.trainable = False
# model_left.trainable = False
# model_right.trainable = False

predictions_all = Concatenate(axis=1)([out_supine,out_left,out_right])

model_all = Model(inputs=[model_3_classes.input, model_supine.input, model_left.input, model_right.input], outputs= [predictions_all])

print('------------------------------------------------------------------------')
print(f'Training for subject {sub} ...')

model_all.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0015),
            metrics=['accuracy'])

checkpoint_filepath = 'bed_posture/ckpt_3/17_classes_deep-{}'.format(sub)

import keras

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath + '-epoch-{epoch}',
    save_weights_only=True,)

model_all.fit([x_train_3_class,x_train_supine,x_train_left,x_train_right], [y_train],
         batch_size=1024,
         epochs=50,
         verbose=1,
         callbacks=[model_checkpoint_callback],
         validation_split=0.2)



import os
scores = []
soure = "bed_posture/ckpt_3"
ckpts = [os.path.join(soure,e) for e in os.listdir(soure) if sub in e]
for ckpt in ckpts:
    print("======================================")
    print(ckpt)
    print("======================================")
    model_all.load_weights(ckpt)
    score = model_supine.evaluate(x_test_only_supine, y_test_only_supine, verbose=0)   
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    scores.append(score[1])

print("======================================")
print("Max accuracy:",max(scores))
print("Best epoch:",ckpts[scores.index(max(scores))])









