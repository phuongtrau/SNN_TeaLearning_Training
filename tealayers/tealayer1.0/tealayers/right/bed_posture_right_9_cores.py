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
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, Concatenate,Average,Permute
from keras.datasets import mnist,fashion_mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
import sys
sys.path.append("../../../../rancutils/rancutils")

# from teaconversion import create_cores,create_packets,get_connections_and_biases
# from packet import Packet
sys.path.append("../")
# from tea import Tea
from additivepooling import AdditivePooling
import helper
from tea import Tea
import random
from sklearn.utils import shuffle
import cv2

# import preprocess
# from output_bus import OutputBus
# from serialization import save as sim_save
# from emulation import write_cores

exp_i_data = helper.load_exp_i_right("../../dataset/experiment-i")
# kernel = np.ones((3,3),np.uint8)*200
# print(len(dataset))
datasets = {"Base":exp_i_data}
# list_train = ["S1","S10","S7","S11","S2","S12","S3","S13","S4","S9","S5","S8",]
# random.shuffle(list_train)
subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]

sub="S13"

subjects.remove(sub)
random.shuffle(subjects)

train_data = helper.Mat_Dataset(datasets,["Base"],subjects)
# kernel = np.ones((5,5),np.uint8)
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


test_data = helper.Mat_Dataset(datasets,["Base"],[sub])
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
    



x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = to_categorical(train_data.labels, 4)
y_test = to_categorical(test_data.labels, 4)

random.seed(1)
(x_train,y_train) = shuffle(x_train,y_train)
random.seed(1)
(x_test,y_test) = shuffle(x_test,y_test)

inputs = Input(shape=(64, 32,9,))

permute = Permute((2,1,3))(inputs)

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

flattened_inputs_1 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_1)
flattened_inputs_2 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_2)
flattened_inputs_3 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_3)
flattened_inputs_4 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_4)
flattened_inputs_5 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_5)
flattened_inputs_6 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_6)
flattened_inputs_7 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_7)
flattened_inputs_8 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_8)
flattened_inputs_9 = Lambda(lambda x : x[:,:2048-256])(flattened_inputs_9)

x1_1  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_1)
x2_1  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_1)
x3_1  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_1)
x4_1  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_1)
x5_1  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_1)
x6_1  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_1)
x7_1  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_1)
x8_1  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_1)
x9_1  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_1)
x10_1  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_1)
x11_1  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_1)
x12_1  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_1)
x13_1  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_1)
x14_1  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_1)
x15_1  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_1)
x16_1  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_1)

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

x1_2  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_2)
x2_2  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_2)
x3_2  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_2)
x4_2  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_2)
x5_2  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_2)
x6_2  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_2)
x7_2  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_2)
x8_2  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_2)
x9_2  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_2)
x10_2  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_2)
x11_2  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_2)
x12_2  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_2)
x13_2  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_2)
x14_2  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_2)
x15_2  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_2)
x16_2  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_2)

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

x1_3  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_3)
x2_3  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_3)
x3_3  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_3)
x4_3  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_3)
x5_3  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_3)
x6_3  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_3)
x7_3  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_3)
x8_3  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_3)
x9_3  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_3)
x10_3  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_3)
x11_3  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_3)
x12_3  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_3)
x13_3  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_3)
x14_3  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_3)
x15_3  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_3)
x16_3  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_3)

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

x1_4  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_4)
x2_4  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_4)
x3_4  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_4)
x4_4  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_4)
x5_4  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_4)
x6_4  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_4)
x7_4  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_4)
x8_4  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_4)
x9_4  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_4)
x10_4  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_4)
x11_4  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_4)
x12_4  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_4)
x13_4  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_4)
x14_4  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_4)
x15_4  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_4)
x16_4  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_4)

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

x1_5  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_5)
x2_5  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_5)
x3_5  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_5)
x4_5  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_5)
x5_5  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_5)
x6_5  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_5)
x7_5  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_5)
x8_5  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_5)
x9_5  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_5)
x10_5  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_5)
x11_5  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_5)
x12_5  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_5)
x13_5  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_5)
x14_5  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_5)
x15_5  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_5)
x16_5  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_5)

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

x1_6  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_6)
x2_6  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_6)
x3_6  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_6)
x4_6  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_6)
x5_6  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_6)
x6_6  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_6)
x7_6  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_6)
x8_6  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_6)
x9_6  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_6)
x10_6  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_6)
x11_6  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_6)
x12_6  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_6)
x13_6  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_6)
x14_6  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_6)
x15_6  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_6)
x16_6  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_6)

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

x1_7  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_7)
x2_7  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_7)
x3_7  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_7)
x4_7  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_7)
x5_7  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_7)
x6_7  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_7)
x7_7  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_7)
x8_7  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_7)
x9_7  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_7)
x10_7  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_7)
x11_7  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_7)
x12_7  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_7)
x13_7  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_7)
x14_7  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_7)
x15_7  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_7)
x16_7  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_7)

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

x1_8  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_8)
x2_8  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_8)
x3_8  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_8)
x4_8  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_8)
x5_8  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_8)
x6_8  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_8)
x7_8  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_8)
x8_8  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_8)
x9_8  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_8)
x10_8  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_8)
x11_8  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_8)
x12_8  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_8)
x13_8  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_8)
x14_8  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_8)
x15_8  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_8)
x16_8  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_8)

x1_8_8  = Tea(64)(x1_8)
x2_8_8  = Tea(64)(x2_8)
x3_8_8  = Tea(64)(x3_8)
x4_8_8  = Tea(64)(x4_8)
x5_8_8  = Tea(64)(x5_8)
x6_8_8  = Tea(64)(x6_8)
x7_8_8  = Tea(64)(x7_8)
x8_8_8  = Tea(64)(x8_8)
x9_8_8  = Tea(64)(x9_8)
x10_8_8  = Tea(64)(x10_8)
x11_8_8  = Tea(64)(x11_8)
x12_8_8  = Tea(64)(x12_8)
x13_8_8  = Tea(64)(x13_8)
x14_8_8  = Tea(64)(x14_8)
x15_8_8  = Tea(64)(x15_8)
x16_8_8  = Tea(64)(x16_8)

x1_9  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs_9)
x2_9  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs_9)
x3_9  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs_9)
x4_9  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs_9)
x5_9  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs_9)
x6_9  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs_9)
x7_9  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs_9)
x8_9  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs_9)
x9_9  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs_9)
x10_9  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs_9)
x11_9  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs_9)
x12_9  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs_9)
x13_9  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs_9)
x14_9  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs_9)
x15_9  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs_9)
x16_9  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs_9)

x1_9_9  = Tea(64)(x1_9)
x2_9_9  = Tea(64)(x2_9)
x3_9_9  = Tea(64)(x3_9)
x4_9_9  = Tea(64)(x4_9)
x5_9_9  = Tea(64)(x5_9)
x6_9_9  = Tea(64)(x6_9)
x7_9_9  = Tea(64)(x7_9)
x8_9_9  = Tea(64)(x8_9)
x9_9_9  = Tea(64)(x9_9)
x10_9_9  = Tea(64)(x10_9)
x11_9_9  = Tea(64)(x11_9)
x12_9_9  = Tea(64)(x12_9)
x13_9_9  = Tea(64)(x13_9)
x14_9_9  = Tea(64)(x14_9)
x15_9_9  = Tea(64)(x15_9)
x16_9_9  = Tea(64)(x16_9)

x1_1_1 = Average()([x1_1_1,x1_2_2,x1_3_3,x1_4_4,x1_5_5,x1_6_6,x1_7_7,x1_8_8,x1_9_9])
x2_1_1 = Average()([x2_1_1,x2_2_2,x2_3_3,x2_4_4,x2_5_5,x2_6_6,x2_7_7,x2_8_8,x2_9_9])
x3_1_1 = Average()([x3_1_1,x3_2_2,x3_3_3,x3_4_4,x3_5_5,x3_6_6,x3_7_7,x3_8_8,x3_9_9])
x4_1_1 = Average()([x4_1_1,x4_2_2,x4_3_3,x4_4_4,x4_5_5,x4_6_6,x4_7_7,x4_8_8,x4_9_9])
x5_1_1 = Average()([x5_1_1,x5_2_2,x5_3_3,x5_4_4,x5_5_5,x5_6_6,x5_7_7,x5_8_8,x5_9_9])
x6_1_1 = Average()([x6_1_1,x6_2_2,x6_3_3,x6_4_4,x6_5_5,x6_6_6,x6_7_7,x6_8_8,x6_9_9])
x7_1_1 = Average()([x7_1_1,x7_2_2,x7_3_3,x7_4_4,x7_5_5,x7_6_6,x7_7_7,x7_8_8,x7_9_9])
x8_1_1 = Average()([x8_1_1,x8_2_2,x8_3_3,x8_4_4,x8_5_5,x8_6_6,x8_7_7,x8_8_8,x8_9_9])
x9_1_1 = Average()([x9_1_1,x9_2_2,x9_3_3,x9_4_4,x9_5_5,x9_6_6,x9_7_7,x9_8_8,x9_9_9])
x10_1_1 = Average()([x10_1_1,x10_2_2,x10_3_3,x10_4_4,x10_5_5,x10_6_6,x10_7_7,x10_8_8,x10_9_9])
x11_1_1 = Average()([x11_1_1,x11_2_2,x11_3_3,x11_4_4,x11_5_5,x11_6_6,x11_7_7,x11_8_8,x11_9_9])
x12_1_1 = Average()([x12_1_1,x12_2_2,x12_3_3,x12_4_4,x12_5_5,x12_6_6,x12_7_7,x12_8_8,x12_9_9])
x13_1_1 = Average()([x13_1_1,x13_2_2,x13_3_3,x13_4_4,x13_5_5,x13_6_6,x13_7_7,x13_8_8,x13_9_9])
x14_1_1 = Average()([x14_1_1,x14_2_2,x14_3_3,x14_4_4,x14_5_5,x14_6_6,x14_7_7,x14_8_8,x14_9_9])
x15_1_1 = Average()([x15_1_1,x15_2_2,x15_3_3,x15_4_4,x15_5_5,x15_6_6,x15_7_7,x15_8_8,x15_9_9])
x16_1_1 = Average()([x16_1_1,x16_2_2,x16_3_3,x16_4_4,x16_5_5,x16_6_6,x16_7_7,x16_8_8,x16_9_9])

x1_1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
x2_1 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
x3_1 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
x4_1 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

x1_1 = Tea(64)(x1_1)
x2_1 = Tea(64)(x2_1)
x3_1 = Tea(64)(x3_1)
x4_1 = Tea(64)(x4_1)

x_out_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
# x_out_2 = Concatenate(axis=1)([x3_1,x4_1])

x_out_1 = Tea(256)(x_out_1)
# x_out_2 = Tea(256)(x_out_2)
# x_out = Concatenate(axis=1)([x_out_1,x_out_2])

x_out = AdditivePooling(4)(x_out_1)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

print('------------------------------------------------------------------------')
print(f'Training for subject {sub} ...')

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0005),
              metrics=['accuracy'])

checkpoint_filepath = '../bed_posture/ckpt_3/4_class_deep-{}'.format(sub)
import keras 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath+ '-epoch-{epoch}',
    save_weights_only=True,)
    # save_freq = "epoch",)

model.load_weights("../bed_posture/ckpt_right_9_cores/4_class_deep-{}".format(sub)) 

model.fit(x_train, y_train,
          batch_size=64,
          epochs=50,
          verbose=1,
          callbacks=[model_checkpoint_callback],
          validation_split=0)

import os
scores = []
soure = "../bed_posture/ckpt_3"
ckpts = [os.path.join(soure,e) for e in os.listdir(soure) if sub in e]
for ckpt in ckpts:
    print("======================================")
    print(ckpt)
    print("======================================")
    model.load_weights(ckpt)      
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    scores.append(score[1])
print("======================================")
print("Max accuracy:",max(scores))
print("Best epoch:",ckpts[scores.index(max(scores))])
