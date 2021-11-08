from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math
import os 

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras import Model
from keras.engine.topology import Layer
from keras import initializers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, Concatenate,Average
from keras.datasets import mnist,fashion_mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
import sys
sys.path.append("../../../rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet
# sys.path.append("../")
# from tea import Tea
from additivepooling import AdditivePooling
import helper
from tea import Tea
# import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

exp_i_data = helper.load_exp_i_supine("../dataset/experiment-i")

# print(len(dataset))
datasets = {"Base":exp_i_data}

subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]
# subjects = ["S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]
# Define per-fold score containers
acc_per_so = []
loss_per_so = []
ls_train_full = subjects.copy()
for sub in ls_train_full:
  subjects.remove(sub)
  train_data = helper.Mat_Dataset(datasets,["Base"],subjects)
  test_data = helper.Mat_Dataset(datasets,["Base"],[sub])
  
  x_train = []

  for i in range(len(train_data.samples)):
      
      train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])

      heat = cv2.applyColorMap(train_data.samples[i], cv2.COLORMAP_JET)
      mask = np.ones_like(heat)
      bin1 = np.array(heat>=mask*63).astype(np.uint8)
      bin2 = np.array(heat>=mask*127).astype(np.uint8)
      bin3 = np.array(heat>=mask*190).astype(np.uint8)
      bin_out = np.concatenate((bin1,bin2,bin3),axis=2)
      x_train.append(bin_out)

  x_test = []

  for i in range(len(test_data.samples)):

      test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
      
      heat = cv2.applyColorMap(test_data.samples[i], cv2.COLORMAP_JET)
      mask = np.ones_like(heat)
      bin1 = np.array(heat>=mask*63).astype(np.uint8)
      bin2 = np.array(heat>=mask*127).astype(np.uint8)
      bin3 = np.array(heat>=mask*190).astype(np.uint8)
      bin_out = np.concatenate((bin1,bin2,bin3),axis=2)

      x_test.append(bin_out)
      
  x_train = np.array(x_train).astype(np.uint8)
  x_test = np.array(x_test).astype(np.uint8)

  y_train = to_categorical(train_data.labels, 9)
  y_test = to_categorical(test_data.labels, 9)

  inputs = Input(shape=(64, 32,9))

  # permute = Permute((2,1,3))(inputs)
  flattened = Flatten()(inputs)

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
  # x1_1 = Average()([x1_1_1,x2_1_1,x3_1_1,x4_1_1,x1_1])
  x2_1 = Tea(64)(x2)
  # x2_1 = Average()([x1_1_1,x2_1_1,x3_1_1,x4_1_1,x2_1])
  x3_1 = Tea(64)(x3)
  # x3_1 = Average()([x1_1_1,x2_1_1,x3_1_1,x4_1_1,x3_1])
  x4_1 = Tea(64)(x4)
  # x4_1 = Average()([x1_1_1,x2_1_1,x3_1_1,x4_1_1,x4_1])


  x_out = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
  x_out = Tea(252)(x_out)

  # x_out = Concatenate(axis=1)([x_out_1,x_out_2])

  x_out = AdditivePooling(9)(x_out)

  predictions = Activation('softmax')(x_out)

  model = Model(inputs=inputs, outputs=predictions)

  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

  # callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=10)

  print('------------------------------------------------------------------------')
  print(f'Training for subject {sub} ...')

  # model.fit(x_train, y_train,
  #         batch_size=64,
  #         epochs=45,
  #         # callbacks=[callback],
  #         verbose=1,)
          # validation_split=0.2)
  
  model.load_weights("bed_posture/ckpt_supine/9_class_deep-{}".format(sub))
  # model.load_weights("bed_posture/ckpt_supine/9_class_deep-S12")
  score = model.evaluate(x_test, y_test, verbose=0)  
  acc_per_so.append(score[1] * 100)
  loss_per_so.append(score[0])
  
  subjects = ls_train_full.copy()

  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  # if not os.path.exists(".mem_files/supine_21_cores_mem/{}".format(sub)):
  #   os.makedirs(".mem_files/supine_21_cores_mem/{}".format(sub))

  # cores_sim = create_cores(model, 16*9+5 , neuron_reset_type=0,num_classes=9) 

  # write_cores(cores_sim,max_xy=(1,16*9+5),output_path=".mem_files/supine_21_cores_mem/{}".format(sub))

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per subject out')
for i in range(0, len(acc_per_so)):
  print('------------------------------------------------------------------------')
  print(f'> Subject {i+1} - Loss: {loss_per_so[i]} - Accuracy: {acc_per_so[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all subject out:')
print(f'> Accuracy: {np.mean(acc_per_so)}% (+- {np.std(acc_per_so)})')
print(f'> Loss: {np.mean(loss_per_so)}')
print('------------------------------------------------------------------------')

for i in range(0, len(acc_per_so)):
  print(f'{acc_per_so[i]}%')
print(f'Accuracy: {np.mean(acc_per_so)}%')
