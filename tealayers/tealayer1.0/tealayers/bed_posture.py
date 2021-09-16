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
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, Concatenate,Average,Permute,Add
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
import cv2

from preprocess import check_in_region,center_out
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

class Bed_Poture_Heat_21(Layer):
    def __init__(self,inputs):
        self.inputs = inputs
    def forward(self):

        R = Lambda(lambda x : x[:,     :2048 ])(self.inputs)
        G = Lambda(lambda x : x[:, 2048:4096 ])(self.inputs)
        B = Lambda(lambda x : x[:, 4096:     ])(self.inputs)

        x1_1  = Lambda(lambda x : x[:,     :256 ])(R)
        x2_1  = Lambda(lambda x : x[:, 119 : 375 ])(R)
        x3_1  = Lambda(lambda x : x[:, 238 :494 ])(R)
        x4_1  = Lambda(lambda x : x[:, 357 : 613])(R)
        x5_1  = Lambda(lambda x : x[:, 476:732])(R)
        x6_1  = Lambda(lambda x : x[:, 595:851])(R)
        x7_1  = Lambda(lambda x : x[:, 714:970])(R)
        x8_1  = Lambda(lambda x : x[:, 833:1089])(R)
        x9_1  = Lambda(lambda x : x[:, 952:1208])(R)
        x10_1  = Lambda(lambda x : x[:, 1071:1327])(R)
        x11_1  = Lambda(lambda x : x[:, 1190:1446])(R)
        x12_1  = Lambda(lambda x : x[:, 1309:1565])(R)
        x13_1  = Lambda(lambda x : x[:, 1428:1684])(R)
        x14_1  = Lambda(lambda x : x[:, 1547:1803])(R)
        x15_1  = Lambda(lambda x : x[:, 1666:1922])(R)
        x16_1  = Lambda(lambda x : x[:, 1785:2041])(R)

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

        x1_2  = Lambda(lambda x : x[:,     :256 ])(G)
        x2_2  = Lambda(lambda x : x[:, 119 : 375 ])(G)
        x3_2  = Lambda(lambda x : x[:, 238 :494 ])(G)
        x4_2  = Lambda(lambda x : x[:, 357 : 613])(G)
        x5_2  = Lambda(lambda x : x[:, 476:732])(G)
        x6_2  = Lambda(lambda x : x[:, 595:851])(G)
        x7_2  = Lambda(lambda x : x[:, 714:970])(G)
        x8_2  = Lambda(lambda x : x[:, 833:1089])(G)
        x9_2  = Lambda(lambda x : x[:, 952:1208])(G)
        x10_2  = Lambda(lambda x : x[:, 1071:1327])(G)
        x11_2  = Lambda(lambda x : x[:, 1190:1446])(G)
        x12_2  = Lambda(lambda x : x[:, 1309:1565])(G)
        x13_2  = Lambda(lambda x : x[:, 1428:1684])(G)
        x14_2  = Lambda(lambda x : x[:, 1547:1803])(G)
        x15_2  = Lambda(lambda x : x[:, 1666:1922])(G)
        x16_2  = Lambda(lambda x : x[:, 1785:2041])(G)

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

        x1_3  = Lambda(lambda x : x[:,     :256 ])(B)
        x2_3  = Lambda(lambda x : x[:, 119 : 375 ])(B)
        x3_3  = Lambda(lambda x : x[:, 238 :494 ])(B)
        x4_3  = Lambda(lambda x : x[:, 357 : 613])(B)
        x5_3  = Lambda(lambda x : x[:, 476:732])(B)
        x6_3  = Lambda(lambda x : x[:, 595:851])(B)
        x7_3  = Lambda(lambda x : x[:, 714:970])(B)
        x8_3  = Lambda(lambda x : x[:, 833:1089])(B)
        x9_3  = Lambda(lambda x : x[:, 952:1208])(B)
        x10_3  = Lambda(lambda x : x[:, 1071:1327])(B)
        x11_3  = Lambda(lambda x : x[:, 1190:1446])(B)
        x12_3  = Lambda(lambda x : x[:, 1309:1565])(B)
        x13_3  = Lambda(lambda x : x[:, 1428:1684])(B)
        x14_3  = Lambda(lambda x : x[:, 1547:1803])(B)
        x15_3  = Lambda(lambda x : x[:, 1666:1922])(B)
        x16_3  = Lambda(lambda x : x[:, 1785:2041])(B)

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

        x1_1_1 = Average()([x1_1,x1_2,x1_3])
        x2_1_1 = Average()([x2_1,x2_2,x2_3])
        x3_1_1 = Average()([x3_1,x3_2,x3_3])
        x4_1_1 = Average()([x4_1,x4_2,x4_3])
        x5_1_1 = Average()([x5_1,x5_2,x5_3])
        x6_1_1 = Average()([x6_1,x6_2,x6_3])
        x7_1_1 = Average()([x7_1,x7_2,x7_3])
        x8_1_1 = Average()([x8_1,x8_2,x8_3])
        x9_1_1 = Average()([x9_1,x9_2,x9_3])
        x10_1_1 = Average()([x10_1,x10_2,x10_3])
        x11_1_1 = Average()([x11_1,x11_2,x11_3])
        x12_1_1 = Average()([x12_1,x12_2,x12_3])
        x13_1_1 = Average()([x13_1,x13_2,x13_3])
        x14_1_1 = Average()([x14_1,x14_2,x14_3])
        x15_1_1 = Average()([x15_1,x15_2,x15_3])
        x16_1_1 = Average()([x16_1,x16_2,x16_3])

        x1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
        x2 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
        x3 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
        x4 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

        x1_1 = Tea(64)(x1)
        x2_1 = Tea(64)(x2)
        x3_1 = Tea(64)(x3)
        x4_1 = Tea(64)(x4)

        x_out_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
        x_out_1 = Tea(255)(x_out_1)
        return x_out_1

class Bed_Poture_Gray_21(Layer):
    def __init__(self,inputs):
        self.inputs = inputs
    def forward(self):

        x1_1  = Lambda(lambda x : x[:,     :256 ])(self.inputs)

        x2_1  = Lambda(lambda x : x[:, 119 : 375 ])(self.inputs)

        x3_1  = Lambda(lambda x : x[:, 238 :494 ])(self.inputs)

        x4_1  = Lambda(lambda x : x[:, 357 : 613])(self.inputs)

        x5_1  = Lambda(lambda x : x[:, 476:732])(self.inputs)

        x6_1  = Lambda(lambda x : x[:, 595:851])(self.inputs)

        x7_1  = Lambda(lambda x : x[:, 714:970])(self.inputs)

        x8_1  = Lambda(lambda x : x[:, 833:1089])(self.inputs)

        x9_1  = Lambda(lambda x : x[:, 952:1208])(self.inputs)

        x10_1  = Lambda(lambda x : x[:, 1071:1327])(self.inputs)

        x11_1  = Lambda(lambda x : x[:, 1190:1446])(self.inputs)

        x12_1  = Lambda(lambda x : x[:, 1309:1565])(self.inputs)

        x13_1  = Lambda(lambda x : x[:, 1428:1684])(self.inputs)

        x14_1  = Lambda(lambda x : x[:, 1547:1803])(self.inputs)

        x15_1  = Lambda(lambda x : x[:, 1666:1922])(self.inputs)

        x16_1  = Lambda(lambda x : x[:, 1785:2041])(self.inputs)

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
        return x_out

class Bed_Poture_Gray_11(Layer):
    
    def __init__(self,inputs):
        self.inputs = inputs
    def forward(self):
        x1_1  = Lambda(lambda x : x[:,     :256 ])(self.inputs)
        x2_1  = Lambda(lambda x : x[:, 256 :512 ])(self.inputs)
        x3_1  = Lambda(lambda x : x[:, 512 :768 ])(self.inputs)
        x4_1  = Lambda(lambda x : x[:, 768 :1024])(self.inputs)
        x5_1  = Lambda(lambda x : x[:, 1024:1280])(self.inputs)
        x6_1  = Lambda(lambda x : x[:, 1280:1536])(self.inputs)
        x7_1  = Lambda(lambda x : x[:, 1536:1792])(self.inputs)
        x8_1  = Lambda(lambda x : x[:, 1792:    ])(self.inputs)

        x1_1  = Tea(64)(x1_1)
        x2_1  = Tea(64)(x2_1)
        x3_1  = Tea(64)(x3_1)
        x4_1  = Tea(64)(x4_1)
        x5_1  = Tea(64)(x5_1)
        x6_1  = Tea(64)(x6_1)
        x7_1  = Tea(64)(x7_1)
        x8_1  = Tea(64)(x8_1)

        x1_1_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
        x2_1_1 = Concatenate(axis=1)([x5_1,x6_1,x7_1,x8_1])

        x1_1 = Tea(128)(x1_1_1)
        x2_1 = Tea(128)(x2_1_1)

        x_out = Concatenate(axis=1)([x1_1,x2_1])
        x_out = Tea(255)(x_out)
        return x_out
