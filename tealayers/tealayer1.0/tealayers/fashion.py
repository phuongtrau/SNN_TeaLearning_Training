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
sys.path.append("../../../rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

class Fashion():
    def __init__(self, input):
        self.flattened_inputs = input
        # self.time_window = time

    def forward(self):
        
        x0 = Lambda(lambda x : x[:,:512])(self.flattened_inputs)
        x1 = Lambda(lambda x : x[:,34:546])(self.flattened_inputs)
        x2 = Lambda(lambda x : x[:,68:580])(self.flattened_inputs)
        x3 = Lambda(lambda x : x[:,102:614])(self.flattened_inputs)
        x4 = Lambda(lambda x : x[:,136:648])(self.flattened_inputs)
        x5 = Lambda(lambda x : x[:,170:682])(self.flattened_inputs)
        x6 = Lambda(lambda x : x[:,204:716])(self.flattened_inputs)
        x7 = Lambda(lambda x : x[:,238:750])(self.flattened_inputs)
        x8 = Lambda(lambda x : x[:,272:])(self.flattened_inputs)

        x0 = Tea(256)(x0)
        x1 = Tea(256)(x1)
        x2 = Tea(256)(x2)
        x3 = Tea(256)(x3)
        x4 = Tea(256)(x4)
        x5 = Tea(256)(x5)
        x6 = Tea(256)(x6)
        x7 = Tea(256)(x7)
        x8 = Tea(256)(x8)

        x_1 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])

        x0 = Lambda(lambda x : x[:,:512])(x_1)
        x1 = Lambda(lambda x : x[:,224:736])(x_1)
        x2 = Lambda(lambda x : x[:,448:960])(x_1)
        x3 = Lambda(lambda x : x[:,672:1184])(x_1)
        x4 = Lambda(lambda x : x[:,896:1408])(x_1)
        x5 = Lambda(lambda x : x[:,1120:1632])(x_1)
        x6 = Lambda(lambda x : x[:,1344:1856])(x_1)
        x7 = Lambda(lambda x : x[:,1568:2080])(x_1)
        x8 = Lambda(lambda x : x[:,1792:])(x_1)

        x0 = Tea(128)(x0)
        x1 = Tea(128)(x1)
        x2 = Tea(128)(x2)
        x3 = Tea(128)(x3)
        x4 = Tea(128)(x4)
        x5 = Tea(128)(x5)
        x6 = Tea(128)(x6)
        x7 = Tea(128)(x7)
        x8 = Tea(128)(x8)

        x_2 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])

        x0 = Lambda(lambda x : x[:,:512])(x_2)
        x1 = Lambda(lambda x : x[:,80:592])(x_2)
        x2 = Lambda(lambda x : x[:,160:672])(x_2)
        x3 = Lambda(lambda x : x[:,240:752])(x_2)
        x4 = Lambda(lambda x : x[:,320:832])(x_2)
        x5 = Lambda(lambda x : x[:,400:912])(x_2)
        x6 = Lambda(lambda x : x[:,480:992])(x_2)
        x7 = Lambda(lambda x : x[:,560:1072])(x_2)
        x8 = Lambda(lambda x : x[:,640:])(x_2)

        x0 = Tea(170)(x0)
        x1 = Tea(170)(x1)
        x2 = Tea(170)(x2)
        x3 = Tea(170)(x3)
        x4 = Tea(170)(x4)
        x5 = Tea(170)(x5)
        x6 = Tea(170)(x6)
        x7 = Tea(170)(x7)
        x8 = Tea(170)(x8)

        x_2 = Concatenate(axis=1)([x0, x1, x2])
        x_3 = Concatenate(axis=1)([x3, x4, x5])
        x_4 = Concatenate(axis=1)([x6, x7, x8])

        x0 = Tea(170)(x_2)
        # x0 = Average()([x0,x1,x2])

        x1 = Tea(170)(x_3)
        # x1 = Average()([x3,x4,x5])

        x2 = Tea(170)(x_4)
        # x2 = Average()([x6,x7,x8])

        # print(x0,x1,x2,x3)
        # Concatenate output of first layer to send into next

        x = Concatenate(axis=1)([x0, x1, x2])

        x = Tea(510)(x)

        x = Average()([x_2,x_3,x_4,x])
        
        return x
        # # Pool spikes and output neurons into 10 classes.

        # x = AdditivePooling(10)(x)

        # self. predictions = Activation('softmax')(x)