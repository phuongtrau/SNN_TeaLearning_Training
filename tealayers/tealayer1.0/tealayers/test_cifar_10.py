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
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, concatenate
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import to_categorical

from tea import Tea
from additivepooling import AdditivePooling

import sys
sys.path.append("/home/phuongdh/Documents/SNN_TeaLearning_Training/rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

# Load MNIST data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# save old labels for later
y_test_not = y_test

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

inputs = Input(shape=(32, 32,3,))
### GET ALL CHANNEL ###  
red = Lambda(lambda x : x[:,:,:,0])(inputs)
green = Lambda(lambda x : x[:,:,:,1])(inputs)
blue = Lambda(lambda x : x[:,:,:,2])(inputs)
### USING KERNEL  ###
# FIRST STEP # STRIDE = 2 
red_1_1 = Lambda(lambda x : x[:,0:16,0:16])(red)
red_1_2 = Lambda(lambda x : x[:,16:,0:16])(red)
red_1_3 = Lambda(lambda x : x[:,0:16,16:])(red)
red_1_4 = Lambda(lambda x : x[:,16:,16:])(red)

green_1_1 = Lambda(lambda x : x[:,0:16,0:16])(green)
green_1_2 = Lambda(lambda x : x[:,16:,0:16])(green)
green_1_3 = Lambda(lambda x : x[:,0:16,16:])(green)
green_1_4 = Lambda(lambda x : x[:,16:,16:])(green)

blue_1_1 = Lambda(lambda x : x[:,0:16,0:16])(blue)
blue_1_2 = Lambda(lambda x : x[:,16:,0:16])(blue)
blue_1_3 = Lambda(lambda x : x[:,0:16,16:])(blue)
blue_1_4 = Lambda(lambda x : x[:,16:,16:])(blue)
# SECOND STEP # 
red_2_1 = Lambda(lambda x : x[:,2:18,2:18])(red)
red_2_2 = Lambda(lambda x : x[:,14:30,2:18])(red)
red_2_3 = Lambda(lambda x : x[:,2:18,14:30])(red)
red_2_4 = Lambda(lambda x : x[:,14:30,14:30])(red)

green_2_1 = Lambda(lambda x : x[:,2:18,2:18])(green)
green_2_2 = Lambda(lambda x : x[:,14:30,2:18])(green)
green_2_3 = Lambda(lambda x : x[:,2:18,14:30])(green)
green_2_4 = Lambda(lambda x : x[:,16:,16:])(green)

blue_2_1 = Lambda(lambda x : x[:,2:18,2:18])(blue)
blue_2_2 = Lambda(lambda x : x[:,14:30,2:18])(blue)
blue_2_3 = Lambda(lambda x : x[:,2:18,14:30])(blue)
blue_2_4 = Lambda(lambda x : x[:,14:30,14:30])(blue)
# THIRTH STEP #
red_3_1 = Lambda(lambda x : x[:,4:20,4:20])(red)
red_3_2 = Lambda(lambda x : x[:,12:28,4:20])(red)
red_3_3 = Lambda(lambda x : x[:,4:20,12:28])(red)
red_3_4 = Lambda(lambda x : x[:,12:28,12:28])(red)

green_3_1 = Lambda(lambda x : x[:,4:20,4:20])(green)
green_3_2 = Lambda(lambda x : x[:,12:28,4:20])(green)
green_3_3 = Lambda(lambda x : x[:,4:20,12:28])(green)
green_3_4 = Lambda(lambda x : x[:,12:28,12:28])(green)

blue_3_1 = Lambda(lambda x : x[:,4:20,4:20])(blue)
blue_3_2 = Lambda(lambda x : x[:,12:28,4:20])(blue)
blue_3_3 = Lambda(lambda x : x[:,4:20,12:28])(blue)
blue_3_4 = Lambda(lambda x : x[:,12:28,12:28])(blue)
# FOURTH STEP #
red_4_1 = Lambda(lambda x : x[:,6:22,6:22])(red)
red_4_2 = Lambda(lambda x : x[:,10:26,6:22])(red)
red_4_3 = Lambda(lambda x : x[:,6:22,10:26])(red)
red_4_4 = Lambda(lambda x : x[:,10:26,10:26])(red)

green_4_1 = Lambda(lambda x : x[:,6:22,6:22])(green)
green_4_2 = Lambda(lambda x : x[:,10:26,6:22])(green)
green_4_3 = Lambda(lambda x : x[:,6:22,10:26])(green)
green_4_4 = Lambda(lambda x : x[:,10:26,10:26])(green)

blue_4_1 = Lambda(lambda x : x[:,6:22,6:22])(blue)
blue_4_2 = Lambda(lambda x : x[:,10:26,6:22])(blue)
blue_4_3 = Lambda(lambda x : x[:,6:22,10:26])(blue)
blue_4_4 = Lambda(lambda x : x[:,10:26,10:26])(blue)
# FINAL STEP #

red_5 = Lambda(lambda x : x[:,8:24,8:24])(red)
green_5 = Lambda(lambda x : x[:,8:24,8:24])(green)
blue_5 = Lambda(lambda x : x[:,8:24,8:24])(blue)








