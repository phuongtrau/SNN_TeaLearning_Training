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
import cv2

exp_i_data = helper.load_exp_i("../dataset/experiment-i")

# print(len(dataset))
datasets = {"Base":exp_i_data}
train_data = helper.Mat_Dataset(datasets,["Base"],["S1","S2","S3","S4","S5","S6","S7","S8","S9"])
cv2.imwrite("img_raw.jpg",train_data.samples[99])
for i in range(len(train_data.samples)):
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])
cv2.imwrite("img_raw_after.jpg",train_data.samples[99])
# print((train_data.samples.shape,train_data.labels.shape))

test_data = helper.Mat_Dataset(datasets,["Base"],["S10","S11","S12","S13"])
for i in range(len(test_data.samples)):
    test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
# print((test_data.samples,test_data.labels))

x_train = train_data.samples.astype('float32')
x_test = test_data.samples.astype('float32')

x_train /= 255
x_test /= 255

# cv2.imwrite("raw.jpg",train_data.samples[0])
# img = cv2.equalizeHist(train_data.samples[0])
# cv2.imwrite("test.jpg",img)
# for e in train_data.labels:
#     print(e)

y_train = to_categorical(train_data.labels, 3)
y_test = to_categorical(test_data.labels, 3)

# for e in y_train:
#     print(e)

# random.seed(0)
(x_train,y_train) = shuffle(x_train,y_train)
# print(x_train_s,y_train_s)

(x_test,y_test) = shuffle(x_test,y_test)
# print(x_train_s,y_train_s)

# inputs = Input(shape=(64, 32,))
# flattened_inputs = Flatten()(inputs)

# x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs)
# x2_1  = Lambda(lambda x : x[:, 256 :512 ])(flattened_inputs)
# x3_1  = Lambda(lambda x : x[:, 512 :768 ])(flattened_inputs)
# x4_1  = Lambda(lambda x : x[:, 768 :1024])(flattened_inputs)
# x5_1  = Lambda(lambda x : x[:, 1024:1280])(flattened_inputs)
# x6_1  = Lambda(lambda x : x[:, 1280:1536])(flattened_inputs)
# x7_1  = Lambda(lambda x : x[:, 1536:1792])(flattened_inputs)
# x8_1  = Lambda(lambda x : x[:, 1792:    ])(flattened_inputs)

# x1_1  = Tea(64)(x1_1)
# x2_1  = Tea(64)(x2_1)
# x3_1  = Tea(64)(x3_1)
# x4_1  = Tea(64)(x4_1)
# x5_1  = Tea(64)(x5_1)
# x6_1  = Tea(64)(x6_1)
# x7_1  = Tea(64)(x7_1)
# x8_1  = Tea(64)(x8_1)

# x1_1_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
# x2_1_1 = Concatenate(axis=1)([x5_1,x6_1,x7_1,x8_1])

# x1_1 = Tea(128)(x1_1_1)
# x2_1 = Tea(128)(x2_1_1)

# x_out = Concatenate(axis=1)([x1_1,x2_1])
# x_out = Tea(255)(x_out)
# x_out = AdditivePooling(3)(x_out)

# predictions = Activation('softmax')(x_out)

# model = Model(inputs=inputs, outputs=predictions)

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=64,
#           epochs=10,
#           verbose=1,
#           validation_split=0.2)

# score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# # weights , biases = get_connections_and_biases(model,11)

# from output_bus import OutputBus
# from serialization import save as sim_save
# from emulation import write_cores
# if score[1] >= 0.975:
#     print("good")
#     cores_sim = create_cores(model, 11,neuron_reset_type=0 ) 

#     write_cores(cores_sim,max_xy=(1,11),output_path="/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/output_mem_bed_posture_11_cores")
