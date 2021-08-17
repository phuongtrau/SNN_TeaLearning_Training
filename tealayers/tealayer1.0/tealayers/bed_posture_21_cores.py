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

import preprocess
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

exp_i_data = helper.load_exp_i("../dataset/experiment-i")

# print(len(dataset))
datasets = {"Base":exp_i_data}
train_data = helper.Mat_Dataset(datasets,["Base"],["S1","S2","S3","S4","S5","S6","S7","S8","S9"])
for i in range(len(train_data.samples)):
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])
    # train_data.samples[i] = preprocess.preprocess(train_data.samples[i])

# print((train_data.samples.shape,train_data.labels.shape))

test_data = helper.Mat_Dataset(datasets,["Base"],["S10","S11","S12","S13"])
for i in range(len(test_data.samples)):
    test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
    # test_data.samples[i] = preprocess.preprocess(test_data.samples[i])
# print((test_data.samples,test_data.labels))

x_train = train_data.samples.astype('float32')
x_test = test_data.samples.astype('float32')

x_train /= 255
x_test /= 255

# cv2.imwrite("raw.jpg",train_data.samples[0])
# img = cv2.equalizeHist(train_data.samples[0])
# cv2.imwrite("test.jpg",img)

y_train = to_categorical(train_data.labels, 17)
y_test = to_categorical(test_data.labels, 17)

# random.seed(0)
(x_train,y_train) = shuffle(x_train,y_train)
# print(x_train_s,y_train_s)

(x_test,y_test) = shuffle(x_test,y_test)
# print(x_train_s,y_train_s)

inputs = Input(shape=(64, 32,))
flattened_inputs = Flatten()(inputs)

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
x_out = AdditivePooling(17)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=25,
          verbose=1,
          validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
if (score[1]>=0.985):
    print("good")
    cores_sim = create_cores(model,21, neuron_reset_type=0) 

    write_cores(cores_sim,output_path="/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/output_mem_bed_posture")

# ## get connection ## 
# weights_1 , biases_1 = get_connections_and_biases(model,11)

# connections_1 = []
# # bias_retrain_1 = []
# for weight_1 in weights_1:

#     connections_1.append(np.clip(np.round(weight_1), 0, 1))

# for i in range(len(connections_1)) :
#     # print(connections_1[i].shape)

#     connections_1[i]=np.reshape(connections_1[i],(256,-1))

# ## pretrained ##

# train_data_1 = helper.Mat_Dataset(datasets,["Base"],["S1","S2","S3","S4","S5","S7","S8","S9","S6"])

# for i in range(len(train_data_1.samples)):

#     train_data_1.samples[i] = cv2.equalizeHist(train_data_1.samples[i])
#     # train_data_1.samples[i] = preprocess.preprocess(train_data_1.samples[i])

# # print((train_data.samples.shape,train_data.labels.shape))

# test_data_1 = helper.Mat_Dataset(datasets,["Base"],["S10","S11","S12","S13"])
# for i in range(len(test_data_1.samples)):
#     test_data_1.samples[i] = cv2.equalizeHist(test_data_1.samples[i])
#     # test_data_1.samples[i] = preprocess.preprocess(test_data_1.samples[i])
# # print((test_data.samples,test_data.labels))

# x_train_1 = train_data_1.samples.astype('float32')
# x_test_1 = test_data_1.samples.astype('float32')

# x_train_1 /= 255
# x_test_1 /= 255

# # cv2.imwrite("raw.jpg",train_data.samples[0])
# # img = cv2.equalizeHist(train_data.samples[0])
# # cv2.imwrite("test.jpg",img)

# y_train_1 = to_categorical(train_data_1.labels, 3)
# y_test_1 = to_categorical(test_data_1.labels, 3)

# # random.seed(0)
# (x_train_1,y_train_1) = shuffle(x_train_1,y_train_1)
# # print(x_train_s,y_train_s)

# (x_test_1,y_test_1) = shuffle(x_test_1,y_test_1)

# inputs_1 = Input(shape=(64, 32,))
# flattened_inputs = Flatten()(inputs_1)

# x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs)
# x2_1  = Lambda(lambda x : x[:, 256 :512 ])(flattened_inputs)
# x3_1  = Lambda(lambda x : x[:, 512 :768 ])(flattened_inputs)
# x4_1  = Lambda(lambda x : x[:, 768 :1024])(flattened_inputs)
# x5_1  = Lambda(lambda x : x[:, 1024:1280])(flattened_inputs)
# x6_1  = Lambda(lambda x : x[:, 1280:1536])(flattened_inputs)
# x7_1  = Lambda(lambda x : x[:, 1536:1792])(flattened_inputs)
# x8_1  = Lambda(lambda x : x[:, 1792:    ])(flattened_inputs)

# x1_1  = Tea(64,init_connection=connections_1[0])(x1_1)
# x2_1  = Tea(64,init_connection=connections_1[1])(x2_1)
# x3_1  = Tea(64,init_connection=connections_1[2])(x3_1)
# x4_1  = Tea(64,init_connection=connections_1[3])(x4_1)
# x5_1  = Tea(64,init_connection=connections_1[4])(x5_1)
# x6_1  = Tea(64,init_connection=connections_1[5])(x6_1)
# x7_1  = Tea(64,init_connection=connections_1[6])(x7_1)
# x8_1  = Tea(64,init_connection=connections_1[7])(x8_1)

# x1_1_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
# x2_1_1 = Concatenate(axis=1)([x5_1,x6_1,x7_1,x8_1])

# x1_1 = Tea(128,init_connection=connections_1[8])(x1_1_1)
# x2_1 = Tea(128,init_connection=connections_1[9])(x2_1_1)

# x_out = Concatenate(axis=1)([x1_1,x2_1])
# x_out = Tea(255,init_connection=connections_1[10])(x_out)
# x_out = AdditivePooling(3)(x_out)

# predictions = Activation('softmax')(x_out)

# model_1 = Model(inputs=inputs_1, outputs=predictions)

# model_1.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

# model_1.fit(x_train_1, y_train_1,
#           batch_size=64,
#           epochs=10,
#           verbose=1,
#           validation_split=0.2)




# score = model_1.evaluate(x_test_1, y_test_1, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# cores_sim = create_cores(model_1, 11,neuron_reset_type=0) 

# write_cores(cores_sim,output_path="/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/output_mem_bed_posture")