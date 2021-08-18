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

exp_i_data = helper.load_exp_i_supine("../dataset/experiment-i")

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

# threshold_train = np.ones_like(x_train)*127
# threshold_test = np.ones_like(x_test)*127

stage_1 = 63
stage_2 = 127
stage_3 = 190

x_train_new = []
for i in range(len(x_train)):
    threshold = np.ones_like(x_train[i])
    e_1 = np.array(x_train[i]>=threshold*stage_1).astype(float)*x_train/255
    e_2 = np.array(x_train[i]>=threshold*stage_2).astype(float)*x_train/255 + e_1
    e_3 = np.array(x_train[i]>=threshold*stage_3).astype(float)*x_train/255 + e_2
    
    # x_train[i] = (e_1+e_2+e_3)/3

    x_train_new.append(np.concatenate([e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis]],axis=2))

x_train_new = np.array(x_train_new)
# print(x_train_new.shape)

x_test_new = []
for i in range(len(x_test)):
    threshold = np.ones_like(x_test[i])
    e_1 = np.array(x_test[i]>=threshold*stage_1).astype(float)*x_test/255
    e_2 = np.array(x_test[i]>=threshold*stage_2).astype(float)*x_test/255 + e_1
    e_3 = np.array(x_test[i]>=threshold*stage_3).astype(float)*x_test/255 + e_2
    
    # x_test[i] = (e_1+e_2+e_3)/3
    
    x_test_new.append(np.concatenate([e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis]],axis=2))
    
x_test_new = np.array(x_test_new)
# print(x_test_new.shape)

# x_train = np.array(x_train>=threshold_train).astype(float)
# x_test = np.array(x_test>=threshold_test).astype(float)


# x_train /= 255
# x_test /= 255

y_train = to_categorical(train_data.labels, 9)
y_test = to_categorical(test_data.labels, 9)

(x_train,y_train) = shuffle(x_train_new,y_train)
# print(x_train_s,y_train_s)

(x_test,y_test) = shuffle(x_test_new,y_test)

inputs = Input(shape=(64, 32,3,))
# print(inputs)
# permute = Permute((2,1,3))(inputs)
# print(permute)
flattened_inputs = Flatten()(inputs)

# flattened_inputs = Flatten()(permute)

# flattened_inputs = Lambda(lambda x : x[:,: 2048])(flattened_inputs)

# flattened_inputs_2 = Lambda(lambda x : x[:,2048:4096])(flattened_inputs)
# flattened_inputs_3 = Lambda(lambda x : x[:,4096:])(flattened_inputs)

# flattened_inputs = Lambda(lambda x : x[:,256: 1792])(flattened_inputs)

# flattened_inputs_2 = Lambda(lambda x : x[:,256: 1792])(flattened_inputs_2)
# flattened_inputs_3 = Lambda(lambda x : x[:,256: 1792])(flattened_inputs_3)

x1_1  = Lambda(lambda x : x[:,     : 256 ])(flattened_inputs)
x2_1  = Lambda(lambda x : x[:, 256 : 512 ])(flattened_inputs)
x3_1  = Lambda(lambda x : x[:, 512 : 768 ])(flattened_inputs)
x4_1  = Lambda(lambda x : x[:, 768 : 1024 ])(flattened_inputs)
x5_1  = Lambda(lambda x : x[:, 1024 : 1280 ])(flattened_inputs)
x6_1  = Lambda(lambda x : x[:, 1280 : 1536])(flattened_inputs)
x7_1  = Lambda(lambda x : x[:, 1536: 1792])(flattened_inputs)
x8_1  = Lambda(lambda x : x[:, 1792: 2048])(flattened_inputs)

x9_1  = Lambda(lambda x : x[:, 2048: 2304 ])(flattened_inputs)
x10_1  = Lambda(lambda x : x[:,2304 : 2560 ])(flattened_inputs)
x11_1  = Lambda(lambda x : x[:,2560 : 2816 ])(flattened_inputs)
x12_1  = Lambda(lambda x : x[:,2816 : 3072 ])(flattened_inputs)
x13_1  = Lambda(lambda x : x[:,3072 : 3328 ])(flattened_inputs)
x14_1  = Lambda(lambda x : x[:,3328 : 3584])(flattened_inputs)
x15_1  = Lambda(lambda x : x[:,3584: 3840])(flattened_inputs)
x16_1  = Lambda(lambda x : x[:,3840: 4096])(flattened_inputs)

x17_1  = Lambda(lambda x : x[:,4096 : 4352 ])(flattened_inputs)
x18_1  = Lambda(lambda x : x[:,4352 : 4608 ])(flattened_inputs)
x19_1  = Lambda(lambda x : x[:,4608 : 4864])(flattened_inputs)
x20_1  = Lambda(lambda x : x[:,4864 : 5120 ])(flattened_inputs)
x21_1  = Lambda(lambda x : x[:,5120 : 5376 ])(flattened_inputs)
x22_1  = Lambda(lambda x : x[:,5376 : 5632])(flattened_inputs)
x23_1  = Lambda(lambda x : x[:,5632: 5888])(flattened_inputs)
x24_1  = Lambda(lambda x : x[:,5888: 6144])(flattened_inputs)

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

x17_1  = Tea(64)(x17_1)
x18_1  = Tea(64)(x18_1)
x19_1  = Tea(64)(x19_1)
x20_1  = Tea(64)(x20_1)
x21_1  = Tea(64)(x21_1)
x22_1  = Tea(64)(x22_1)
x23_1  = Tea(64)(x23_1)
x24_1  = Tea(64)(x24_1)

x1_1_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])
x2_1_1 = Concatenate(axis=1)([x5_1,x6_1,x7_1,x8_1])
x3_1_1 = Concatenate(axis=1)([x9_1,x10_1,x11_1,x12_1])
x4_1_1 = Concatenate(axis=1)([x13_1,x14_1,x15_1,x16_1])
x5_1_1 = Concatenate(axis=1)([x17_1,x18_1,x19_1,x20_1])
x6_1_1 = Concatenate(axis=1)([x21_1,x22_1,x23_1,x24_1])

x1_1 = Tea(128)(x1_1_1)
x2_1 = Tea(128)(x2_1_1)
x3_1 = Tea(128)(x3_1_1)
x4_1 = Tea(128)(x4_1_1)
x5_1 = Tea(128)(x5_1_1)
x6_1 = Tea(128)(x6_1_1)

x_out_1 = Concatenate(axis=1)([x1_1,x2_1])
x_out_2 = Concatenate(axis=1)([x3_1,x4_1])
x_out_3 = Concatenate(axis=1)([x5_1,x6_1])

x_out_1 = Tea(252)(x_out_1)
x_out_2 = Tea(252)(x_out_2)
x_out_3 = Tea(252)(x_out_3)

x_out = Concatenate(axis=1)([x_out_1,x_out_2,x_out_3])


# x1_2  = Lambda(lambda x : x[:,   3 : 256 ])(flattened_inputs_2)
# x2_2  = Lambda(lambda x : x[:, 185 : 441 ])(flattened_inputs_2)
# x3_2  = Lambda(lambda x : x[:, 367 : 623 ])(flattened_inputs_2)
# x4_2  = Lambda(lambda x : x[:, 549 : 805 ])(flattened_inputs_2)
# x5_2  = Lambda(lambda x : x[:, 731 : 987 ])(flattened_inputs_2)
# x6_2  = Lambda(lambda x : x[:, 913 : 1169])(flattened_inputs_2)
# x7_2  = Lambda(lambda x : x[:, 1095: 1351])(flattened_inputs_2)
# x8_2  = Lambda(lambda x : x[:, 1277: 1533])(flattened_inputs_2)

# x1_2  = Tea(64)(x1_2)
# x2_2  = Tea(64)(x2_2)
# x3_2  = Tea(64)(x3_2)
# x4_2  = Tea(64)(x4_2)
# x5_2  = Tea(64)(x5_2)
# x6_2  = Tea(64)(x6_2)
# x7_2  = Tea(64)(x7_2)
# x8_2  = Tea(64)(x8_2)

# x1_2_2 = Concatenate(axis=1)([x1_2,x2_2,x3_2,x4_2])
# x2_2_2 = Concatenate(axis=1)([x5_2,x6_2,x7_2,x8_2])

# x1_2 = Tea(128)(x1_2_2)
# x2_2 = Tea(128)(x2_2_2)

# x_out_2 = Concatenate(axis=1)([x1_2,x2_2])
# x_out_2 = Tea(252)(x_out_2)

# x1_3  = Lambda(lambda x : x[:,   3 : 256 ])(flattened_inputs_3)
# x2_3  = Lambda(lambda x : x[:, 185 : 441 ])(flattened_inputs_3)
# x3_3  = Lambda(lambda x : x[:, 367 : 623 ])(flattened_inputs_3)
# x4_3  = Lambda(lambda x : x[:, 549 : 805 ])(flattened_inputs_3)
# x5_3  = Lambda(lambda x : x[:, 731 : 987 ])(flattened_inputs_3)
# x6_3  = Lambda(lambda x : x[:, 913 : 1169])(flattened_inputs_3)
# x7_3  = Lambda(lambda x : x[:, 1095: 1351])(flattened_inputs_3)
# x8_3  = Lambda(lambda x : x[:, 1277: 1533])(flattened_inputs_3)

# x1_3  = Tea(64)(x1_3)
# x2_3  = Tea(64)(x2_3)
# x3_3  = Tea(64)(x3_3)
# x4_3  = Tea(64)(x4_3)
# x5_3  = Tea(64)(x5_3)
# x6_3  = Tea(64)(x6_3)
# x7_3  = Tea(64)(x7_3)
# x8_3  = Tea(64)(x8_3)

# x1_3_3 = Concatenate(axis=1)([x1_3,x2_3,x3_3,x4_3])
# x2_3_3 = Concatenate(axis=1)([x5_3,x6_3,x7_3,x8_3])

# x1_3 = Tea(128)(x1_3_3)
# x2_3 = Tea(128)(x2_3_3)

# x_out_3 = Concatenate(axis=1)([x1_3,x2_3])
# x_out_3 = Tea(252)(x_out_3)

# x_out = Average()([x_out_1,x_out_2,x_out_3])

x_out = AdditivePooling(9)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=20,
          verbose=1,
          validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])