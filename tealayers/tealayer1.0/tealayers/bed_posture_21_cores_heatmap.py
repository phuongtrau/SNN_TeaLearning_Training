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

# from preprocess import check_in_region,center_out
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

exp_i_data = helper.load_exp_i("../dataset/experiment-i")
kernel_ero = np.ones((3,3),np.uint8)
kernel_dil = np.ones((5,5),np.uint8)
# print(len(dataset))
datasets = {"Base":exp_i_data}

train_data = helper.Mat_Dataset(datasets,["Base"],["S1","S2","S3","S4","S5","S6","S7","S8","S9"])

x_train = []

for i in range(len(train_data.samples)):
    
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])

    heat = cv2.applyColorMap(train_data.samples[i], cv2.COLORMAP_JET)
    x_train.append(heat)
 
    
test_data = helper.Mat_Dataset(datasets,["Base"],["S10","S11","S12","S13"])

x_test = []

for i in range(len(test_data.samples)):
    
    test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
    
    heat = cv2.applyColorMap(test_data.samples[i], cv2.COLORMAP_JET)
    x_test.append(heat)
    
x_train = np.array(x_train).astype(float)
x_test = np.array(x_test).astype(float)

x_train /= 255
x_test /= 255

y_train = to_categorical(train_data.labels, 17)
y_test = to_categorical(test_data.labels, 17)

inputs = Input(shape=(64, 32,3))

# permute = Permute((2,1,3))(inputs)
flattened = Flatten()(inputs)

flattened_inputs_1 = Lambda(lambda x : x[:,      :2048*3 ])(flattened)

# flattened_inputs_2 = Lambda(lambda x : x[:,2048*3: ])(flattened)
# flattened_inputs_2 = Lambda(lambda x : x[:,2048*3:2048*6 ])(flattened)
# flattened_inputs_3 = Lambda(lambda x : x[:,2048*6: ])(flattened)

R = Lambda(lambda x : x[:,     :2048 ])(flattened_inputs_1)
G = Lambda(lambda x : x[:, 2048:4096 ])(flattened_inputs_1)
B = Lambda(lambda x : x[:, 4096:     ])(flattened_inputs_1)

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

x_out = AdditivePooling(17)(x_out_1)

predictions = Activation('softmax')(x_out)

# save_model = Model(inputs=inputs, outputs=predictions)

# save_model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

# save_model.load_weights("bed_posture/ckpt/17_class-50-1.44.hdf5")      
# score = save_model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1]*100)        

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=30,mode='max',min_delta=0.0001)

checkpoint_filepath = 'bed_posture/ckpt/17_class-{epoch:02d}-acc{val_acc:.3f}'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

model.fit(x_train, y_train,
          batch_size=64,
          epochs=100,
          verbose=1,
          callbacks=[callback,model_checkpoint_callback],
          validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)
if score[1]>=0.76:
    print("good")
    model.save_weights("bed_posture/ckpt/17_class_weight")
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)



