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
import random
from sklearn.utils import shuffle
import cv2

# import preprocess
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

exp_i_data = helper.load_exp_i_supine_norm("../dataset/experiment-i")
# kernel = np.ones((3,3),np.uint8)*200
# print(len(dataset))
datasets = {"Base":exp_i_data}

train_data = helper.Mat_Dataset(datasets,["Base"],["S1","S2","S3","S4","S5","S6","S7","S8","S9"])
kernel = np.ones((5,5),np.uint8)
for i in range(len(train_data.samples)):
    
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])
    _,thresh1=cv2.threshold(train_data.samples[i],0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_eros = cv2.erode(thresh1,kernel,iterations=1)
    img_dila = cv2.dilate(img_eros,kernel,iterations=1)
    train_data.samples[i] = cv2.equalizeHist(img_dila*train_data.samples[i])

test_data = helper.Mat_Dataset(datasets,["Base"],["S10","S11","S12","S13"])

for i in range(len(test_data.samples)):
    
    test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
    _,thresh1=cv2.threshold(test_data.samples[i],0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_eros = cv2.erode(thresh1,kernel,iterations=1)
    img_dila = cv2.dilate(img_eros,kernel,iterations=1)
    test_data.samples[i]= cv2.equalizeHist(test_data.samples[i]*img_dila)

x_train = train_data.samples.astype('float32')
x_test = test_data.samples.astype('float32')

x_train /= 255
x_test /= 255

y_train = to_categorical(train_data.labels, 6)
y_test = to_categorical(test_data.labels, 6)
random.seed(0)
(x_train,y_train) = shuffle(x_train,y_train)
random.seed(1)
(x_test,y_test) = shuffle(x_test,y_test)

inputs = Input(shape=(64, 32,))

permute = Permute((2,1))(inputs)

flattened_inputs = Flatten()(permute)

flattened_inputs = Lambda(lambda x : x[:,128: 1920])(flattened_inputs)

x1_1  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs)
x2_1  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs)
x3_1  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs)
x4_1  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs)
x5_1  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs)
x6_1  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs)
x7_1  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs)
x8_1  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs)
x9_1  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs)
x10_1  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs)
x11_1  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs)
x12_1  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs)
x13_1  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs)
x14_1  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs)
x15_1  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs)
x16_1  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs)

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

x1_1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
x2_1 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
x3_1 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
x4_1 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

x1_1 = Tea(64)(x1_1)
x2_1 = Tea(64)(x2_1)
x3_1 = Tea(64)(x3_1)
x4_1 = Tea(64)(x4_1)

x_out = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])

x_out = Tea(252)(x_out)

x_out = AdditivePooling(6)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          verbose=1,
          validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)
model.summary()
model.save_weights("blabla_2")

print('Test loss:', score[0])
print('Test accuracy:', score[1])

inputs = Input(shape=(64, 32,))
# print(inputs)
permute = Permute((2,1))(inputs)
# print(permute)
flattened_inputs = Flatten()(permute)

flattened_inputs = Lambda(lambda x : x[:,128: 1920])(flattened_inputs)

x1_1  = Lambda(lambda x : x[:,   3 : 259 ])(flattened_inputs)
x2_1  = Lambda(lambda x : x[:, 105 : 361 ])(flattened_inputs)
x3_1  = Lambda(lambda x : x[:, 207 : 463 ])(flattened_inputs)
x4_1  = Lambda(lambda x : x[:, 309 : 565 ])(flattened_inputs)
x5_1  = Lambda(lambda x : x[:, 411 : 667 ])(flattened_inputs)
x6_1  = Lambda(lambda x : x[:, 513 : 769 ])(flattened_inputs)
x7_1  = Lambda(lambda x : x[:, 615 : 871 ])(flattened_inputs)
x8_1  = Lambda(lambda x : x[:, 717 : 973 ])(flattened_inputs)
x9_1  = Lambda(lambda x : x[:, 819 : 1075])(flattened_inputs)
x10_1  = Lambda(lambda x : x[:,921 : 1177])(flattened_inputs)
x11_1  = Lambda(lambda x : x[:,1023: 1279])(flattened_inputs)
x12_1  = Lambda(lambda x : x[:,1125: 1381])(flattened_inputs)
x13_1  = Lambda(lambda x : x[:,1227: 1483])(flattened_inputs)
x14_1  = Lambda(lambda x : x[:,1329: 1585])(flattened_inputs)
x15_1  = Lambda(lambda x : x[:,1431: 1687])(flattened_inputs)
x16_1  = Lambda(lambda x : x[:,1533: 1789])(flattened_inputs)

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

x1_1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
x2_1 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
x3_1 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
x4_1 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

x1_1 = Tea(64)(x1_1)
x2_1 = Tea(64)(x2_1)
x3_1 = Tea(64)(x3_1)
x4_1 = Tea(64)(x4_1)

x_out = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])

x_out = Tea(252)(x_out)

x_out = AdditivePooling(6)(x_out)

predictions = Activation('softmax')(x_out)

saved_model= Model(inputs=inputs, outputs=predictions)

saved_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

saved_model.load_weights("blabla")
score = saved_model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])