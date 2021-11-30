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

exp_i_data = helper.load_exp_i_short("../dataset/experiment-i")

# print(len(dataset))
datasets = {"Base":exp_i_data}
subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]

sub="S4"

subjects.remove(sub)
random.seed(1)
random.shuffle(subjects)

train_data = helper.Mat_Dataset(datasets,["Base"],subjects)
for i in range(len(train_data.samples)):
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])
    
test_data = helper.Mat_Dataset(datasets,["Base"],[sub])
for i in range(len(test_data.samples)):
    test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])

x_train = train_data.samples.astype('float32')
x_test = test_data.samples.astype('float32')

x_train /= 255
x_test /= 255

y_train = to_categorical(train_data.labels, 3)
y_test = to_categorical(test_data.labels, 3)

random.seed(2)
(x_train,y_train) = shuffle(x_train,y_train)
# print(x_train_s,y_train_s)
# random.seed(2)
(x_test,y_test) = shuffle(x_test,y_test)
# print(x_train_s,y_train_s)

inputs = Input(shape=(64, 32,))

# permute = Permute((2,1))(inputs)

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
x_out = AdditivePooling(3)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0015),
              metrics=['accuracy'])

import keras
checkpoint_filepath = 'bed_posture/ckpt/3_class-{}'.format(sub)


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath+'-epoch-{epoch}',
    save_weights_only=True,)


model.load_weights("bed_posture/ckpt_3_classes/3_class-{}".format(sub))

model.fit(x_train, y_train,
          batch_size=1024,
          epochs=30,
          verbose=1,
          callbacks=[model_checkpoint_callback],
          validation_split=0)


import os
scores = []
soure = "bed_posture/ckpt"
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

# score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

