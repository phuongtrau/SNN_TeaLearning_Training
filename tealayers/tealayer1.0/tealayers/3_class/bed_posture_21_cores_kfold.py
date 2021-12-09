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
sys.path.append("../../../../rancutils/rancutils")
import os

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet
sys.path.append("../")
# from tea import Tea
from additivepooling import AdditivePooling
import helper
from tea import Tea
import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

exp_i_data = helper.load_exp_i_short("../../dataset/experiment-i")

# print(len(dataset))
datasets = {"Base":exp_i_data}
subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]
train_data = helper.Mat_Dataset(datasets,["Base"],subjects)

# Define per-fold score containers
for i in range(len(train_data.samples)):
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])

x_train = train_data.samples.astype('float32')

x_train /= 255

y_train = to_categorical(train_data.labels, 3)

random.seed(2)
(x_train,y_train) = shuffle(x_train,y_train)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits = 5, shuffle = False)
fold_no = 1
for train,test in kfold.split(x_train,y_train):
      
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
  x_out = AdditivePooling(3)(x_out)
  
  predictions = Activation('softmax')(x_out)

  model = Model(inputs=inputs, outputs=predictions)

  model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0005),
              metrics=['accuracy'])

  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  model.fit(x_train[train], y_train[train],
            batch_size=64,
            epochs=30,
            verbose=1,
            validation_split=0.2)

  score = model.evaluate(x_train[test], y_train[test], verbose=0)
  
  acc_per_fold.append(score[1] * 100)
  loss_per_fold.append(score[0])
  
  print('Test loss:', score[0])
  print('Test accuracy:', score[1]*100,'%')

  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
