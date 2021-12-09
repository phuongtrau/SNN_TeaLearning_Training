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
# import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

exp_i_data = helper.load_exp_i_short("../../dataset/experiment-i",preprocess=False)

# print(len(dataset))
datasets = {"Base":exp_i_data}
subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]

# Define per-fold score containers
acc_per_so = []
loss_per_so = []
ls_train_full = subjects.copy()
for sub in ls_train_full:

  subjects.remove(sub)
  train_data = helper.Mat_Dataset(datasets,["Base"],subjects)
  test_data = helper.Mat_Dataset(datasets,["Base"],[sub])
  
  # for i in range(len(train_data.samples)):
  #   train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])
  # for i in range(len(test_data.samples)):
  #   test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])

  x_train = train_data.samples.astype('float32')
  x_test = test_data.samples.astype('float32')

  x_train /= 255
  x_test /=255
  y_train = to_categorical(train_data.labels, 3)
  y_test = to_categorical(test_data.labels,3)
  
  (x_train,y_train) = shuffle(x_train,y_train)

  (x_test,y_test) = shuffle(x_test,y_test)
      
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
              optimizer=Adam(),
              metrics=['accuracy'])

  print('------------------------------------------------------------------------')
  print(f'Training for subject {sub} ...')

  model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          verbose=1,)
          # validation_split=0.2)
  # model.load_weights("bed_posture/ckpt_3_classes_median/3_class-{}".format(sub))

  # model.load_weights("bed_posture/ckpt_3_classes/3_class-S8_1")
  

  score = model.evaluate(x_test, y_test, verbose=0)

  acc_per_so.append(score[1] * 100)
  loss_per_so.append(score[0])
  
  subjects = ls_train_full.copy()

  print('Test loss:', score[0])
  print('Test accuracy:', score[1]*100)
  # if not os.path.exists("./mem_files/3_classes_21_cores_mem/{}".format(sub)):
  #   os.makedirs("./mem_files/3_classes_21_cores_mem/{}".format(sub))

  # cores_sim = create_cores(model, 21 , neuron_reset_type=0,num_classes=4) 

  # write_cores(cores_sim,max_xy=(1,21),output_path="./mem_files/3_classes_21_cores_mem/{}".format(sub))

  # fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per subject out')
for i in range(0, len(acc_per_so)):
  print('------------------------------------------------------------------------')
  print(f'> Subject {i+1} - Loss: {loss_per_so[i]} - Accuracy: {acc_per_so[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all subject out:')
print(f'> Accuracy: {np.mean(acc_per_so)} (+- {np.std(acc_per_so)})')
print(f'> Loss: {np.mean(loss_per_so)}')
print('------------------------------------------------------------------------')

for i in range(0, len(acc_per_so)):
  print(f'{acc_per_so[i]}')
print(f'Accuracy: {np.mean(acc_per_so)}')

# weights , biases = get_connections_and_biases(model,11)

# from output_bus import OutputBus
# from serialization import save as sim_save
# from emulation import write_cores

# cores_sim = create_cores(model, 11,neuron_reset_type=0 ) 

# write_cores(cores_sim,output_path="/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/output_mem_bed_posture")