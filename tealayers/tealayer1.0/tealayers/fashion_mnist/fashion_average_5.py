from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math

from numpy.core.fromnumeric import mean

import tensorflow as tf
# from tensorflow import squeeze
import numpy as np
# from keras import backend as K
from keras import Model
from keras.layers import Flatten, Activation, Input, Lambda,Average, Add,BatchNormalization ,Concatenate
from keras.datasets import fashion_mnist,mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append("../../../../rancutils/rancutils")
sys.path.append("../")

from tea import Tea
from additivepooling import AdditivePooling
import random
from sklearn.utils import shuffle

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet
from fashion import Fashion
from emulation import write_cores
import cv2

# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train_copy = np.empty_like(x_train)
x_train_copy[:,:,:]=x_train

x_test_copy = np.empty_like(x_test)
x_test_copy[:,:,:]=x_test

del x_train
del x_test

train_data = x_train_copy
test_data = x_test_copy
print(train_data[0].shape)

x_train = []
for i in range(len(train_data)):
    mask = np.ones_like(train_data[i])
    
    e_1 = np.array(train_data[i]>=mask*25).astype(float)
    # e_2 = np.array(train_data[i]>=mask*50).astype(float)
    e_2 = np.array(train_data[i]>=mask*75).astype(float)
    # e_4 = np.array(train_data[i]>=mask*100).astype(float)
    e_3 = np.array(train_data[i]>=mask*125).astype(float)
    # e_6 = np.array(train_data[i]>=mask*150).astype(float)
    e_4 = np.array(train_data[i]>=mask*175).astype(float)
    # e_8 = np.array(train_data[i]>=mask*200).astype(float)
    e_5 = np.array(train_data[i]>=mask*225).astype(float)

    x_train.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                    e_4[:,:,np.newaxis],e_5[:,:,np.newaxis]),axis=2))


x_test = []
for i in range(len(test_data)):
    mask = np.ones_like(test_data[i])
    
    e_1 = np.array(test_data[i]>=mask*25).astype(float)
    # e_2 = np.array(test_data[i]>=mask*50).astype(float)
    e_2 = np.array(test_data[i]>=mask*75).astype(float)
    # e_4 = np.array(test_data[i]>=mask*100).astype(float)
    e_3 = np.array(test_data[i]>=mask*125).astype(float)
    # e_6 = np.array(test_data[i]>=mask*150).astype(float)
    e_4 = np.array(test_data[i]>=mask*175).astype(float)
    # e_8 = np.array(test_data[i]>=mask*200).astype(float)
    e_5 = np.array(test_data[i]>=mask*225).astype(float)


    x_test.append(np.concatenate((e_1[:,:,np.newaxis],e_2[:,:,np.newaxis],e_3[:,:,np.newaxis],\
                                    e_4[:,:,np.newaxis],e_5[:,:,np.newaxis]),axis=2))
x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

random.seed(1)
(x_train,y_train) = shuffle(x_train,y_train)
random.seed(1)
(x_test,y_test) = shuffle(x_test,y_test)

tea_0_1 = Tea(64)
tea_0_2 = Tea(64)
tea_0_3 = Tea(64)
tea_0_4 = Tea(64)
tea_0_5 = Tea(64)
tea_0_6 = Tea(64)
tea_0_7 = Tea(64)
tea_0_8 = Tea(64)
tea_0_9 = Tea(64)
tea_0_10 = Tea(64)
tea_0_11 = Tea(64)
tea_0_12 = Tea(64)
tea_0_13 = Tea(64)
tea_0_14 = Tea(64)
tea_0_15 = Tea(64)
tea_0_16 = Tea(64)

tea_1_1 = Tea(64)
tea_1_2 = Tea(64)
tea_1_3 = Tea(64)
tea_1_4 = Tea(64)

tea_2_1 = Tea(250)

inputs = Input(shape=(28, 28, 5,))
flattened_inputs = Flatten()(inputs)

flattened_inputs_1 = Lambda(lambda x : x[:,     :1*784])(flattened_inputs)
flattened_inputs_2 = Lambda(lambda x : x[:,1*784:2*784])(flattened_inputs)
flattened_inputs_3 = Lambda(lambda x : x[:,2*784:3*784])(flattened_inputs)
flattened_inputs_4 = Lambda(lambda x : x[:,3*784:4*784])(flattened_inputs)
flattened_inputs_5 = Lambda(lambda x : x[:,4*784:5*784])(flattened_inputs)
flattened_inputs_6 = Lambda(lambda x : x[:,5*784:6*784])(flattened_inputs)
flattened_inputs_7 = Lambda(lambda x : x[:,6*784:7*784])(flattened_inputs)
flattened_inputs_8 = Lambda(lambda x : x[:,7*784:8*784])(flattened_inputs)
flattened_inputs_9 = Lambda(lambda x : x[:,8*784:      ])(flattened_inputs)

x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_1)
x2_1  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_1)
x3_1  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_1)
x4_1  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_1)
x5_1  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_1)
x6_1  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_1)
x7_1  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_1)
x8_1  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_1)
x9_1  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_1)
x10_1  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_1)
x11_1  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_1)
x12_1  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_1)
x13_1  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_1)
x14_1  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_1)
x15_1  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_1)
x16_1  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_1)

x1_1   = tea_0_1(x1_1)
x2_1   = tea_0_2(x2_1)
x3_1   = tea_0_3(x3_1)
x4_1   = tea_0_4(x4_1)
x5_1   = tea_0_5(x5_1)
x6_1   = tea_0_6(x6_1)
x7_1   = tea_0_7(x7_1)
x8_1   = tea_0_8(x8_1)
x9_1   = tea_0_9(x9_1)
x10_1  = tea_0_10(x10_1)
x11_1  = tea_0_11(x11_1)
x12_1  = tea_0_12(x12_1)
x13_1  = tea_0_13(x13_1)
x14_1  = tea_0_14(x14_1)
x15_1  = tea_0_15(x15_1)
x16_1  = tea_0_16(x16_1)

x1_2  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_2)
x2_2  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_2)
x3_2  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_2)
x4_2  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_2)
x5_2  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_2)
x6_2  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_2)
x7_2  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_2)
x8_2  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_2)
x9_2  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_2)
x10_2  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_2)
x11_2  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_2)
x12_2  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_2)
x13_2  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_2)
x14_2  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_2)
x15_2  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_2)
x16_2  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_2)

x1_2  = tea_0_1(x1_2)
x2_2  = tea_0_2(x2_2)
x3_2  = tea_0_3(x3_2)
x4_2  = tea_0_4(x4_2)
x5_2  = tea_0_5(x5_2)
x6_2  = tea_0_6(x6_2)
x7_2  = tea_0_7(x7_2)
x8_2  = tea_0_8(x8_2)
x9_2  = tea_0_9(x9_2)
x10_2  = tea_0_10(x10_2)
x11_2  = tea_0_11(x11_2)
x12_2  = tea_0_12(x12_2)
x13_2  = tea_0_13(x13_2)
x14_2  = tea_0_14(x14_2)
x15_2  = tea_0_15(x15_2)
x16_2  = tea_0_16(x16_2)

x1_3  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_3)
x2_3  = Lambda(lambda x : x[:, 35:291 ])(flattened_inputs_3)
x3_3  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_3)
x4_3  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_3)
x5_3  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_3)
x6_3  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_3)
x7_3  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_3)
x8_3  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_3)
x9_3  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_3)
x10_3  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_3)
x11_3  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_3)
x12_3  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_3)
x13_3  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_3)
x14_3  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_3)
x15_3  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_3)
x16_3  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_3)

x1_3  = tea_0_1(x1_3)
x2_3  = tea_0_2(x2_3)
x3_3  = tea_0_3(x3_3)
x4_3  = tea_0_4(x4_3)
x5_3  = tea_0_5(x5_3)
x6_3  = tea_0_6(x6_3)
x7_3  = tea_0_7(x7_3)
x8_3  = tea_0_8(x8_3)
x9_3  = tea_0_9(x9_3)
x10_3  = tea_0_10(x10_3)
x11_3  = tea_0_11(x11_3)
x12_3  = tea_0_12(x12_3)
x13_3  = tea_0_13(x13_3)
x14_3  = tea_0_14(x14_3)
x15_3  = tea_0_15(x15_3)
x16_3  = tea_0_16(x16_3)

x1_4  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_4)
x2_4  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_4)
x3_4  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_4)
x4_4  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_4)
x5_4  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_4)
x6_4  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_4)
x7_4  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_4)
x8_4  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_4)
x9_4  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_4)
x10_4  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_4)
x11_4  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_4)
x12_4  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_4)
x13_4  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_4)
x14_4  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_4)
x15_4  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_4)
x16_4  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_4)

x1_4  = tea_0_1(x1_4)
x2_4  = tea_0_2(x2_4)
x3_4  = tea_0_3(x3_4)
x4_4  = tea_0_4(x4_4)
x5_4  = tea_0_5(x5_4)
x6_4  = tea_0_6(x6_4)
x7_4  = tea_0_7(x7_4)
x8_4  = tea_0_8(x8_4)
x9_4  = tea_0_9(x9_4)
x10_4  = tea_0_10(x10_4)
x11_4  = tea_0_11(x11_4)
x12_4  = tea_0_12(x12_4)
x13_4  = tea_0_13(x13_4)
x14_4  = tea_0_14(x14_4)
x15_4  = tea_0_15(x15_4)
x16_4  = tea_0_16(x16_4)

x1_5  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_5)
x2_5  = Lambda(lambda x : x[:, 35:291])(flattened_inputs_5)
x3_5  = Lambda(lambda x : x[:, 70:326 ])(flattened_inputs_5)
x4_5  = Lambda(lambda x : x[:, 105:361])(flattened_inputs_5)
x5_5  = Lambda(lambda x : x[:, 140:396])(flattened_inputs_5)
x6_5  = Lambda(lambda x : x[:, 175:431])(flattened_inputs_5)
x7_5  = Lambda(lambda x : x[:, 210:466])(flattened_inputs_5)
x8_5  = Lambda(lambda x : x[:, 245:501])(flattened_inputs_5)
x9_5  = Lambda(lambda x : x[:, 280:536])(flattened_inputs_5)
x10_5  = Lambda(lambda x : x[:, 315:571])(flattened_inputs_5)
x11_5  = Lambda(lambda x : x[:, 350:606])(flattened_inputs_5)
x12_5  = Lambda(lambda x : x[:, 385:641])(flattened_inputs_5)
x13_5  = Lambda(lambda x : x[:, 420:676])(flattened_inputs_5)
x14_5  = Lambda(lambda x : x[:, 455:711])(flattened_inputs_5)
x15_5  = Lambda(lambda x : x[:, 490:746])(flattened_inputs_5)
x16_5  = Lambda(lambda x : x[:, 525:781])(flattened_inputs_5)

x1_5  = tea_0_1(x1_5)
x2_5  = tea_0_2(x2_5)
x3_5  = tea_0_3(x3_5)
x4_5  = tea_0_4(x4_5)
x5_5  = tea_0_5(x5_5)
x6_5  = tea_0_6(x6_5)
x7_5  = tea_0_7(x7_5)
x8_5  = tea_0_8(x8_5)
x9_5  = tea_0_9(x9_5)
x10_5  = tea_0_10(x10_5)
x11_5  = tea_0_11(x11_5)
x12_5  = tea_0_12(x12_5)
x13_5  = tea_0_13(x13_5)
x14_5  = tea_0_14(x14_5)
x15_5  = tea_0_15(x15_5)
x16_5  = tea_0_16(x16_5)

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

x1_1_2 = Average()([x1_2,x1_3,x1_4])
x2_1_2 = Average()([x2_2,x2_3,x2_4])
x3_1_2 = Average()([x3_2,x3_3,x3_4])
x4_1_2 = Average()([x4_2,x4_3,x4_4])
x5_1_2 = Average()([x5_2,x5_3,x5_4])
x6_1_2 = Average()([x6_2,x6_3,x6_4])
x7_1_2 = Average()([x7_2,x7_3,x7_4])
x8_1_2 = Average()([x8_2,x8_3,x8_4])
x9_1_2 = Average()([x9_2,x9_3,x9_4])
x10_1_2 = Average()([x10_2,x10_3,x10_4])
x11_1_2 = Average()([x11_2,x11_3,x11_4])
x12_1_2 = Average()([x12_2,x12_3,x12_4])
x13_1_2 = Average()([x13_2,x13_3,x13_4])
x14_1_2 = Average()([x14_2,x14_3,x14_4])
x15_1_2 = Average()([x15_2,x15_3,x15_4])
x16_1_2 = Average()([x16_2,x16_3,x16_4])

x1_1_3 = Average()([x1_3,x1_4,x1_5])
x2_1_3 = Average()([x2_3,x2_4,x2_5])
x3_1_3 = Average()([x3_3,x3_4,x3_5])
x4_1_3 = Average()([x4_3,x4_4,x4_5])
x5_1_3 = Average()([x5_3,x5_4,x5_5])
x6_1_3 = Average()([x6_3,x6_4,x6_5])
x7_1_3 = Average()([x7_3,x7_4,x7_5])
x8_1_3 = Average()([x8_3,x8_4,x8_5])
x9_1_3 = Average()([x9_3,x9_4,x9_5])
x10_1_3 = Average()([x10_3,x10_4,x10_5])
x11_1_3 = Average()([x11_3,x11_4,x11_5])
x12_1_3 = Average()([x12_3,x12_4,x12_5])
x13_1_3 = Average()([x13_3,x13_4,x13_5])
x14_1_3 = Average()([x14_3,x14_4,x14_5])
x15_1_3 = Average()([x15_3,x15_4,x15_5])
x16_1_3 = Average()([x16_3,x16_4,x16_5])

x1_1 = Concatenate(axis=1)([x1_1_1,x2_1_1,x3_1_1,x4_1_1])
x2_1 = Concatenate(axis=1)([x5_1_1,x6_1_1,x7_1_1,x8_1_1])
x3_1 = Concatenate(axis=1)([x9_1_1,x10_1_1,x11_1_1,x12_1_1])
x4_1 = Concatenate(axis=1)([x13_1_1,x14_1_1,x15_1_1,x16_1_1])

x1_2 = Concatenate(axis=1)([x1_1_2,x2_1_2,x3_1_2,x4_1_2])
x2_2 = Concatenate(axis=1)([x5_1_2,x6_1_2,x7_1_2,x8_1_2])
x3_2 = Concatenate(axis=1)([x9_1_2,x10_1_2,x11_1_2,x12_1_2])
x4_2 = Concatenate(axis=1)([x13_1_2,x14_1_2,x15_1_2,x16_1_2])

x1_3 = Concatenate(axis=1)([x1_1_3,x2_1_3,x3_1_3,x4_1_3])
x2_3 = Concatenate(axis=1)([x5_1_3,x6_1_3,x7_1_3,x8_1_3])
x3_3 = Concatenate(axis=1)([x9_1_3,x10_1_3,x11_1_3,x12_1_3])
x4_3 = Concatenate(axis=1)([x13_1_3,x14_1_3,x15_1_3,x16_1_3])

x1_1 = tea_1_1(x1_1)
x2_1 = tea_1_2(x2_1)
x3_1 = tea_1_3(x3_1)
x4_1 = tea_1_4(x4_1)

x1_2 = tea_1_1(x1_2)
x2_2 = tea_1_2(x2_2)
x3_2 = tea_1_3(x3_2)
x4_2 = tea_1_4(x4_2)

x1_3 = tea_1_1(x1_3)
x2_3 = tea_1_2(x2_3)
x3_3 = tea_1_3(x3_3)
x4_3 = tea_1_4(x4_3)

x_out_1 = Average()([x1_1,x1_2,x1_3])
x_out_2 = Average()([x2_1,x2_2,x2_3])
x_out_3 = Average()([x3_1,x3_2,x3_3])
x_out_4 = Average()([x4_1,x4_2,x4_3])

x_out = Concatenate(axis=1)([x_out_1,x_out_2,x_out_3,x_out_4])

x_out = tea_2_1(x_out)

x_out = AdditivePooling(10)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

checkpoint_filepath = './ckpt_fashion/10_class_fashion'

import keras 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath + '-epoch-{epoch}',
    save_weights_only=True,)
# model.load_weights('./ckpt_fashion/10_class_fashion-epoch-7')
model.fit(x_train, y_train,
          batch_size=256,
          epochs=30,
          verbose=1,
          callbacks=[model_checkpoint_callback],
          validation_split=0.2)

import os
scores = []
soure = "./ckpt_fashion"
ckpts = [os.path.join(soure,e) for e in os.listdir(soure)]
for ckpt in ckpts:
    print("======================================")
    print(ckpt)
    print("======================================")
    model.load_weights(ckpt)      
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    scores.append(score[1])

print("Max accuracy:",max(scores))
print("Best epoch:",ckpts[scores.index(max(scores))])

# cores_sim = create_cores(model, 20,neuron_reset_type=0 ) 
# write_cores(cores_sim,output_path="/home/phuongdh/Documents/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/out_new")