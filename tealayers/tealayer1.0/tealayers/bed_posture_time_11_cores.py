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

for i in range(len(train_data.samples)):
    train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])
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

y_train = to_categorical(train_data.labels, 17)
y_test = to_categorical(test_data.labels, 17)

# random.seed(0)
(x_train,y_train) = shuffle(x_train,y_train)
# print(x_train_s,y_train_s)

(x_test,y_test) = shuffle(x_test,y_test)
# print(x_train_s,y_train_s)


time_win = 5
filter_win = []
x_tr = []
x_ts = []
# for i in range(time_win):
filter_wins = []

for i in range(time_win):
    filter_win = np.random.poisson(lam= 1.85+i*0.05,size=(64,32))
    filter_wins.append(filter_win)
    cv2.imwrite(str(i)+".jpg",filter_win*255)

# print(filter_win.shape)

for ele in x_train:
    # x_tr.append(x_train)
    eles = []
    for filter_win in filter_wins:
        encode = (ele > filter_win).astype('float32') 
        eles.append(encode*ele)
    for i in range(4):
        eles[i+1] = eles[i+1] + eles[i]
        eles[i+1] = eles[i+1] + eles[i+1] * (eles[i+1] > np.ones((64,32))).astype('float32')  
    ele = np.array(eles)
    # print(ele.shape)
    ele = np.moveaxis(ele,0,-1)
    # print(ele.shape)
    x_tr.append(ele)

for ele in x_test:
    eles = []
    for filter_win in filter_wins:
        encode = (ele > filter_win).astype('float32') 
        eles.append(encode*ele)
    for i in range(4):
        eles[i+1] = eles[i+1] + eles[i]
        eles[i+1] = eles[i+1] + eles[i+1] * (eles[i+1] > np.ones((64,32))).astype('float32')
    ele = np.array(eles)
    ele = np.moveaxis(ele, 0, -1)
    x_ts.append(ele)

x_tr = np.array(x_tr)
# print(x_tr[0][:,:,4].shape)
# cv2.imwrite("time.jpg",x_tr[0][:,:,4]*255)
x_ts = np.array(x_ts)


inputs = Input(shape=(64, 32, 5,))

flattened_inputs = Flatten()(inputs)
flattened_inputs_1 = Lambda(lambda x : x[:,:2048])(flattened_inputs)
flattened_inputs_2 = Lambda(lambda x : x[:,2048:4096])(flattened_inputs)
flattened_inputs_3 = Lambda(lambda x : x[:,4096:6144])(flattened_inputs)
flattened_inputs_4 = Lambda(lambda x : x[:,6144:8192])(flattened_inputs)
flattened_inputs_5 = Lambda(lambda x : x[:,8192:])(flattened_inputs)

x1_1  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_1)
x2_1  = Lambda(lambda x : x[:, 256 :512 ])(flattened_inputs_1)
x3_1  = Lambda(lambda x : x[:, 512 :768 ])(flattened_inputs_1)
x4_1  = Lambda(lambda x : x[:, 768 :1024])(flattened_inputs_1)
x5_1  = Lambda(lambda x : x[:, 1024:1280])(flattened_inputs_1)
x6_1  = Lambda(lambda x : x[:, 1280:1536])(flattened_inputs_1)
x7_1  = Lambda(lambda x : x[:, 1536:1792])(flattened_inputs_1)
x8_1  = Lambda(lambda x : x[:, 1792:    ])(flattened_inputs_1)

x1_1  = Tea(64)(x1_1)
x2_1  = Tea(64)(x2_1)
x3_1  = Tea(64)(x3_1)
x4_1  = Tea(64)(x4_1)
x5_1  = Tea(64)(x5_1)
x6_1  = Tea(64)(x6_1)
x7_1  = Tea(64)(x7_1)
x8_1  = Tea(64)(x8_1)

x1_1_1 = Concatenate(axis=1)([x1_1,x2_1,x3_1,x4_1])

x2_1_1 = Concatenate(axis=1)([x5_1,x6_1,x7_1,x8_1])

x1_1 = Tea(128)(x1_1_1)
x2_1 = Tea(128)(x2_1_1)

x_out_1 = Concatenate(axis=1)([x1_1,x2_1])

x_out_1 = Tea(255)(x_out_1)

x1_2  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_2)
x2_2  = Lambda(lambda x : x[:, 256 :512 ])(flattened_inputs_2)
x3_2  = Lambda(lambda x : x[:, 512 :768 ])(flattened_inputs_2)
x4_2  = Lambda(lambda x : x[:, 768 :1024])(flattened_inputs_2)
x5_2  = Lambda(lambda x : x[:, 1024:1280])(flattened_inputs_2)
x6_2  = Lambda(lambda x : x[:, 1280:1536])(flattened_inputs_2)
x7_2  = Lambda(lambda x : x[:, 1536:1792])(flattened_inputs_2)
x8_2  = Lambda(lambda x : x[:, 1792:    ])(flattened_inputs_2)

x1_2  = Tea(64)(x1_2)
x2_2  = Tea(64)(x2_2)
x3_2  = Tea(64)(x3_2)
x4_2  = Tea(64)(x4_2)
x5_2  = Tea(64)(x5_2)
x6_2  = Tea(64)(x6_2)
x7_2  = Tea(64)(x7_2)
x8_2  = Tea(64)(x8_2)

x1_2_2 = Concatenate(axis=1)([x1_2,x2_2,x3_2,x4_2])

x2_2_2 = Concatenate(axis=1)([x5_2,x6_2,x7_2,x8_2])

x1_2 = Tea(128)(x1_2_2)
x2_2 = Tea(128)(x2_2_2)

x_out_2 = Concatenate(axis=1)([x1_2,x2_2])

x_out_2 = Tea(255)(x_out_2)

x1_3  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_3)
x2_3  = Lambda(lambda x : x[:, 256 :512 ])(flattened_inputs_3)
x3_3  = Lambda(lambda x : x[:, 512 :768 ])(flattened_inputs_3)
x4_3  = Lambda(lambda x : x[:, 768 :1024])(flattened_inputs_3)
x5_3  = Lambda(lambda x : x[:, 1024:1280])(flattened_inputs_3)
x6_3  = Lambda(lambda x : x[:, 1280:1536])(flattened_inputs_3)
x7_3  = Lambda(lambda x : x[:, 1536:1792])(flattened_inputs_3)
x8_3  = Lambda(lambda x : x[:, 1792:    ])(flattened_inputs_3)

x1_3  = Tea(64)(x1_3)
x2_3  = Tea(64)(x2_3)
x3_3  = Tea(64)(x3_3)
x4_3  = Tea(64)(x4_3)
x5_3  = Tea(64)(x5_3)
x6_3  = Tea(64)(x6_3)
x7_3  = Tea(64)(x7_3)
x8_3  = Tea(64)(x8_3)

x1_3_3 = Concatenate(axis=1)([x1_3,x2_3,x3_3,x4_3])

x2_3_3 = Concatenate(axis=1)([x5_3,x6_3,x7_3,x8_3])

x1_3 = Tea(128)(x1_3_3)
x2_3 = Tea(128)(x2_3_3)

x_out_3 = Concatenate(axis=1)([x1_3,x2_3])

x_out_3 = Tea(255)(x_out_3)

x1_4  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_4)
x2_4  = Lambda(lambda x : x[:, 256 :512 ])(flattened_inputs_4)
x3_4  = Lambda(lambda x : x[:, 512 :768 ])(flattened_inputs_4)
x4_4  = Lambda(lambda x : x[:, 768 :1024])(flattened_inputs_4)
x5_4  = Lambda(lambda x : x[:, 1024:1280])(flattened_inputs_4)
x6_4  = Lambda(lambda x : x[:, 1280:1536])(flattened_inputs_4)
x7_4  = Lambda(lambda x : x[:, 1536:1792])(flattened_inputs_4)
x8_4  = Lambda(lambda x : x[:, 1792:    ])(flattened_inputs_4)

x1_4  = Tea(64)(x1_4)
x2_4  = Tea(64)(x2_4)
x3_4  = Tea(64)(x3_4)
x4_4  = Tea(64)(x4_4)
x5_4  = Tea(64)(x5_4)
x6_4  = Tea(64)(x6_4)
x7_4  = Tea(64)(x7_4)
x8_4  = Tea(64)(x8_4)

x1_4_4 = Concatenate(axis=1)([x1_4,x2_4,x3_4,x4_4])

x2_4_4 = Concatenate(axis=1)([x5_4,x6_4,x7_4,x8_4])

x1_4 = Tea(128)(x1_4_4)
x2_4 = Tea(128)(x2_4_4)

x_out_4 = Concatenate(axis=1)([x1_4,x2_4])

x_out_4 = Tea(255)(x_out_4)

x1_5  = Lambda(lambda x : x[:,     :256 ])(flattened_inputs_5)
x2_5  = Lambda(lambda x : x[:, 256 :512 ])(flattened_inputs_5)
x3_5  = Lambda(lambda x : x[:, 512 :768 ])(flattened_inputs_5)
x4_5  = Lambda(lambda x : x[:, 768 :1024])(flattened_inputs_5)
x5_5  = Lambda(lambda x : x[:, 1024:1280])(flattened_inputs_5)
x6_5  = Lambda(lambda x : x[:, 1280:1536])(flattened_inputs_5)
x7_5  = Lambda(lambda x : x[:, 1536:1792])(flattened_inputs_5)
x8_5  = Lambda(lambda x : x[:, 1792:    ])(flattened_inputs_5)

x1_5  = Tea(64)(x1_5)
x2_5  = Tea(64)(x2_5)
x3_5  = Tea(64)(x3_5)
x4_5  = Tea(64)(x4_5)
x5_5  = Tea(64)(x5_5)
x6_5  = Tea(64)(x6_5)
x7_5  = Tea(64)(x7_5)
x8_5  = Tea(64)(x8_5)

x1_5_5 = Concatenate(axis=1)([x1_5,x2_5,x3_5,x4_5])

x2_5_5 = Concatenate(axis=1)([x5_5,x6_5,x7_5,x8_5])

x1_5 = Tea(128)(x1_5_5)
x2_5 = Tea(128)(x2_5_5)

x_out_5 = Concatenate(axis=1)([x1_5,x2_5])

x_out_5 = Tea(255)(x_out_5)

x_out = Average()([x_out_1,x_out_2,x_out_3,x_out_4,x_out_5])

x_out = AdditivePooling(17)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_tr, y_train,
          batch_size=64,
          epochs=20,
          verbose=1,
          validation_split=0.2)

score = model.evaluate(x_ts, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])




# weights , biases = get_connections_and_biases(model,11)

# from output_bus import OutputBus
# from serialization import save as sim_save
# from emulation import write_cores

# cores_sim = create_cores(model, 11,neuron_reset_type=0 ) 

# write_cores(cores_sim,output_path="/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/output_mem_bed_posture")