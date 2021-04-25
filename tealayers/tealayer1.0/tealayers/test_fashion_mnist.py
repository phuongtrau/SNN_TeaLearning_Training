from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math

import tensorflow as tf
from tensorflow import squeeze
import numpy as np
from keras import backend as K
from keras import Model
from keras.engine.topology import Layer
from keras import initializers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, AveragePooling1D, Reshape, Concatenate, MaxPooling1D,Average 
from keras.datasets import fashion_mnist
from keras.optimizers import Adam,SGD
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tea import Tea
from additivepooling import AdditivePooling

import sys
sys.path.append("/home/hoangphuong/Documents/FPGA/SNN_TeaLearning_Training/rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train_1 =  x_train / 255
x_test_1  = x_test / 255

x_train_2 = x_train
x_test_2 = x_test
for ele in x_train_2:
        for i in range(28):
                for j in range(28):
                        if ele[i][j] <= 128:
                                ele[i][j]= ele[i][j] / 128
                        else:
                                ele[i][j]=(ele[i][j]-128)/128 

for ele in x_test_2:
        for i in range(28):
                for j in range(28):
                        if ele[i][j] <= 128:
                                ele[i][j]= ele[i][j] / 128
                        else:
                                ele[i][j]=(ele[i][j]-128)/128

x_train = np.concatenate([x_train_1,x_train_2],axis=0)
x_test = np.concatenate([x_test_1,x_test_2],axis=0)
y_train = np.concatenate([y_train,y_train],axis=0)
y_test = np.concatenate([y_test,y_test],axis=0)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# x_train = datagen.flow(x_train[:,:,:,np.newaxis],y_train)


# save old labels for later
# y_test_not = y_test

# convert class vectors to binary class matrices


# Define model (use functional API to follow fan-in and 
# fan-out constraints)
inputs = Input(shape=(28, 28,))

flattened_inputs = Flatten()(inputs)
# Send input into 4 different cores (256 axons each)
x0 = Lambda(lambda x : x[:,:256])(flattened_inputs)
x0_res = Reshape((256,1)) (x0)
x0_res  = AveragePooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x0_res)
x0_res  = Lambda(lambda x : squeeze(x,2))(x0_res)

x1 = Lambda(lambda x : x[:,176:432])(flattened_inputs)
x1_res = Reshape((256,1)) (x1)
x1_res  = AveragePooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x1_res)
x1_res  = Lambda(lambda x : squeeze(x,2))(x1_res)

x2 = Lambda(lambda x : x[:,352:608])(flattened_inputs)
x2_res = Reshape((256,1)) (x2)
x2_res  = AveragePooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x2_res)
x2_res  = Lambda(lambda x : squeeze(x,2))(x2_res)

x3 = Lambda(lambda x : x[:,528:])(flattened_inputs)
x3_res = Reshape((256,1)) (x3)
x3_res  = AveragePooling1D(pool_size=2, strides=2, padding="valid", data_format="channels_last")(x3_res)
x3_res  = Lambda(lambda x : squeeze(x,2))(x3_res)

x0 = Tea(128)(x0)
x0 = Average()([x0,x0_res])

x1 = Tea(128)(x1)
x1 = Average()([x1,x1_res])

x2 = Tea(128)(x2)
x2 = Average()([x2,x2_res])

x3 = Tea(128)(x3)
x3 = Average()([x3,x3_res])
# print(x0,x1,x2,x3)
# Concatenate output of first layer to send into next

x2_1 = Concatenate(axis=1)([x0,x1])
x2_1 = Tea(128)(x2_1)
x2_1 = Average()([x2_1,x0,x1])

x2_2 = Concatenate(axis=1)([x2,x3])
x2_2 = Tea(128)(x2_2)
x2_2 = Average()([x2_2,x2,x3])

x2_3 = Concatenate(axis=1)([x0,x2])
x2_3 = Tea(128)(x2_3)
x2_3 = Average()([x2_3,x0,x2])

x2_4 = Concatenate(axis=1)([x1,x3])
x2_4 = Tea(128)(x2_4)
x2_4 = Average()([x2_4,x3,x1])

x2_5 = Concatenate(axis=1)([x0,x3])
x2_5 = Tea(128)(x2_5)
x2_5 = Average()([x2_5,x0,x3])

x2_6 = Concatenate(axis=1)([x2,x1])
x2_6 = Tea(128)(x2_6)
x2_6 = Average()([x2_6,x2,x1])

x3_1 = Concatenate(axis=1)([x2_1,x2_2])
x3_1 = Tea(128)(x3_1)
x3_1 = Average()([x3_1,x2_1,x2_2])

x3_2 = Concatenate(axis=1)([x2_3,x2_4])
x3_2 = Tea(128)(x3_2)
x3_2 = Average()([x3_2,x2_3,x2_4])

x3_3 = Concatenate(axis=1)([x2_5,x2_6])
x3_3 = Tea(128)(x3_3)
x3_3 = Average()([x3_3,x2_5,x2_6])

x4_1 = Concatenate(axis=1)([x3_1,x3_2])
x4_1 = Tea(250)(x4_1)
x4_2 = Concatenate(axis=1)([x3_1,x3_3])
x4_2 = Tea(250)(x4_2)
x4_3 = Concatenate(axis=1)([x3_2,x3_3])
x4_3 = Tea(250)(x4_3)


x_out = Concatenate(axis=1)([x4_1,x4_2,x4_3]) 
# print(x_out)
x_out = Reshape((750,1)) (x_out)
x_out  = AveragePooling1D(pool_size=3, strides=3, padding="valid", data_format="channels_last")(x_out)
x_out  = Lambda(lambda x : squeeze(x,2))(x_out)

# Pool spikes and output neurons into 10 classes.
x_out = AdditivePooling(10)(x_out)

predictions = Activation('softmax')(x_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr= 0.01,beta_1 = 0.99,beta_2 = 0.9999,epsilon = 1e-07,amsgrad = True),
              metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=128,epochs=50,verbose=1,validation_split=0.2)
# model.save("mnist_12_3_1.h5")
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



# cores_sim = create_cores(model, 5 ,neuron_reset_type=0) 
# # weights = []
# # for layer in model.layers:
# #     weights.append(np.reshape(layer.get_weights(),(256,-1)))
# connections, biases = get_connections_and_biases(model,5)
# # print(connections)
# # print(connections)
# for i in range(len(connections)):
#     connections[i]= np.reshape(connections[i],(256,-1))
#     # print(connections[i])
#     # connections[i] = np.round(connections[i])
#     # print(connections[i])
#     # connections[i]= tf.keras.layers.reshape((256,-1))(connections[i])
    
#     connections[i]= K.clip(connections[i], 0, 1)
#     connections[i]= K.round(connections[i])



# x0 = Lambda(lambda x : x[:,:256])(flattened_inputs)
# x1 = Lambda(lambda x : x[:,176:432])(flattened_inputs)
# x2 = Lambda(lambda x : x[:,352:608])(flattened_inputs)
# x3 = Lambda(lambda x : x[:,528:])(flattened_inputs)
# x0 = Tea(64,init_connection=connections[0])(x0)
# x1 = Tea(64,init_connection=connections[1])(x1)
# x2 = Tea(64,init_connection=connections[2])(x2)
# x3 = Tea(64,init_connection=connections[3])(x3)

# # Concatenate output of first layer to send into next
# x = concatenate([x0, x1, x2, x3])
# x = Tea(250,init_connection=connections[4])(x)

# # Pool spikes and output neurons into 10 classes.
# x = AdditivePooling(10)(x)
# predictions = Activation('softmax')(x)

# model_1 = Model(inputs=inputs, outputs=predictions)

# model_1.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

# model_1.fit(x_train, y_train,
#           batch_size=128,
#           epochs=5,
#           verbose=1,
#           validation_split=0.2)

# score_1 = model_1.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score_1[0])
# print('Test accuracy:', score_1[1])

# from output_bus import OutputBus
# from serialization import save as sim_save
# # from rancutils.teaconversion import create_cores, create_packets, Packet
# # 

# x_test_flat = x_test.reshape((10000, 784))
# partitioned_packets = []

# # Use absolute/hard reset by specifying neuron_reset_type=0
# cores_sim = create_cores(model, 2, neuron_reset_type=0) 
# # Partition the packets into groups as they will be fed into each of the input cores
# partitioned_packets.append(x_test_flat[:, :256])
# partitioned_packets.append(x_test_flat[:, 176:432])
# partitioned_packets.append(x_test_flat[:, 352:608])
# partitioned_packets.append(x_test_flat[:, 528:])
# packets_sim = create_packets(partitioned_packets)
# output_bus_sim = OutputBus((0, 2), num_outputs=250)

# This file can then be used as an input json to the RANC Simulator through the "input file" argument.
# sim_save("mnist_config.json", cores_sim, packets_sim, output_bus_sim, indent=2)


# Additionally, output the tensorflow predictions and correct labels for later cross validation
# np.save("mnist_tf_preds.txt", test_predictions)
# np.save("mnist_correct_preds.txt", y_test)

# # TODO: Add usage example for outputting to emulation via rancutils.emulation.write_cores, et