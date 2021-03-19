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
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, concatenate
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import to_categorical

from tea import Tea
from additivepooling import AdditivePooling

import sys
sys.path.append("/home/hoangphuong/Documents/FPGA/SNN_TeaLearning_Training/rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# save old labels for later
y_test_not = y_test

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define model (use functional API to follow fan-in and 
# fan-out constraints)
inputs = Input(shape=(28, 28,))
flattened_inputs = Flatten()(inputs)
# Send input into 4 different cores (256 axons each)
x0 = Lambda(lambda x : x[:,:256])(flattened_inputs)
x1 = Lambda(lambda x : x[:,176:432])(flattened_inputs)
x2 = Lambda(lambda x : x[:,352:608])(flattened_inputs)
x3 = Lambda(lambda x : x[:,528:])(flattened_inputs)
x0 = Tea(64)(x0)
x1 = Tea(64)(x1)
x2 = Tea(64)(x2)
x3 = Tea(64)(x3)
# Concatenate output of first layer to send into next
x = concatenate([x0, x1, x2, x3])
x = Tea(250)(x)
# Pool spikes and output neurons into 10 classes.
x = AdditivePooling(10)(x)

predictions = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_split=0.2)

# for layer in model.layers:
#     print(layer)
#     print(layer.get_weights())

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# cores_sim = create_cores(model, 5 ,neuron_reset_type=0) 
weights , biases = get_connections_and_biases(model,5)
connections = []
for weight in weights:
    connections.append(np.clip(np.round(weight), 0, 1))
for i in range(len(connections)) :
    connections[i]=np.reshape(connections[i],(256,-1))
for e in connections:
    print(e)
    print(e.shape)


x0 = Lambda(lambda x : x[:,:256])(flattened_inputs)
x1 = Lambda(lambda x : x[:,176:432])(flattened_inputs)
x2 = Lambda(lambda x : x[:,352:608])(flattened_inputs)
x3 = Lambda(lambda x : x[:,528:])(flattened_inputs)
x0 = Tea(64,init_connection=connections[0])(x0)
x1 = Tea(64,init_connection=connections[1])(x1)
x2 = Tea(64,init_connection=connections[2])(x2)
x3 = Tea(64,init_connection=connections[3])(x3)
# Concatenate output of first layer to send into next
x = concatenate([x0, x1, x2, x3])
x = Tea(250,init_connection=connections[4])(x)
# Pool spikes and output neurons into 10 classes.
x = AdditivePooling(10)(x)

predictions = Activation('softmax')(x)

model_1 = Model(inputs=inputs, outputs=predictions)

model_1.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model_1.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_split=0.2)

score_1 = model_1.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score_1[0])
print('Test accuracy:', score_1[1])

# from output_bus import OutputBus
# from serialization import save as sim_save
# # from rancutils.teaconversion import create_cores, create_packets, Packet
# # 

# x_test_flat = x_test.reshape((10000, 784))
# partitioned_packets = []

# # # Use absolute/hard reset by specifying neuron_reset_type=0
# # cores_sim = create_cores(model, 2, neuron_reset_type=0) 
# # # Partition the packets into groups as they will be fed into each of the input cores
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