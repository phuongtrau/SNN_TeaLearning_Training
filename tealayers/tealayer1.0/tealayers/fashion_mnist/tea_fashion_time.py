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
import sys
sys.path.append("../../../../rancutils/rancutils")
sys.path.append("../")

from tea import Tea
from additivepooling import AdditivePooling

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet
from fashion import Fashion
# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32')
# print(x_train.shape)
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

## change the input threshold spike ## 

x_train =  x_train / 255

x_test  = x_test / 255

time_win = 5


inputs = Input(shape=(28, 28,))
print(inputs)
flattened = Flatten()(inputs)
print(flattened)
fas_out = tf.zeros([510,])
fas_in_temp = tf.zeros_like(flattened)

for i in range(time_win):
    filter_win = tf.random.normal([-1,784],mean = 0.5)
    print(filter_win)
    mask = tf.math.greater(flattened,filter_win)
    print(mask)
    mask = tf.cast(mask,tf.float32)
    print(mask)
    flattened = tf.math.multiply(flattened ,mask)

    fas_in_temp = tf.math.add(fas_in_temp,flattened) 
    # print(flattened_inputs)
    
    fas_in = Fashion(fas_in_temp)
    print(fas_in)
    fas_in = fas_in.forward_1()
    print(fas_in)
    fas_out = tf.math.add(fas_out,fas_in)
    print(fas_out)

fas_out = fas_out / time_win
print(fas_out)

final = AdditivePooling(10)(fas_out)
print(final)
predictions = Activation('softmax')(final)
print(predictions)

model = Model(inputs= inputs, outputs= predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=50,
          verbose=1,
          validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])