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
sys.path.append("../../../rancutils/rancutils")

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

time_window = 5

filter = []
x_tr = []
x_ts = []
for k in range(time_window):
    filter.append(list(np.random.rand(28,28)))
    x_train_temp = x_train > np.array(filter[k])
    x_train_temp = x_train_temp.astype('float32')
    x_tr.append(x_train_temp * x_train)
    x_test_temp = x_test > np.array(filter[k])
    x_test_temp = x_test_temp.astype('float32')
    x_ts.append(x_test_temp * x_test)
    # x_ts.append(x_train)

# x_train = np.concatenate(x_tr[:,:,:,np.newaxis],axis= 3)
# x_test = np.concatenate(x_ts[:,:,:,np.newaxis],axis= 3)


print(np.array(x_tr).shape)
x_tr = np.reshape(x_tr,(-1,28,28,5))
# print(x_tr.shape)
# np.save("x_train_2bit",x_train)
# x_test = np.concatenate([x_test_1[:,:,:,np.newaxis],x_test_2[:,:,:,np.newaxis]],axis=3)
# np.save("x_test_2bit",x_test)
# np.save("y_train",y_train)
# np.save("y_test",y_test)

inputs = Input(shape=(28, 28,5,))

flattened = Flatten()(inputs)
# flattened_inputs = []
flattened_inputs_1 = Lambda(lambda x : x[:,:784])(flattened)
flattened_inputs_2 = Lambda(lambda x : x[:,784:2*784])(flattened)
flattened_inputs_3 = Lambda(lambda x : x[:,2*784:3*784])(flattened)
flattened_inputs_4 = Lambda(lambda x : x[:,3*784:4*784])(flattened)
flattened_inputs_5 = Lambda(lambda x : x[:,4*784:5*784])(flattened)

fas_1 = Fashion(flattened_inputs_1)
fas_2 = Fashion(flattened_inputs_2)
fas_3 = Fashion(flattened_inputs_3)
fas_4 = Fashion(flattened_inputs_4)
fas_5 = Fashion(flattened_inputs_5)

fas_1 = fas_1.forward()
fas_2 = fas_2.forward()
fas_3 = fas_3.forward()
fas_4 = fas_4.forward()
fas_5 = fas_5.forward()

final = np.concatenate([fas_1,fas_2,fas_3,fas_4,fas_5])
final = AdditivePooling(10)(final)

predictions = Activation('softmax')(final)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_tr, y_train,
          batch_size=64,
          epochs=50,
          verbose=1,
          validation_split=0.2)

score = model.evaluate(x_ts, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])