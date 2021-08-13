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
from deep_tea import DeepTea 
from additivepooling import AdditivePooling

# import sys
# sys.path.append("../../../rancutils/rancutils")

# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

## change the input threshold spike ## 

x_train =  x_train / 255

x_test  = x_test / 255

# for e in x_train:
#     for i in range(28):
#         for j in range(28):
#             if e[i][j]>1:
#                 e[i][j]=1

# for e in x_test:
#     for i in range(28):
#         for j in range(28):
#             if e[i][j]>1:
#                 e[i][j]=1

inputs = Input(shape=(28, 28,))
flattened_inputs = Flatten()(inputs) 
# print(flattened_inputs.shape)

dt1 = DeepTea(depth=2,units=250,stride=100)(flattened_inputs)
# dt1=tf.reshape(dt1,[1,dt1.shape[-1]])
print(dt1)
dt2 = DeepTea(depth=1,units=250,stride=1000)(dt1)
print(dt2)
out = AdditivePooling(10)(dt1)

predictions = Activation('softmax')(out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_split=0.2)



