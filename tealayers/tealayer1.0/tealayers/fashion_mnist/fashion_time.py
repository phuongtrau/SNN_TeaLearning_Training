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
from keras.layers import Flatten, Activation, Input, Lambda,Average, Add,BatchNormalization 
from keras.datasets import fashion_mnist,mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append("../../../../rancutils/rancutils")
sys.path.append("../")

from tea import Tea
from additivepooling import AdditivePooling

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet
from fashion import Fashion
from emulation import write_cores

# Load FASHION_MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

## change the input threshold spike ## 

x_train =  x_train / 255

x_test  = x_test / 255

time_win = 4
filter_win = []
x_tr = []
x_ts = []
# for i in range(time_win):
filter_wins = np.random.rand(time_win,28,28)
# print(filter_win.shape)

for ele in x_train:
    # x_tr.append(x_train)
    eles = []
    for filter_win in filter_wins:
        encode = (ele > filter_win).astype('float32') 
        eles.append(encode*ele)
    for i in range(3):
        eles[i+1] = eles[i+1] + eles[i]
        eles[i+1] = eles[i+1] + eles[i+1] * (eles[i+1] > np.ones((28,28))).astype('float32')  
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
    for i in range(3):
        eles[i+1] = eles[i+1] + eles[i]
        eles[i+1] = eles[i+1] + eles[i+1] * (eles[i+1] > np.ones((28,28))).astype('float32')
    ele = np.array(eles)
    ele = np.moveaxis(ele, 0, -1)
    x_ts.append(ele)

x_tr = np.array(x_tr)
x_ts = np.array(x_ts)

paddings = tf.constant([[0, 0,], [137, 137]])
inputs = Input(shape=(28, 28, 4,))
flattened_inputs = Flatten()(inputs)

fas_in_1 = Lambda(lambda x : x[:,:784])(flattened_inputs)
fas_in_2 = Lambda(lambda x : x[:,784:1568])(flattened_inputs)
fas_in_3 = Lambda(lambda x : x[:,1568:2352])(flattened_inputs)
fas_in_4 = Lambda(lambda x : x[:,2352:3136])(flattened_inputs)


fas_in_1 = Fashion(fas_in_1)
fas_in_2 = Fashion(fas_in_2)
fas_in_3 = Fashion(fas_in_3)
fas_in_4 = Fashion(fas_in_4)

fas_in_1 = fas_in_1.forward()
# fas_in_1 = Tea(512)(fas_in_1)
# fas_in_1 = tf.pad(fas_in_1, paddings, "SYMMETRIC")

# fas_in_2 = Fashion(Add()([fas_in_2 , fas_in_1]))
fas_in_2 = fas_in_2.forward()
# fas_in_2 = Tea(512)(fas_in_2)
# fas_in_2 = tf.pad(fas_in_2, paddings, "SYMMETRIC")

# fas_in_3 = Fashion(Add()([fas_in_3,fas_in_2]))
fas_in_3 = fas_in_3.forward()
# fas_in_3 = Tea(512)(fas_in_3)
# fas_in_3 = tf.pad(fas_in_3, paddings, "SYMMETRIC")

# fas_in_4 = Fashion(Add()([fas_in_4,fas_in_3]))
fas_in_4 = fas_in_4.forward()


# fas_out = Average()([fas_in_1,fas_in_2,fas_in_3,fas_in_4, fas_in_5, fas_in_6, fas_in_7, fas_in_8, fas_in_9, fas_in_10])
fas_out = Average()([fas_in_1,fas_in_2,fas_in_3,fas_in_4])

fas_out = AdditivePooling(10)(fas_out)

predictions = Activation('softmax')(fas_out)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_tr, y_train,
          batch_size=32,
          epochs=50,
          verbose=1,
          validation_split=0.2)

score = model.evaluate(x_ts, y_test, verbose=0)

print('Test loss new:', score[0])
print('Test accuracy new:', score[1])

# cores_sim = create_cores(model, 20,neuron_reset_type=0 ) 
# write_cores(cores_sim,output_path="/home/phuongdh/Documents/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/out_new")