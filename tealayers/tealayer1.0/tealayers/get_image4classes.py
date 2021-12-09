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
from keras.layers import Dropout, Flatten, Activation, Input, Lambda, Concatenate,Average,Permute
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
import random
from sklearn.utils import shuffle
import cv2

# import preprocess
from output_bus import OutputBus
from serialization import save as sim_save
from emulation import write_cores

exp_i_data = helper.load_exp_i("../dataset/experiment-i")

datasets = {"Base":exp_i_data}
subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]
supine = [0,7,8,9,10,11,14,15,16]
left = [2,5,6,13]
right = [1,3,4,12]

for sub in subjects:
    data = helper.Mat_Dataset(datasets,["Base"],[sub])
    
    for i in range(len(data.samples)):    
        data.samples[i] = cv2.equalizeHist(data.samples[i])
        if data.labels[i] in supine :
            class_1 = 0
            class_2 = supine.index(data.labels[i])
            # print(data.samples[i].shape)
            heat = cv2.applyColorMap(data.samples[i], cv2.COLORMAP_JET)
            print("supine")
            cv2.imwrite("/home/hoangphuong/Documents/image_Pmatdata/{}/{}_{}_{}.jpg".format(sub,i,class_1,class_2),heat)
        elif data.labels[i] in left :
            class_1 = 1
            class_2 = left.index(data.labels[i])
            print("left")
            cv2.imwrite("/home/hoangphuong/Documents/image_Pmatdata/{}/{}_{}_{}.jpg".format(sub,i,class_1,class_2),data.samples[i])
        else :
            class_1 = 2
            class_2 = right.index(data.labels[i])
            print("right")
            cv2.imwrite("/home/hoangphuong/Documents/image_Pmatdata/{}/{}_{}_{}.jpg".format(sub,i,class_1,class_2),data.samples[i])
        # print("/home/hoangphuong/Documents/image_Pmatdata/{}/{}_{}_{}.jpg".format(sub,i,class_1,class_2))
    print("done {}".format(sub))