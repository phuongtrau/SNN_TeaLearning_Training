# import cv2 
# import numpy as np 
# # kernel = np.ones((16,16),np.uint8)

# # for i in range (20):
# img = cv2.imread("/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/image_test_inclined/8_1343.jpg")

# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # print(img)
# # kernel = np.ones((3,3),np.uint8)

# # # img_eros = cv2.erode(img,kernel,iterations=1)
# # # img_dila = cv2.dilate(img_eros,kernel,iterations=1)
# # # cv2.imwrite("/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/image_test_inclined/img_raw_eros_dila_{}.jpg".format(i),img_dila)
# # ret,thresh1 = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # img_eros = cv2.erode(thresh1,kernel,iterations=1)
# # img_dila = cv2.dilate(img_eros,kernel,iterations=1)
# # thresh1 = cv2.equalizeHist(img_dila*img)

# np.savetxt("/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/image_test_inclined/img_8_1343.txt",img.astype(int),fmt="%d")
# # print(thresh1)
#     # thresh1 = cv2.dilate(thresh1,kernel,iterations=1)
#     # cv2.imwrite("/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/home/phuongdh/Documents/SNN/SNN_TeaLearning_Training/tealayers/tealayer1.0/tealayers/image_test_inclined/img_raw_after_bin_{}.jpg".format(i),thresh1)


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
import time
import matplotlib.pyplot as plt
import matplotlib

def check_in_region(indexs):
    out_list = []
    list_cor = list(zip(indexs[0], indexs[1]))
    out_list.append(list_cor[0])
    
    for cor in list_cor:
        F = 1 
        for i in range(len(out_list)):
            
            if cor[0] > out_list[i][0] + 20 or cor[1] > out_list[i][1]+20:
                F = F * 1
            else:
                F = F * 0
        if F:
            out_list.append(cor)
    # for out in out_list:
    #     if out[0] > 32 and 
    return out_list


def center_out (out_list):
    out = []
    for i in range(len(out_list)):
        if out_list[i][0] >=16 and out_list[i][0] <=59 and out_list[i][1] >=16 and out_list[i][1] <= 27:
            out.append([out_list[i][0]-16,out_list[i][1]-16])
            out.append([out_list[i][0]+16,out_list[i][1]+16])
        elif out_list[i][0] < 16 and out_list[i][1] < 16:
            out.append([0,0])
            out.append([out_list[i][0]+16,out_list[i][1]+16])
        elif out_list[i][0] > 169 and out_list[i][1] >27 :
            out.append([out_list[i][0]-16,out_list[i][1]-16])
            out.append([64,32])
        elif out_list[i][0] < 16:
            out.append([0,out_list[i][1]-16])
            out.append([out_list[i][0]+16,out_list[i][1]+16])
        elif out_list[i][1] < 16:
            out.append([out_list[i][0]-16,0])
            out.append([out_list[i][0]+16,out_list[i][1]+16])
        elif out_list[i][0] > 59:
            out.append([out_list[i][0]-16,out_list[i][1]-16])
            out.append([64,out_list[i][1]+16])
        elif out_list[i][1] > 27:
            out.append([out_list[i][0]-16,out_list[i][1]-16])
            out.append([out_list[i][0]+16,32])
    return out


exp_i_data = helper.load_exp_i("../dataset/experiment-i")
kernel = np.ones((3,3),np.uint8)
# print(len(dataset))
datasets = {"Base":exp_i_data}

# train_data = helper.Mat_Dataset(datasets,["Base"],["S1","S2","S3","S4","S16","S6","S7","S8","S9"])
test_data = helper.Mat_Dataset(datasets,["Base"],["S1"])

for i in range(len(test_data.samples)):
    
    print("class: ",test_data.labels[i])
    # max_val = np.amax(test_data.samples[i])
    
    # indexs = np.where(test_data.samples[i] == max_val)
    
    # out_list = check_in_region(indexs)
    
    # out_cen = center_out(out_list)
    heat = cv2.applyColorMap(test_data.samples[i], cv2.COLORMAP_JET)
    cv2.imwrite("/home/phuongdh/Documents/SNN/heatmap_all/{}_{}_Heat_Raw.jpg".format(test_data.labels[i],i),heat)

    hsvImage = cv2.cvtColor(heat, cv2.COLOR_BGR2HSV)
    cv2.imwrite("/home/phuongdh/Documents/SNN/heatmap_all/{}_{}_HSV_Raw.jpg".format(test_data.labels[i],i),hsvImage)

    # for j in range(int(len(out_cen)/2)):
    #     test_data.samples[i][out_cen[2*j][0]:out_cen[2*j+1][0],out_cen[2*j][1]:out_cen[2*j+1][1]]=\
    #     cv2.equalizeHist(test_data.samples[i][out_cen[2*j][0]:out_cen[2*j+1][0],out_cen[2*j][1]:out_cen[2*j+1][1]])
    test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
    
    # cv2.imwrite("/home/phuongdh/Documents/SNN/heatmap_all/{}_{}_EH_RE.jpg".format(test_data.labels[i],i),test_data.samples[i])

    heat = cv2.applyColorMap(test_data.samples[i], cv2.COLORMAP_JET)
    cv2.imwrite("/home/phuongdh/Documents/SNN/heatmap_all/{}_{}_EH_Heat.jpg".format(test_data.labels[i],i),heat)

    hsvImage = cv2.cvtColor(heat, cv2.COLOR_BGR2HSV)
    cv2.imwrite("/home/phuongdh/Documents/SNN/heatmap_all/{}_{}_EH_HSV.jpg".format(test_data.labels[i],i),hsvImage)

    avg = np.round((hsvImage+heat)/2)
    cv2.imwrite("/home/phuongdh/Documents/SNN/heatmap_all/{}_{}_AVG.jpg".format(test_data.labels[i],i),avg)
print("done")

