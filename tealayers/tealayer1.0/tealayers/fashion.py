from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math

import tensorflow as tf
from tensorflow import squeeze
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Lambda,Concatenate,Average,BatchNormalization 
from tea import Tea

import sys
sys.path.append("../../../rancutils/rancutils")

from teaconversion import create_cores,create_packets,get_connections_and_biases
from packet import Packet

class Fashion(Layer):
    def __init__(self,inputs):
        self.inputs = inputs
    def forward(self):
        # inputs = Input(shape=(784,))
        x0 = Lambda(lambda x : x[:,:512])(self.inputs)
        x1 = Lambda(lambda x : x[:,34:546])(self.inputs)
        x2 = Lambda(lambda x : x[:,68:580])(self.inputs)
        x3 = Lambda(lambda x : x[:,102:614])(self.inputs)
        x4 = Lambda(lambda x : x[:,136:648])(self.inputs)
        x5 = Lambda(lambda x : x[:,170:682])(self.inputs)
        x6 = Lambda(lambda x : x[:,204:716])(self.inputs)
        x7 = Lambda(lambda x : x[:,238:750])(self.inputs)
        x8 = Lambda(lambda x : x[:,272:])(self.inputs)

        x0 = Tea(256)(x0)
        x1 = Tea(256)(x1)
        x2 = Tea(256)(x2)
        x3 = Tea(256)(x3)
        x4 = Tea(256)(x4)
        x5 = Tea(256)(x5)
        x6 = Tea(256)(x6)
        x7 = Tea(256)(x7)
        x8 = Tea(256)(x8)

        x_1 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_1 = BatchNormalization()(x_1)
        x0 = Lambda(lambda x : x[:,:512])(x_1)
        x1 = Lambda(lambda x : x[:,224:736])(x_1)
        x2 = Lambda(lambda x : x[:,448:960])(x_1)
        x3 = Lambda(lambda x : x[:,672:1184])(x_1)
        x4 = Lambda(lambda x : x[:,896:1408])(x_1)
        x5 = Lambda(lambda x : x[:,1120:1632])(x_1)
        x6 = Lambda(lambda x : x[:,1344:1856])(x_1)
        x7 = Lambda(lambda x : x[:,1568:2080])(x_1)
        x8 = Lambda(lambda x : x[:,1792:])(x_1)

        x0 = Tea(128)(x0)
        x1 = Tea(128)(x1)
        x2 = Tea(128)(x2)
        x3 = Tea(128)(x3)
        x4 = Tea(128)(x4)
        x5 = Tea(128)(x5)
        x6 = Tea(128)(x6)
        x7 = Tea(128)(x7)
        x8 = Tea(128)(x8)

        x_2 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_2 = BatchNormalization()(x_2)
        x0 = Lambda(lambda x : x[:,:512])(x_2)
        x1 = Lambda(lambda x : x[:,80:592])(x_2)
        x2 = Lambda(lambda x : x[:,160:672])(x_2)
        x3 = Lambda(lambda x : x[:,240:752])(x_2)
        x4 = Lambda(lambda x : x[:,320:832])(x_2)
        x5 = Lambda(lambda x : x[:,400:912])(x_2)
        x6 = Lambda(lambda x : x[:,480:992])(x_2)
        x7 = Lambda(lambda x : x[:,560:1072])(x_2)
        x8 = Lambda(lambda x : x[:,640:])(x_2)

        x0 = Tea(170)(x0)
        x1 = Tea(170)(x1)
        x2 = Tea(170)(x2)
        x3 = Tea(170)(x3)
        x4 = Tea(170)(x4)
        x5 = Tea(170)(x5)
        x6 = Tea(170)(x6)
        x7 = Tea(170)(x7)
        x8 = Tea(170)(x8)
        # x_br_2 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7])

        x_2 = Concatenate(axis=1)([x0, x1, x2])
        x_3 = Concatenate(axis=1)([x3, x4, x5])
        x_4 = Concatenate(axis=1)([x6, x7, x8])

        x0 = Tea(170)(x_2)
        # x0 = Average()([x0,x1,x2])

        x1 = Tea(170)(x_3)
        # x1 = Average()([x3,x4,x5])

        x2 = Tea(170)(x_4)
        # x2 = Average()([x6,x7,x8])

        # print(x0,x1,x2,x3)
        # Concatenate output of first layer to send into next

        x = Concatenate(axis=1)([x0, x1, x2])
        # x = BatchNormalization()(x)
        x = Tea(510)(x)

        x = Average()([x_2,x_3,x_4,x])

        
        return x

    def forward_1(self):
        x0 = Lambda(lambda x : x[:,:512])(self.inputs)
        x1 = Lambda(lambda x : x[:,91:603])(self.inputs)
        x2 = Lambda(lambda x : x[:,181:693])(self.inputs)
        x3 = Lambda(lambda x : x[:,272:])(self.inputs)
        x0 = Tea(128)(x0)
        x1 = Tea(128)(x1)
        x2 = Tea(128)(x2)
        x3 = Tea(128)(x3)
        # print(x0,x1,x2,x3)
        # Concatenate output of first layer to send into next
        x = Concatenate(axis=1)([x0, x1, x2, x3])
        x = Tea(510)(x)
        return x

    def forward_2(self):
        # inputs = Input(shape=(784,))
        x0 = Lambda(lambda x : x[:,:512])(self.inputs)
        x1 = Lambda(lambda x : x[:,34:546])(self.inputs)
        x2 = Lambda(lambda x : x[:,68:580])(self.inputs)
        x3 = Lambda(lambda x : x[:,102:614])(self.inputs)
        x4 = Lambda(lambda x : x[:,136:648])(self.inputs)
        x5 = Lambda(lambda x : x[:,170:682])(self.inputs)
        x6 = Lambda(lambda x : x[:,204:716])(self.inputs)
        x7 = Lambda(lambda x : x[:,238:750])(self.inputs)
        x8 = Lambda(lambda x : x[:,272:])(self.inputs)

        x0 = Tea(256)(x0)
        x1 = Tea(256)(x1)
        x2 = Tea(256)(x2)
        x3 = Tea(256)(x3)
        x4 = Tea(256)(x4)
        x5 = Tea(256)(x5)
        x6 = Tea(256)(x6)
        x7 = Tea(256)(x7)
        x8 = Tea(256)(x8)

        x_1 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_1 = BatchNormalization()(x_1)
        x0 = Lambda(lambda x : x[:,:512])(x_1)
        x1 = Lambda(lambda x : x[:,224:736])(x_1)
        x2 = Lambda(lambda x : x[:,448:960])(x_1)
        x3 = Lambda(lambda x : x[:,672:1184])(x_1)
        x4 = Lambda(lambda x : x[:,896:1408])(x_1)
        x5 = Lambda(lambda x : x[:,1120:1632])(x_1)
        x6 = Lambda(lambda x : x[:,1344:1856])(x_1)
        x7 = Lambda(lambda x : x[:,1568:2080])(x_1)
        x8 = Lambda(lambda x : x[:,1792:])(x_1)

        x0 = Tea(128)(x0)
        x1 = Tea(128)(x1)
        x2 = Tea(128)(x2)
        x3 = Tea(128)(x3)
        x4 = Tea(128)(x4)
        x5 = Tea(128)(x5)
        x6 = Tea(128)(x6)
        x7 = Tea(128)(x7)
        x8 = Tea(128)(x8)

        x_2 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_2 = BatchNormalization()(x_2)
        #### branch 1 ####
        xbr1_0 = Lambda(lambda x : x[:,:512])(x_2)
        xbr1_1 = Lambda(lambda x : x[:,80:592])(x_2)
        xbr1_2 = Lambda(lambda x : x[:,160:672])(x_2)
        xbr1_3 = Lambda(lambda x : x[:,240:752])(x_2)
        xbr1_4 = Lambda(lambda x : x[:,320:832])(x_2)
        xbr1_5 = Lambda(lambda x : x[:,400:912])(x_2)
        xbr1_6 = Lambda(lambda x : x[:,480:992])(x_2)
        xbr1_7 = Lambda(lambda x : x[:,560:1072])(x_2)
        xbr1_8 = Lambda(lambda x : x[:,640:])(x_2)

        xbr1_0 = Tea(170)(xbr1_0)
        xbr1_1 = Tea(170)(xbr1_1)
        xbr1_2 = Tea(170)(xbr1_2)
        xbr1_3 = Tea(170)(xbr1_3)
        xbr1_4 = Tea(170)(xbr1_4)
        xbr1_5 = Tea(170)(xbr1_5)
        xbr1_6 = Tea(170)(xbr1_6)
        xbr1_7 = Tea(170)(xbr1_7)
        xbr1_8 = Tea(170)(xbr1_8)
        
        xbr1_0 = Concatenate(axis=1)([xbr1_0, xbr1_1, xbr1_2])
        xbr1_1 = Concatenate(axis=1)([xbr1_3, xbr1_4, xbr1_5])
        xbr1_2 = Concatenate(axis=1)([xbr1_6, xbr1_7, xbr1_8])

        xbr1_0 = Tea(256)(xbr1_0)
        xbr1_1 = Tea(256)(xbr1_1)
        xbr1_2 = Tea(256)(xbr1_2)

        xbr1 = Concatenate(axis=1)([xbr1_0, xbr1_1, xbr1_2])
        ### branch 2 ###

        xbr2_0 = Lambda(lambda x : x[:,:512])(x_2)
        xbr2_1 = Lambda(lambda x : x[:,128:640)(x_2)
        xbr2_2 = Lambda(lambda x : x[:,256:768])(x_2)
        xbr2_3 = Lambda(lambda x : x[:,384:896])(x_2)
        xbr2_4 = Lambda(lambda x : x[:,512:1024])(x_2)
        xbr2_5 = Lambda(lambda x : x[:,640:])(x_2)
        
        xbr2_0 = Tea(256)(xbr2_0)
        xbr2_1 = Tea(256)(xbr2_1)
        xbr2_2 = Tea(256)(xbr2_2)
        xbr2_3 = Tea(256)(xbr2_3)
        xbr2_4 = Tea(256)(xbr2_4)
        xbr2_5 = Tea(256)(xbr2_5)

        xbr2_0 = Concatenate(axis=1)([xbr2_0, xbr2_1])
        xbr2_1 = Concatenate(axis=1)([xbr2_2, xbr2_3])
        xbr2_2 = Concatenate(axis=1)([xbr2_4, xbr2_5])

        xbr2_0 = Tea(256)(xbr2_0_1)
        xbr2_1 = Tea(256)(xbr2_0_2)
        xbr2_2 = Tea(256)(xbr2_0_3)

        xbr2 = Concatenate(axis=1)([xbr2_0, xbr2_1, xbr2_2])

        ### merge 2 branch ###

        x = Average()([xbr1,xbr2]) 

        ### branch 1 ### 
        
        xbr1_0 = Lambda(lambda x : x[:,:512])(x)
        xbr1_1 = Lambda(lambda x : x[:,32:544])(x)
        xbr1_2 = Lambda(lambda x : x[:,64:576])(x)
        xbr1_3 = Lambda(lambda x : x[:,96:608])(x)
        xbr1_4 = Lambda(lambda x : x[:,128:640])(x)
        xbr1_5 = Lambda(lambda x : x[:,160:672])(x)
        xbr1_6 = Lambda(lambda x : x[:,192:704])(x)
        xbr1_7 = Lambda(lambda x : x[:,224:736])(x)
        xbr1_8 = Lambda(lambda x : x[:,256:])(x)

        xbr1_0 = Tea(170)(xbr1_0)
        xbr1_1 = Tea(170)(xbr1_1)
        xbr1_2 = Tea(170)(xbr1_2)
        xbr1_3 = Tea(170)(xbr1_3)
        xbr1_4 = Tea(170)(xbr1_4)
        xbr1_5 = Tea(170)(xbr1_5)
        xbr1_6 = Tea(170)(xbr1_6)
        xbr1_7 = Tea(170)(xbr1_7)
        xbr1_8 = Tea(170)(xbr1_8)
        
        xbr1_0 = Concatenate(axis=1)([xbr1_0, xbr1_1, xbr1_2])
        xbr1_1 = Concatenate(axis=1)([xbr1_3, xbr1_4, xbr1_5])
        xbr1_2 = Concatenate(axis=1)([xbr1_6, xbr1_7, xbr1_8])

        xbr1_0 = Tea(170)(xbr1_0)
        xbr1_1 = Tea(170)(xbr1_1)
        xbr1_2 = Tea(170)(xbr1_2)

        x_out_1 = Concatenate(axis=1)([xbr1_0,xbr1_1,xbr1_2])

        x_out_1 = Tea(510)(x_out_1)
        ### branch 2 ###

        xbr2_0 = Lambda(lambda x : x[:,:512])(x)
        xbr2_1 = Lambda(lambda x : x[:,128:640])(x)
        xbr2_2 = Lambda(lambda x : x[:,256:])(x)
        
        xbr2_0 = Tea(170)(xbr2_0)
        xbr2_1 = Tea(170)(xbr2_1)
        xbr2_2 = Tea(170)(xbr2_2)

        x_out_2 = Concatenate(axis=1)([xbr2_0,xbr2_1,xbr2_2])

        x_out_2 = Tea(510)(x_out_2)
        
        ### x_out ### 
        x_out = Average()([x_out_1,x_out_2])
        
        # x = Concatenate(axis=1)([x0, x1, x2])
        
        # x = Tea(510)(x)

        # x = Average()([x_2,x_3,x_4,x])

        # x_br = Concatenate(axis=1)([xbr2_0_1_t, xbr2_0_2_t, xbr2_0_3_t])
        
        # x_br = Tea(510)(x_br)
        
        # x_br = Average()([xbr2_0_1, xbr2_0_2, xbr2_0_3,x_br])
        
        # x = Average()([x,x_br])

        return x_out
        
    def forward_3(self):
        x0 = Lambda(lambda x : x[:,:512])(self.inputs)
        x1 = Lambda(lambda x : x[:,34:546])(self.inputs)
        x2 = Lambda(lambda x : x[:,68:580])(self.inputs)
        x3 = Lambda(lambda x : x[:,102:614])(self.inputs)
        x4 = Lambda(lambda x : x[:,136:648])(self.inputs)
        x5 = Lambda(lambda x : x[:,170:682])(self.inputs)
        x6 = Lambda(lambda x : x[:,204:716])(self.inputs)
        x7 = Lambda(lambda x : x[:,238:750])(self.inputs)
        x8 = Lambda(lambda x : x[:,272:])(self.inputs)

        x0 = Tea(256)(x0)
        x1 = Tea(256)(x1)
        x2 = Tea(256)(x2)
        x3 = Tea(256)(x3)
        x4 = Tea(256)(x4)
        x5 = Tea(256)(x5)
        x6 = Tea(256)(x6)
        x7 = Tea(256)(x7)
        x8 = Tea(256)(x8)

        x_1 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_1 = BatchNormalization()(x_1)
        x0 = Lambda(lambda x : x[:,:512])(x_1)
        x1 = Lambda(lambda x : x[:,224:736])(x_1)
        x2 = Lambda(lambda x : x[:,448:960])(x_1)
        x3 = Lambda(lambda x : x[:,672:1184])(x_1)
        x4 = Lambda(lambda x : x[:,896:1408])(x_1)
        x5 = Lambda(lambda x : x[:,1120:1632])(x_1)
        x6 = Lambda(lambda x : x[:,1344:1856])(x_1)
        x7 = Lambda(lambda x : x[:,1568:2080])(x_1)
        x8 = Lambda(lambda x : x[:,1792:])(x_1)

        x0 = Tea(128)(x0)
        x1 = Tea(128)(x1)
        x2 = Tea(128)(x2)
        x3 = Tea(128)(x3)
        x4 = Tea(128)(x4)
        x5 = Tea(128)(x5)
        x6 = Tea(128)(x6)
        x7 = Tea(128)(x7)
        x8 = Tea(128)(x8)

        x_2 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_2 = BatchNormalization()(x_2)
        x0 = Lambda(lambda x : x[:,:512])(x_2)
        x1 = Lambda(lambda x : x[:,80:592])(x_2)
        x2 = Lambda(lambda x : x[:,160:672])(x_2)
        x3 = Lambda(lambda x : x[:,240:752])(x_2)
        x4 = Lambda(lambda x : x[:,320:832])(x_2)
        x5 = Lambda(lambda x : x[:,400:912])(x_2)
        x6 = Lambda(lambda x : x[:,480:992])(x_2)
        x7 = Lambda(lambda x : x[:,560:1072])(x_2)
        x8 = Lambda(lambda x : x[:,640:])(x_2)

        x0 = Tea(170)(x0)
        x1 = Tea(170)(x1)
        x2 = Tea(170)(x2)
        x3 = Tea(170)(x3)
        x4 = Tea(170)(x4)
        x5 = Tea(170)(x5)
        x6 = Tea(170)(x6)
        x7 = Tea(170)(x7)
        x8 = Tea(170)(x8)
        
        x_br_2 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7])

        xbr2_0 = Lambda(lambda x : x[:,:512])(x_br_2)
        xbr2_1 = Lambda(lambda x : x[:,106:618])(x_br_2)
        xbr2_2 = Lambda(lambda x : x[:,212:724])(x_br_2)
        xbr2_3 = Lambda(lambda x : x[:,318:830])(x_br_2)
        xbr2_4 = Lambda(lambda x : x[:,424:936])(x_br_2)
        xbr2_5 = Lambda(lambda x : x[:,530:1042])(x_br_2)
        xbr2_6 = Lambda(lambda x : x[:,636:1148])(x_br_2)
        xbr2_7 = Lambda(lambda x : x[:,742:1254])(x_br_2)
        xbr2_8 = Lambda(lambda x : x[:,848:])(x_br_2)

        xbr2_0 = Tea(170)(xbr2_0)
        xbr2_1 = Tea(170)(xbr2_1)
        xbr2_2 = Tea(170)(xbr2_2)
        xbr2_3 = Tea(170)(xbr2_3)
        xbr2_4 = Tea(170)(xbr2_4)
        xbr2_5 = Tea(170)(xbr2_5)
        xbr2_6 = Tea(170)(xbr2_6)
        xbr2_7 = Tea(170)(xbr2_7)
        xbr2_8 = Tea(170)(xbr2_8)

        x_2 = Concatenate(axis=1)([x0, x1, x2])
        x_3 = Concatenate(axis=1)([x3, x4, x5])
        x_4 = Concatenate(axis=1)([x6, x7, x8])

        xbr2_0_1 = Concatenate(axis=1)([xbr2_0, xbr2_1, xbr2_2])
        xbr2_0_2 = Concatenate(axis=1)([xbr2_3, xbr2_4, xbr2_5])
        xbr2_0_3 = Concatenate(axis=1)([xbr2_6, xbr2_7, xbr2_8])

        xbr2_0_1_t = Tea(170)(xbr2_0_1)
        xbr2_0_2_t = Tea(170)(xbr2_0_2)
        xbr2_0_3_t = Tea(170)(xbr2_0_3)

        x0 = Tea(170)(x_2)
        # x0 = Average()([x0,x1,x2])

        x1 = Tea(170)(x_3)
        # x1 = Average()([x3,x4,x5])

        x2 = Tea(170)(x_4)
        # x2 = Average()([x6,x7,x8])

        # print(x0,x1,x2,x3)
        # Concatenate output of first layer to send into next

        x = Concatenate(axis=1)([x0, x1, x2])
        # x = BatchNormalization()(x)
        x = Tea(392)(x)

        # x = Average()([x_2,x_3,x_4,x])

        x_br = Concatenate(axis=1)([xbr2_0_1_t, xbr2_0_2_t, xbr2_0_3_t])
        x_br = Tea(392)(x_br)
        # x_br = Average()([xbr2_0_1, xbr2_0_2, xbr2_0_3,x_br])
        # x = Average()([x,x_br])
        x = Concatenate(axis=1)([x,x_br])
        
        x0 = Lambda(lambda x : x[:,:512])(x)
        x1 = Lambda(lambda x : x[:,34:546])(x)
        x2 = Lambda(lambda x : x[:,68:580])(x)
        x3 = Lambda(lambda x : x[:,102:614])(x)
        x4 = Lambda(lambda x : x[:,136:648])(x)
        x5 = Lambda(lambda x : x[:,170:682])(x)
        x6 = Lambda(lambda x : x[:,204:716])(x)
        x7 = Lambda(lambda x : x[:,238:750])(x)
        x8 = Lambda(lambda x : x[:,272:])(x)

        x0 = Tea(256)(x0)
        x1 = Tea(256)(x1)
        x2 = Tea(256)(x2)
        x3 = Tea(256)(x3)
        x4 = Tea(256)(x4)
        x5 = Tea(256)(x5)
        x6 = Tea(256)(x6)
        x7 = Tea(256)(x7)
        x8 = Tea(256)(x8)

        x_1 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_1 = BatchNormalization()(x_1)
        x0 = Lambda(lambda x : x[:,:512])(x_1)
        x1 = Lambda(lambda x : x[:,224:736])(x_1)
        x2 = Lambda(lambda x : x[:,448:960])(x_1)
        x3 = Lambda(lambda x : x[:,672:1184])(x_1)
        x4 = Lambda(lambda x : x[:,896:1408])(x_1)
        x5 = Lambda(lambda x : x[:,1120:1632])(x_1)
        x6 = Lambda(lambda x : x[:,1344:1856])(x_1)
        x7 = Lambda(lambda x : x[:,1568:2080])(x_1)
        x8 = Lambda(lambda x : x[:,1792:])(x_1)

        x0 = Tea(128)(x0)
        x1 = Tea(128)(x1)
        x2 = Tea(128)(x2)
        x3 = Tea(128)(x3)
        x4 = Tea(128)(x4)
        x5 = Tea(128)(x5)
        x6 = Tea(128)(x6)
        x7 = Tea(128)(x7)
        x8 = Tea(128)(x8)

        x_2 = Concatenate(axis=1)([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        # x_2 = BatchNormalization()(x_2)
        x0 = Lambda(lambda x : x[:,:512])(x_2)
        x1 = Lambda(lambda x : x[:,80:592])(x_2)
        x2 = Lambda(lambda x : x[:,160:672])(x_2)
        x3 = Lambda(lambda x : x[:,240:752])(x_2)
        x4 = Lambda(lambda x : x[:,320:832])(x_2)
        x5 = Lambda(lambda x : x[:,400:912])(x_2)
        x6 = Lambda(lambda x : x[:,480:992])(x_2)
        x7 = Lambda(lambda x : x[:,560:1072])(x_2)
        x8 = Lambda(lambda x : x[:,640:])(x_2)

        x0 = Tea(170)(x0)
        x1 = Tea(170)(x1)
        x2 = Tea(170)(x2)
        x3 = Tea(170)(x3)
        x4 = Tea(170)(x4)
        x5 = Tea(170)(x5)
        x6 = Tea(170)(x6)
        x7 = Tea(170)(x7)
        x8 = Tea(170)(x8)
        
        x_2 = Concatenate(axis=1)([x0, x1, x2])
        x_3 = Concatenate(axis=1)([x3, x4, x5])
        x_4 = Concatenate(axis=1)([x6, x7, x8])
        
        x0 = Tea(170)(x_2)
        # x0 = Average()([x0,x1,x2])

        x1 = Tea(170)(x_3)
        # x1 = Average()([x3,x4,x5])

        x2 = Tea(170)(x_4)
        # x2 = Average()([x6,x7,x8])

        # print(x0,x1,x2,x3)
        # Concatenate output of first layer to send into next

        x = Concatenate(axis=1)([x0, x1, x2])
        # x = BatchNormalization()(x)
        x = Tea(510)(x)

        x = Average()([x_2,x_3,x_4,x])
        return x