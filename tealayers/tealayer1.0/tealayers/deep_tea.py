from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.topology import Layer
from keras.layers import Lambda,Concatenate, Reshape
from tea import Tea
import tensorflow as tf


class DeepTea(Layer):
    def __init__(self,
                depth,
                units,
                stride,
                inputmax=256,
                **kwargs):
        # assert type(depth) == 'int'
        # assert type(stride) == 'int'
        # assert type(units) == 'int'
        
        self.depth = depth
        self.units = units
        self.stride = stride
        self.inputmax = inputmax
        super(DeepTea, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        # print(input_shape)
        self.num_Tea = (input_shape[-1]-self.inputmax)//self.stride 
        # print(self.num_Tea)
        super(DeepTea, self).build(input_shape)

    def call(self,input_layer):
        out_depth=[]
        if self.depth ==1: 
            
            if self.num_Tea == 1: 
                tea_input = Lambda(lambda x : x[:,:self.inputmax])(input_layer)
                # print(tea_input)
                return Tea(self.units)(tea_input)
            else:
                out = []
                for i in range(self.num_Tea):
                    # print((i*self.stride)+self.inputmax,(i+1)*self.stride+self.inputmax)
                    tea_input = Lambda(lambda x : x[:,i*self.stride:i*self.stride+self.inputmax])(input_layer)
                    # print(tea_input)
                    out.append(Tea(self.units)(tea_input))
                    # print(self.out)
                # print(len(out))
                out = Concatenate(axis=1)(out)
                # out = Reshape([1,out.shape[-1]])(out)
                return out
        else:
            for j in range(self.depth):
                
                out = []
                
                for i in range(self.num_Tea):
                    # print((i*self.stride)+self.inputmax,(i+1)*self.stride+self.inputmax)
                    tea_input = Lambda(lambda x : x[:,(i*self.stride)+self.inputmax:(i+1)*self.stride+self.inputmax])(input_layer)
                    # print(tea_input)
                    out.append(Tea(self.units)(tea_input))
                    # print(self.out)
                
                out = Concatenate(axis=1)(out)
                
                out_depth.append(out)
            out_depth = Concatenate(axis=1)(out_depth)
            # out_depth = Reshape([1,out_depth.shape[-1]])(out_depth)
            return out_depth
        
        # for i in range(num_Tea):

        # shape = (input_shape[-1], self.units)



