import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 10
batch_size  = 100
learning_rate = 1e-3
num_epochs = 100 # max epoch

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

class TeaTorch(nn.Module):
    def __init__(self, 
                units,
                connection = None,
                round_input= True,
                round_connections = True,
                round_bias = True,
                clip_connections = True):
        
        super(TeaTorch, self).__init__()
        
        self.units = units

        self.round_input = round_input
        self.round_connections = round_connections
        self.clip_connections = clip_connections
        self.round_bias = round_bias

        if connection is not None:
            self.connections = connection

    def forward(self, input):
        

        return outputs


def tea_weight_initializer(shape, dtype=np.float32):
    """Returns a tensor of alternating 1s and -1s, which is (kind of like)
    how IBM initializes their weight matrix in their TeaLearning
    literature.

    Arguments:
        shape -- The shape of the weights to intialize.

    Keyword Arguments:
        dtype -- The data type to use to initialize the weights.
                 (default: {np.float32})"""
    num_axons = shape[0]
    num_neurons = shape[1]
    ret_array = np.zeros((int(num_axons), int(num_neurons)), dtype=dtype)
    for axon_num, axon in enumerate(ret_array):
        if axon_num % 2 == 0:
            for i in range(len(axon)):
                ret_array[axon_num][i] = 1
        else:
            for i in range(len(axon)):
                ret_array[axon_num][i] = -1
    return torch.tensor(ret_array)
