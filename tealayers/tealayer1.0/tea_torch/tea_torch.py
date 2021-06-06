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
    def __init__(self):
        super(TeaTorch, self).__init__()
        
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        if connection_initializer is not None:
            self.connection_initializer = connection_initializer
        else:
            self.connection_initializer = initializers.TruncatedNormal(mean=0.5)
            # print(self.connection_initializer)
        if weight_initializer is not None:
            self.weight_initializer = weight_initializer
        else:
            self.weight_initializer = tea_weight_initializer
            # print(connection_initializer.shape)
        self.bias_initializer = bias_initializer
        self.connection_regularizer = regularizers.get(connection_regularizer)
        self.connection_constraint = constraints.get(connection_constraint)
        
        self.input_width = None
        self.round_input = round_input
        self.round_connections = round_connections
        self.clip_connections = clip_connections
        self.round_bias = round_bias
        self.constrain_outputs_after_trianing = \
            constrain_outputs_after_trianing
        # Needs to be set to `True` to use the `K.in_train_phase` function.
        self.uses_learning_phase = True

        self.connections = init_connection
        
        super(Tea, self).__init__(**kwargs)

    def forward(self, input, time_window = 20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2,x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
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
    return torch.tens(ret_array)
