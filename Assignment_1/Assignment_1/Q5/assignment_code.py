import torch
import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
# from torch.autograd import Variable
# from torch.nn import Parameter
# from torch import Tensor
# import torch.nn.functional as F
import math


# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
"""
Parameters
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

Inputs: input, hidden
input of shape (batch, input_size): tensor containing input features
hidden of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.

Outputs: h'
h’ of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
"""
class GRUCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_i_r = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_i_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_h_r = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_h_z = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_i_n = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_h_n = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, inputs, hidden=False):
        if hidden is False: 
            hidden = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        x, h = inputs, hidden
        r = torch.sigmoid(self.W_i_r(x) + self.W_h_r(h))
        z = torch.sigmoid(self.W_i_z(x) + self.W_h_z(h))
        n = torch.tanh(self.W_i_n(x) + r * self.W_h_n(h))
        return (1 - z) * n + z * h


# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
"""
Parameters
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

Inputs: input, (h_0, c_0)
input of shape (batch, input_size): tensor containing input features
h_0 of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
c_0 of shape (batch, hidden_size): tensor containing the initial cell state for each element in the batch.
If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

Outputs: (h_1, c_1)
h_1 of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
c_1 of shape (batch, hidden_size): tensor containing the next cell state for each element in the batch
"""


class LSTMCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.W_i_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_h_i = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_i_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_h_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_i_g = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_h_g = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_i_o = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_h_o = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, inputs, h_0=False, c_0=False):
        if h_0 is False:
            h_0 = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        if c_0 is False:
            c_0 = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        x, h, c = inputs, h_0, c_0

        i = torch.sigmoid(self.W_i_i(x) + self.W_h_i(h))
        f = torch.sigmoid(self.W_i_f(x) + self.W_h_f(h))
        g = torch.tanh(self.W_i_g(x) + self.W_h_g(h))
        o = torch.sigmoid(self.W_i_o(x) + self.W_h_o(h))
        c_prime = f * c + i * g
        h_prime = o * torch.tanh(c_prime)

        return h_prime, c_prime
