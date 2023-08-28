# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:04:56 2022

@author: windows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()
        
        self.eps = eps
        self.hidden_size = hidden_size
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        
    def forward(self, z):
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a2 + self.b2
        return ln_out
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        # lstm weights
        self.weight_fh = nn.Linear(hidden_size, hidden_size)
        self.weight_ih = nn.Linear(hidden_size, hidden_size)
        self.weight_ch = nn.Linear(hidden_size, hidden_size)
        self.weight_oh = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_cx = nn.Linear(input_size, hidden_size)
        self.weight_ox = nn.Linear(input_size, hidden_size)
        # decoder
        self.decoder = nn.Linear(hidden_size, output_size)
        # layer normalization
        self.lnx = LayerNormalization(hidden_size)
        self.lnh = LayerNormalization(hidden_size)
        self.lnc = LayerNormalization(hidden_size)

    def forward(self, inp, h_0, c_0):
        # forget gate
        f_g = F.sigmoid(self.lnx(self.weight_fx(inp)) + self.lnh(self.weight_fh(h_0)))
        # input gate
        i_g = F.sigmoid(self.lnx(self.weight_ix(inp)) + self.lnh(self.weight_ih(h_0)))
        # intermediate cell state
        c_tilda = F.tanh(self.lnx(self.weight_cx(inp)) + self.lnh(self.weight_ch(h_0)))
        # current cell state
        cx = f_g * c_0 + i_g * c_tilda
        # output gate
        o_g = F.sigmoid(self.lnx(self.weight_ox(inp)) + self.lnh(self.weight_oh(h_0)))
        # hidden state
        hx = o_g * F.tanh(self.lnc(cx))
        hx1 = torch.Tensor(hx.tolist()[0][-1])

        out = self.decoder(hx1.view(1,-1))

        return out, hx, cx

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size))
        c_0 = Variable(torch.zeros(1, self.hidden_size))
        return h_0, c_0