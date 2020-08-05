#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
### YOUR CODE HERE for part 1d

class Highway(nn.Module):
    def __init__(self, embed_size, dropout_rate):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate

        self.proj_projection = nn.Linear(self.embed_size, self.embed_size)
        self.gate_projection = nn.Linear(self.embed_size, self.embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x_conv_out):
        """Take a minibatch of x_conv_out vectors and return
        x_highway as in the assignment PDF

        @param x_conv_out: shape (batch_size, embed_size)

        @returns x_highway: shape (batch_size, embed_size)
        """
        x_proj_pre = self.proj_projection(x_conv_out)
        x_proj = nn.ReLU()(x_proj_pre)

        x_gate_pre = self.gate_projection(x_conv_out)
        x_gate = nn.Sigmoid()(x_gate_pre)

        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1 - x_gate), x_conv_out)
        return x_highway
### END YOUR CODE

