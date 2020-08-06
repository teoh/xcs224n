#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
### YOUR CODE HERE for part 1e

class Cnn(nn.Module):
    def __init__(self, embed_size, kernel_size, num_filters):
        super(Cnn, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.embed_size = embed_size

        self.conv = nn.Conv1d(in_channels=self.embed_size,
                              out_channels=self.num_filters,
                              kernel_size=self.kernel_size,)

    def forward(self, x_reshaped):
        """Take a minibatch of x_reshaped vectors and return
        x_conv_out as in the assignment PDF

        @param x_reshaped: shape (batch_size, embed_size, max_word_length)

        @returns x_conv_out: shape (batch_size, embed_size)
        """
        x_conv = self.conv(x_reshaped)
        x_conv_relu = nn.ReLU()(x_conv)
        x_conv_out = torch.max(x_conv_relu, dim=-1)[0]

        return x_conv_out


### END YOUR CODE

