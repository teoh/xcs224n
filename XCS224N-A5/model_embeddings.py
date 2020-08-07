#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        pad_token_idx = vocab.char2id['<pad>']
        self.char_embedding_size = 50
        self.kernel_size = 5
        self.dropout_rate = 0.3

        self.word_embedding_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id),
                                       self.char_embedding_size,
                                       padding_idx=pad_token_idx)
        self.cnn = CNN(self.char_embedding_size,
                       kernel_size=self.kernel_size,
                       num_filters=self.word_embedding_size)
        self.highway = Highway(self.word_embedding_size,
                               self.dropout_rate)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        char_embeddings_raw = self.embeddings(input_tensor)
        char_embeddings_mword_last = torch.transpose(char_embeddings_raw, -1, -2).contiguous()
        sentence_length, batch_size, char_embedding_size, max_word_length = \
            char_embeddings_mword_last.shape

        # TODO: if problems look here for causes
        char_embeddings_megabatch = char_embeddings_mword_last.view(
            -1, char_embedding_size, max_word_length)

        x_conv_out_raw = self.cnn(char_embeddings_megabatch)

        x_highway_raw = self.highway(x_conv_out_raw)

        output = x_highway_raw.reshape(sentence_length, batch_size, -1).contiguous()

        return output
        ### END YOUR CODE
