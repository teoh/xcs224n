#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.char_embedding_size = char_embedding_size
        self.hidden_size = hidden_size
        self.target_vocab = target_vocab
        self.target_vocab_size = len(self.target_vocab.char2id)

        self.charDecoder = nn.LSTM(self.char_embedding_size,
                                   self.hidden_size)
        self.char_output_projection = nn.Linear(in_features=self.hidden_size,
                                                out_features=self.target_vocab_size)
        self.decoderCharEmb = nn.Embedding(num_embeddings=self.target_vocab_size,
                                           embedding_dim=self.char_embedding_size,
                                           padding_idx=self.target_vocab.char2id['<pad>'])
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_embeddings = self.decoderCharEmb(input)
        if dec_hidden is not None:
            lstm_output, (h_n, c_n) = self.charDecoder(input_embeddings, dec_hidden)
        else:
            lstm_output, (h_n, c_n) = self.charDecoder(input_embeddings)
        scores = self.char_output_projection(lstm_output)

        return scores, (h_n, c_n)
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, (h_n, c_n) = self.forward(char_sequence, dec_hidden=dec_hidden)

        # TODO: if something goes wrong look here
        # hint: use boolean indexing after a view
        scores_sliced = scores[:-1]
        char_sequence_sliced = char_sequence[1:]
        pad_mask = char_sequence_sliced != 0

        scores_shaped = scores_sliced.reshape(-1, self.target_vocab_size)
        char_shaped = char_sequence_sliced.reshape(-1)
        pad_mask_shaped = pad_mask.reshape(-1)

        scores_masked = scores_shaped[pad_mask_shaped]
        char_masked = char_shaped[pad_mask_shaped]

        loss = self.cross_entropy_loss(scores_masked, char_masked)

        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        state = initialStates
        batch_size = initialStates[0].shape[1]
        sequences = torch.zeros((max_length + 1, batch_size),
                                dtype=torch.long,
                                device=device)
        sequences[0] = self.target_vocab.start_of_word

        for step in range(max_length):
            scores, state = self.forward(sequences[step].unsqueeze(0),
                                         state)
            sequences[step + 1] = scores.argmax(-1)

        chars_raw = list(map(lambda id_list: [self.target_vocab.id2char[i] for i in id_list],
                             sequences.transpose(0, 1).numpy()))
        decodedWords = list(map(lambda chars: ''.join(self._clean_char_list(chars)),
                                chars_raw))
        return decodedWords
        ### END YOUR CODE

    def _clean_char_list(self, chars_raw):
        end_ind = 0
        while end_ind < len(chars_raw) and chars_raw[end_ind] not in ['}', '<pad>']:
            end_ind += 1
        return chars_raw[1:end_ind]
