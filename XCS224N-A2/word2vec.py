#!/usr/bin/env python

from utils.gradcheck import gradcheck_naive
from utils.sanity_checks import *
from utils.utils import softmax
import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1. / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    center_word_vec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outside_word_idx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outside_vectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    grad_center_vec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    grad_outside_vecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)

     Note:
     we usually use column vector convention (i.e., vectors are in column form) for vectors in matrix U and V (in the handout)
     but for ease of implementation/programming we usually use row vectors (representing vectors in row form).
    """

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.

    ### YOUR CODE HERE
    scores = outside_vectors @ center_word_vec
    y_hat = softmax(scores)
    loss = -np.log(y_hat)[outside_word_idx]
    y = np.array([int(i == outside_word_idx) for i in range(len(outside_vectors))])

    pred_diff = y_hat - y
    grad_center_vec = pred_diff @ outside_vectors
    grad_outside_vecs = np.outer(pred_diff, center_word_vec)


    ### END YOUR CODE

    return loss, grad_center_vec, grad_outside_vecs


def get_negative_samples(outside_word_idx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    neg_sample_word_indices = [None] * K
    for k in range(K):
        newidx = dataset.sample_token_idx()
        while newidx == outside_word_idx:
            newidx = dataset.sample_token_idx()
        neg_sample_word_indices[k] = newidx
    return neg_sample_word_indices


def neg_sampling_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset,
        K=10
):
    """ Negative sampling loss function for word2vec models

     Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
     K is the number of negative samples to take.

     """

    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices

    grad_center_vec = np.zeros(center_word_vec.shape)
    grad_outside_vecs = np.zeros(outside_vectors.shape)

    labels = np.array([1] + [-1 for k in range(K)])
    vecs = outside_vectors[indices, :]

    t = sigmoid(vecs.dot(center_word_vec) * labels)
    loss = -np.sum(np.log(t))

    delta = labels * (t - 1)
    grad_center_vec = delta.reshape((1, K + 1)).dot(vecs).flatten()
    grad_outside_vecs_temp = delta.reshape((K + 1, 1)).dot(center_word_vec.reshape(
        (1, center_word_vec.shape[0])))
    for k in range(K + 1):
        grad_outside_vecs[indices[k]] += grad_outside_vecs_temp[k, :]

    return loss, grad_center_vec, grad_outside_vecs


def skipgram(current_center_word, window_size, outside_words, word2ind,
             center_word_vectors, outside_vectors, dataset,
             word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_center_word -- a string of the current center word
    window_size -- integer, context window size
    outside_words -- list of no more than 2*window_size strings, the outside words
    word2ind -- a dictionary that maps words to their indices in
              the word vector list
    center_word_vectors -- center word vectors (as rows) for all words in vocab
                          (V in pdf handout)
    outside_vectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vec_loss_and_gradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    grad_center_vecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    grad_outside_vectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)
    grad_outside_vectors = np.zeros(outside_vectors.shape)

    center_word_idx = word2ind[current_center_word]
    center_word_vec = center_word_vectors[center_word_idx]
    ### YOUR CODE HERE
    for word in outside_words:
        outside_word_idx = word2ind[word]
        current_loss, grad_center_vec, grad_outside_vecs = \
            word2vec_loss_and_gradient(center_word_vec, outside_word_idx,
                                       outside_vectors, dataset)
        loss += current_loss
        grad_center_vecs[center_word_idx] += grad_center_vec
        grad_outside_vectors += grad_outside_vecs
    ### END YOUR CODE

    return loss, grad_center_vecs, grad_outside_vectors


def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset,
                         window_size,
                         word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:int(N / 2), :]
    outside_vectors = word_vectors[int(N / 2):, :]
    for i in range(batchsize):
        window_size_1 = random.randint(1, window_size)
        center_word, context = dataset.get_random_context(window_size_1)

        c, gin, gout = word2vec_model(
            center_word, window_size_1, context, word2ind, center_word_vectors,
            outside_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def test_naive_softmax_loss_and_gradient():
    print("\n\n\t\t\tnaive_softmax_loss_and_gradient\t\t\t")

    dataset, dummy_vectors, dummy_tokens = dummy()

    print("\nYour Result:")
    loss, dj_dv, dj_du = naive_softmax_loss_and_gradient(
        inputs['test_naivesoftmax']['center_word_vec'],
        inputs['test_naivesoftmax']['outside_word_idx'],
        inputs['test_naivesoftmax']['outside_vectors'],
        dataset
    )

    print(
        "Loss: {}\nGradient wrt Center Vector (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(loss,
                                                                                                                  dj_dv,
                                                                                                                  dj_du))

    print("Expected Result: Value should approximate these:")
    print(
        "Loss: {}\nGradient wrt Center Vectors(dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            outputs['test_naivesoftmax']['loss'],
            outputs['test_naivesoftmax']['dj_dvc'],
            outputs['test_naivesoftmax']['dj_du']))


def test_sigmoid():
    print("\n\n\t\t\ttest sigmoid\t\t\t")

    x = inputs['test_sigmoid']['x']
    s = sigmoid(x)

    print("\nYour Result:")
    print(s)
    print("Expected Result: Value should approximate these:")
    print(outputs['test_sigmoid']['s'])


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset, dummy_vectors, dummy_tokens = dummy()

    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naive_softmax_loss_and_gradient),
                    dummy_vectors, "naive_softmax_loss_and_gradient Gradient")

    print("\n\n\t\t\tSkip-Gram with naive_softmax_loss_and_gradient\t\t\t")

    print("\nYour Result:")
    loss, dj_dv, dj_du = skipgram(inputs['test_word2vec']['current_center_word'], inputs['test_word2vec']['window_size'],
                                  inputs['test_word2vec']['outside_words'],
                                  dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                                  naive_softmax_loss_and_gradient)
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(loss,
                                                                                                                   dj_dv,
                                                                                                                   dj_du))

    print("Expected Result: Value should approximate these:")
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            outputs['test_word2vec']['loss'],
            outputs['test_word2vec']['dj_dv'],
            outputs['test_word2vec']['dj_du']))


if __name__ == "__main__":
    test_word2vec()
    test_naive_softmax_loss_and_gradient()
    test_sigmoid()
