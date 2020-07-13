#!/usr/bin/env python

import sys
import os
import numpy as np
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from functools import reduce

sys.path.append(os.path.abspath(os.path.join('..')))

from utils.sanity_checks import *

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

### SOLUTION BEGIN
    distinct_set = reduce(lambda s1, s2: s1.union(s2), [set(doc) for doc in corpus])
    corpus_words = sorted(list(distinct_set))
    num_corpus_words = len(corpus_words)
### SOLUTION END

    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurrence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    def get_rough_window(center_idx, window_size, doc):
        w_lower, w_higher = center_idx - window_size, center_idx + window_size + 1
        return range(max(0, w_lower),
                     min(len(doc), w_higher))

    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))

### SOLUTION BEGIN
    word2Ind = {word: ind for ind, word in enumerate(words)}
    for doc in corpus:
        for center_idx, center_word in enumerate(doc):
            for window_idx in get_rough_window(center_idx, window_size, doc):
                if window_idx != center_idx:
                    window_word_idx = word2Ind[doc[window_idx]]
                    center_word_idx = word2Ind[center_word]
                    M[center_word_idx][window_word_idx] += 1


### SOLUTION END

    return M, word2Ind


def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of number of corpus words)): co-occurrence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    np.random.seed(4355)
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    ### SOLUTION BEGIN
    M_reduced = TruncatedSVD(n_components=k, n_iter=n_iters).fit_transform(M)
    ### SOLUTION END

    print("Done.")
    return M_reduced

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def test_distinct_words():
    print("\n\t\t\t Testing distinct_words \t\t\t")

    test_corpus = toy_corpus()
    test_corpus_words, num_corpus_words = distinct_words(test_corpus)

    ans_test_corpus_words = sorted(list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))
    ans_num_corpus_words = len(ans_test_corpus_words)

    print("\nYour Result:")
    print(
        "Words in corpus: {}\n Number of words in corpus: {}\n".format(test_corpus_words,
                                                                        num_corpus_words
                                                                        )
    )

    print("Expected Result:")
    print(
        "Words in corpus: {}\n Number of words in corpus: {}\n".format(ans_test_corpus_words,
                                                                        ans_num_corpus_words
                                                                        )
    )

def test_compute_co_occurrence_matrix():
    print("\n\t\t\t Testing compute_co_occurrence_matrix \t\t\t")

    test_corpus = toy_corpus()
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=2)

    M_test_ans, word2Ind_test_ans = toy_corpus_co_occurrence()

    for w1 in word2Ind_test_ans.keys():
        idx1 = word2Ind_test_ans[w1]
        for w2 in word2Ind_test_ans.keys():
            idx2 = word2Ind_test_ans[w2]
            student = M_test[idx1, idx2]
            correct = M_test_ans[idx1, idx2]
            if student != correct:
                print("Correct M:")
                print(M_test_ans)
                print("Your M: ")
                print(M_test)
                raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

    print("\nYour Result:")
    print(
        "Shape of co-occurrence matrix: {}\n Word to index map: {}\n".format(M_test.shape,
                                                                        word2Ind_test
                                                                        )
    )

    print("\nExpected Result:")
    print(
        "Shape of co-occurrence matrix: {}\n Word to index map: {}\n".format(M_test_ans.shape,
                                                                        word2Ind_test_ans
                                                                        )
    )

def test_reduce_to_k_dim():
    print("\n\t\t\t Testing reduceToKDim \t\t\t")

    M_test_ans, word2Ind_test_ans = toy_corpus_co_occurrence()
    M_test_reduced = reduce_to_k_dim(M_test_ans, k=2)

    print("\nYour Result:")
    print(
        "Shape of reduced dim co-occurrence matrix: {}\n".format(M_test_reduced.shape
                                                                        )
    )

    print("\nExpected Result:")
    print(
        "Shape of reduced dim co-occurrence matrix: {}\n".format((10, 2)
                                                                        )
    )

if __name__ == "__main__":
    test_distinct_words()
    test_compute_co_occurrence_matrix()
    test_reduce_to_k_dim()
