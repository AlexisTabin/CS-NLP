#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices
    gradOutsideVectors = np.zeros(outsideVectors.shape)
    gradCenterVector = np.zeros(centerWordVec.shape)
    loss = 0.0

    # =============  YOUR CODE HERE ====================

    context = outsideVectors[outsideWordIdx]
    loss -= np.log(sigmoid(np.sum(np.dot(centerWordVec, context))))
    negSampleSum = 0
    for i in negSampleWordIndices:
        ctxi = np.dot(centerWordVec, outsideVectors[i])
        tmp_nswi = np.log(sigmoid(-np.sum(ctxi)))
        negSampleSum += tmp_nswi
    # print("vector_tmp : {tmp_vector}")
    loss = loss - negSampleSum

    ctx = np.dot(centerWordVec, context)
    tmp_sigmoid = sigmoid(np.sum(ctx)) - 1
    gradCenterVector += np.dot(tmp_sigmoid, context)

    gradOutsideVectors[outsideWordIdx] = np.dot(tmp_sigmoid, np.sum(centerWordVec, axis=0))
    for i in negSampleWordIndices:
        context = outsideVectors[i]
        tmp_minus_sig = 1 - sigmoid(-np.sum(np.dot(centerWordVec, context)))
        gradCenterVector += np.dot(tmp_minus_sig, context)
        gradOutsideVectors[i] += np.dot(tmp_minus_sig, np.sum(centerWordVec, axis=0))

    return loss, gradCenterVector, gradOutsideVectors


def skipgram(CurrentWordngramsIdx, windowSize, outsideWords, word2Ind,
             centernGramVectors, outsideVectors, dataset,
             word2vecLossAndGradient=negSamplingLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    CurrentWordngramsIdx -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center ngram vectors (as rows) for all words in vocab
                        (Z in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dZ in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """
    loss = 0.0
    gradCenterVectors = np.zeros(centernGramVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    centerWordVector = np.array([centernGramVectors[i] for i in CurrentWordngramsIdx])

    for word in outsideWords:
        outsideWordIdx = word2Ind[word]
        loss_updated, gradCenterVec, gradOutsideVecs = word2vecLossAndGradient(centerWordVector, outsideWordIdx, outsideVectors, dataset)
        loss += loss_updated
        gradCenterVectors[CurrentWordngramsIdx] += gradCenterVec
        gradOutsideVectors += gradOutsideVecs

    return loss, gradCenterVectors, gradOutsideVectors

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=negSamplingLossAndGradient):
    batchsize = 50
    loss = 0.0

    # N = wordVectors.shape[0]
    centerWordVectors, outsideVectors  = wordVectors
    grad_in = np.zeros(centerWordVectors.shape)
    grad_out = np.zeros(outsideVectors.shape)

    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, centerword_ngrams, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerword_ngrams, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad_in += gin / batchsize
        grad_out += gout / batchsize

    return loss, [grad_in, grad_out]


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)],  [random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
                    [dummy_vectors[:5,:], dummy_vectors[5:,:]], "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")


if __name__ == "__main__":
    test_word2vec()
