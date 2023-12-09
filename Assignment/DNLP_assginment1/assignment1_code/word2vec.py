#!/usr/bin/env python

import argparse
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid_fct(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    sig -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)

    #sigmoid(x)=1/(1+exp(-x))
    sig=1/(1+np.exp(-x))

    ### END YOUR CODE

    return sig


def NaiveSoftmaxLossGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss and gradient function for word2vec models
    The goal is to create a function that can calculate the naive softmax loss and gradients 
    between the embeddings of a center word and an outside word. 
    This function will serve as a foundation for our word2vec models. 
    For individuals who are not familiar with numpy notation, 
    it's important to note that a numpy ndarray with a shape of (x, ) is essentially a one-dimensional array 
    that can be thought of as a vector with a length of x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    # softmax = exp((u_o)^T*v_c)/SUM(exp((u_w)^T*v_c)) with w in vocab
    naive_softmax = softmax(np.dot(outsideVectors, centerWordVec))
    # this calculates the softmax of each word

    #------- IGNORE THE FOLLOWING LINE ---------#
    # outsideVectors is a matrix of N word (each word is a vector M)
    # u is MxN => uT is NxM
    # centerWordVec is 1xM vector, but it is treated as Mx1 matrix cause numpy auto reshape it depend on the second array
    # value of dot product is Nx1
    # apply softmax we have probability of each word that can be an outside word when knowing the center word

    # J = -log(softmax(O=outside_word))
    loss = -np.log(naive_softmax[outsideWordIdx])

    # ∂J/∂v_c=-u_o+SUM(y_hat_x*u_x)
    # with u_o is the input outsideVector, y_hat is the softmax and u is all the outsideVector
    GradCenterVec = -outsideVectors[outsideWordIdx] + np.dot(naive_softmax, outsideVectors)

    # ∂J/∂u_o=v_c*(-y_hat_o)
    # ∂J/∂u_w=v_c*y_hat_w with all w!=o
    # So first we calculate all the gradient of u_w
    GradOutsideVecs = np.dot(naive_softmax.reshape(-1, 1), centerWordVec.reshape(1, -1))
    # And then replace the u_o with the true value
    GradOutsideVecs[outsideWordIdx] = np.dot((naive_softmax[outsideWordIdx].reshape(-1, 1) - 1.0),
                                             centerWordVec.reshape(1, -1))
    ### END YOUR CODE

    return loss, GradCenterVec, GradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def NegSamplingLossGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    To create a building block for word2vec models, 
    a function is needed that can calculate the negative sampling loss and gradients for a center word vector 
    and an outside word index. This function should take K negative samples. 
    
    It's worth noting that the same word can be negatively sampled multiple times, 
    which means that if an outside word is sampled more than once, 
    the gradient with respect to that word must be counted multiple times as well 
    (twice if sampled twice, thrice if sampled three times, etc.).

    Arguments/Return Specifications: same as NaiveSoftmaxLossGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)
    # value=u_w^T*v_c
    values = np.dot(outsideVectors[indices], centerWordVec)
    # calculate sigmoid(u_o^T*v_c) and sigmoid(-u_k^T*v_c), o is outsideWord and k is k negative samples
    sigOutsideWords = sigmoid_fct(values[0])
    sigNegativeWords = sigmoid_fct(-values[1:])
    # J =-log(sigmoid(u_o^T*v_c)) - SUM(log(sigmoid(-u_k^T*v_c)))
    loss = -np.log(sigOutsideWords) - np.sum(np.log(sigNegativeWords))
    # ∂J/∂v_c=(sigmoid(u_o^T*v_c)-1)*u_o - SUM((sigmoid(-u_k^T*v_c)-1)*u_k)
    GradCenterVec = np.dot((sigOutsideWords - 1.0), outsideVectors[outsideWordIdx]) - \
                    np.sum((sigNegativeWords - 1.0)[:, np.newaxis] * outsideVectors[negSampleWordIndices], axis=0)
    # initialize the gradient with value = 0
    GradOutsideVecs = np.zeros_like(outsideVectors)
    # ∂J/∂u_o=(sigmoid(u_o^T*v_c)-1)*v_c
    GradOutsideVecs[outsideWordIdx] = (sigOutsideWords - 1.0) * centerWordVec
    for i in range(K):
        # ∂J/∂u_w=(1-sigmoid(-u_k^T*v_c))*v_c
        GradOutsideVecs[negSampleWordIndices[i]] += (1.0 - sigNegativeWords[i]) * centerWordVec
    ### END YOUR CODE

    return loss, GradCenterVec, GradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=NaiveSoftmaxLossGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    GradCenterVecs = np.zeros(centerWordVectors.shape)
    GradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    loss_gradients = []
    # go through each of the outsideWords to calculate grad and loss
    for word in outsideWords:
        # apply the function to get
        lossGrad = list(word2vecLossAndGradient(
            centerWordVectors[word2Ind[currentCenterWord]], word2Ind[word], outsideVectors, dataset))
        # update loss
        loss += lossGrad[0]
        # update grad for center vector
        GradCenterVecs[word2Ind[currentCenterWord]] += lossGrad[1]
        # update grad for outside vectors
        GradOutsideVectors += lossGrad[2]
    ### END YOUR CODE
    
    return loss, GradCenterVecs, GradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vecSGD_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=NaiveSoftmaxLossGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad

def test_sigmoid():
    """ Test sigmoid function """
    print("=== Sanity check for sigmoid ===")
    assert sigmoid_fct(0) == 0.5
    assert np.allclose(sigmoid_fct(np.array([0])), np.array([0.5]))
    assert np.allclose(sigmoid_fct(np.array([1,2,3])), np.array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")

def getDummyObjects():
    """ Helper method for NaiveSoftmaxLossGradient and NegSamplingLossGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    return dataset, dummy_vectors, dummy_tokens

def test_NaiveSoftmaxLossGradient():
    """ Test NaiveSoftmaxLossGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for NaiveSoftmaxLossGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = NaiveSoftmaxLossGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "NaiveSoftmaxLossGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = NaiveSoftmaxLossGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "NaiveSoftmaxLossGradient gradOutsideVecs")

def test_NegSamplingLossGradient():
    """ Test NegSamplingLossGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for NegSamplingLossGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = NegSamplingLossGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "NegSamplingLossGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = NegSamplingLossGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "NegSamplingLossGradient gradOutsideVecs")

def test_skipgram():
    """ Test skip-gram with NaiveSoftmaxLossGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with NaiveSoftmaxLossGradient ====")
    gradcheck_naive(lambda vec: word2vecSGD_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, NaiveSoftmaxLossGradient),
        dummy_vectors, "NaiveSoftmaxLossGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with NegSamplingLossGradient ====")
    gradcheck_naive(lambda vec: word2vecSGD_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, NegSamplingLossGradient),
        dummy_vectors, "NegSamplingLossGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, NegSamplingLossGradient)

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    test_sigmoid()
    test_NaiveSoftmaxLossGradient()
    test_NegSamplingLossGradient()
    test_skipgram()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()
    if args.function == 'sigmoid':
        test_sigmoid()
    elif args.function == 'NaiveSoftmaxLossGradient':
        test_NaiveSoftmaxLossGradient()
    elif args.function == 'NegSamplingLossGradient':
        test_NegSamplingLossGradient()
    elif args.function == 'skipgram':
        test_skipgram()
    elif args.function == 'all':
        test_word2vec()
