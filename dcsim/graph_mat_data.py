import sys, os, string
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import cPickle
import random


neg = 40

dim = 128

def is_in(X, e):
    for x in X:
        if np.array_equal(x, e):
            return True
    return False


def load_googlejam_data_newencoding(neg_ratio = 1.0, pos_ratio = 1.0, add_positive_sample = True):
    train_filepath = "../detector/dataset/training/googlejam_newencoding/train4_128.npy"
    test_filepath = "../detector/dataset/training/googlejam_newencoding/test4_128.npy"
    dataset = cPickle.load(open(train_filepath, 'r'))
    train_data_dict, train_file_dest, train_infos = dataset['data'], dataset['file_dest'], dataset['infos']
    dataset = cPickle.load(open(test_filepath, 'r'))
    test_data_dict, test_file_dest, test_infos = dataset['data'], dataset['file_dest'], dataset['infos']
    trainX_left, trainX_right, trainY = make_pairs_encoding(train_data_dict, neg_ratio, pos_ratio=pos_ratio, add_positive_sample = add_positive_sample)
    testX_left, testX_right, testY = make_pairs_encoding_test(test_data_dict, neg_ratio, add_positive_sample = add_positive_sample)
    return trainX_left, trainX_right, trainY, testX_left, testX_right, testY


def make_pairs_10_fold(X, Y, pos_ratio = 1.0, neg_ratio=1.0, add_all_neg=False):
    indices = np.random.permutation(np.shape(Y)[0])
    X = np.array(X)[indices]
    Y = np.array(Y, dtype=np.int)[indices]
    y_dist = np.bincount(Y)
    positive_count = reduce(lambda n1, n2: n1+n2, map(lambda num: num*num/2,
                                          y_dist.tolist()))
    X_left = []
    X_right = []
    trainY = []
    p = positive_count * neg_ratio * pos_ratio / (len(X) * len(X) / 2)
    for i in xrange(len(X)):
        for j in xrange(i + 1, len(X)):
            if Y[i] == Y[j] and np.random.rand(1)[0] <= pos_ratio:
                X_left.append(X[i])
                X_right.append(X[j])
                trainY.append([0, 1])
            elif np.random.rand(1)[0] <= p or add_all_neg:
                X_left.append(X[i])
                X_right.append(X[j])
                trainY.append([1, 0])

    indices = np.random.permutation(np.shape(trainY)[0])
    sample_X_left = np.array(X_left)[indices]
    sample_X_right = np.array(X_right)[indices]
    sample_Y = np.array(trainY, dtype=np.float32)[indices]
    return sample_X_left, sample_X_right, sample_Y

# for sda_unsup and sda_base
def make_pairs_10_fold_old_model(X, Y, pos_ratio = 1.0, neg_ratio=1.0,
                        add_all_neg=False):
    indices = np.random.permutation(np.shape(Y)[0])
    X = np.array(X)[indices]
    Y = np.array(Y, dtype=np.int)[indices]
    y_dist = np.bincount(Y)
    positive_count = reduce(lambda n1, n2: n1+n2, map(lambda num: num*num/2,
                                          y_dist.tolist()))
    X_left = []
    X_right = []
    trainY = []
    p = positive_count * neg_ratio * pos_ratio / (len(X) * len(X) / 2)
    for i in xrange(len(X)):
        for j in xrange(i + 1, len(X)):
            if Y[i] == Y[j] and np.random.rand(1)[0] <= pos_ratio:
                X_left.append(X[i])
                X_right.append(X[j])
                trainY.append([0, 1])
            elif np.random.rand(1)[0] <= p or add_all_neg:
                X_left.append(X[i])
                X_right.append(X[j])
                trainY.append([1, 0])
    m = min(np.bincount(np.argmax(np.array(trainY), axis=1)))
    sample_X_left = []
    sample_X_right = []
    sample_Y = []
    iter = 0
    for i in xrange(len(X_left)):
        if iter >= m:
            break
        if trainY[i][0] ==1:
            sample_X_left.append(X_left[i])
            sample_X_right.append(X_right[i])
            sample_Y.append([1, 0])
            iter += 1
    iter = 0
    for i in xrange (len(X_left)):
        if iter >= m:
            break
        if trainY[i][1] == 1:
            sample_X_left.append(X_left[i])
            sample_X_right.append(X_right[i])
            sample_Y.append([0, 1])
            iter += 1
    indices = np.random.permutation(np.shape(sample_Y)[0])
    sample_X_left = np.array(sample_X_left)[indices]
    sample_X_right = np.array(sample_X_right)[indices]
    sample_Y = np.array(sample_Y, dtype=np.float32)[indices]
    return sample_X_left, sample_X_right, sample_Y


def make_pairs_10_fold_for_predict(X, Y):
    X_left = []
    X_right = []
    trainY = []
    for i in xrange(len(X)):
        for j in xrange(i + 1, len(X)):
            if Y[i] == Y[j]:
                X_left.append(X[i])
                X_right.append(X[j])
                trainY.append([0, 1])
            else:
                X_left.append(X[i])
                X_right.append(X[j])
                trainY.append([1, 0])
    sample_X_left = np.array(X_left)
    sample_X_right = np.array(X_right)
    sample_Y = np.array(trainY, dtype=np.float32)
    return sample_X_left, sample_X_right, sample_Y