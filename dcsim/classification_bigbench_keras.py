# -*- coding: utf-8 -*-

from __future__ import print_function, division

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils import np_utils, Sequence
from keras.utils.vis_utils import plot_model
import keras as K
import numpy as np
import pandas as pd
import os
import time

import matplotlib
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight, shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# matplotlib.rcParams['font.family']=''
matplotlib.rcParams['font.weight'] = 'bold'

import graph_mat_data
import preprocessing_bigbench

bin_vec_dim = 88
embedding_dim = 6
dim = 128
keep_prob = 0.6

batch_size = 256
test_size = 256


# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logdir = '/tmp/logs'

kernel_init = K.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                             distribution='uniform')
bias_init = K.initializers.Constant(value=0.01)


def stat_by_type(y_true, y_pred, ts, fout=None):
    print('*' * 40 + " Performance by Type " + '*' * 40)
    # T1
    indices = np.where(ts==0)
    accuracy = accuracy_score(y_true[indices], y_pred[indices])
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(y_true[indices], y_pred[indices],
                                        average='binary')
    print("T1: accuracy: %.4f, recall: %.4f, "
          "precision: %.4f, f1 score: %.4f\n" % (
              accuracy, recall, precision, fscore))
    if fout is not None:
        fout.write("T1: accuracy: %.4f, recall: %.4f, "
              "precision: %.4f, f1 score: %.4f\n" % (
                  accuracy, recall, precision, fscore))
    
    #T2
    indices = np.where(ts == 1)
    accuracy = accuracy_score(y_true[indices], y_pred[indices])
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(y_true[indices], y_pred[indices],
                                        average='binary')
    print("T2: accuracy: %.4f, recall: %.4f, "
          "precision: %.4f, f1 score: %.4f\n" % (
              accuracy, recall, precision, fscore))
    if fout is not None:
        fout.write("T2: accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                       accuracy, recall, precision, fscore))
    
    # ST3
    indices = np.where(ts == 2)
    accuracy = accuracy_score(y_true[indices], y_pred[indices])
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(y_true[indices], y_pred[indices],
                                        average='binary')
    print("ST3: accuracy: %.4f, recall: %.4f, "
          "precision: %.4f, f1 score: %.4f\n" % (
              accuracy, recall, precision, fscore))
    if fout is not None:
        fout.write("ST3: accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                       accuracy, recall, precision, fscore))
    
    #MT3
    indices = np.where(ts == 3)
    accuracy = accuracy_score(y_true[indices], y_pred[indices])
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(y_true[indices], y_pred[indices],
                                        average='binary')
    print("MT3: accuracy: %.4f, recall: %.4f, "
          "precision: %.4f, f1 score: %.4f\n" % (
              accuracy, recall, precision, fscore))
    if fout is not None:
        fout.write("MT3: accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                       accuracy, recall, precision, fscore))
    
    indices = np.where(ts == 4)
    accuracy = accuracy_score(y_true[indices], y_pred[indices])
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(y_true[indices], y_pred[indices],
                                        average='binary')
    print("WT3/T4: accuracy: %.4f, recall: %.4f, "
          "precision: %.4f, f1 score: %.4f\n" % (
              accuracy, recall, precision, fscore))
    if fout is not None:
        fout.write("WT3/T4: accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                       accuracy, recall, precision, fscore))

def from_sparse_arr(sparse_arr):
    mat = np.zeros((dim, dim, bin_vec_dim), dtype=np.float32)
    for (i, j, k) in sparse_arr:
        mat[i, j, k] = 1
    return mat

def from_sparse_arrs(sparse_arrs):
    mats = []
    for sparse_arr in sparse_arrs:
        mats.append(from_sparse_arr(sparse_arr))
    mats = np.array(mats, dtype=np.float32)
    return mats

def fit_generator(Xl, Xr, Y):
    '''
    Best set worker=1, use_multiprocessing=False
    :param Xl:
    :param Xr:
    :param Y:
    :return:
    '''
    while True:
        Xl, Xr, Y = shuffle(Xl, Xr, Y)
        batch_Xl = []
        batch_Xr = []
        batch_y = []
        count = 0
        for (xl, xr, y) in zip(Xl, Xr, Y):
            batch_Xl.append(from_sparse_arr(xl))
            batch_Xr.append(from_sparse_arr(xr))
            batch_y.append(y)
            count += 1
            if len(batch_y) == batch_size or count == np.shape(Y)[0]:
                yield ([np.array(batch_Xl), np.array(batch_Xr)],
                       np.expand_dims(np.array(batch_y, dtype=np.float32),
                                      axis=1))
                batch_Xl = []
                batch_Xr = []
                batch_y = []

class SequenceSamples(Sequence):
    def __init__(self, Xl, Xr, Y, batch_size):
        self.Xl, self.Xr, self.Y = Xl, Xr, Y
        self.batch_size = batch_size
    
    def __len__(self):
        return np.ceil(np.shape(self.Y)[0] / batch_size)
        
    def __getitem__(self, item):
        batch_Xl = from_sparse_arrs(self.Xl[item * self.batch_size:(item + 1) * self.batch_size])
        batch_Xr = from_sparse_arrs(self.Xr[item * self.batch_size:(item + 1) * self.batch_size])
        # Y shouldn't be (256,), it should has the same shape as the model's
        # output
        batch_Y = self.Y[item * self.batch_size:(item+1)*self.batch_size]\
                        .reshape(batch_size, 1)
        print("Batch size: ", batch_Xl.shape[0], batch_Xr.shape[0],
              batch_Y.shape[0])
        return ([batch_Xl, batch_Xr], batch_Y)
    

def feed_forward(x):
    x = Lambda(lambda input: K.backend.reshape(input, (-1, bin_vec_dim)),
               batch_input_shape=K.backend.get_variable_shape(x))(x)
    x = Dense(embedding_dim,
              kernel_initializer=kernel_init,
              bias_initializer=bias_init)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Lambda(
        lambda input: K.backend.reshape(input, (-1, dim * embedding_dim)))(x)
    x = Dense(256, kernel_initializer=kernel_init,
              bias_initializer=bias_init)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dropout(keep_prob)(x)
    
    x = Dense(64,
              kernel_initializer=kernel_init,
              bias_initializer=bias_init)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dropout(keep_prob)(x)
    x = Lambda(lambda input: K.backend.reshape(input, (-1, dim, 64)))(x)
    x = GlobalAveragePooling1D()(x)  # (batch_size, 64)
    return x

def classification(x1, x2):
    input = Input(shape=(dim, dim, bin_vec_dim))
    # share layers
    feed_forward_model = Model(inputs=input, outputs=feed_forward(input))
    x1 = feed_forward_model(x1)
    x2 = feed_forward_model(x2)
    concat_input = Input(shape=(128,))
    # share layers
    merge_model = Model(inputs=concat_input,
                        outputs=Activation(activation='relu')(
                            BatchNormalization()(
                                Dense(32, kernel_initializer=kernel_init,
                                      bias_initializer=bias_init,
                                      input_shape=(128,))(
                                    concat_input))))
    
    xc1 = K.layers.concatenate([x1, x2])
    xc1 = merge_model(xc1)
    
    xc2 = K.layers.concatenate([x2, x1])
    xc2 = merge_model(xc2)
    
    xc = K.layers.average([xc1, xc2])
    
    x = Dense(1, use_bias=False, activation='sigmoid',
              kernel_initializer=kernel_init,
              batch_input_shape=K.backend.get_variable_shape(xc))(xc)
    
    return x

def model_summary():
    X_left = Input((dim, dim, bin_vec_dim))
    X_right = Input((dim, dim, bin_vec_dim))
    predictions = classification(X_left, X_right)
    model = Model(inputs=[X_left, X_right], outputs=predictions)
    model.compile(optimizer=K.optimizers.adam(lr=0.0005),
                  loss=K.losses.binary_crossentropy,
                  metrics=['accuracy'])
    
    # plot_model(model, to_file='./result/plot/whole_model.png', show_shapes=True)

def train_10_fold_balanced():
    
    skf = StratifiedKFold(n_splits=10)
    
    Xl, Xr, y, ts = preprocessing_bigbench.load_dataset()
    
    fold_index = 0
    avg_accuracy = 0.
    avg_recall = 0.
    avg_precision = 0.
    avg_f1_score = 0.
    fout = open('result/10_fold_balanced.txt', 'w')
    if os.path.exists('result') is not True:
        os.mkdir("result")
    if os.path.exists("10_fold_balanced") is not True:
        os.mkdir("10_fold_balanced")
    for train_idx, test_idx in skf.split(Xl, y):
        t_beg = time.clock()
        
        print ('*' * 40 + str(fold_index) + '*' * 40)
        fold_path = os.path.join("10_fold_balanced", str(fold_index))
        if os.path.exists(fold_path) is not True:
            os.mkdir(fold_path)
        
        train_X_left = Xl[train_idx]
        train_X_right = Xr[train_idx]
        train_Y = y[train_idx]
        
        train_Yt = train_Y[train_Y == 0]
        train_Xlt = train_X_left[train_Y == 0]
        train_Xrt = train_X_right[train_Y == 0]
        train_Xl = train_X_left[train_Y == 1][:5 * train_Yt.shape[0]]
        train_Xr = train_X_right[train_Y == 1][:5 * train_Yt.shape[0]]
        train_y = train_Y[train_Y == 1][:5 * train_Yt.shape[0]]
        train_X_left = np.concatenate((train_Xlt, train_Xl), axis=0)
        train_X_right = np.concatenate((train_Xrt, train_Xr), axis=0)
        train_Y = np.concatenate((train_Yt, train_y), axis=0)
        train_X_left, train_X_right, train_Y = shuffle(train_X_left,
                                                       train_X_right, train_Y)
        
        test_X_left = Xl[test_idx]
        test_X_right = Xr[test_idx]
        test_Y = y[test_idx]
        test_ts = ts[test_idx]
        
        validate_X_left = from_sparse_arrs(test_X_left[:256])
        validate_X_right = from_sparse_arrs(test_X_right[:256])
        validate_Y = test_Y[:256]

        X_left = Input(shape=(dim, dim, bin_vec_dim))
        X_right = Input(shape=(dim, dim, bin_vec_dim))

        predictions = classification(X_left, X_right)

        model = Model(inputs=[X_left, X_right], outputs=predictions)

        model.compile(optimizer=K.optimizers.adam(lr=0.001),
                      loss=K.losses.binary_crossentropy,
                      metrics=['accuracy'])
        samples_generator = SequenceSamples(train_X_left,train_X_right,
                                            train_Y, batch_size)
        model.fit_generator(fit_generator(train_X_left, train_X_right, train_Y),
                            steps_per_epoch=np.ceil(train_Y.shape[0]/batch_size),
                            epochs=1, verbose=1,
                            workers=1, use_multiprocessing=False,
                            validation_data=([validate_X_left, validate_X_right], validate_Y))
        
        t_end = time.clock()
        print('Time cost: %.2f' % (t_end - t_beg))
        
        model.save(filepath=os.path.join(fold_path, 'model.ckpt'))
        
        print("Evaluation:")

        test_samples_generator = SequenceSamples(test_X_left, test_X_right,
                                                 test_Y, batch_size),
        y_pred = model.predict_generator(fit_generator(test_X_left,
                                                       test_X_right, test_Y),
                            steps=np.ceil(test_Y.shape[0] / batch_size),
                            workers=1, use_multiprocessing=False)
        y_pred = np.round(y_pred)
        accuracy = accuracy_score(test_Y, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(test_Y,
                                                                 y_pred, average='binary')
        print("Fold index: %d, accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                   fold_index, accuracy, recall, precision, fscore))
        fout.write('*' * 80 + '\n')
        fout.write('Fold %d:\n' % (fold_index))
        fout.write('Time cost: %.2f\n' % (t_end - t_beg))
        fout.write("Fold index: %d, accuracy: %.4f, recall: %.4f, "
                   "precision: %.4f, f1 score: %.4f\n" % (
                   fold_index, accuracy, recall, precision, fscore))
        stat_by_type(test_Y, y_pred, test_ts, fout)
        fout.flush()
        avg_accuracy += accuracy
        avg_precision += precision
        avg_recall += recall
        avg_f1_score += fscore
        
    avg_accuracy /= 10.0
    avg_precision /= 10.0
    avg_recall /= 10.0
    avg_f1_score /= 10.0
    print('Avg accuracy: %.4f, avg recall: %.4f, avg precision: %.4f, avg f1 '
          'score: %.4f' % (
              avg_accuracy, avg_recall, avg_precision, avg_f1_score))
    fout.write('*' * 80 + '\n')
    fout.write(
        'Avg accuracy: %.4f, avg recall: %.4f, avg precision: %.4f, avg f1 '
        'score: %.4f' % (avg_accuracy, avg_recall, avg_precision, avg_f1_score))
    fout.close()

def train_on_selected_id():
    t_beg = time.clock()

    Xl_selected, Xr_selected, y_selected, ts_selected, Xl, Xr, y, ts = preprocessing_bigbench.load_train_test(id=4)
    
    train_X_left = Xl_selected
    train_X_right = Xr_selected
    train_Y = y_selected

    train_Yt = train_Y[train_Y == 0]
    train_Xlt = train_X_left[train_Y == 0]
    train_Xrt = train_X_right[train_Y == 0]
    train_Xl = train_X_left[train_Y == 1][:5 * train_Yt.shape[0]]
    train_Xr = train_X_right[train_Y == 1][:5 * train_Yt.shape[0]]
    train_y = train_Y[train_Y == 1][:5 * train_Yt.shape[0]]
    train_X_left = np.concatenate((train_Xlt, train_Xl), axis=0)
    train_X_right = np.concatenate((train_Xrt, train_Xr), axis=0)
    train_Y = np.concatenate((train_Yt, train_y), axis=0)
    train_X_left, train_X_right, train_Y = shuffle(train_X_left,
                                                   train_X_right, train_Y)
    print("Training data size: ", train_Y.shape[0])
    
    test_X_left = Xl
    test_X_right = Xr
    test_Y = y
    test_ts = ts
    
    validate_X_left = from_sparse_arrs(test_X_left[:256])
    validate_X_right = from_sparse_arrs(test_X_right[:256])
    validate_Y = test_Y[:256]
    
    X_left = Input(shape=(dim, dim, bin_vec_dim))
    X_right = Input(shape=(dim, dim, bin_vec_dim))
    
    predictions = classification(X_left, X_right)
    
    model = Model(inputs=[X_left, X_right], outputs=predictions)
    
    model.compile(optimizer=K.optimizers.adam(lr=0.001),
                  loss=K.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.fit_generator(fit_generator(train_X_left, train_X_right, train_Y),
                        steps_per_epoch=np.ceil(train_Y.shape[0] / batch_size),
                        epochs=1, verbose=1,
                        workers=1, use_multiprocessing=False,
                        validation_data=(
                        [validate_X_left, validate_X_right], validate_Y))
    
    t_end = time.clock()
    print('Time cost: %.2f' % (t_end - t_beg))
    
    model.save(filepath=os.path.join('./model', 'model_id4.ckpt'))
    
    print("Evaluation:")
    
    y_pred = model.predict_generator(fit_generator(test_X_left,
                                                   test_X_right, test_Y),
                                     steps=np.ceil(test_Y.shape[0] /
                                                   batch_size),
                                     workers=1, use_multiprocessing=False)
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(test_Y, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_Y,
                                                                   y_pred,
                                                                   average='binary')
    print("accuracy: %.4f, recall: %.4f, "
          "precision: %.4f, f1 score: %.4f\n" % (
              accuracy, recall, precision, fscore))
    
    stat_by_type(test_Y, y_pred, test_ts)


if __name__ == '__main__':
    # model_summary()
    beg = time.time()
    train_10_fold_balanced()
    st = time.time()
    print("Total time: ", st-beg)