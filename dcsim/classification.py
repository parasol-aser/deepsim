# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time

import matplotlib
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set(style='white')
# matplotlib.rcParams['font.family']=''
matplotlib.rcParams['font.weight'] = 'bold'

import graph_mat_data

bin_vec_dim = 88
embedding_dim = 6
dim = 128
keep_prob = 0.75

batch_size = 256
test_size = 256

beta = 0.00003
# beta = 0.00001 # for model with batch normalization
reg_term = None

# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logdir = '/tmp/tf_logs'


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def relu(x, alpha=0., max_value=None):
    """Rectified linear unit.
    With default values, it returns element-wise `max(x, 0)`.
    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: Saturation threshold.
    # Returns
        A tensor.
    """
    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        max_value = _to_tensor(max_value, x.dtype.base_dtype)
        zero = _to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)
    if alpha != 0.:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x


def batch_act(h, act, phase, scope):
    with tf.variable_scope(scope):
        return act(h)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def init_weights(shape, name):
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                           initializer=tf.contrib.layers.variance_scaling_initializer(
                               factor=1.0, mode='FAN_AVG', uniform=True))

def init_bias(shape, name):
    if len(shape) > 1:
        raise Exception('Bias should be a vector.')
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                           initializer=tf.constant_initializer(
                               0.01))

def model(X, dropout, phase):
    global reg_term
    num = tf.shape(X)[0]
    with tf.name_scope('emb_layer'):
        wf = init_weights([bin_vec_dim, embedding_dim], 'wf')
        reg_term = tf.nn.l2_loss(wf)
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, wf)
        variable_summaries(wf)
        bf = init_bias([embedding_dim], 'bf')
        variable_summaries(bf)
        X = tf.reshape(X, [num * dim * dim, bin_vec_dim])
        # h0 = tf.nn.elu(tf.nn.bias_add(tf.matmul(X, wf), bf))
        h0 = tf.nn.bias_add(tf.matmul(X, wf), bf)
        h0 = batch_act(h0, phase=phase, act=tf.nn.elu, scope='emb_layer_bn')
        h0 = tf.reshape(h0, [num * dim, dim * embedding_dim])
        h0 = tf.nn.dropout(h0, dropout)
    with tf.name_scope('row_fc_layer1'):
        wr1 = init_weights([embedding_dim * dim, 256], 'wr1')  # 128
        reg_term += tf.nn.l2_loss(wr1)
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, wr1)
        br1 = init_bias([256], 'br1')
        # h1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h0, wr1), br1))
        h1 = tf.nn.bias_add(tf.matmul(h0, wr1), br1)
        h1 = batch_act(h1, phase=phase, act=tf.nn.elu, scope='row_fc_layer1_bn')
        h1 = tf.nn.dropout(h1, dropout)
    with tf.name_scope('row_fc_layer2'):
        wr2 = init_weights([256, 64], 'wr2')  # 32
        reg_term += tf.nn.l2_loss(wr2)
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, wr2)
        br2 = init_bias([64], 'br2')
        # h2 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h1, wr2), br2))
        h2 = tf.nn.bias_add(tf.matmul(h1, wr2), br2)
        h2 = batch_act(h2, phase=phase, act=tf.nn.elu, scope='row_fc_layer2_bn')
        h2 = tf.reshape(h2, [num, dim, 64])  # 32
    with tf.name_scope('avg_pooling'):
        h3 = tf.reduce_mean(h2, 1)
    return h3


def classification(X1, X2, dropout, phase):
    global reg_term
    with tf.variable_scope('encoding') as scope:
        h31 = model(X1, dropout, phase)
        scope.reuse_variables()
        h32 = model(X2, dropout, phase)
    # h4 = tf.concat(1, [h31, h32])  #old tf version
    h41 = tf.concat(values=[h31, h32], axis=1)
    with tf.name_scope('fc_layer1_1'):
        w5 = init_weights([128, 32], 'w5')  # 64 16
        reg_term += tf.nn.l2_loss(w5)
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w5)
        b5 = init_bias([32], 'b5')
        # h5_1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h41, w5), b5))
        h5_1 = tf.nn.bias_add(tf.matmul(h41, w5), b5)
        h5_1 = batch_act(h5_1, phase=phase, act=tf.nn.elu,
                         scope='fc_layer1_1_bn')
        # h5_1 = tf.nn.dropout(h5_1, dropout)
    h42 = tf.concat(values=[h32, h31], axis=1)
    with tf.name_scope('fc_layer1_2'):
        # h5_2 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h42, w5), b5))
        h5_2 = tf.nn.bias_add(tf.matmul(h42, w5), b5)
        h5_2 = batch_act(h5_2, phase=phase, act=tf.nn.elu,
                         scope='fc_layer1_2_bn')
        # h5_2 = tf.nn.dropout(h5_2, dropout)
    h5 = (h5_1 + h5_2) / 2.
    with tf.name_scope('sm_layer'):
        w7 = init_weights([32, 2], 'w7')
        reg_term += tf.nn.l2_loss(w7)
        variable_summaries(w7)
        o = tf.matmul(h5, w7)
    return o


def classification_predict(hl, hr, dropout, phase):
    h41 = tf.concat(values=[hl, hr], axis=1)
    with tf.name_scope('fc_layer1_1'):
        w5 = init_weights([128, 32], 'w5')  # 64 16
        b5 = init_bias([32], 'b5')
        h5_1 = tf.nn.bias_add(tf.matmul(h41, w5), b5)
        h5_1 = batch_act(h5_1, phase=phase, act=tf.nn.elu,
                         scope='fc_layer1_1_bn')
    h42 = tf.concat(values=[hr, hl], axis=1)
    with tf.name_scope('fc_layer1_2'):
        # h5_2 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h42, w5), b5))
        h5_2 = tf.nn.bias_add(tf.matmul(h42, w5), b5)
        h5_2 = batch_act(h5_2, phase=phase, act=tf.nn.elu,
                         scope='fc_layer1_2_bn')
        # h5_2 = tf.nn.dropout(h5_2, dropout)
    h5 = (h5_1 + h5_2) / 2.
    with tf.name_scope('sm_layer'):
        w7 = init_weights([32, 2], 'w7')
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w7)
        variable_summaries(w7)
        o = tf.matmul(h5, w7)
    return o


def emb_transform(X):
    with tf.variable_scope('encoding'):
        wf = init_weights([bin_vec_dim, embedding_dim], 'wf')
        bf = init_bias([embedding_dim], 'bf')
        emb = tf.nn.bias_add(tf.matmul(X, wf), bf)
        emb = tf.nn.elu(emb)
    return emb


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


def train():
    global reg_term
    with tf.name_scope('input'):
        X_left = tf.placeholder(tf.float32, [None, dim, dim, bin_vec_dim])
        X_right = tf.placeholder(tf.float32, [None, dim, dim, bin_vec_dim])
        Y = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool, name='phase')

    py_x = classification(X_left, X_right, dropout, phase)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    tf.summary.scalar('cost', cost)
    # regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
    # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_cost = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    # tf.summary.scalar('reg_cost', reg_cost)
    cost = tf.reduce_mean(cost + beta * reg_term)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        predict_op = tf.argmax(py_x, 1)

    train_X_left, train_X_right, train_Y, test_X_left, test_X_right, test_Y = graph_mat_data.load_googlejam_data_newencoding(
        neg_ratio=1.3, pos_ratio=1.0)
    # train_X_left = train_X_left[:256*100]
    # train_X_right = train_X_right[:256*100]
    # train_Y = train_Y[:256*100]
    t_beg = time.clock()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/home/zg/Desktop/dev/logs',
                                             sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        for epoch in xrange(4):
            dense_test_X_left = from_sparse_arrs(test_X_left[0:test_size])
            dense_test_X_right = from_sparse_arrs(test_X_right[0:test_size])
            iter = 0
            for start, end in zip(
                    range(0, np.shape(train_X_left)[0], batch_size),
                    range(batch_size, np.shape(train_X_left)[0] + 1,
                          batch_size)):
                dense_train_X_left = from_sparse_arrs(train_X_left[start:end])
                dense_train_X_right = from_sparse_arrs(train_X_right[start:end])
                summary, _ = sess.run([merged, train_op],
                                      feed_dict={X_left: dense_train_X_left,
                                                 X_right: dense_train_X_right,
                                                 Y: train_Y[start:end],
                                                 dropout: keep_prob, phase: 1})
                train_writer.add_summary(summary, iter)
                print('epoch %d, iteration %d\n' % (epoch, iter))
                iter += 1

            predict_Y = sess.run(predict_op,
                                 feed_dict={X_left: dense_test_X_left,
                                            X_right: dense_test_X_right,
                                            dropout: 1.0,
                                            phase: 0})  # no dropout
            print(
            epoch, np.mean(np.argmax(test_Y[:test_size], axis=1) == predict_Y))
            saver.save(sess=sess,
                       save_path='models/model4_' + str(epoch) + '.ckpt')

        saver.save(sess, "models/model4.ckpt")
        print "model saved."
        t_end = time.clock()
        print('Time cost: %.2f' % (t_end - t_beg))


def train_10_fold_balanced():
    global reg_term
    with tf.name_scope('input'):
        X_left = tf.placeholder(tf.float32, [None, dim, dim, bin_vec_dim])
        X_right = tf.placeholder(tf.float32, [None, dim, dim, bin_vec_dim])
        Y = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool, name='phase')
    sample_weights = tf.placeholder(tf.float32, [batch_size])

    py_x = classification(X_left, X_right, dropout, phase)
    cost = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(logits=py_x, onehot_labels=Y,
                                        weights=sample_weights))
    tf.summary.scalar('cost', cost)
    cost = tf.reduce_mean(cost + beta * reg_term)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # learning_rate = 0.001 # for 1:1 dataset
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
            cost)
        predict_op = tf.argmax(py_x, 1)

    skf = StratifiedKFold(n_splits=10)
    file_path = "data/googlejam_newencoding/g4_128.npy"
    dataset = np.load(open(file_path, 'r'))
    X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
    # shuffle
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
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
    for train_idx, test_idx in skf.split(X, y):
        print ('*' * 40 + str(fold_index) + '*' * 40)
        fold_path = os.path.join("10_fold_balanced", str(fold_index))
        if os.path.exists(fold_path) is not True:
            os.mkdir(fold_path)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_X_left, train_X_right, train_Y = \
            graph_mat_data.make_pairs_10_fold(X_train, y_train, neg_ratio=10.0,
                                              pos_ratio=1.0, add_all_neg=True)
        test_X_left, test_X_right, test_Y = \
            graph_mat_data.make_pairs_10_fold(X_test, y_test, neg_ratio=1.0,
                                              pos_ratio=1.0, add_all_neg=True)

        # compute the class weights
        classes_numbers = np.bincount(np.argmax(train_Y, axis=1))
        classes_weights = np.array([classes_numbers[1] * 2.0 /
                                     (classes_numbers[0] + classes_numbers[1]),
                                     classes_numbers[0] * 1.0 /
                                     (classes_numbers[0] + classes_numbers[1])],
                                    dtype=np.float32)
        classes_weights = np.reshape(classes_weights, newshape=[2,1])

        t_beg = time.clock()
        # tf.reset_default_graph() # reset the model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                logdir, sess.graph)
            saver = tf.train.Saver(max_to_keep=3)
            step = 0
            for epoch in xrange(4):
                # re-shuffle for each epoch
                indices = np.random.permutation(train_X_left.shape[0])
                train_X_left = train_X_left[indices]
                train_X_right = train_X_right[indices]
                train_Y = train_Y[indices]
                # for small test
                dense_test_X_left = from_sparse_arrs(test_X_left[0:test_size])
                dense_test_X_right = from_sparse_arrs(test_X_right[0:test_size])

                for start, end in zip(
                        range(0, np.shape(train_X_left)[0], batch_size),
                        range(batch_size, np.shape(train_X_left)[0] + 1,
                              batch_size)):
                    dense_train_X_left = from_sparse_arrs(
                        train_X_left[start:end])
                    dense_train_X_right = from_sparse_arrs(
                        train_X_right[start:end])
                    # compute batch sample weights
                    batch_samples_weights = np.matmul(train_Y[start:end],
                                                     classes_weights)
                    batch_samples_weights = np.reshape(batch_samples_weights,
                                                       newshape=[batch_size])
                    _ = sess.run([train_op],
                                          feed_dict={X_left: dense_train_X_left,
                                                     X_right: dense_train_X_right,
                                                     Y: train_Y[start:end],
                                                     sample_weights:
                                                         batch_samples_weights,
                                                     dropout: keep_prob,
                                                     phase: 1})
                    print('epoch %d, iteration %d\n' % (epoch, step))
                    step += 1
                    if step % 100 == 0 and step != 0:
                        batch_samples_weights = np.matmul(test_Y[:test_size],
                                                          classes_weights)
                        batch_samples_weights = np.reshape(
                            batch_samples_weights,
                            newshape=[test_size])
                        predict_Y, summary = sess.run([predict_op, merged],
                                             feed_dict={
                                                 X_left: dense_test_X_left,
                                                 X_right: dense_test_X_right,
                                                 Y: test_Y[:test_size],
                                                 sample_weights:batch_samples_weights,
                                                 dropout: 1.0,
                                                 phase: 0})  # no dropout
                        train_writer.add_summary(summary, step)
                        print(epoch, np.mean(
                            np.argmax(test_Y[:test_size], axis=1) == predict_Y))
                        # saver.save(sess=sess, save_path='models/model4_' + str(epoch) + '.ckpt')
            saver.save(sess, os.path.join(fold_path, 'mode.ckpt'))
            print "model saved."
            t_end = time.clock()
            print('Time cost: %.2f' % (t_end - t_beg))

            # validation
            overall_accuracy = 0.
            overall_predict_Y = []
            iter = 0
            for start, end in zip(
                    range(0, np.shape(test_X_left)[0], batch_size),
                    range(batch_size, np.shape(test_X_left)[0] + 1,
                          batch_size)):
                dense_test_X_left = from_sparse_arrs(test_X_left[start:end])
                dense_test_X_right = from_sparse_arrs(test_X_right[start:end])
                predict_Y = sess.run(predict_op,
                                     feed_dict={X_left: dense_test_X_left,
                                                X_right: dense_test_X_right,
                                                dropout: 1.0,
                                                phase: 0})  # no dropout
                overall_predict_Y.extend(predict_Y.tolist())
                accuracy = np.mean(
                    np.argmax(test_Y[start:end], axis=1) == predict_Y)
                iter += 1
                overall_accuracy += accuracy

            print('Overall accuracy: %.5f' % (overall_accuracy / iter))
            t_end = time.clock()
            print('Time cost: %.2f' % (t_end - t_beg))
            fout.write('*' * 80 + '\n')
            fout.write('Fold %d:\n' % (fold_index))
            fout.write('Overall accuracy: %.5f\n' % (overall_accuracy / iter))
            fout.write('Time cost: %.2f\n' % (t_end - t_beg))
            recall, precision, f1_score = stat(
                np.argmax(test_Y[:len(overall_predict_Y)], axis=1),
                np.array(overall_predict_Y, dtype=np.int32), fout=fout)
            # fout.write("Fold index: %d, accuracy: %.4f, recall: %.4f, "
            #            "precision: %.4f, f1 score: %.4f\n" % (
            #            fold_index, overall_accuracy /
            #            iter, recall, precision, f1_score))
            fout.flush()
            avg_accuracy += overall_accuracy / iter
            avg_recall += recall
            avg_precision += precision
            avg_f1_score += f1_score
        print('*' * 80)
        fold_index += 1
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


def stat(Y, predicted_Y, fout=None):
    real_positive_count = 0
    predict_positive_count = 0
    recall = 0
    precision = 0
    for i in xrange(Y.shape[0]):
        if Y[i] == 1:
            real_positive_count += 1
            if predicted_Y[i] == 1:
                recall += 1
        if predicted_Y[i] == 1:
            predict_positive_count += 1
            if Y[i] == 1:
                precision += 1
    retrieved_positive_count = recall
    recall /= real_positive_count * 1.0
    precision /= max(predict_positive_count * 1.0, 1.0)
    f1_score = 2 * recall * precision / max(
    recall + precision, 0.00001)
    print "Clone pairs: %d, non-clone pairs: %d " % (
    real_positive_count, Y.shape[0] - real_positive_count)
    print "Recall: %f, precision: %f, f1 score: %f" % (
    recall, precision, f1_score)
    print "Predicted_positive_count: %d, recall truly positive: %d, false positive: %d, missed true positive: %d" \
          % (predict_positive_count, retrieved_positive_count,
             predict_positive_count - retrieved_positive_count,
             real_positive_count - retrieved_positive_count)
    if fout is not None:
        fout.write("Clone pairs: %d, non-clone pairs: %d\n" % (
    real_positive_count, Y.shape[0] - real_positive_count))
        fout.write("Recall: %.4f, precision: %.4f, f1 score: %.4f\n" % (
    recall, precision, f1_score))
        fout.write("Predicted_positive_count: %d, recall truly positive: %d, "
                   "false positive: %d, missed true positive: %d\n" \
          % (predict_positive_count, retrieved_positive_count,
             predict_positive_count - retrieved_positive_count,
             real_positive_count - retrieved_positive_count))
    return recall, precision, f1_score


def predict_on_full_dataset():
    with tf.name_scope('input'):
        X_left = tf.placeholder(tf.float32, [None, dim, dim, bin_vec_dim])
        X_right = tf.placeholder(tf.float32, [None, dim, dim, bin_vec_dim])
        Y = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool, name='phase')

    with tf.variable_scope('encoding'):
        h_op = model(X_left, dropout, phase)

    h_left = tf.placeholder(tf.float32, [None, 64])
    h_right = tf.placeholder(tf.float32, [None, 64])
    py_x = classification_predict(h_left, h_right, dropout, phase)
    predict_op = tf.argmax(py_x, 1)

    file_path = "../detector/dataset/training/googlejam_newencoding/g4_128.npy"
    dataset = np.load(open(file_path, 'r'))
    X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
    
    t_beg = time.clock()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, '10_fold_balanced/2/mode.ckpt')

    iter = 0
    X_reps = []
    for start, end in zip(range(0, np.shape(X)[0], batch_size), \
                     range(batch_size, np.shape(X)[0] + 1, batch_size)):
        dense_X = from_sparse_arrs(X[start:end])
        h_val = sess.run(h_op, feed_dict={X_left: dense_X, dropout: 1.0,
                                          phase:0})
        X_reps.extend(h_val.tolist())
    dense_X = from_sparse_arrs(X[end:])
    h_val = sess.run(h_op, feed_dict={X_left: dense_X, dropout: 1.0, phase:0})
    X_reps.extend(h_val.tolist())
    test_X_left = []
    test_X_right = []
    test_Y = []
    for i in xrange(y.shape[0]):
        for j in xrange(i+1, y.shape[0]):
            if y[i] == y[j]:
                test_X_left.append(X_reps[i])
                test_X_right.append(X_reps[j])
                test_Y.append([0, 1])
            else:
                test_X_left.append(X_reps[i])
                test_X_right.append(X_reps[j])
                test_Y.append([1, 0])
    test_X_left = np.array(test_X_left)
    test_X_right = np.array(test_X_right)
    test_Y = np.array(test_Y, dtype=np.float32)
    

    overall_predict_Y = []
    for start, end in zip(range(0, np.shape(test_X_left)[0], batch_size),
                          range(batch_size, np.shape(test_X_left)[0] + 1,
                                batch_size)):
        predict_Y = sess.run(predict_op,
                             feed_dict={h_left: test_X_left[start:end],
                                        h_right: test_X_right[start:end],
                                        dropout: 1.0, phase: 0})  # no dropout
        overall_predict_Y.extend(predict_Y.tolist())
        iter += 1

    stat(np.argmax(test_Y[:end], axis=1),
         np.array(overall_predict_Y, dtype=np.int32))


if __name__ == '__main__':
    # balanced version of 10-fold using penalty for 0-class
    train_10_fold_balanced()
    st = time.time()
    predict_on_full_dataset()
    print "Predict time on the full dataset: ", time.time() - st