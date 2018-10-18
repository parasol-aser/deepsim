import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import StratifiedKFold

import graph_mat_data

dim = graph_mat_data.dim
bin_vec_dim = 8

w_0_s = [dim * dim * bin_vec_dim, 1024]; b_0_s = [1024]
w_1_s = [1024, 256]; b_1_s = [256]
w_2_s = [256, 128]; b_2_s = [128]
w_m_s = [256, 64]; b_m_s = [64]
w_o_s = [64, 2]

keep_prob = 0.75
beta = 0.000003



logdir = '/tmp/tf_logs'



def init_weights(shape, name):
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                      uniform=True),
                           regularizer=tf.contrib.layers.l2_regularizer(scale=beta))


def bias_variable(shape, name):
    if len(shape) > 1:
        raise Exception('Bias should be a vector.')
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.01))

def model(X_left, X_right, dropout):
    with tf.name_scope("Encoder_Layer_1"):
        with tf.name_scope('weights'):
            w_0 = init_weights(w_0_s, 'w_0')
        with tf.name_scope('bias'):
            b_0 = bias_variable(b_0_s, 'b_0')
        with tf.name_scope('drop_out'):
            X_left = tf.nn.dropout(X_left, dropout)
            X_right = tf.nn.dropout(X_right, dropout)
        with tf.name_scope('hidden'):
            a_1_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(X_left, w_0), b_0))
            a_1_r = tf.nn.elu(tf.nn.bias_add(tf.matmul(X_right, w_0), b_0))
    with tf.name_scope("Encoder_Layer_2"):
        with tf.name_scope('weights'):
            w_1 = init_weights(w_1_s, 'w_1')
        with tf.name_scope('bias'):
            b_1 = bias_variable(b_1_s, 'b_1')
        with tf.name_scope('drop_out'):
            a_1_l = tf.nn.dropout(a_1_l, dropout)
            a_1_r = tf.nn.dropout(a_1_r, dropout)
        with tf.name_scope('hidden'):
            a_2_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_1_l, w_1), b_1))
            a_2_r = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_1_r, w_1), b_1))
    with tf.name_scope("Encoder_Layer3"):
        with tf.name_scope('weights'):
            w_2 = init_weights(w_2_s, 'w_2')
        with tf.name_scope('bias'):
            b_2 = bias_variable(b_2_s, 'b_2')
        with tf.name_scope('drop_out'):
            a_2_l = tf.nn.dropout(a_2_l, dropout)
            a_2_r = tf.nn.dropout(a_2_r, dropout)
        with tf.name_scope('hidden'):
            a_3_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_2_l, w_2), b_2))
            a_3_r = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_2_r, w_2), b_2))
    with tf.name_scope('Merge_Layer'):
        input = tf.concat(axis=1, values=[a_3_l, a_3_r])
        with tf.name_scope('weights'):
            w_m = init_weights(w_m_s, 'w_m')
        with tf.name_scope('bias'):
            b_m = init_weights(b_m_s, 'b_m')
        with tf.name_scope('hidden'):
            a_in = tf.nn.elu(tf.nn.bias_add(tf.matmul(input, w_m), b_m))
    with tf.name_scope('output_layer'):
        with tf.name_scope('weights'):
            w_o = init_weights(w_o_s, 'w_o')
        with tf.name_scope('out'):
            a_o = tf.matmul(a_in, w_o)
    return a_o


def from_sparse_arr(sparse_arr):
    mat = np.zeros((dim, dim, bin_vec_dim), dtype=np.float32)
    for (i,j,k) in sparse_arr:
        if k <= 2:
            mat[i,j,0] = k
        elif k <= 6:
            mat[i,j,1] = k - 3
        elif k <= 18:
            mat[i,j,2] = k - 7
        elif k <= 21:
            mat[i,j,3] = k - 19
        elif k <= 25:
            mat[i,j,4] = k - 22
        elif k <= 37:
            mat[i,j,5] = k - 26
        elif k <= 80:
            mat[i,j,6] = k - 38
        else:
            mat[i,j,7] = k - 81
    return mat

def from_sparse_arrs(sparse_arrs):
    mats = []
    for sparse_arr in sparse_arrs:
        mats.append(from_sparse_arr(sparse_arr))
    mats = np.array(mats, dtype=np.float32)
    mats = np.reshape(mats, [mats.shape[0], dim*dim*bin_vec_dim])
    return mats


def train_10_fold():
    batch_size = 128
    test_size = 128

    skf = StratifiedKFold(n_splits=10)
    file_path = "../dataset/g4_128.npy"
    dataset = np.load(open(file_path, 'r'))
    X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
    
    with tf.name_scope('Input'):
        X_left = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        X_right = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        Y = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32, name='dropout')
    sample_weights = tf.placeholder(tf.float32, [batch_size])
    
    py_x = model(X_left, X_right, dropout)
    
    cost = tf.reduce_mean(
       tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    reg_term = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    train_op = tf.train.AdamOptimizer().minimize(cost + reg_term)
    predict_op = tf.argmax(py_x, 1)

    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    fold_index = 0
    avg_accuracy = 0.
    avg_recall = 0.
    avg_precision = 0.
    avg_f1_score = 0.
    fout = open('result/sda_base_10_fold.txt', 'w')
    if os.path.exists('result') is not True:
        os.mkdir("result")
    if os.path.exists("old_models_10_fold") is not True:
        os.mkdir("old_models_10_fold")
    for train_idx, test_idx in skf.split(X, y):
        print ('*' * 40 + str(fold_index) + '*' * 40)
        fold_path = os.path.join("old_models_10_fold/sda_base", str(fold_index))
        if os.path.exists(fold_path) is not True:
            os.mkdir(fold_path)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_X_left, train_X_right, train_Y = \
            graph_mat_data.make_pairs_10_fold_old_model(X_train, y_train,
                                                        neg_ratio=1.3,
                                              pos_ratio=1.0, add_all_neg=False)
        test_X_left, test_X_right, test_Y = \
            graph_mat_data.make_pairs_10_fold(X_test, y_test, neg_ratio=1.0,
                                              pos_ratio=1.0, add_all_neg=True)
        # compute the class weights
        classes_numbers = np.bincount(np.argmax(train_Y, axis=1))
        classes_weights = np.array([classes_numbers[1] * 1.0 /
                                    (classes_numbers[0] + classes_numbers[1]),
                                    classes_numbers[0] * 1.0 /
                                    (classes_numbers[0] + classes_numbers[1])],
                                   dtype=np.float32)
        classes_weights = np.reshape(classes_weights, newshape=[2, 1])
    
        t_beg = time.clock()
        
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(logdir,
                                                 sess.graph)
            tf.global_variables_initializer().run()
            
            dense_test_X_left = from_sparse_arrs(test_X_left[0:test_size])
            dense_test_X_right = from_sparse_arrs(test_X_right[0:test_size])
            step = 0
            for epoch in xrange(6):
                indices = np.random.permutation(train_X_left.shape[0])
                train_X_left = train_X_left[indices]
                train_X_right = train_X_right[indices]
                train_Y = train_Y[indices]
                for start, end in zip(
                        range(0, np.shape(train_X_left)[0], batch_size),
                        range(batch_size, np.shape(train_X_left)[0] + 1,
                              batch_size)):
                    dense_train_X_left = from_sparse_arrs(train_X_left[start:end])
                    dense_train_X_right = from_sparse_arrs(train_X_right[start:end])
                    batch_samples_weights = np.matmul(train_Y[start:end],
                                                      classes_weights)
                    batch_samples_weights = np.reshape(batch_samples_weights,
                                                       newshape=[batch_size])
                    sess.run(train_op, feed_dict={X_left: dense_train_X_left,
                                                  X_right: dense_train_X_right,
                                                  Y: train_Y[start:end],
                                                  dropout: keep_prob,
                                                  sample_weights:
                                                      batch_samples_weights
                                                  })
                    step += 1
                    if step % 100 == 0 and step != 0:
                        print('Epoch: %d, Iter: %d\n' % (epoch, step))
                
                predict_Y = sess.run(predict_op,
                                     feed_dict={X_left: dense_test_X_left,
                                                X_right: dense_test_X_right,
                                                dropout: 1.0})
                print(
                epoch, np.mean(np.argmax(test_Y[0:test_size], axis=1) == predict_Y))
            
            t_end = time.clock()
            print('time cost: %.2f' % (t_end - t_beg))
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(fold_path, 'mode.ckpt'))
            print "model saved."

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
                                                dropout: 1.0})  # no dropout
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
    st = time.time()
    file_path = "../dataset/g4_128.npy"
    dataset = np.load(open(file_path, 'r'))
    X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
    test_X_left, test_X_right, test_Y = \
        graph_mat_data.make_pairs_10_fold_for_predict(X, y)
    print "File reading time: ", time.time() - st
    
    
    with tf.name_scope('Input'):
        X_left = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        X_right = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        Y = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    py_x = model(X_left, X_right, dropout)
    predict_op = tf.argmax(py_x, 1)
    
    batch_size = 256
    test_size = 256
    
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'old_models_10_fold/sda_base/0/mode.ckpt')
    
    iter = 0
    overall_predict_Y = []
    for start, end in zip(range(0, np.shape(test_X_left)[0], batch_size),
                          range(batch_size, np.shape(test_X_left)[0] + 1,
                                batch_size)):
        dense_test_X_left = from_sparse_arrs(test_X_left[start:end])
        dense_test_X_right = from_sparse_arrs(test_X_right[start:end])
        predict_Y = sess.run(predict_op,
                             feed_dict={X_left: dense_test_X_left,
                                        X_right: dense_test_X_right,
                                        dropout: 1.0})  # no dropout
        overall_predict_Y.extend(predict_Y.tolist())
        iter += 1

    stat(np.argmax(test_Y[:end], axis=1),
         np.array(overall_predict_Y, dtype=np.int32))


if __name__ == '__main__':
    
    st = time.time()
    train_10_fold()
    print 'Total 10-fold time: ', time.time() - st

    st = time.time()
    predict_on_full_dataset()
    print "Total predicting time: ", time.time() - st