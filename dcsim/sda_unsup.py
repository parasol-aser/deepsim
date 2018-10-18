import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import StratifiedKFold

import graph_mat_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# w_0_s = [28 * 28, 256]; b_0_s = [256]
# w_1_s = [256, 50]; b_1_s = [50]
# w_m_s = [100, 20]; b_m_s = [20]
# w_o_s = [20, 2]

# w_0_s = [28 * 28 * 2, 512]; b_0_s = [512]
# w_1_s = [512, 100]; b_1_s = [100]
# w_m_s = [100, 20]; b_m_s = [20]
# w_o_s = [20, 2]

dim = graph_mat_data.dim
bin_vec_dim = 8

w_0_s = [dim * dim * bin_vec_dim, 1024]; b_0_s = [1024]
w_1_s = [1024, 512]; b_1_s = [512]
w_2_s = [512, 256]; b_2_s = [256]
w_m_s = [512, 128]; b_m_s = [128]
w_o_s = [128, 2]
h_dim = 256

keep_prob = 0.75
beta_1 = 0.0003
beta_2 = 0.001

noise_g_std = 0.1

reg_term = None
cls_reg_term = None

logdir = '/tmp/tf_logs'


def variable_summaries(var, collections):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean, collections)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev, collections)
    tf.summary.scalar('max', tf.reduce_max(var), collections)
    tf.summary.scalar('min', tf.reduce_min(var), collections)
    tf.summary.histogram('histogram', var, collections)

def add_gaussian_noise(h):
    noise = tf.random_normal(shape=tf.shape(h), mean=0., stddev=noise_g_std, dtype=tf.float32)
    return h + noise


def init_weights(shape, name):
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                           initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                      uniform=True))
    # return tf.Variable(tf.random_normal(shape, stddev=0.1)) #better to use tf.get_variable with regularizer and initializer

def bias_variable(shape, name):
    if len(shape) > 1:
        raise Exception('Bias should be a vector.')
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.01))

def encoder(X, dropout):
    global reg_term
    with tf.name_scope("Encoder_Layer_1"):
        with tf.variable_scope('encoder_weights'):
            w_0 = init_weights(w_0_s, 'w_0')
        b_0 = bias_variable(b_0_s, 'b_0')
        X = tf.nn.dropout(X, dropout)
        a_1_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(X, w_0), b_0))
    reg_term = tf.nn.l2_loss(w_0)
    with tf.name_scope("Encoder_Layer_2"):
        with tf.variable_scope('encoder_weights'):
            w_1 = init_weights(w_1_s, 'w_1')
        b_1 = bias_variable(b_1_s, 'b_1')
        a_1_l = tf.nn.dropout(a_1_l, dropout)
        a_2_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_1_l, w_1), b_1))
    reg_term += tf.nn.l2_loss(w_1)
    with tf.name_scope("Encoder_Layer3"):
        with tf.variable_scope('encoder_weights'):
            w_2 = init_weights(w_2_s, 'w_2')
        b_2 = bias_variable(b_2_s, 'b_2')
        # a_2_l = tf.nn.dropout(a_2_l, dropout)
        a_3_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_2_l, w_2), b_2))
    reg_term += tf.nn.l2_loss(w_2)
    return a_3_l

def decoder(X):
    with tf.name_scope("Decoder_Layer_1"):
        with tf.variable_scope('encoder_weights', reuse=True):
            w_0 = tf.transpose(init_weights(w_2_s, 'w_2'))
        b_0 = bias_variable(b_1_s, 'b_1_s')
        a_1_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(X, w_0), b_0))
    with tf.name_scope("Decoder_Layer_2"):
        with tf.variable_scope('encoder_weights', reuse=True):
            w_1 = tf.transpose(init_weights(w_1_s, 'w_1'))
        b_1 = bias_variable(b_0_s, 'b_0_s')
        a_2_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_1_l, w_1), b_1))
    with tf.name_scope("Decoder_Layer3"):
        with tf.variable_scope('encoder_weights', reuse=True):
            w_2 = tf.transpose(init_weights(w_0_s, 'w_0'))
        b_2 = bias_variable([dim * dim * bin_vec_dim], 'b_s')
        a_3_l = tf.nn.elu(tf.nn.bias_add(tf.matmul(a_2_l, w_2), b_2))
    return a_3_l

def binary_classification(h_left, h_right, dropout):
    global cls_reg_term
    with tf.name_scope('Merge_Layer'):
        input = tf.concat(axis=1, values=[h_left, h_right])
        with tf.name_scope('weights'):
            w_m = init_weights(w_m_s, 'w_m')
        with tf.name_scope('bias'):
            b_m = init_weights(b_m_s, 'b_m')
        with tf.name_scope('hidden'):
            a_in = tf.nn.elu(tf.nn.bias_add(tf.matmul(input, w_m), b_m))
        cls_reg_term = tf.nn.l2_loss(w_m)
    with tf.name_scope('output_layer'):
        with tf.name_scope('weights'):
            w_o = init_weights(w_o_s, 'w_o')
            variable_summaries(w_o, ['cls'])
        with tf.name_scope('out'):
            a_o = tf.matmul(a_in, w_o)
        cls_reg_term += tf.nn.l2_loss(w_o)
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
    global cls_reg_term, reg_term
    
    t_beg = time.clock()
    
    batch_size = 256
    test_size = 256

    skf = StratifiedKFold(n_splits=10)
    file_path = "../dataset/g4_128.npy"
    dataset = np.load(open(file_path, 'r'))
    X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int32)
    
    with tf.name_scope('Input'):
        X_left = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        X_right = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        Y = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32, name='dropout')
    sample_weights = tf.placeholder(tf.float32, [batch_size])
    
    with tf.variable_scope('encoder-decoder'):
        z = decoder(encoder(X_left, dropout))
    
    cost = tf.reduce_mean(tf.nn.l2_loss(z - X_left))
    tf.summary.scalar('ae_cost', cost, collections=['ae'])
    encoding_trian_op = tf.train.AdamOptimizer().minimize(
        cost + beta_1 * reg_term)
    
    with tf.variable_scope('encoder-decoder', reuse=True):
        h_left_op = encoder(X_left, dropout)
        h_right_op = encoder(X_right, dropout)
    h_left = tf.placeholder(tf.float32, [None, h_dim])
    h_right = tf.placeholder(tf.float32, [None, h_dim])
    py_x = binary_classification(h_left, h_right, dropout)
    
    cls_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    tf.summary.scalar('cls_cost', cls_cost, ['cls'])
    cls_train_op = tf.train.AdamOptimizer().minimize(
        cls_cost + beta_2 * cls_reg_term)
    predict_op = tf.argmax(py_x, 1)

    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    fold_index = 0
    avg_accuracy = 0.
    avg_recall = 0.
    avg_precision = 0.
    avg_f1_score = 0.
    fout = open('result/sda_unsup_10_fold.txt', 'w')
    if os.path.exists('result') is not True:
        os.mkdir("result")
    if os.path.exists("old_models_10_fold") is not True:
        os.mkdir("old_models_10_fold")
    for train_idx, test_idx in skf.split(X, y):
        print ('*' * 40 + str(fold_index) + '*' * 40)
        fold_path = os.path.join("old_models_10_fold/sda_unsup",
                                 str(fold_index))
        if os.path.exists(fold_path) is not True:
            os.mkdir(fold_path)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_X_left, train_X_right, train_Y = \
            graph_mat_data.make_pairs_10_fold_old_model(X_train, y_train,
                                                        neg_ratio=1.3,
                                                        pos_ratio=1.0,
                                                        add_all_neg=False)
        test_X_left, test_X_right, test_Y = \
            graph_mat_data.make_pairs_10_fold(X_test, y_test, neg_ratio=1.0,
                                              pos_ratio=1.0, add_all_neg=True)
    
        classes_numbers = np.bincount(np.argmax(train_Y, axis=1))
        classes_weights = np.array([classes_numbers[1] * 2.0 /
                                    (classes_numbers[0] + classes_numbers[1]),
                                    classes_numbers[0] * 1.0 /
                                    (classes_numbers[0] + classes_numbers[1])],
                                   dtype=np.float32)
        classes_weights = np.reshape(classes_weights, newshape=[2, 1])
    
        t_beg = time.clock()
    
        t_end = time.clock()
        # print('Preparing time: %.2f' % (t_end - t_beg))
        
        with tf.Session() as sess:
            merged = tf.summary.merge_all(key='ae')
            train_writer = tf.summary.FileWriter(
                logdir, sess.graph)
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            dense_test_X_left = from_sparse_arrs(test_X_left[0:test_size])
            dense_test_X_right = from_sparse_arrs(test_X_right[0:test_size])

            it = 0
            for epoch in xrange(300):
                indices = np.random.permutation(X_train.shape[0])
                shuffle_train_X = X_train[indices]
                for start, end in zip(
                        xrange(0, np.shape(X_train)[0], batch_size),
                        xrange(batch_size, np.shape(X_train)[0] + 1,
                               batch_size)):
                    dense_train_X_left = from_sparse_arrs(
                        shuffle_train_X[start:end])
                    # dense_train_X_right = from_sparse_arrs(train_X_right[start:end])
                    _ = sess.run([encoding_trian_op],
                                 feed_dict={X_left: dense_train_X_left,
                                            dropout: keep_prob})
                    # train_writer.add_summary(summary, it)
                    if (it % 100 == 0):
                        print('Epoch: %d, Iter: %d\n' % (epoch, it))
                        h_left_val = sess.run(z,
                                              feed_dict={
                                                  X_left: dense_test_X_left,
                                                  dropout: 1.0})
                        print('l2 cost: %.3f\n' % (np.mean(
                            np.linalg.norm(h_left_val - dense_test_X_left,
                                           axis=1))))
                    it += 1

            t_end = time.clock()
            print('Unsup phase time: %.2f' % (t_end - t_beg))

            merged = tf.summary.merge_all('cls')
            best_test_acc = 0.
            it = 0
            for epoch in xrange(3):
                indices = np.random.permutation(train_X_left.shape[0])
                train_X_left = train_X_left[indices]
                train_X_right = train_X_right[indices]
                train_Y = train_Y[indices]
                for start, end in zip(
                        range(0, np.shape(train_X_left)[0], batch_size),
                        range(batch_size, np.shape(train_X_left)[0] + 1,
                              batch_size)):
                    batch_samples_weights = np.matmul(train_Y[start:end],
                                                      classes_weights)
                    batch_samples_weights = np.reshape(
                        batch_samples_weights,
                        newshape=[batch_size])
                    dense_train_X_left = from_sparse_arrs(
                        train_X_left[start:end])
                    dense_train_X_right = from_sparse_arrs(
                        train_X_right[start:end])
                    h_left_val = sess.run(h_left_op,
                                          feed_dict={
                                              X_left: dense_train_X_left,
                                              dropout: 1.0})
                    h_right_val = sess.run(h_right_op,
                                           feed_dict={
                                               X_right: dense_train_X_right,
                                               dropout: 1.0})
                    _ = sess.run([cls_train_op],
                                 feed_dict={h_left: h_left_val,
                                            h_right: h_right_val,
                                            Y: train_Y[start:end],
                                            dropout: keep_prob,
                                            sample_weights: batch_samples_weights})
                    # train_writer.add_summary(summary, it)
                    if (it % 100 == 0):
                        print('Epoch: %d, Iter: %d\n' % (epoch, it))
                    it += 1
    
                h_left_val = sess.run(h_left_op,
                                      feed_dict={X_left: dense_test_X_left,
                                                 dropout: 1.0})
                h_right_val = sess.run(h_right_op,
                                       feed_dict={
                                           X_right: dense_test_X_right,
                                           dropout: 1.0})
                predict_Y = sess.run(predict_op,
                                     feed_dict={h_left: h_left_val,
                                                h_right: h_right_val,
                                                dropout: 1.0})
                stat(np.argmax(test_Y[0:test_size], axis=1), np.array(
                    predict_Y, dtype=np.float32))
                test_acc = np.mean(
                    np.argmax(test_Y[0:test_size], axis=1) == predict_Y)
                print(epoch, test_acc)
    
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(fold_path, 'mode.ckpt'))
                print "model saved."

            t_end = time.clock()
            print('time cost: %.2f' % (t_end - t_beg))

            overall_accuracy = 0.
            overall_predict_Y = []
            iter = 0
            # saver = tf.train.Saver()
            # saver.restore(sess, os.path.join(fold_path, 'mode.ckpt'))
            for start, end in zip(
                    range(0, np.shape(test_X_left)[0], batch_size),
                    range(batch_size, np.shape(test_X_left)[0] + 1,
                          batch_size)):
                dense_test_X_left = from_sparse_arrs(test_X_left[start:end])
                dense_test_X_right = from_sparse_arrs(
                    test_X_right[start:end])
                h_left_val = sess.run(h_left_op,
                                      feed_dict={X_left: dense_test_X_left,
                                                 dropout: 1.0})
                h_right_val = sess.run(h_right_op,
                                       feed_dict={
                                           X_right: dense_test_X_right,
                                           dropout: 1.0})
                predict_Y = sess.run(predict_op,
                                     feed_dict={h_left: h_left_val,
                                                h_right: h_right_val,
                                                Y: train_Y[
                                                   start:end],
                                                dropout: 1.0})
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
            fout.write(
                'Overall accuracy: %.5f\n' % (overall_accuracy / iter))
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
        fout.write(
            "Predicted_positive_count: %d, recall truly positive: %d, "
            "false positive: %d, missed true positive: %d\n" \
            % (predict_positive_count, retrieved_positive_count,
               predict_positive_count - retrieved_positive_count,
               real_positive_count - retrieved_positive_count))
    return recall, precision, f1_score


def predict_on_full_dataset():
    '''
    Test the time performance on the full dataset after evaluation using
    10-fold cross-validation
    :return:
    '''
    st = time.time()
    file_path = "../dataset/g4_128.npy"
    dataset = np.load(open(file_path, 'r'))
    X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
    # test_X_left, test_X_right, test_Y = \
    #     graph_mat_data.make_pairs_10_fold_for_predict(X, y)
    print "File reading time: ", time.time() - st
    
    with tf.name_scope('Input'):
        X_left = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        X_right = tf.placeholder(tf.float32, [None, dim * dim * bin_vec_dim])
        Y = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    with tf.variable_scope('encoder-decoder'):
        h = encoder(X_left, dropout)
        z = decoder(h)
    
    with tf.variable_scope('encoder-decoder', reuse=True):
        h_left_op = encoder(X_left, dropout)
        h_right_op = encoder(X_right, dropout)
    h_left = tf.placeholder(tf.float32, [None, h_dim])
    h_right = tf.placeholder(tf.float32, [None, h_dim])
    py_x = binary_classification(h_left, h_right, dropout)
    
    predict_op = tf.argmax(py_x, 1)
    
    batch_size = 256
    
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'old_models_10_fold/sda_unsup/0/mode.ckpt')
    
    iter = 0
    overall_predict_Y= []
    X_reps = []
    for start, end in zip(range(0, np.shape(X)[0], batch_size), \
                     range(batch_size, np.shape(X)[0] + 1, batch_size)):
        dense_X = from_sparse_arrs(X[start:end])
        h_val = sess.run(h_left_op, feed_dict={X_left: dense_X, dropout: 1.0})
        X_reps.extend(h_val.tolist())
    dense_X = from_sparse_arrs(X[end:])
    h_val = sess.run(h_left_op, feed_dict={X_left: dense_X, dropout: 1.0})
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
    
    for start, end in zip(range(0, np.shape(test_X_left)[0], batch_size),
                          range(batch_size, np.shape(test_X_left)[0] + 1,
                                batch_size)):
        predict_Y = sess.run(predict_op, feed_dict={h_left: test_X_left[
                                                            start:end],
                                                    h_right: test_X_right[
                                                             start:end],
                                                    dropout: 1.0})
        overall_predict_Y.extend(predict_Y.tolist())
        iter += 1

    stat(np.argmax(test_Y[:end], axis=1),
         np.array(overall_predict_Y, dtype=np.int32))


if __name__ == '__main__':

    st = time.time()
    train_10_fold()
    print 'Total 10-fold cross-validation time: ', time.time() - st
    st = time.time()
    predict_on_full_dataset()
    print 'Total predicting time on the full dataset: ', time.time() - st