#!/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import os, sys
import cPickle as pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


cfg_folder = "../BigCloneBench/dcsim" \
             "/cfgs_antlr"

labels_folder = "../BigCloneBench/dcsim" \
                "/labels_antlr"

seed = 233


def read_sample_sparse(filepath):
    sparse_arr = []
    row = 0
    for line in open(filepath, 'r'):
        xs = line.split('\t')[:-1]  # remove '\n'
        if (len(xs) != 128):
            continue
        col = 0
        for x in xs:
            if x != '{}':
                x = x[1:-1]
                if len(x) > 0:
                    indices = x.split(',')
                    for index in indices:
                        sparse_arr.append((row, col, int(index)))
            col += 1
        row += 1
    return sparse_arr


def read_data_info(filepath):
    """
    Read information about each node in the graph for each method.
    :param filepath: The file path of the *.info file
    :return:
    """
    fin = open(filepath, 'r')
    line = fin.readline()
    var_count = int(line.split(' ')[0])
    block_count = int(line.split(' ')[1])
    types = []
    for _ in xrange(var_count):
        line = fin.readline()
        types.append(int(line))
    for _ in xrange(block_count):
        line = fin.readline()
        types.append(int(line))
    return types, var_count, block_count


def flatten_clones_set(clones_set):
    X = []
    y = []
    file_dest = []
    infos = []
    for (l, clones) in clones_set.items():
        for m, file_path, info in zip(clones['data'], clones['file_dest'], clones['infos']):
            X.append(m)
            y.append(l)
            file_dest.append(file_path)
            infos.append(info)
    # X = np.array(X, dtype=np.float)
    y = np.array(y, dtype=np.int)
    return X, y, file_dest, infos


def select_by_functionality_id(id=None):
    cfgs = {}
    Xl = []
    Xr = []
    y = []
    ts = []
    Xl_selected = []
    Xr_selected = []
    y_selected = []
    ts_selected = []
    label_files = os.listdir(labels_folder)
    for label_file in label_files:
        type = int(label_file[1])
        if label_file.find('FP') != -1:
            pairs = None
            # check if the file is empty
            if os.path.getsize(os.path.join(labels_folder, label_file)) == 0:
                continue
            try:
                pairs = pd.read_csv(os.path.join(labels_folder, label_file),
                                    sep=',',
                                    header=None).values
            except IOError:
                print("IO error for pairs file: " + os.path.join(
                    labels_folder, label_file))
                continue
            for idx in xrange(pairs.shape[0]):
                pair = pairs[idx]
                p1 = int(pair[0])
                p2 = int(pair[1])
                label = int(pair[2])
                if cfgs.has_key(p1) is not True:
                    cfg_path = os.path.join(cfg_folder, str(p1) + '.mat')
                    if os.path.exists(cfg_path) is not True:
                        continue
                    cfgs[p1] = read_sample_sparse(cfg_path)
                if cfgs.has_key(p2) is not True:
                    cfg_path = os.path.join(cfg_folder, str(p2) + '.mat')
                    if os.path.exists(cfg_path) is not True:
                        continue
                    cfgs[p2] = read_sample_sparse(cfg_path)
                if id is not None and label==id:
                    Xl_selected.append(cfgs[p1])
                    Xr_selected.append(cfgs[p2])
                    y_selected.append(0)
                    ts_selected.append(type)
                else:
                    Xl.append(cfgs[p1])
                    Xr.append(cfgs[p2])
                    y.append(0)
                    ts.append(type)
        else:
            pairs = pd.read_csv(os.path.join(labels_folder, label_file),
                                sep=',',
                                header=None).values
            for idx in xrange(pairs.shape[0]):
                pair = pairs[idx]
                p1 = int(pair[0])
                p2 = int(pair[1])
                label = pair[2]
                if cfgs.has_key(p1) is not True:
                    cfg_path = os.path.join(cfg_folder, str(p1) + '.mat')
                    if os.path.exists(cfg_path) is not True:
                        continue
                    cfgs[p1] = read_sample_sparse(cfg_path)
                if cfgs.has_key(p2) is not True:
                    cfg_path = os.path.join(cfg_folder, str(p2) + '.mat')
                    if os.path.exists(cfg_path) is not True:
                        continue
                    cfgs[p2] = read_sample_sparse(cfg_path)
                if id is not None and label == id:
                    Xl_selected.append(cfgs[p1])
                    Xr_selected.append(cfgs[p2])
                    y_selected.append(1)
                    ts_selected.append(type)
                else:
                    Xl.append(cfgs[p1])
                    Xr.append(cfgs[p2])
                    y.append(1)
                    ts.append(type)
    return np.array(Xl), np.array(Xr), np.array(y), np.array(ts),\
           np.array(Xl_selected), np.array(Xr_selected),\
           np.array(y_selected), np.array(ts_selected)
            

def load_train_test(id=None):
    if id is not None and os.path.exists("./data/bigbench/id_" + str(id) +
                                         "_train.npy"):
        train_dataset = pickle.load(open('./data/bigbench/id_'
                                                   + str(id) + '_train.npy', 'r'))
        test_dataset = pickle.load(open('./data/bigbench/id_'
                                         + str(id) + '_test.npy', 'r'))
        return train_dataset['Xl'], train_dataset['Xr'], train_dataset['y'],\
                train_dataset['ts'], test_dataset['Xl'], test_dataset['Xr'],\
                test_dataset['y'], test_dataset['ts']

    Xl, Xr, y, ts, Xl_selected, Xr_selected, y_selected, ts_selected = \
        select_by_functionality_id(id)
    Xl, Xr, y, ts = shuffle(Xl, Xr, y, ts, random_state=seed)
    Xl_selected, Xr_selected, y_selected, ts_selected = \
        shuffle(Xl_selected, Xr_selected, y_selected, ts_selected, random_state=seed)
    
    if id is None:
        return train_test_split(Xl, Xr, y, ts, test_size=0.2, random_state=seed)
    else:
        dataset = {}
        dataset['Xl'] = Xl_selected
        dataset['Xr'] = Xr_selected
        dataset['y'] = y_selected
        dataset['ts'] = ts_selected
        pickle.dump(dataset, open('./data/bigbench/id_' + str(id) +
                                  '_train.npy', 'w'))
        dataset = {}
        dataset['Xl'] = Xl
        dataset['Xr'] = Xr
        dataset['y'] = y
        dataset['ts'] = ts
        pickle.dump(dataset, open('./data/bigbench/id_' + str(id) +
                                  '_test.npy', 'w'))
        return Xl_selected, Xr_selected, y_selected, ts_selected, Xl, Xr, y, ts


def load_dataset():
    if os.path.exists("./data/bigbench/full.npy") is not True:
        Xl, Xr, y, ts, Xl_selected, Xr_selected, y_selected, ts_selected = \
            select_by_functionality_id()
        
        dataset = {}
        dataset['Xl'] = Xl
        dataset['Xr'] = Xr
        dataset['y'] = y
        dataset['ts'] = ts
        pickle.dump(dataset, open('./data/bigbench/full.npy', 'w'))
        return shuffle(Xl, Xr, y, ts, random_state=seed)
    else:
        dataset = pickle.load(open('./data/bigbench/full.npy', 'r'))
        return shuffle(dataset['Xl'], dataset['Xr'], dataset['y'],
                       dataset['ts'], random_state=seed)


if __name__ == '__main__':
    Xl, Xr, y, ts = load_dataset()
    print("Y test distribution: ", np.bincount(y))
    print("T1: ", (np.bincount(y[ts==0])))
    print("T2: ", (np.bincount(y[ts == 1])))
    print("ST3: ", (np.bincount(y[ts == 2])))
    print("M3: ", (np.bincount(y[ts == 3])))
    print("WT3/T4 ", (np.bincount(y[ts == 4])))