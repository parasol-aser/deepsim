import os, sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
import cPickle

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from sklearn import (manifold, decomposition)
from PIL import Image

mat_dim = 128


def remove_invalid_samples(dir):
    list_dirs = os.walk(dir)
    samples_count = 0
    for root, dirs, _ in list_dirs:
        for d in dirs:
            data_folder = os.path.join(root, d)
            list_sub_dirs = os.walk(data_folder)
            for subroot, subdirs, data_files in list_sub_dirs:
                # print "Read googlejam_data in folder ", subroot
                if len(data_files) > 2:  # .txt and .info
                    # os.rmdir(data_folder) #cannot delete unempty dirs
                    shutil.rmtree(data_folder)
                if data_folder.find('$') != -1:
                    # os.rmdir(data_folder)
                    if os.path.exists(data_folder):
                        shutil.rmtree(data_folder)
                    related_data_folder = os.path.join(root, d[:d.index('$')])
                    if os.path.exists(related_data_folder):
                        # os.rmdir(related_data_folder)
                        shutil.rmtree(related_data_folder)
                if os.path.exists(data_folder):
                    for data_file in data_files:
                        if data_file.find('main') == -1:
                            shutil.rmtree(data_folder)
                            break

    print "Clean all folder contains more than 1 functions."


def record_valid_pairs(dir):
    list_dirs = os.walk(dir)
    samples_count = 0
    clones_sets = dict()
    for root, dirs, _ in list_dirs:
        for d in dirs:
            list_sub_dirs = os.walk(os.path.join(root, d))
            for subroot, subdirs, data_files in list_sub_dirs:
                # print "Read googlejam_data in folder ", subroot
                if len(data_files) > 2:
                    continue
                relative_root = os.path.join("googlejam", d)
                print "googlejam_data file: ", data_files[1]
                jam_dir = subroot[subroot.rindex('/') + 1:]
                jam_dir = jam_dir[:jam_dir.index('.')]
                jam_number = int(jam_dir[9:])
                if clones_sets.has_key(jam_number):
                    clones_sets[jam_number].append(os.path.join(relative_root, data_files[0]))
                else:
                    clones_sets[jam_number] = []
                    clones_sets[jam_number].append(os.path.join(relative_root, data_files[0]))
                samples_count = samples_count + 1

    print "%d methods remain" % (samples_count)

    fout = open('googlejam_data/googlejam.txt', 'w')
    for i in xrange(len(clones_sets)):
        clones = clones_sets[i + 1]
        for j in xrange(len(clones)):
            for k in xrange(j + 1, len(clones)):
                fout.write("%s %s\n" % (clones[j], clones[k]))
    fout.close()


def input_data(file_dir):
    min = 0
    max = 2 ** 24 - 1
    list_dirs = os.walk(file_dir)
    X = []
    file_dest = []
    vcount_list = []
    for root, dirs, files in list_dirs:
        for d in dirs:
            list_sub_dirs = os.walk(os.path.join(root, d))
            for sub_root, sub_dirs, sub_files in list_sub_dirs:
                for datafile in sub_files:
                    if os.path.splitext(datafile)[1] != '.txt':
                        continue
                    mat = pd.read_csv(os.path.join(sub_root, datafile), delim_whitespace=True, header=None).values
                    vcount_list.append(mat[-1][-1])
                    mat = (mat - min) * 1.0 / max
                    # if np.sum(np.sum(mat, axis=1), axis=0) < 0.00001:
                    #     continue
                    X.append(mat)
                    file_dest.append(os.path.join(sub_root, datafile))
        for file in files:
            if os.path.splitext(file)[1] != '.txt':
                continue
            mat = pd.read_csv(os.path.join(root, file), delim_whitespace=True, header=None).values
            vcount_list.append(mat[-1][-1])
            mat = (mat - min) * 1.0 / max
            # if np.sum(np.sum(mat, axis=1), axis=0) < 0.00001:
            #     continue
            X.append(mat)
            file_dest.append(os.path.join(root, file))
    X = np.array(X, dtype=float)
    X = X.reshape(len(X), mat_dim * mat_dim)
    return X, file_dest, vcount_list


def store(file_dir, save_dir):
    X, file_dest, _ = input_data(file_dir)
    for x, file in zip(X, file_dest):
        new_file = os.path.splitext(file)[0] + ".txt.npy"
        np.save(new_file, x)
    file = save_dir + '/mat.npy'
    np.save(file, X)
    file = save_dir + '/file_dest.npy'
    np.save(file, file_dest)


def store_sub(file_dir, save_dir):
    list_dirs = os.walk(file_dir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            X, file_dest, vcount_list = input_data(os.path.join(root, d))
            save_file = save_dir + "/" + d + "_mat.npy"
            np.save(save_file, X)
            save_file = save_dir + "/" + d + "_file_dest.npy"
            np.save(save_file, file_dest)
            save_file = save_dir + "/" + d + "_vcount.npy"  # varCount + blockCount
            np.save(save_file, vcount_list)
        break


def load_dict(file_path):
    dataset = cPickle.load(open(file_path, 'r'))
    data_dict, file_dest, infos = dataset['data'], dataset['file_dest'], dataset['infos']
    return data_dict, file_dest, infos


def load_pairs(file_path):
    dataset = cPickle.load(open(file_path))
    return dataset['pairs'], dataset['matrices'], dataset['label']


def extract_test_pair_src_file():
    file_path = "/Users/zg/Desktop/dev/java/deepcode/code_search/detector/dataset/training/googlejam/dataset4_dict_128.npy"
    testing_save_path = "/Users/zg/Desktop/dev/java/deepcode/code_search/detector/dataset/training/fixed_data/googlejam4_testing_128.npy"
    data_dict, file_dest, infos = load_dict(file_path)
    pairs, _, labels = load_pairs(testing_save_path)
    test_file_set = set()
    for (i, j) in pairs:
        test_file_set.add(i)
        test_file_set.add(j)
    save_path = save_path = "/Users/zg/Desktop/benchmark/google_code_jam/complete/googlejam4_classes_test_pair_128.txt"
    fout = open(save_path, 'w')
    for i in test_file_set:
        mat_file = file_dest[i]
        tokens = mat_file.split('/')
        src_path = ""
        src_path += tokens[-3] + '/'
        src_path += '/'.join(tokens[-2].split('.')) + ".java"
        fout.write(src_path + "\n")
    fout.close()
    testing_label_save_path = "/Users/zg/Desktop/benchmark/google_code_jam/complete/googlejam4_classes_test_pair_128_label.txt"
    fout = open(testing_label_save_path, 'w')
    for ((i, j), l) in zip(pairs, labels):
        fout.write("%s %s %d\n" % (file_dest[i].split('/')[-2] + ".java", file_dest[j].split('/')[-2] + ".java", l))
    fout.close()



def extract_test_src_file(file, save_path):
    data_dict, file_dest, infos = load_dict(file)
    fout = open(save_path, 'w')
    for mat_file in file_dest:
        tokens = mat_file.split('/')
        src_path = ""
        src_path += tokens[-3] + '/'
        src_path += '/'.join(tokens[-2].split('.')) + ".java"
        fout.write(src_path + "\n")
    fout.close()


def copy_test_src_file(file_list_path, src_files_folder, dest_folder):
    fin = open(file_list_path, 'r')
    line = fin.readline()
    while line is not None and len(line) > 0:
        src_file = os.path.join(src_files_folder, "/".join(line.split('/')[1:])[:-1])
        tokens = line.split('/')
        dest_file = ".".join(tokens[1:])[:-1]
        dest_file = os.path.join(dest_folder, tokens[0] + "/" + dest_file)
        if os.path.exists(src_file):
            shutil.copyfile(src_file, dest_file)
        if os.path.exists(src_file) is not True:
            print "file not exist: ", src_file
        line = fin.readline()



def extract_src_file():
    file_dir = "/Users/zg/Desktop/benchmark/google_code_jam/complete/googlejam4_classes_128"
    list_dirs = os.walk(file_dir)
    fout = open("/Users/zg/Desktop/benchmark/google_code_jam/complete/googlejam4_classes_128.txt", 'w')
    for root, dirs, _ in list_dirs:
        for d in dirs:
            set_folder = os.path.join(root, d)
            list_sub_dirs = os.walk(set_folder)
            for subroot, subdirs, _ in list_sub_dirs:
                if len(subdirs) == 0:
                    continue
                fout.write("%s %d\n" % (d, len(subdirs)))
                for subdir in subdirs:
                    src_path = subdir.replace('.', '/') + '.java'
                    fout.write(src_path + "\n")
    fout.close()


def copy_src_files():
    src_list_file = "/Users/zg/Desktop/benchmark/google_code_jam/complete/googlejam4_classes_128.txt"
    src_folder = "/Users/zg/Desktop/benchmark/google_code_jam/benchmark4"
    dest_folder = "/Users/zg/Desktop/benchmark/google_code_jam/complete/googlejam4_src"
    fin = open(src_list_file, 'r')
    line = fin.readline()
    while line is not None and len(line) > 0:
        label, num = line.split(' ')
        num = int(num)
        sub_dest_folder = os.path.join(dest_folder, label)
        if os.path.exists(sub_dest_folder) is not True:
            os.mkdir(sub_dest_folder)
        for _ in xrange(num):
            short_file_name = fin.readline()[:-1]
            file_name = os.path.join(src_folder, short_file_name)
            dest_short_file_name = short_file_name.replace('/', '.')
            # dest_file = os.path.join(sub_dest_folder, file_name[file_name.rindex('/') + 1:])
            dest_file = os.path.join(sub_dest_folder, dest_short_file_name)
            if os.path.exists(file_name) is not True:   #class name is not the java file name
                file_name = os.listdir(os.path.join(src_folder, short_file_name[:short_file_name.rindex('/')]))[0]
                file_name = os.path.join(src_folder, short_file_name[:short_file_name.rindex('/') + 1] + file_name)
                dest_short_file_name = dest_short_file_name[:dest_short_file_name.rindex(".")]
                dest_short_file_name = dest_short_file_name[:dest_short_file_name.rindex(".") + 1]
                dest_file = os.path.join(sub_dest_folder, dest_short_file_name + file_name[file_name.rindex('/') + 1:])
            shutil.copyfile(file_name, dest_file)
        line = fin.readline()
