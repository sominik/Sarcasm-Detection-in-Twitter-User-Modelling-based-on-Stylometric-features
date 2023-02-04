#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import pickle
from collections import defaultdict
import sys
import pandas as pd
from ast import literal_eval


TRAIN_MAP_FILE = "../csv_files/clean_datasets/train_tweets_dataset.csv"
TEST_MAP_FILE = "../csv_files/clean_datasets/test_tweets_dataset.csv"


def build_data_cv(data_folder, clean_string=True):
    revs = []

    sarcastic_train_file = data_folder[0]
    sarcastic_test_file = data_folder[1]

    train_data = np.asarray(pd.read_csv(sarcastic_train_file, header=None))
    test_data = np.asarray(pd.read_csv(sarcastic_test_file, header=None))

    vocab = defaultdict(float)

    for line in train_data:
        rev = []
        label_str = line[6]
        if label_str == 'non_sarcastic':
            label = [1, 0]
        else:
            label = [0, 1]
        rev.append(line[1].strip())
        if clean_string:
            orig_rev = " ".join(literal_eval(line[5]))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum = {"y": int(1),
                 "id": line[0],
                 "text": orig_rev,
                 "author": line[2],
                 "label": label,
                 "num_words": len(orig_rev.split()),
                 "split": int(1)}
        revs.append(datum)

    for line in test_data:
        rev = []
        label_str = line[6]
        if label_str == 'non_sarcastic':
            label = [1, 0]
        else:
            label = [0, 1]
        rev.append(line[1].strip())
        if clean_string:
            orig_rev = " ".join(literal_eval(line[5]))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum = {"y": int(1),
                 "id": line[0],
                 "text": orig_rev,
                 "author": line[2],
                 "label": label,
                 "num_words": len(orig_rev.split()),
                 "split": int(0)}
        revs.append(datum)

    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_fasttext(fname, vocab):
    """
    Loads 300x1 word vecs from Fasttext
    """
    print("Loading FastText Model")
    f = open(fname, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        if word in vocab:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


if __name__ == "__main__":
    w2v_file = sys.argv[1]
    data_folder = [TRAIN_MAP_FILE, TEST_MAP_FILE]
    # data_folder = ["./csv_files/test.csv", "./csv_files/test2.csv"]   # # should delete
    print("loading data...")
    revs, vocab = build_data_cv(data_folder, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: ", len(revs))
    print("vocab size: ", len(vocab))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...")
    w2v = load_fasttext(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    print("num words in word2vec after adding unknown words: " + str(len(w2v)))
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open("../main_balanced_pickle.p", "wb"))
    print("dataset created!")
