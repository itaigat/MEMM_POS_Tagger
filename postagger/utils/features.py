import numpy as np
from .common import poss
from scipy.sparse import lil_matrix, hstack, vstack, csr_matrix


def uni(x, y, sentence):
    """
    Compose unigram features per sample
    :param x: history tuple
    :param y: tag
    :param sentence:
    :return: feature vector of shape (m,)
    """
    lil_mtx_shape = (len(poss), 1)
    feature_vec = lil_matrix(lil_mtx_shape)

    # unigram: search for index of current y
    # i = unigram_dict[y]  # fails for tags not on train_dev
    for i, pos in enumerate(poss):
        if y == pos:
            feature_vec[i] = 1

    return feature_vec.transpose()


def build_features(x, y, sentences, features_fncs):
    """
    applies predefined functions on one sample
    :param x: history tuple <u,v,sentence,i>
    :param y: tag at place i
    :param sentences: list of sentences which are list of words
    :param features_fncs: functions to apply
    :return: features vector (concatenated)
    """
    features_vec = lil_matrix((0,0))
    sent_idx = x[2]
    sentence = sentences[sent_idx]
    for func in features_fncs:
        f = func(x, y, sentence)
        # assumes func outputs sparse vectors in shape (1,features_num)
        features_vec = hstack([features_vec, f])

    # sparse vector of shape (1,total_features_num)
    return features_vec


def compute_y_x_matrix(X, sentences, features_fncs):
    """ SPARSE
    for each x in X, compute all y's for x:
    works by vertical stacking of shape (1,m)
    output of shape (|Y|*|X|,m)
    """
    feature_matrix = lil_matrix((0,0))
    for i, x in enumerate(X):
        for j, pos in enumerate(poss):
            f = build_features(x, pos, sentences, features_fncs)  # f shape: (m,)
            feature_matrix = vstack([feature_matrix, f])  # add another row to matrix

    return csr_matrix(feature_matrix)


def build_feature_matrix(X, y, sentences, features_fncs):
    """ SPARSE
    stacks f(x_i,y_i) vertically, output shape: (|X|, m)
    """
    # build features matrix
    feature_matrix = lil_matrix((0, 0))
    len_dataset = len(X)

    for i in range(len_dataset):
        f = build_features(X[i], y[i], sentences, features_fncs)  # f shape: (m,)
        feature_matrix = vstack([feature_matrix, f])  # add another row to matrix

    return csr_matrix(feature_matrix)