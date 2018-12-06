import numpy as np
from .common import poss

def uni(x, y, sentence):
    """
    Compose unigram features per sample
    :param x: history tuple
    :param y: tag
    :param sentence:
    :return: feature vector of shape (m,)
    """
    feature = [0 for i in range(len(poss))]  # TODO : np.array
    for i, pos in enumerate(poss):
        if y == pos:
            feature[i] = 1

    return feature


def build_features(x, y, sentences, features_fncs):
    """
    applies predefined functions on one sample
    :param x: history tuple <u,v,sentence,i>
    :param y: tag at place i
    :param sentences: list of sentences which are list of words
    :param features_fncs: functions to apply
    :return: features vector (concatenated)
    """
    features_vec = []  # TODO : np.array
    sent_idx = x[2]
    sentence = sentences[sent_idx]
    for func in features_fncs:
        f = func(x, y, sentence)
        features_vec.append(f)  # TODO: np.concat

    return np.array(features_vec)
