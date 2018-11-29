import numpy as np


def uni(x, y, sentence):
    """
    Compose unigram features per sample
    :param x: history tuple
    :param y: tag
    :param sentence:
    :return: feature vector of shape (m,)
    """
    poss = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'WDT', 'WP', 'WP$', 'WRB']
    feature = [0 for i in range(len(poss))]
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
    :return:
    """
    features_vec = []
    sent_idx = x[2]
    sentence = sentences[sent_idx]
    for func in features_fncs:
        f = func(x, y, sentence)
        features_vec.append(f)

    return np.array(features_vec)