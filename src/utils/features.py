from scipy.sparse import lil_matrix, hstack, vstack

from .common import poss


def uni(**kwargs):
    """
    Compose unigram features per sample
    y: tag
    :return: feature vector of shape (m,)
    """
    y = kwargs['y']
    features_matrix = lil_matrix((len(y), len(poss)))

    for i, pos in enumerate(poss):
        for label_idx, label in enumerate(y):
            if label == pos:
                features_matrix[label_idx, i] = 1

    return features_matrix


def build_features(X, y, sentences, features_funcs):
    """
    applies predefined functions on one sample
    :param X: history tuple <u,v,sentence,i>
    :param y: tag at place i
    :param sentences: list of sentences which are list of words
    :param features_funcs: functions to apply
    :return:
    """
    features_vec = []
    sent_idx = X[2]
    sentence = sentences[sent_idx]

    for func in features_funcs:
        f = func(X=X, y=y, sentence=sentence)
        features_vec.append(f)

    return hstack(features_vec)


def get_feature_matrix(iterable_sentences, features_funcs):
    feature_matrix_lst = []

    for tuples, tags, sentence in iterable_sentences:
        for i in range(len(tuples)):
            f = build_features(tuples[i], tags[i], sentence, features_funcs)  # f shape: (m,)
            feature_matrix_lst.append(f)

    return vstack(feature_matrix_lst)
