import os
import time
from copy import copy
from os.path import join, dirname
from postagger.utils.features import build_y_x_matrix
from postagger.utils.params import Params
from postagger.utils.features import init_callable_features
import numpy as np


def read_file(file_path):
    """
    Securely read a file from a given path
    :param file_path: Path of the file
    :return: The content of the text file
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print('Could not find file in the path')
    except Exception as e:
        print(e)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_data_path(data_file='train_dev.wtag'):
    if os.name == 'nt':
        path = join(dirname(os.getcwd()), 'resources', data_file)
    else:
        path = join(os.getcwd(), 'resources', data_file)

    return path


def get_probabilities(x, tags, sentences, features_funcs, w):
    """
    computes q(v|u,t,sent_id,word_id)
    """
    # ('NN', 'VB', 'I ate food', 2)
    # x = build_feature_matrix_()
    y_x_matrix = build_y_x_matrix(X=x, poss=tags, sentences=sentences, feature_functions=features_funcs)  # shape (|Y|*|X|, m)
    dot_prod = y_x_matrix.dot(w)  # shape (|Y|*|X|, 1)
    dot_prod = dot_prod.reshape(-1, len(x))  # shape (|Y|, |X|)
    scores = np.exp(dot_prod)
    norma = np.sum(scores)  # shape (1,)
    probs_matrix = scores / norma  # shape (|Y|, |X|)
    probs_matrix = probs_matrix.reshape(-1, 1)  # shape (|Y|*|X|, 1)

    return probs_matrix


def max_probabilities(probability_dic, sk2, u, v, w, sentence, word_id):
    max_probability = 0
    argmax_probability = ''
    probabilities_vector = get_probabilities(u, v, w, sentence, word_id)

    for tag in enumerate(sk2):
        tmp = probability_dic[(word_id, tag, u)] * probabilities_vector[v]
        if tmp > max_probability:
            max_probability = tmp
            argmax_probability = tag

    return max_probability, argmax_probability


def pie_arg_max(probability_dic, sentence):
    u_max, v_max = '', ''
    max_probability = 0
    last_idx_sentence = len(sentence) - 1

    for u in poss:
        for v in poss:
            if probability_dic[(last_idx_sentence, u, v)] > max_probability:
                max_probability = probability_dic[(last_idx_sentence, u, v)]
                u_max, v_max = u, v

    return u_max, v_max


def init_s(idx):
    s = copy(poss)

    if idx == 0:
        return s, ['*'], ['*']
    elif idx == 1:
        return s, s, ['*']
    else:
        return s, s, s


def viterbi(sentence, w):
    sentence_lst = sentence.split(' ')
    len_sentence = len(sentence_lst)
    tags = ['' for i in range(len_sentence)]

    probability_dic = {(0, '*', '*'): 1}
    bp = {}

    for idx, word in enumerate(sentence_lst):
        s_current, sk1, sk2 = init_s(idx)

        for v in s_current:
            for u in sk1:
                # TODO: Check what bp gets
                probability_dic[(idx, u, v)], bp[(idx, u, v)] = max_probabilities(probability_dic, sk2,
                                                                                  u, v, w, sentence, idx)

    tags[len_sentence - 1], tags[len_sentence] = pie_arg_max(probability_dic, sentence_lst)

    for k in range(len_sentence - 2, -1, -1):
        tag = bp[(k + 2, tags[k + 1], tags[k + 2])]
        tags.append(tag)

    return reversed(tags)


'''
Part Of Speech from the template
poss = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WP$', 'WRB', '#', '$', "''", '``', '(', ')', ',', '.', ':']
'''

poss = ['RBR', '``', 'JJS', ',', 'VBG', 'VBZ', 'TO', 'MD', 'JJ', 'RB', 'VBP', '-LRB-', 'DT', 'WP$', 'PDT', 'CD', 'NN',
        'WP', 'VB', '$', 'POS', 'WRB', 'IN', 'VBN', 'NNP', 'RP', 'EX', 'JJR', 'PRP', '-RRB-', "''", 'VBD', '.', 'RBS',
        ':', 'PRP$', 'NNS', 'WDT', 'CC', 'UH']

if __name__ == '__main__':
    # test get_probab
    preprocess_dict = {
        'wordtag-f100': [('the', 'DT')],
        'suffix-f101': [('ing', 'VBG')],
        'prefix-f102': [('pre', 'NN')],
        'trigram-f103': [('DT', 'JJ', 'NN')],  # TODO: broken, param optimized is zero
        'bigram-f104': [('DT', 'JJ')],
        'unigram-f105': ['DT'],
        'previousword-f106': [('the', 'NNP')],  # TODO: broken, param optimized is zero
        'nextword-f107': [('the', 'VB')],
        'starting_capital': ['DT'],
        'capital_inside': ['NN'],
        'number_inside': ['CD']
    }
    x = [('DT', 'NN', 0, 2)]
    tags = poss
    sentences = [['The', 'dog', 'walks']]
    w = np.random.rand(11)
    callables = init_callable_features(tags, Params, preprocess_dict)
    v = get_probabilities(x, tags, sentences, callables, w)
    print(v)