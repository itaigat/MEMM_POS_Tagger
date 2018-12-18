import os
import time
from copy import copy
from os.path import join, dirname
import pickle
import numpy as np

from postagger.utils.features import build_y_x_matrix


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
        path = join('resources', data_file)
    else:
        path = join(os.getcwd(), 'resources', data_file)

    return path


def get_probabilities(x, w, sentences, callable_functions):
    """
    computes q(v|u,t,sent_id,word_id)
    """
    # ('NN', 'VB', 'I ate food', 2)
    # x = build_feature_matrix_()
    tags = copy(poss)
    # shape (|Y|*|X|, m)
    try:
        y_x_matrix = build_y_x_matrix(X=x, poss=tags, sentences=sentences, feature_functions=callable_functions)
    except Exception as e:
        print(x)
        print(tags)
        print(sentences)
        print(e)

    dot_prod = y_x_matrix.dot(w)  # shape (|Y|*|X|, 1)
    dot_prod = dot_prod.reshape(-1, len(x))  # shape (|Y|, |X|)

    scores = np.exp(dot_prod)
    norma = np.sum(scores)  # shape (1,)

    probs_matrix = scores / norma  # shape (|Y|, |X|)
    probs_matrix = probs_matrix.reshape(-1, 1)  # shape (|Y|*|X|, 1)

    return probs_matrix


def max_probabilities(probability_dic, sk2, u, v, w, sentence_id, sentence_list, word_id, tmp_probabilities_dic,
                      callable_functions):
    max_probability = 0
    argmax_probability = ''
    v_index = poss.index(v)

    for tag in sk2:
        if (u, tag) not in tmp_probabilities_dic.keys():
            tmp_probabilities_dic[(u, tag)] = get_probabilities([(tag, u, sentence_id, word_id)], w, sentence_list,
                                                                callable_functions)

        tmp = probability_dic[(word_id - 1, tag, u)] * tmp_probabilities_dic[(u, tag)][v_index][0]

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


def viterbi(sentence_id, sentence_lst, w, callable_functions):
    sentence = sentence_lst[sentence_id]
    len_sentence = len(sentence)
    tags = ['' for i in range(len_sentence)]

    tmp_probabilities_dic = {}
    probability_dic = {(-1, '*', '*'): 1}
    bp = {}

    for idx, word in enumerate(sentence):
        s_current, sk1, sk2 = init_s(idx)
        for v in s_current:
            for u in sk1:
                # TODO: Check what bp gets
                probability_dic[(idx, u, v)], bp[(idx, u, v)] = max_probabilities(probability_dic, sk2, u, v, w,
                                                                                  sentence_id, sentence_lst, idx,
                                                                                  tmp_probabilities_dic,
                                                                                  callable_functions)
    tags[len_sentence - 2], tags[len_sentence - 1] = pie_arg_max(probability_dic, sentence)

    for k in range(len_sentence - 3, -1, -1):
        tag = bp[(k + 2, tags[k + 1], tags[k + 2])]
        tags[k] = tag

    return list(reversed(tags))


def pickle_load(filename):
    try:
        with open(filename, 'rb') as handle:
            pickled = pickle.load(handle)
            return pickled
    except Exception as e:
        print("Pickle load failed with filename: " + str(filename))
        print("Exception raised: " + str(e))
        return None


def pickle_save(obj, filename):
    try:
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True
    except Exception as e:
        print("Pickle save failed with filename, object: " + str(filename) + ',' + str(obj))
        print("Exception raised: " + str(e))
        return None


'''
Part Of Speech from the template
poss = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WP$', 'WRB', '#', '$', "''", '``', '(', ')', ',', '.', ':']
'''

poss = ['RBR', '``', 'JJS', ',', 'VBG', 'VBZ', 'TO', 'MD', 'JJ', 'RB', 'VBP', '-LRB-', 'DT', 'WP$', 'PDT', 'CD', 'NN',
        'WP', 'VB', '$', 'POS', 'WRB', 'IN', 'VBN', 'NNP', 'RP', 'EX', 'JJR', 'PRP', '-RRB-', "''", 'VBD', '.', 'RBS',
        ':', 'PRP$', 'NNS', 'WDT', 'CC', 'UH']

# if __name__ == '__main__':
#     # test get_probab
#     preprocess_dict = {
#         'wordtag-f100': [('the', 'DT')],
#         'suffix-f101': [('ing', 'VBG')],
#         'prefix-f102': [('pre', 'NN')],
#         'trigram-f103': [('DT', 'JJ', 'NN')],  # TODO: broken, param optimized is zero
#         'bigram-f104': [('DT', 'JJ')],
#         'unigram-f105': ['DT'],
#         'previousword-f106': [('the', 'NNP')],  # TODO: broken, param optimized is zero
#         'nextword-f107': [('the', 'VB')],
#         'starting_capital': ['DT'],
#         'capital_inside': ['NN'],
#         'number_inside': ['CD']
#     }
#     x = [('DT', 'NN', 0, 2), ('DT', 'DT', 0, 1)]
#     tags = poss
#     sentences = [['The', 'dog', 'walks']]
#     w = np.random.rand(11)
#     callables = init_callable_features(tags, Params, preprocess_dict)
#     v = get_probabilities(x, sentences, w, callables)
#     print(viterbi(2, sentences[0], w, callables))
#     print(v)
