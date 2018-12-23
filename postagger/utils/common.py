import os
from copy import copy
from os.path import join
import pickle
import numpy as np
import time

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
    tags = copy(poss)
    # shape (|Y|*|X|, m)
    y_x_matrix, help_matrix = build_y_x_matrix(X=x, poss=tags, sentences=sentences,
                                               feature_functions=callable_functions)
    sums = 1.0 / (help_matrix.transpose() * (help_matrix * np.exp(y_x_matrix * w)))
    probs_matrix = sums * np.exp(y_x_matrix * w)
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
    path = []

    tmp_probabilities_dic = {}
    probability_dic = {(-1, '*', '*'): 1}
    bp = {}
    start = time.time()
    for idx, word in enumerate(sentence):
        s_current, sk1, sk2 = init_s(idx)
        for v in s_current:
            for u in sk1:
                probability_dic[(idx, u, v)], bp[(idx, u, v)] = max_probabilities(probability_dic, sk2, u, v, w,
                                                                                  sentence_id, sentence_lst, idx,
                                                                                  tmp_probabilities_dic,
                                                                                  callable_functions)
    tags[len_sentence - 2], tags[len_sentence - 1] = pie_arg_max(probability_dic, sentence)

    for k in range(len_sentence - 3, -1, -1):
        tag = bp[(k + 2, tags[k + 1], tags[k + 2])]
        tags[k] = tag

    end = time.time()
    print(end - start)

    return list(reversed(tags))


def viterbi_s(sentence_id, sentence_lst, w, callable_functions):
    start = time.time()
    V = [{}]
    count = 0

    path = {}

    first_state = get_probabilities([('*', '*', sentence_id, 0)], w, sentence_lst, callable_functions)

    for idx, y in enumerate(poss):
        curr_prob = first_state[idx]
        V[0][y] = curr_prob
        path[y] = [y]

    for t in range(1, len(sentence_lst[sentence_id])):
        V.append({})
        new_path = {}
        # curr_obs = features_list[t]
        all_proba_dict_dict = {}
        for y0 in poss:
            if t != 1:
                all_proba_dict_dict[y0] = get_probabilities([(y0, path[y0][-2], sentence_id, t)], w,
                                                            sentence_lst, callable_functions)
            else:
                all_proba_dict_dict[y0] = get_probabilities([(y0, '*', sentence_id, t)], w, sentence_lst,
                                                            callable_functions)
        for y_idx, y in enumerate(poss):
            max_prob = - 1
            former_state = None
            for y0 in poss:
                curr_prob = V[t - 1][y0]
                # curr_obs = obs_dict[y0]
                proba_dict = all_proba_dict_dict[y0]
                curr_prob = curr_prob * proba_dict[y_idx]

                if curr_prob > max_prob:
                    max_prob = curr_prob
                    former_state = y0
            V[t][y] = max_prob
            new_path[y] = path[former_state] + [y]

        path = new_path

    prob = -1
    for y in poss:
        cur_prob = V[len(sentence_lst[sentence_id]) - 1][y]
        if cur_prob > prob:
            prob = cur_prob
            state = y
    end = time.time()
    print(end - start)

    # return prob, path[state]
    return path[state]


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