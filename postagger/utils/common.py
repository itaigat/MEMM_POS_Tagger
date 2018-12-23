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


def get_probabilities(x, w, sentences, callable_functions, poss):
    """
    computes q(v|u,t,sent_id,word_id)
    """
    tags = poss
    # shape (|Y|*|X|, m)
    y_x_matrix, help_matrix = build_y_x_matrix(X=x, poss=tags, sentences=sentences,
                                               feature_functions=callable_functions)
    sums = 1.0 / (help_matrix.transpose() * (help_matrix * np.exp(y_x_matrix * w)))
    probs_matrix = sums * np.exp(y_x_matrix * w)
    return probs_matrix



def viterbi_s(sentence_id, sentence_lst, w, callable_functions, poss):
    start = time.time()
    path_dict, V_paths = {}, [{}]
    state = None

    first_state = get_probabilities([('*', '*', sentence_id, 0)], w, sentence_lst, callable_functions, poss)

    for idx, y in enumerate(poss):
        current_probability = first_state[idx]
        V_paths[0][y] = current_probability
        path_dict[y] = [y]

    for t in range(1, len(sentence_lst[sentence_id])):
        V_paths.append({})
        new_path_dict = {}
        all_probabilities_dict_dict = {}

        for y0 in poss:
            if t != 1:
                all_probabilities_dict_dict[y0] = get_probabilities([(y0, path_dict[y0][-2], sentence_id, t)], w,
                                                                    sentence_lst, callable_functions, poss)
            else:
                all_probabilities_dict_dict[y0] = get_probabilities([(y0, '*', sentence_id, t)], w, sentence_lst,
                                                                    callable_functions, poss)

        for y_idx, y in enumerate(poss):
            max_prob = - 1
            last = None
            for y0 in poss:
                current_probability = V_paths[t - 1][y0]
                probabilities_dict = all_probabilities_dict_dict[y0]
                current_probability = current_probability * probabilities_dict[y_idx]

                if current_probability > max_prob:
                    max_prob = current_probability
                    last = y0

            V_paths[t][y] = max_prob
            new_path_dict[y] = path_dict[last] + [y]

        path_dict = new_path_dict
    prob = -1

    for y in poss:
        cur_prob = V_paths[len(sentence_lst[sentence_id]) - 1][y]
        if cur_prob > prob:
            prob = cur_prob
            state = y

    end = time.time()
    print(end - start)

    return path_dict[state]


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


def get_tags(file):
    poss = []

    if file == 'train.wtag':
        poss = ['RBR', '``', 'JJS', ',', 'VBG', 'VBZ', 'TO', 'MD', 'JJ', 'RB', 'VBP', '-LRB-', 'DT', 'WP$', 'PDT', 'CD',
                'NN', 'WP', 'VB', '$', 'POS', 'WRB', 'IN', 'VBN', 'NNP', 'RP', 'EX', 'JJR', 'PRP', '-RRB-', "''", 'VBD',
                '.', 'RBS', ':', 'PRP$', 'NNS', 'WDT', 'CC', 'UH', 'SYM', 'NNPS', 'FW', '#']
    elif file == 'train2.wtag':
        poss = ['WP', 'JJS', 'LS', 'RBS', 'SYM', 'RBR', 'NNP', 'JJR', 'RP', 'EX', 'RB', 'PRP', 'VBP', 'IN', 'NN',
                  'DT', 'JJ', 'MD', 'VB', 'TO', 'NNS', 'VBD', 'CC', 'WDT', 'VBG', 'VBN', 'VBZ', 'CD', 'FW', 'WRB',
                  'PDT', ',', ':', '.']

    return poss


def get_poss_dict(poss):
    pos_dic = {}

    for idx, pos in enumerate(poss):
        pos_dic[pos] = idx

    return pos_dic

'''
Part Of Speech from the template
poss = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WP$', 'WRB', '#', '$', "''", '``', '(', ')', ',', '.', ':']

poss = ['RBR', '``', 'JJS', ',', 'VBG', 'VBZ', 'TO', 'MD', 'JJ', 'RB', 'VBP', '-LRB-', 'DT', 'WP$', 'PDT', 'CD', 'NN',
        'WP', 'VB', '$', 'POS', 'WRB', 'IN', 'VBN', 'NNP', 'RP', 'EX', 'JJR', 'PRP', '-RRB-', "''", 'VBD', '.', 'RBS',
        ':', 'PRP$', 'NNS', 'WDT', 'CC', 'UH']

poss_train = ['RBR', '``', 'JJS', ',', 'VBG', 'VBZ', 'TO', 'MD', 'JJ', 'RB', 'VBP', '-LRB-', 'DT', 'WP$', 'PDT', 'CD',
            'NN', 'WP', 'VB', '$', 'POS', 'WRB', 'IN', 'VBN', 'NNP', 'RP', 'EX', 'JJR', 'PRP', '-RRB-', "''", 'VBD',
            '.', 'RBS', ':', 'PRP$', 'NNS', 'WDT', 'CC', 'UH', 'SYM', 'NNPS', 'FW', '#']

poss_2 = ['WP', 'JJS', 'LS', 'RBS', 'SYM', 'RBR','NNP', 'JJR', 'RP', 'EX', 'RB', 'PRP', 'VBP', 'IN', 'NN',
          'DT', 'JJ',  'MD','VB', 'TO','NNS', 'VBD', 'CC', 'WDT','VBG', 'VBN', 'VBZ', 'CD', 'FW', 'WRB',
          'PDT',',', ':', '.']
'''