from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix

#   TODO: not ready yet - integrate this module, add method for build_feature_matrix

poss = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WP$', 'WRB', '#', '$', '\'\'', '``', '(', ')', ',', '.', ':']


class FeatureFunction(ABC):
    """
    abstract class to hold the feature function and its output size
    needed for computing the shape of the matrix ahead of time (sparse)
    """
    def __init__(self):
        self.m = self.compute_size()

    @abstractmethod
    def compute_size(self):
        """:return: feature vector size """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """actual feature function"""
        pass


class Unigram(FeatureFunction):

    name = 'unigram'

    def __init__(self, poss):
        self.tags = poss
        super().__init__()

    def compute_size(self):
        return len(self.tags)

    def __call__(self, *args, **kwargs):
        # unpack needed vars
        data, i, j = [], [], []
        y = kwargs['y']
        if y in self.tags:
            index = self.tags.index(y)
            data.append(1)
            # relative indices (will be shifted later)
            i.append(0)  # this will always be 0 since we compute a row vector
            j.append(index)

        return data, i, j


class Paramsb:
    features_fncs = [Unigram]


def build_features(x, y, sentences, features_fncs, i_shift=0):
    """applies predefined functions on one sample
    returns shifted (corrected) data,i,j

    i_shift is the row number, calling function is responsible setting it
    """
    data, i, j = [], [], []
    for ind, f in enumerate(features_fncs):
        j_shift = f.m
        cur_data, cur_i, cur_j = f(y=y)
        # shift j (only if not first feature)
        if ind != 0:
            cur_j = [x+j_shift for x in cur_j]
        cur_i = [x+i_shift for x in cur_i]
        # append
        data += cur_data
        i += cur_i
        j += cur_j

    return data, i, j


def build_y_x_matrix(X, poss, sentences, feature_fncs):
    """build y_x feature matrix in coo format

    output shape: (|Y|*|X|, m)
    """
    # get size
    m = 0
    for f in feature_fncs:
        m += f.m

    matrix_shape = (len(X)*len(poss), m)

    # build features matrix
    current_row = 0
    data, row, col = [], [], []
    for i, x in enumerate(X):
        for j, pos in enumerate(poss):
            cur_data, cur_i, cur_j = build_features(x, pos, sentences, feature_fncs, i_shift=current_row)
            # append
            data += cur_data
            row += cur_i
            col += cur_j
            current_row += 1

    data = np.array(data)
    row = np.array(row)
    col = np.array(col)
    matrix = csr_matrix((data, (row, col)), shape=matrix_shape)

    return matrix

if __name__ == '__main__':
    # tests
    """
    f = Unigram(poss)
    print(f.m)
    data, i, j = f(y='DT')
    data = np.array(data)
    i = np.array(i)
    j = np.array(j)
    print(data.shape)
    print(i)
    print(j)
    """

    # init feature funcs
    """
    callables = []
    for f in Paramsb.features_fncs:
        if f.name == 'unigram':
            arg = poss
        callables.append(f(arg))

    data, i, j = build_features(x=None, y='DT', sentences=None, features_fncs=callables, i_shift=0)
    print(data, i, j)
    """
    pass