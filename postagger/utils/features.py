from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix


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
        """actual feature function - applied per sample
        :return: lists (data, row, col)"""
        pass


class Unigram(FeatureFunction):
    """
    unigram features, takes tags list as parameter
    """
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

    matrix = csr_matrix((data, (row, col)), shape=matrix_shape)

    return matrix


def build_feature_matrix_(X, y, sentences, feature_fncs):
    """
    build feature matrix from training set, i.e., for each (x_i, y_i)
    output shape: (|X|, m)
    """
    # get size
    m = 0
    for f in feature_fncs:
        m += f.m

    matrix_shape = (len(X), m)

    # build features matrix
    data, row, col = [], [], []
    for i, x in enumerate(X):
        cur_data, cur_i, cur_j = build_features(x, y[i], sentences, feature_fncs, i_shift=i)
        # append
        data += cur_data
        row += cur_i
        col += cur_j

    matrix = csr_matrix((data, (row, col)), shape=matrix_shape)

    return matrix
