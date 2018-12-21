from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix


def extract_current_word(**kwargs):
    sentence = kwargs['sentence']
    word_id = kwargs['x'][3]
    word = sentence[word_id]
    return word


def extract_prev_word(**kwargs):
    sentence = kwargs['sentence']
    word_id = kwargs['x'][3]
    if word_id > 0:
        prev_word = sentence[word_id - 1]
        return prev_word
    else:
        return None


def extract_pp_word(**kwargs):
    sentence = kwargs['sentence']
    word_id = kwargs['x'][3]
    if word_id > 1:
        prev_word = sentence[word_id - 2]
        return prev_word
    else:
        return None


def extract_next_word(**kwargs):
    sentence = kwargs['sentence']
    word_id = kwargs['x'][3]
    if word_id < len(sentence)-1:
        next_word = sentence[word_id + 1]
        return next_word
    else:
        return None


def extract_nn_word(**kwargs):
    sentence = kwargs['sentence']
    word_id = kwargs['x'][3]
    if word_id < len(sentence)-2:
        next_word = sentence[word_id + 2]
        return next_word
    else:
        return None


class FeatureFunction(ABC):
    """
    abstract class to hold the feature function and its output size
    """
    def __init__(self, tags, tuples):
        self.tags = tags
        self.tuples = tuples
        self.m = self.compute_size()

    def compute_size(self):
        """:return: feature vector size """
        return len(self.tuples)

    @abstractmethod
    def __call__(self, **kwargs):
        """actual feature function - applied per sample
        :return: lists (data, row, col)"""
        pass


class Wordtag(FeatureFunction):
    name = 'wordtag-f100'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        word = extract_current_word(**kwargs)
        tag = kwargs['y']
        tup = (word, tag)
        if tup in self.tuples:
            index = self.tuples[tup]
            data.append(1)
            i.append(0)
            j.append(index)

        return data, i, j


class Suffix(FeatureFunction):
    name = 'suffix-f101'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        word = extract_current_word(**kwargs)
        suffixes = []
        if len(word) >= 4:
            suffixes = [word[-i:] for i in range(1, 5)]
        tag = kwargs['y']
        st_tuples = [(suffix, tag) for suffix in suffixes]
        for tup in st_tuples:
            if tup in self.tuples:
                index = self.tuples[tup]
                data.append(1)
                i.append(0)
                j.append(index)

        return data, i, j


class Prefix(FeatureFunction):

    name = 'prefix-f102'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        word = extract_current_word(**kwargs)
        prefixes = []
        if len(word) >= 4:
            prefixes = [word[:i] for i in range(1, 5)]
        tag = kwargs['y']
        pt_tuples = [(prefix, tag) for prefix in prefixes]
        for tup in pt_tuples:
            if tup in self.tuples:
                index = self.tuples[tup]
                data.append(1)
                i.append(0)
                j.append(index)

        return data, i, j


class Trigram(FeatureFunction):
    name = 'trigram-f103'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        pre_pre_tag = kwargs['x'][0]
        pre_tag = kwargs['x'][1]
        tag = kwargs['y']
        tup = (pre_pre_tag, pre_tag, tag)
        if tup in self.tuples:
            index = self.tuples[tup]
            data.append(1)
            i.append(0)
            j.append(index)

        return data, i, j


class Bigram(FeatureFunction):
    name = 'bigram-f104'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        pre_tag = kwargs['x'][1]
        tag = kwargs['y']
        tup = (pre_tag, tag)
        if tup in self.tuples:
            index = self.tuples[tup]
            data.append(1)
            i.append(0)
            j.append(index)

        return data, i, j


class Unigram(FeatureFunction):
    name = 'unigram-f105'

    def __call__(self, **kwargs):
        # unpack needed vars
        data, i, j = [], [], []
        y = kwargs['y']
        if y in self.tuples:
            index = self.tuples[y]
            data.append(1)
            # relative indices (will be shifted later)
            i.append(0)  # this will always be 0 since we compute a row vector
            j.append(index)

        return data, i, j


class PreviousWord(FeatureFunction):
    name = 'previousword-f106'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        prev_word = extract_prev_word(**kwargs)
        if not prev_word:
            return data, i, j
        tag = kwargs['y']
        tup = (prev_word, tag)
        if tup in self.tuples:
            index = self.tuples[tup]
            data.append(1)
            i.append(0)
            j.append(index)

        return data, i, j


class NextWord(FeatureFunction):
    name = 'nextword-f107'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        next_word = extract_next_word(**kwargs)
        if not next_word:
            return data, i, j
        tag = kwargs['y']
        tup = (next_word, tag)
        if tup in self.tuples:
            index = self.tuples[tup]
            data.append(1)
            i.append(0)
            j.append(index)

        return data, i, j


class CapitalStart(FeatureFunction):
    name = 'starting_capital'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        tag = kwargs['y']
        if tag not in self.tuples:
            return data, i, j
        word = extract_current_word(**kwargs)
        if word[0].isupper():
            data.append(1)
            i.append(0)
            j.append(0)

        return data, i, j


class Capital(FeatureFunction):
    name = 'capital_inside'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        tag = kwargs['y']
        word = extract_current_word(**kwargs)
        if tag not in self.tuples or len(word) <= 1:
            return data, i, j
        for c in word[1:]:
            if c.isupper():
                data.append(1)
                i.append(0)
                j.append(0)
                break

        return data, i, j


class Numeric(FeatureFunction):
    name = 'number_inside'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        tag = kwargs['y']
        if tag not in self.tuples:
            return data, i, j
        word = extract_current_word(**kwargs)
        for c in word:
            if c.isdigit():
                data.append(1)
                i.append(0)
                j.append(0)
                break

        return data, i, j


class Hyphen(FeatureFunction):
    name = 'hyphen_inside'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        tag = kwargs['y']
        if tag not in self.tuples:
            return data, i, j
        word = extract_current_word(**kwargs)
        if '-' in word:
            data.append(1)
            i.append(0)
            j.append(0)

        return data, i, j


class PrePreWord(FeatureFunction):
    name = 'pre_pre_word'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        prev_word = extract_pp_word(**kwargs)
        if not prev_word:
            return data, i, j
        tag = kwargs['y']
        tup = (prev_word, tag)
        if tup in self.tuples:
            index = self.tuples[tup]
            data.append(1)
            i.append(0)
            j.append(index)

        return data, i, j


class NextNextWord(FeatureFunction):
    name = 'next_next_word'

    def __call__(self, **kwargs):
        data, i, j = [], [], []
        next_word = extract_nn_word(**kwargs)
        if not next_word:
            return data, i, j
        tag = kwargs['y']
        tup = (next_word, tag)
        if tup in self.tuples:
            index = self.tuples[tup]
            data.append(1)
            i.append(0)
            j.append(index)

        return data, i, j


def build_features(x, y, sentence, features_functions, i_shift=0):
    """applies predefined functions on one sample
    returns shifted (corrected) data,i,j

    i_shift is the row number, calling function is responsible setting it
    """
    data, i, j = [], [], []
    for ind, f in enumerate(features_functions):
        cur_data, cur_i, cur_j = f(x=x, y=y, sentence=sentence)
        # shift j (only if not first feature)
        # sum previous sizes
        j_shift = 0
        for ind_ in range(ind-1, -1, -1):
            j_shift += features_functions[ind_].m
        cur_j = [x+j_shift for x in cur_j]
        cur_i = [x+i_shift for x in cur_i]
        # append
        data += cur_data
        i += cur_i
        j += cur_j

    return data, i, j


def build_y_x_matrix(X, poss, sentences, feature_functions):
    """build y_x feature matrix in coo format

    output shape: (|Y|*|X|, m)
    """
    # get size
    m = 0
    for f in feature_functions:
        m += f.m

    matrix_shape = (len(X)*len(poss), m)

    # build features matrix
    current_row = 0
    data, row, col = [], [], []
    data_help, row_help, col_help = [], [], []
    help_j = 0

    for i, x in enumerate(X):
        for j, pos in enumerate(poss):
            cur_data, cur_i, cur_j = build_features(x, pos, sentences[x[2]], feature_functions, i_shift=current_row)

            # append
            data += cur_data
            row += cur_i
            col += cur_j
            current_row += 1

            data_help += [1]
            row_help += [i]
            col_help += [help_j]

            help_j += 1

    matrix = csr_matrix((data, (row, col)), shape=matrix_shape)
    help_matrix = csr_matrix((data_help, (row_help, col_help)), shape=(len(X), help_j))

    return matrix, help_matrix


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
        cur_data, cur_i, cur_j = build_features(x, y[i], sentences[x[2]], feature_fncs, i_shift=i)
        # append
        data += cur_data
        row += cur_i
        col += cur_j

    matrix = csr_matrix((data, (row, col)), shape=matrix_shape)

    return matrix


def init_callable_features(tags, params, preprocess_dict):
    """
    takes output of preprocess which is a list of tuples for each feature
    e.g., for Bigram the module should provide a list [('DT', 'NN')..]
    :return:
    """
    # init callable features
    callables = []
    total_features = 0
    for f in params.features_functions:
        if f.name in preprocess_dict:
            tuples = preprocess_dict[f.name]
            tuples_dict = {x: i for i, x in enumerate(tuples)}
            callable = f(tags, tuples_dict)
            callables.append(callable)
            total_features += callable.m
            print("Loaded feature function:" + f.name + ',' + "m=%d" % callable.m )

    print("Total features chosen: %d" % total_features)
    return callables
