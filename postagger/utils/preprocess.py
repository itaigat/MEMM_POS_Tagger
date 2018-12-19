import collections
from postagger.utils.common import pickle_load, pickle_save, timeit

preprocess_dict = {
    # needed here only for the keys
    'wordtag-f100': [('the', 'DT')],
    'suffix-f101': [('ing', 'VBG')],
    'prefix-f102': [('pre', 'NN')],
    'trigram-f103': [('DT', 'JJ', 'NN')],
    'bigram-f104': [('DT', 'JJ')],
    'unigram-f105': ['DT'],
    'previousword-f106': [('the', 'NNP')],
    'nextword-f107': [('the', 'VB')],
    'starting_capital': ['DT'],
    'capital_inside': ['NN'],
    'number_inside': ['CD']
}

# default values
MIN_OCCURRENCE = 50
TOP = 100
TOP_COMMON_WORDS = 100


class Preprocessor:

    def __init__(self, iterable_sentences, top_words=None):
        self.X, self.y, self.sentences = [], [], []
        self.words = []
        for tuples, tags, sentence in iterable_sentences:
            for i in range(len(tuples)):
                self.X.append(tuples[i])
                self.y.append(tags[i])
                self.words.append(sentence[tuples[i][3]])

            self.sentences.append(sentence)
        # init dict
        self.pdict = {i: [] for i in preprocess_dict.keys()}
        self.min_occurrence = {i: MIN_OCCURRENCE for i in preprocess_dict.keys()}
        self.top = {i: TOP for i in preprocess_dict.keys()}
        self.common_words = self.get_common_words(top_words)

    def count(self):
        """
        produces the preprocess_dict
        """
        N = len(self.X)
        for i in range(N):
            self.extract_features(self.X[i], self.y[i], self.sentences[self.X[i][2]])
            # extract features

    def count_features_num(self):
        # helper for extract features
        total = 0
        for key, value in self.pdict.items():
            m = len(value)
            print("Preprocessor: features extracted for " + key + ' %d' % m)
            total += m
        print("Preprocessor: total features extracted: %d" % total)

    def summarize_counts(self, method, dict):
        """
        summarize the counts, using one of the methods
        :param method: 'cut' or 'top'
        :param dict: parameters dict
        :return: preprocess_dict, input for classifier
        """

        if method == 'cut':
            return self.cut(dict)
        elif method == 'top':
            return self.top_occ(dict)

    def get_common_words(self, top=None):
        if top is None:
            top = TOP_COMMON_WORDS

        all_train_words = self.words
        counter_obj = collections.Counter(all_train_words)
        most_common_words = [x[0] for x in counter_obj.most_common(top)]
        return most_common_words

    def cut(self, min_occurence=None):
        """process the dict results, apply minimum occurrence option"""
        # first each of the list possibly contain duplicate tuples
        # we will count them and cast them into sets
        pdict_cut = {i: [] for i in preprocess_dict.keys()}
        print("Summarizing counts")
        if min_occurence is None:
            min_occurence = self.min_occurrence
        for feature, tuple_list in self.pdict.items():
            counts_dict = collections.Counter(tuple_list)
            # extract relevant counts
            for key, value in counts_dict.items():
                if value > min_occurence[feature]:
                    pdict_cut[feature].append(key)

        return pdict_cut

    def top_occ(self, top=None):
        """Top features by occurrence"""
        pdict_top = {i: [] for i in preprocess_dict.keys()}
        print("Preprocessor: summarizing counts")
        if top is None:
            top = self.top
        for feature, tuple_list in self.pdict.items():
            counts_dict = collections.Counter(tuple_list)
            # extract relevant counts
            for elem, count in counts_dict.most_common(top[feature]):
                pdict_top[feature].append(elem)

        return pdict_top

    def extract_features(self, x, y, sentence):
        # common words
        word_id = x[3]
        word = sentence[word_id]
        if word in self.common_words:
            self.f100(x, y, sentence)
        # rare words
        else:
            self.f101(x, y, sentence)
            self.f102(x, y, sentence)
            self.f_capital_inside(x, y, sentence)
            self.f_starting_capital(x, y, sentence)
            self.f_number_inside(x, y, sentence)
        # all words
        self.f103(x, y, sentence)
        self.f104(x, y, sentence)
        self.f105(x, y, sentence)
        self.f106(x, y, sentence)
        self.f107(x, y, sentence)

    def f100(self, x, y, sentence):
        word_id = x[3]
        word = sentence[word_id]
        tag = y
        tup = (word, tag)
        self.pdict['wordtag-f100'].append(tup)

    def f101(self, x, y, sentence):
        # suffix
        word_id = x[3]
        word = sentence[word_id]
        tag = y
        if len(word) >= 4:
            suffixes = [(word[-i:], tag) for i in range(1, 5)]
            self.pdict['suffix-f101'] = self.pdict['suffix-f101'] + suffixes

    def f102(self, x, y, sentence):
        # prefix
        word_id = x[3]
        word = sentence[word_id]
        tag = y
        if len(word) >= 4:
            prefixes = [(word[:i], tag) for i in range(1, 5)]
            self.pdict['prefix-f102'] = self.pdict['prefix-f102'] + prefixes

    def f103(self, x, y, sentence):
        # trigram
        word_id = x[3]
        if word_id >= 2:
            pre_pre_tag = x[0]
            pre_tag = x[1]
            tag = y
            tup = (pre_pre_tag, pre_tag, tag)
            self.pdict['trigram-f103'].append(tup)

    def f104(self, x, y, sentence):
        # bigram
        word_id = x[3]
        if word_id >= 1:
            pre_tag = x[1]
            tag = y
            tup = (pre_tag, tag)
            self.pdict['bigram-f104'].append(tup)

    def f105(self, x, y, sentence):
        # unigram
        tag = y
        self.pdict['unigram-f105'].append(tag)

    def f106(self, x, y, sentence):
        # previous word
        word_id = x[3]
        if word_id >= 1:
            prev_word_id = word_id-1
            prev_word = sentence[prev_word_id]
            tag = y
            tup = (prev_word, tag)
            self.pdict['previousword-f106'].append(tup)

    def f107(self, x, y, sentence):
        # next word
        word_id = x[3]
        if word_id < len(sentence)-1:
            next_word_id = word_id + 1
            next_word = sentence[next_word_id]
            tag = y
            tup = (next_word, y)
            self.pdict['nextword-f107'].append(tup)

    def f_starting_capital(self, x, y, sentence):
        word_id = x[3]
        word = sentence[word_id]
        if word[0].isupper():
            tag = y
            self.pdict['starting_capital'].append(tag)

    def f_capital_inside(self, x, y, sentence):
        word_id = x[3]
        word = sentence[word_id]
        for w in word[1:]:
            if w.isupper():
                tag = y
                self.pdict['capital_inside'].append(tag)
                break

    def f_number_inside(self, x, y, sentence):
        word_id = x[3]
        word = sentence[word_id]
        for w in word:
            if w.isdigit():
                tag = y
                self.pdict['number_inside'].append(tag)
                break

@timeit
def load_save_preprocessed_data(filename, iterable_sentences, top_words, load):
    """return a preprocessor which applied count()"""

    # try loading
    if load:
        p = pickle_load(filename)
        if p is not None:
            return p

    # save and return loaded
    p = Preprocessor(iterable_sentences, top_words)
    p.count()
    p.count_features_num()
    ret = pickle_save(p, filename)
    if ret is None:
        return ret
    print("Preprocessor object saved successfuly")
    return p


if __name__ == '__main__':
    # test
    """
    import os
    path = get_data_path(os.path.realpath('../../resources/train.wtag'))
    iterable_sentences = CompData(path)
    p = Preprocessor(iterable_sentences)
    import time; t = time.time()
    p.count()
    print("Counting took: %f s" % (time.time() - t))
    for k, v in p.pdict.items():
        print(k)
        print(v)
    pdict_cut = p.cut()
    for k, v in pdict_cut.items():
        print(k)
        print(v)

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
