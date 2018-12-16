from postagger.utils.common import timeit, get_data_path
from postagger.utils.decoder import CompData
import collections

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

class Preprocessor:

    def __init__(self, iterable_sentences, min_occurence=None):
        self.X, self.y, self.sentences = [], [], []
        for tuples, tags, sentence in iterable_sentences:
            for i in range(len(tuples)):
                self.X.append(tuples[i])
                self.y.append(tags[i])
            self.sentences.append(sentence)
        # init dict
        self.pdict = {i: [] for i in preprocess_dict.keys()}
        self.min_occurence = {i: 6 for i in preprocess_dict.keys()}  # TODO: get from params
        self.pdict_cut = {i: [] for i in preprocess_dict.keys()}

    def count(self):
        """
        produces the preprocess_dict
        """
        N = len(self.X)
        for i in range(N):
            self.extract_features(self.X[i], self.y[i], self.sentences[self.X[i][2]])
            # extract features
        self.summarize()
        return self.pdict_cut

    def summarize(self):
        """process the dict results, apply minimum occurence option"""
        # first each of the list possibly contain duplicate tuples
        # we will count them and cast them into sets
        print("Summarizing counts")
        for feature, tuple_list in self.pdict.items():
            print(feature)
            counts_dict = collections.counter(tuple_list)
            # extract relevant counts
            for key, value in counts_dict.items():
                if value > self.min_occurence[feature]:
                    self.pdict_cut[feature].append(key)


    def extract_features(self, x, y, sentence):
        self.f100(x, y, sentence)
        self.f102(x, y, sentence)

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
            suffixes = [(word[-i:],tag) for i in range(1,5)]
            self.pdict['suffix-f101'] = self.pdict['suffix-f101'] + suffixes

    def f102(self, x, y, sentence):
        # prefix
        word_id = x[3]
        word = sentence[word_id]
        tag = y
        if len(word) >= 4:
            suffixes = [(word[:i],tag) for i in range(1,5)]
            self.pdict['prefix-f102'] = self.pdict['prefix-f102'] + suffixes

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
        for w in word[1:]:
            if w.digit():
                tag = y
                self.pdict['capital_inside'].append(tag)
                break

    # TODO: minimum occurrence option


if __name__ == '__main__':
    # test
    import os
    path = get_data_path(os.path.realpath('../../resources/train.wtag'))
    iterable_sentences = CompData(path)
    p = Preprocessor(iterable_sentences)
    p.count()
    for key, value in p.pdict.items():
        print(key)
        print(value)
