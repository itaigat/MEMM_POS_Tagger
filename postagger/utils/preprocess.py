from postagger.utils.common import timeit, get_data_path
from postagger.utils.decoder import CompData

preprocess_dict = {
    # used only for the keys
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

    def __init__(self, iterable_sentences):
        self.X, self.y, self.sentences = [], [], []
        for tuples, tags, sentence in iterable_sentences:
            for i in range(len(tuples)):
                self.X.append(tuples[i])
                self.y.append(tags[i])
            self.sentences.append(sentence)
        self.pdict = {i: [] for i in preprocess_dict.keys()}

    def count(self):
        """
        produces the preprocess_dict
        """
        N = len(self.X)
        for i in range(N):
            self.extract_features(self.X[i], self.y[i], self.sentences[self.X[i][2]])
            # extract features

    def extract_features(self, x, y, sentence):
        self.f100(x, y, sentence)

    def f100(self, x, y, sentence):
        word_id = x[3]
        word = sentence[word_id]
        tag = y
        tup = (word, tag)
        self.pdict['wordtag-f100'].append(tup)

    # TODO: add all other features
    # TODO: minimum occurrence option


if __name__ == '__main__':
    # test
    import os
    path = get_data_path(os.path.realpath('../../resources/train.wtag'))
    iterable_sentences = CompData(path)
    p = Preprocessor(iterable_sentences)
    p.count()
    for key,value in p.pdict.items():
        print(key)
        print(value)