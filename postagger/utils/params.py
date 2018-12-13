from postagger.utils.features import Unigram, Bigram, Trigram, Capital, CapitalStart


class Params:
    features_functions = [Unigram, Bigram, Trigram, Capital, CapitalStart]
