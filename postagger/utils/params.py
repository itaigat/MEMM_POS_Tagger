from postagger.utils.features import Wordtag, Suffix, Prefix, Unigram, Bigram, Trigram, PreviousWord, \
    NextWord, Capital, CapitalStart, Numeric


class Params:
    features_functions = [Wordtag, Suffix, Prefix, Unigram, Bigram, Trigram, PreviousWord, NextWord,
                          Capital, CapitalStart, Numeric]
