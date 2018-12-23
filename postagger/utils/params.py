from postagger.utils.features import Wordtag, Suffix, Prefix, Unigram, Bigram, Trigram, PreviousWord, \
    NextWord, Capital, CapitalStart, Numeric, Hyphen, PrePreWord, NextNextWord


class Params:
    features_functions = [Wordtag, Suffix, Prefix, Unigram, Bigram, Trigram, PreviousWord, NextWord,
                          Capital, CapitalStart, Numeric, Hyphen, PrePreWord, NextNextWord]

    preprocess_dict = {
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
        'number_inside': ['CD'],
        'hyphen_inside': ['NN'],
        'pre_pre_word': ['IN'],
        'next_next_word': ['DT']
    }
