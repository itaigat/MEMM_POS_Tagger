from src.utils.features import unigram_f
from src.utils.features import bigram_f


class Params:
    feature_lst_functions = [unigram_f]  # List of feature functions
    feature_functions = [bigram_f]
