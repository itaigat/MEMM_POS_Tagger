from src.utils.decoder import Comp
from src.utils.common import create_all_features
from src.utils.params import Params
from os.path import join, dirname

import os


bla = Comp(join(dirname(os.getcwd()), 'resources', 'train.wtag'))

for tuples, sentence in bla:
    create_all_features(Params.feature_lst_functions, tuples, sentence)

