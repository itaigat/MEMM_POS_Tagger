from src.utils.decoder import Comp
from src.utils.common import create_all_features
from src.utils.params import Params
from os.path import join, dirname

import numpy as np
import os


# For Windows
comp_dataset = Comp(join(dirname(os.getcwd()), 'resources', 'train_dev.wtag'))

# For linux
# comp_dataset = Comp(join(os.getcwd(), 'resources', 'train.wtag'))

x_lst = []

for tuples, sentence in comp_dataset:
    x_lst.append(create_all_features(Params.feature_functions, Params.feature_lst_functions, tuples, sentence))

x = np.concatenate(tuple(x_lst), axis=0)

print(x.shape)
