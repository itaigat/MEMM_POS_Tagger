import os
from os.path import join, dirname

import numpy as np

from src.utils.common import create_all_features
from src.utils.decoder import Comp, CompData, create_dataset
from src.utils.params import Params

# For Windows
#comp_dataset = Comp(join(dirname(os.getcwd()), 'resources', 'train_dev.wtag'))

# For linux
comp_dataset = Comp(join(os.getcwd(), 'resources', 'train_dev.wtag'))

# x_lst = []
#
# for tuples, sentence in comp_dataset:
#     print(sentence)
#     print(tuples)
#     x_lst.append(create_all_features(Params.feature_functions, Params.feature_lst_functions, tuples, sentence))
#
# x = np.concatenate(tuple(x_lst), axis=0)
# print(x)
# print(x.shape)


iterable_sentences = CompData(join(os.getcwd(), 'resources', 'train_dev.wtag'))
X, y, sentences_tuple = create_dataset(iterable_sentences)

print("Sentences:" + str(sentences_tuple[0]))
print("Sizes: " + str(sentences_tuple[1]))


