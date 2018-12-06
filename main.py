import os
from os.path import join, dirname

from src.utils.decoder import CompData
from src.utils.classifier import MaximumEntropyClassifier
from tests.test_clf import *

data_file = 'train_dev_500.wtag'

if os.name == 'nt':
    path = join(dirname(os.getcwd()), 'resources', data_file)
else:
    path = join(os.getcwd(), 'resources', data_file)

iterable_sentences = CompData(path)
clf = MaximumEntropyClassifier(iterable_sentences)
# tests (on train_dev, and unigram features only)
#test_first_loss(clf, clf.feature_matrix, clf.X, clf.y, clf.sentences)
#test_first_grad(clf, clf.feature_matrix, clf.X, clf.y, clf.sentences)
clf.fit(max_iter=1)
