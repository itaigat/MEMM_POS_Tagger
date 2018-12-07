import os
import sys
from os.path import join, dirname

from postagger.utils.decoder import CompData
from postagger.utils.classifier import MaximumEntropyClassifier
from tests.test_clf import *


def main(argv):
    data_file = 'train_dev_50.wtag'

    if os.name == 'nt':
        path = join(dirname(os.getcwd()), 'resources', data_file)
    else:
        path = join(os.getcwd(), 'resources', data_file)

    iterable_sentences = CompData(path)
    clf = MaximumEntropyClassifier(iterable_sentences)
    # tests (pass on train_dev and unigram features only)
    #test_first_loss(clf, clf.feature_matrix, clf.X, clf.y, clf.sentences)
    #test_first_grad(clf, clf.feature_matrix, clf.X, clf.y, clf.sentences)
    clf.fit(reg=0, max_iter=1)

if __name__ == '__main__':
    main(sys.argv[1:])
