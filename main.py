import sys
from postagger.utils.decoder import CompData
from postagger.utils.classifier import MaximumEntropyClassifier
from postagger.utils.common import timeit, get_data_path
from tests.test_clf import run_clf_tests


@timeit
def main(argv):
    path = get_data_path('train_dev_500.wtag')
    iterable_sentences = CompData(path)
    clf = MaximumEntropyClassifier(iterable_sentences)
    # tests (pass on train_dev and unigram features only): CURRENTLY BREAKS (fix of numeric issues)
    # run_clf_tests(clf, clf.feature_matrix, clf.X, clf.y, clf.sentences)
    clf.fit(reg=100, max_iter=10, verbose=1)


if __name__ == '__main__':
    main(sys.argv[1:])
