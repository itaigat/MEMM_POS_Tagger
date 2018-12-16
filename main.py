import sys

from postagger.utils.classifier import MaximumEntropyClassifier
from postagger.utils.common import timeit, get_data_path
from postagger.utils.decoder import CompData


@timeit
def main(argv):
    # comp_path = get_data_path('comp.wtag')
    # comp_sentences = CompData(comp_path, comp=True)
    # for bla in comp_sentences:
    #     print(bla)

    train_path = get_data_path('train.wtag')
    train_sentences = CompData(train_path)
    clf = MaximumEntropyClassifier(train_sentences)
    clf.fit(reg=10, verbose=1)

    test_path = get_data_path('test.wtag')
    test_sentences = CompData(test_path)
    t_predict = clf.predict(test_sentences)
    print(t_predict)

    # tests (pass on train_dev and unigram features only): CURRENTLY BREAKS (fix of numeric issues)
    # run_clf_tests(clf, clf.feature_matrix, clf.X, clf.y, clf.sentences)
    # clf.fit(reg=10, verbose=1)


if __name__ == '__main__':
    main(sys.argv[1:])
